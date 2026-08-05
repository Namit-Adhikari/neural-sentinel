"""
src/generation/dataset_builder.py
------------------------------------
Phase 10 — Final Dataset Construction.

Assembles the four final output CSV files and three JSON reports
from all upstream phases.

Outputs (saved to data/interim/ by default)
-------------------------------------------
synthetic_accounts.csv              — clean accounts
synthetic_transactions.csv          — enriched + validated + fraud-labelled transactions
graph_edges.csv                     — lightweight edge list for graph agents
ml_features.csv                     — ML-ready feature set with is_suspicious_tx label

schema_report.json                  — Phase 1 schema statistics
validation_report.json              — Phase 9 constraint violations
generation_log.json                 — timing, row counts, fraud rates per scenario
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns for each output file
# ---------------------------------------------------------------------------

_GRAPH_EDGE_COLS = [
    "row_index",
    "Sender_account",
    "Receiver_account",
    "amount_local_npr",
    "Date",
    "Time",
]

_ML_FEATURE_COLS = [
    "Date", "Time", "Sender_account", "Receiver_account",
    "amount_local_npr", "log_amount", "amount_zscore",
    "above_1M_NPR", "above_10M_NPR",
    "hour_of_day", "day_of_week", "is_weekend", "month",
    "sender_country_risk", "receiver_country_risk",
    "cross_border_flag", "currency_mismatch",
    "velocity_sum_10tx", "tx_count_10", "tx_count_30",
    "sender_account_age_days", "receiver_account_age_days",
    "sender_is_person", "sender_pep", "sender_sanctions",
    "receiver_pep", "receiver_sanctions",
    "transmode_A", "transmode_B", "transmode_E", "transmode_F",
    "transmode_J", "transmode_P", "transmode_Z",
    "is_suspicious_tx",
]


class DatasetBuilder:
    """Assemble and save all final output files.

    Parameters
    ----------
    output_dir : Path
        Directory where all outputs will be saved.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log: dict[str, Any] = {"start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        accounts: pd.DataFrame,
        transactions: pd.DataFrame,
        schema_report: dict | None = None,
        validation_report: dict | None = None,
    ) -> dict[str, Path]:
        """Build and save all outputs.

        Parameters
        ----------
        accounts : pd.DataFrame
            Validated synthetic accounts.
        transactions : pd.DataFrame
            Fully enriched, feature-engineered, validated transactions.
        schema_report : dict, optional
            Phase 1 schema statistics.
        validation_report : dict, optional
            Phase 9 validation results.

        Returns
        -------
        dict mapping output name → Path
        """
        t0 = time.time()
        paths: dict[str, Path] = {}

        # 1. Accounts
        paths["synthetic_accounts"] = self._save_accounts(accounts)

        # 2. Transactions (full enriched)
        paths["synthetic_transactions"] = self._save_transactions(transactions)

        # 3. Graph edges
        paths["graph_edges"] = self._save_graph_edges(transactions)

        # 4. ML features
        paths["ml_features"] = self._save_ml_features(transactions, accounts)

        # 5. Reports
        if schema_report:
            paths["schema_report"] = self._save_json(schema_report, "schema_report.json")

        if validation_report:
            paths["validation_report"] = self._save_json(
                validation_report, "validation_report.json"
            )

        # 6. Generation log
        self._log["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._log["elapsed_seconds"] = round(time.time() - t0, 2)
        self._log["row_counts"] = {
            "accounts": len(accounts),
            "transactions": len(transactions),
        }
        self._log["fraud_stats"] = self._compute_fraud_stats(transactions)
        paths["generation_log"] = self._save_json(self._log, "generation_log.json")

        logger.info("=== Phase 10 Complete — Dataset built in %.1fs ===", self._log["elapsed_seconds"])
        for name, path in paths.items():
            logger.info("  %-30s → %s", name, path.name)

        return paths

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def _save_accounts(self, df: pd.DataFrame) -> Path:
        path = self.output_dir / "synthetic_accounts.csv"
        clean = df.copy()
        # Drop internal validation columns if present
        clean = clean.drop(columns=["_validation_passed", "_violation_codes"], errors="ignore")
        clean.to_csv(path, index=False)
        logger.info("synthetic_accounts.csv — %d rows", len(clean))
        return path

    def _save_transactions(self, df: pd.DataFrame) -> Path:
        path = self.output_dir / "synthetic_transactions.csv"
        clean = df.copy()
        clean = clean.drop(columns=["_validation_passed", "_violation_codes", "_datetime"], errors="ignore")
        clean.to_csv(path, index=False)
        logger.info("synthetic_transactions.csv — %d rows, %d cols", len(clean), len(clean.columns))
        return path

    def _save_graph_edges(self, df: pd.DataFrame) -> Path:
        path = self.output_dir / "graph_edges.csv"
        edges = pd.DataFrame()

        # Normalise column names
        sender_col   = _pick(df, ["Sender_account", "sender_account_id"])
        receiver_col = _pick(df, ["Receiver_account", "receiver_account_id"])
        date_col     = _pick(df, ["Date", "transaction_date"])
        time_col     = _pick(df, ["Time", "transaction_time"])
        amount_col   = _pick(df, ["amount_local_npr"])

        edges["row_index"]         = range(len(df))
        edges["Sender_account"]    = df[sender_col].values   if sender_col   else ""
        edges["Receiver_account"]  = df[receiver_col].values if receiver_col else ""
        edges["amount_local_npr"]  = df[amount_col].values   if amount_col   else 0.0
        edges["Date"]              = df[date_col].values     if date_col     else ""
        edges["Time"]              = df[time_col].values     if time_col     else ""

        edges.to_csv(path, index=False)
        logger.info("graph_edges.csv — %d edges", len(edges))
        return path

    def _save_ml_features(self, df: pd.DataFrame, accounts: pd.DataFrame) -> Path:
        path = self.output_dir / "ml_features.csv"
        ml = df.copy()

        # Normalise column names to match original ml_features.csv schema
        col_renames = {
            "sender_account_id": "Sender_account",
            "receiver_account_id": "Receiver_account",
            "transaction_date": "Date",
            "transaction_time": "Time",
            "is_fraud": "is_suspicious_tx",
            "cross_border_flag": "cross_border_flag",
            "is_cross_border": "cross_border_flag",
            "sender_pep_flag": "sender_pep",
            "receiver_pep_flag": "receiver_pep",
        }
        for old, new in col_renames.items():
            if old in ml.columns and new not in ml.columns:
                ml = ml.rename(columns={old: new})

        # Derive sender_is_person from accounts
        if "sender_is_person" not in ml.columns:
            sender_col = _pick(ml, ["Sender_account", "sender_account_id"])
            if sender_col and "is_person" in accounts.columns:
                person_map = accounts.set_index("account_id")["is_person"].to_dict()
                ml["sender_is_person"] = ml[sender_col].astype(str).map(person_map).fillna(1).astype(int)
            else:
                ml["sender_is_person"] = 1

        # Fill missing columns with 0
        for col in _ML_FEATURE_COLS:
            if col not in ml.columns:
                ml[col] = 0

        # Select only the ML feature columns that exist
        out_cols = [c for c in _ML_FEATURE_COLS if c in ml.columns]
        ml[out_cols].to_csv(path, index=False)
        logger.info("ml_features.csv — %d rows, %d cols", len(ml), len(out_cols))
        return path

    def _save_json(self, obj: dict, filename: str) -> Path:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        logger.info("%s saved", filename)
        return path

    def _compute_fraud_stats(self, df: pd.DataFrame) -> dict:
        fraud_col = _pick(df, ["is_fraud", "is_suspicious_tx"])
        if not fraud_col:
            return {}
        labels = df[fraud_col].fillna(0).astype(int)
        stats: dict[str, Any] = {
            "total_rows": len(df),
            "fraud_rows": int(labels.sum()),
            "fraud_rate": round(float(labels.mean()), 6),
        }
        fraud_type_col = _pick(df, ["fraud_type"])
        if fraud_type_col:
            type_counts = df.loc[labels == 1, fraud_type_col].value_counts().to_dict()
            stats["fraud_by_type"] = {str(k): int(v) for k, v in type_counts.items()}
        return stats


# ------------------------------------------------------------------
# Schema Report (Phase 1 statistics)
# ------------------------------------------------------------------

def build_schema_report(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> dict:
    """Build a Phase 1 schema report dict with distribution statistics."""
    report: dict[str, Any] = {}

    for name, df in [("transactions", transactions), ("accounts", accounts)]:
        tbl: dict[str, Any] = {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": {},
        }
        for col in df.columns:
            series = df[col]
            col_info: dict[str, Any] = {
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(float(series.isna().mean() * 100), 2),
            }
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) > 0:
                    col_info["mean"]   = round(float(numeric.mean()), 4)
                    col_info["median"] = round(float(numeric.median()), 4)
                    col_info["std"]    = round(float(numeric.std(ddof=0)), 4)
                    col_info["min"]    = round(float(numeric.min()), 4)
                    col_info["max"]    = round(float(numeric.max()), 4)
                    col_info["q25"]    = round(float(numeric.quantile(0.25)), 4)
                    col_info["q75"]    = round(float(numeric.quantile(0.75)), 4)
                    col_info["q99"]    = round(float(numeric.quantile(0.99)), 4)
            else:
                vc = series.dropna().value_counts()
                col_info["unique_count"] = int(vc.shape[0])
                col_info["top5"] = {str(k): int(v) for k, v in vc.head(5).items()}
            tbl["columns"][col] = col_info
        report[name] = tbl

    return report


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
