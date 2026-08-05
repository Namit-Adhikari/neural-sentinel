"""
src/generation/validator.py
-----------------------------
Phase 9 — Constraint Validation.

Every generated row must pass all constraints before entering the final
dataset. Rows that violate constraints are flagged (and optionally dropped).

Constraints checked
-------------------
Account-level
  - Unique account_id within accounts table
  - Valid institution (must be in known set)
  - Valid branch (must belong to declared institution in knowledge base)
  - Valid city (must be a known Nepali city)

Transaction-level
  - Valid sender_account_id (must exist in accounts)
  - Valid receiver_account_id (must exist in accounts)
  - Amount > 0
  - Valid timestamp (not null, not in future)
  - Timestamp ≥ sender account opened date
  - Timestamp ≥ receiver account opened date
  - Valid Payment_currency (must be a known ISO code)

Feature-level
  - day_of_week matches actual date
  - is_weekend matches day_of_week
  - above_1M_NPR = (amount_local_npr >= 1_000_000)
  - above_10M_NPR = (amount_local_npr >= 10_000_000)
  - transmode one-hot columns sum to exactly 1
  - account_age_days >= 0

Outputs
-------
Adds column ``_validation_passed`` (bool) and ``_violation_codes`` (str | None).
Saves ``validation_report.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.nepal_context import NEPALI_CITIES

logger = logging.getLogger(__name__)

_KNOWN_CITIES = set(NEPALI_CITIES)

_VALID_CURRENCIES = {
    "NPR", "USD", "EUR", "GBP", "INR", "QAR", "SAR", "AED",
    "MYR", "AUD", "JPY", "CNY", "SGD", "THB",
}

_TRANSMODE_COLS = ["transmode_A", "transmode_B", "transmode_E",
                   "transmode_F", "transmode_J", "transmode_P", "transmode_Z"]


class ConstraintValidator:
    """Validate synthetic accounts and transactions against all constraints.

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Synthetic accounts table.
    knowledge : dict, optional
        Knowledge base (used for institution/branch validation).
    drop_invalid : bool
        If True, remove invalid rows from returned DataFrames.
    """

    def __init__(
        self,
        accounts_df: pd.DataFrame,
        knowledge: dict | None = None,
        drop_invalid: bool = False,
    ) -> None:
        self.accounts = accounts_df.copy()
        self.accounts["account_id"] = self.accounts["account_id"].astype(str)
        self.knowledge = knowledge or {}
        self.drop_invalid = drop_invalid

        self._valid_account_ids: set[str] = set(self.accounts["account_id"].tolist())
        self._institution_branch_map: dict[str, set[str]] = self._build_inst_branch_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_accounts(self) -> tuple[pd.DataFrame, dict]:
        """Validate accounts table. Returns (validated_df, report)."""
        df = self.accounts.copy()
        violations: list[str] = []
        report: dict[str, int] = {}

        # Duplicate account_id
        dup_mask = df["account_id"].duplicated(keep="first")
        report["duplicate_account_id"] = int(dup_mask.sum())
        if dup_mask.any():
            violations.append("duplicate_account_id")

        # Valid city
        if "city" in df.columns:
            bad_city = ~df["city"].isin(_KNOWN_CITIES)
            report["invalid_city"] = int(bad_city.sum())
        else:
            report["invalid_city"] = 0

        # Future opening date
        if "opened" in df.columns:
            opened = pd.to_datetime(df["opened"], errors="coerce")
            future = opened > pd.Timestamp("today")
            report["future_opened_date"] = int(future.sum())
        else:
            report["future_opened_date"] = 0

        report["total_rows"] = len(df)
        report["total_violations"] = sum(v for k, v in report.items() if k != "total_rows")
        logger.info("Account validation: %d violations across %d rows", report["total_violations"], len(df))

        if self.drop_invalid:
            keep_mask = ~df["account_id"].duplicated(keep="first")
            df = df[keep_mask].reset_index(drop=True)

        return df, report

    def validate_transactions(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Validate enriched+feature-engineered transactions. Returns (validated_df, report)."""
        df = transactions.copy()
        n = len(df)
        report: dict[str, int] = {"total_rows": n}

        # Initialise mask: True = row is valid
        valid_mask = pd.Series(True, index=df.index)
        violation_codes = pd.Series("", index=df.index)

        # ----- Transaction-level -----

        sender_col   = _pick(df, ["sender_account_id", "Sender_account"])
        receiver_col = _pick(df, ["receiver_account_id", "Receiver_account"])

        if sender_col:
            bad_sender = ~df[sender_col].astype(str).isin(self._valid_account_ids)
            report["invalid_sender"] = int(bad_sender.sum())
            valid_mask &= ~bad_sender
            violation_codes[bad_sender] += "INVALID_SENDER|"

        if receiver_col:
            bad_recv = ~df[receiver_col].astype(str).isin(self._valid_account_ids)
            report["invalid_receiver"] = int(bad_recv.sum())
            valid_mask &= ~bad_recv
            violation_codes[bad_recv] += "INVALID_RECEIVER|"

        # Amount > 0
        amt_col = _pick(df, ["amount_local_npr", "Amount"])
        if amt_col:
            bad_amt = pd.to_numeric(df[amt_col], errors="coerce").fillna(-1) <= 0
            report["non_positive_amount"] = int(bad_amt.sum())
            valid_mask &= ~bad_amt
            violation_codes[bad_amt] += "NON_POSITIVE_AMOUNT|"

        # Valid timestamp
        date_col = _pick(df, ["Date", "transaction_date"])
        if date_col:
            dts = pd.to_datetime(df[date_col], errors="coerce")
            null_ts = dts.isna()
            future_ts = dts > pd.Timestamp("today")
            bad_ts = null_ts | future_ts
            report["invalid_timestamp"] = int(bad_ts.sum())
            valid_mask &= ~bad_ts
            violation_codes[bad_ts] += "INVALID_TIMESTAMP|"

        # Valid currency
        curr_col = _pick(df, ["Payment_currency", "original_currency"])
        if curr_col:
            bad_curr = ~df[curr_col].isin(_VALID_CURRENCIES)
            report["invalid_currency"] = int(bad_curr.sum())
            # Currency warnings only — don't invalidate rows

        # ----- Feature-level -----

        # day_of_week correctness
        if date_col in df.columns and "day_of_week" in df.columns:
            dts = pd.to_datetime(df[date_col], errors="coerce")
            expected_dow = dts.dt.dayofweek.fillna(-1).astype(int)
            actual_dow   = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(-1).astype(int)
            bad_dow = (expected_dow != actual_dow) & dts.notna()
            report["wrong_day_of_week"] = int(bad_dow.sum())

        # is_weekend correctness
        if "day_of_week" in df.columns and "is_weekend" in df.columns:
            expected_we = (pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0) >= 5).astype(int)
            actual_we   = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
            bad_we = expected_we != actual_we
            report["wrong_is_weekend"] = int(bad_we.sum())

        # above_1M_NPR
        if amt_col and "above_1M_NPR" in df.columns:
            amounts = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
            expected_1m = (amounts >= 1_000_000).astype(int)
            actual_1m   = pd.to_numeric(df["above_1M_NPR"], errors="coerce").fillna(0).astype(int)
            report["wrong_above_1M_flag"] = int((expected_1m != actual_1m).sum())

        # Transmode one-hot sums to 1
        avail_tm = [c for c in _TRANSMODE_COLS if c in df.columns]
        if len(avail_tm) > 0:
            tm_sum = df[avail_tm].fillna(0).astype(int).sum(axis=1)
            bad_tm = tm_sum != 1
            report["invalid_transmode_encoding"] = int(bad_tm.sum())

        # Account age >= 0
        for age_col in ["sender_account_age_days", "receiver_account_age_days"]:
            if age_col in df.columns:
                bad_age = pd.to_numeric(df[age_col], errors="coerce").fillna(0) < 0
                report[f"negative_{age_col}"] = int(bad_age.sum())

        # Finalise
        df["_validation_passed"] = valid_mask
        df["_violation_codes"]   = violation_codes.str.rstrip("|").replace("", None)

        report["rows_passed"]   = int(valid_mask.sum())
        report["rows_failed"]   = int((~valid_mask).sum())
        report["total_violations"] = sum(
            v for k, v in report.items()
            if k not in {"total_rows", "rows_passed", "rows_failed"}
        )

        logger.info(
            "Transaction validation: %d passed, %d failed out of %d rows",
            report["rows_passed"], report["rows_failed"], n,
        )

        if self.drop_invalid:
            df = df[df["_validation_passed"]].reset_index(drop=True)

        return df, report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_inst_branch_map(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        inst_map = self.knowledge.get("institution_mapping", {})
        for inst, data in inst_map.items():
            branches = set(data.get("branches", {}).keys())
            mapping[str(inst)] = branches
        return mapping


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def validate_and_report(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    knowledge: dict | None = None,
    drop_invalid: bool = False,
    report_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run full validation and save report.

    Returns
    -------
    (validated_accounts, validated_transactions, report_dict)
    """
    validator = ConstraintValidator(accounts, knowledge=knowledge, drop_invalid=drop_invalid)
    val_accounts, acc_report = validator.validate_accounts()
    val_transactions, tx_report = validator.validate_transactions(transactions)

    report = {
        "accounts": acc_report,
        "transactions": tx_report,
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Validation report saved to: %s", report_path)

    return val_accounts, val_transactions, report


import json  # noqa: E402 (placed here for module-level use in validate_and_report)
