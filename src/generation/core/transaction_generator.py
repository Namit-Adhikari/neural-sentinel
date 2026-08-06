"""
src/generation/core/transaction_generator.py
---------------------------------------------
Phase 5 — Core Transaction Generation.

Uses SMOTE to learn the distribution of the cleaned transactions and
generate ``n`` new synthetic transaction rows.

The output DataFrame preserves the schema of ``data/interim/transactions.parquet``
(the canonical cleaned transactions produced by phase2_cleaning) so that
downstream phases (AML injection, enrichment, feature engineering) work
without modification.

Default behaviour
-----------------
- Seed data  : ``data/interim/transactions.parquet``
- Output     : ``data/generated/synthetic_transactions.csv``
  (written automatically when ``merge_with_original=True``)
- Merge      : original rows prepended to synthetic rows (``data_source``
  column marks each row as 'original' or 'synthetic')
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]

_DEFAULT_SEED_PATH   = _PROJECT_ROOT / "data" / "interim"   / "transactions.parquet"
_DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "data" / "generated" / "synthetic_transactions.csv"


class TransactionGenerator:
    """Generate synthetic transactions using SMOTE.

    Parameters
    ----------
    knowledge : dict
        Knowledge base (kept for API compatibility; not used by SMOTE).
    accounts_df : pd.DataFrame
        Synthetic accounts (kept for API compatibility; not used by SMOTE).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        knowledge: dict,
        accounts_df: pd.DataFrame,
        seed: int = 42,
    ) -> None:
        self.knowledge = knowledge
        self.accounts = accounts_df
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n: int,
        output_path: Path | str | None = None,
        merge_with_original: bool = True,
        original_data_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """Generate *n* synthetic transactions using SMOTE.

        Parameters
        ----------
        n : int
            Number of synthetic transaction rows to generate.
        output_path : Path | str | None
            Where to save the output CSV.  Defaults to
            ``data/generated/synthetic_transactions.csv`` when
            ``merge_with_original=True``.
        merge_with_original : bool, default ``True``
            Prepend original seed rows to the synthetic output.
            A ``data_source`` column ('original' / 'synthetic') is added.
        original_data_path : Path | str | None
            Path to the seed parquet/CSV.  Defaults to
            ``data/interim/transactions.parquet``.

        Returns
        -------
        pd.DataFrame
            Transactions with columns matching the canonical seed schema,
            plus ``data_source`` when ``merge_with_original=True``.
        """
        from src.generation.synthesizers.smote_generator import SMOTEGenerator

        # ── Resolve seed path ──────────────────────────────────────────
        seed_path = Path(original_data_path) if original_data_path else _DEFAULT_SEED_PATH
        if not seed_path.exists():
            raise FileNotFoundError(
                f"Seed transactions file not found: {seed_path}\n"
                "Run phase2_cleaning.ipynb first to produce this file."
            )

        logger.info("Phase 5 · Loading seed transactions from: %s", seed_path)
        if seed_path.suffix == ".parquet":
            seed_df = pd.read_parquet(seed_path)
        else:
            seed_df = pd.read_csv(seed_path)
        logger.info("Seed data: %d rows, %d columns", len(seed_df), len(seed_df.columns))

        # ── Fit + generate ─────────────────────────────────────────────
        gen = SMOTEGenerator(config={"k_neighbors": 5, "random_state": self.seed})
        gen.fit(seed_df)

        logger.info("Generating %d synthetic transactions via SMOTE...", n)
        synthetic_df = gen.generate(num_rows=n)
        logger.info("SMOTE generation complete: %d rows", len(synthetic_df))

        # ── Ensure required downstream columns exist ───────────────────
        # AMLPatternInjector (Phase 6) and TransactionEnricher (Phase 7)
        # need these columns.  If SMOTE dropped or renamed them, restore defaults.
        synthetic_df = self._ensure_required_columns(synthetic_df)

        # ── Merge ──────────────────────────────────────────────────────
        if merge_with_original:
            seed_df = seed_df.copy()
            seed_df = self._ensure_required_columns(seed_df)
            seed_df["data_source"] = "original"
            synthetic_df["data_source"] = "synthetic"
            df = pd.concat([seed_df, synthetic_df], ignore_index=True)
            logger.info(
                "Merged: %d original + %d synthetic = %d total rows",
                len(seed_df), len(synthetic_df), len(df),
            )
        else:
            df = synthetic_df

        # ── Save ───────────────────────────────────────────────────────
        out = Path(output_path) if output_path else (
            _DEFAULT_OUTPUT_PATH if merge_with_original else None
        )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix == ".parquet":
                df.to_parquet(out, index=False)
            else:
                df.to_csv(out, index=False)
            logger.info("Saved transactions to: %s", out)

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add any columns the downstream pipeline requires that are missing."""
        # Normalise sender/receiver column names
        if "sender_account_id" in df.columns and "Sender_account" not in df.columns:
            df = df.rename(columns={"sender_account_id": "Sender_account"})
        if "receiver_account_id" in df.columns and "Receiver_account" not in df.columns:
            df = df.rename(columns={"receiver_account_id": "Receiver_account"})

        # Columns the AML injector and enricher reference
        defaults: dict[str, object] = {
            "is_fraud":           0,
            "fraud_type":         None,
            "aml_risk_indicator": 0,
            "is_cross_border":    0,
            "remittance_corridor": None,
            "fx_rate_to_npr":     1.0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        return df
