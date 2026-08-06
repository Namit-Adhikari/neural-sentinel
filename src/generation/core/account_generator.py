"""
src/generation/core/account_generator.py
-----------------------------------------
Phase 4 — Synthetic Account Generation.

Uses SMOTE to learn the distribution of the original accounts and generate
``n`` new synthetic account rows.  The generated DataFrame has the same
column schema as ``data/original/accounts.csv`` plus a ``data_source``
provenance column.

Default behaviour
-----------------
- Seed data  : ``data/original/accounts.csv``
- Output     : ``data/generated/synthetic_accounts.csv``
  (written automatically when ``merge_with_original=True``)
- Merge      : original rows prepended to synthetic rows (``data_source``
  column marks each row as 'original' or 'synthetic')
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Project root = 3 levels above this file:
#   src/generation/core/account_generator.py → src/generation/core → src/generation → src → <root>
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]

_DEFAULT_SEED_PATH      = _PROJECT_ROOT / "data" / "original"  / "accounts.csv"
_DEFAULT_OUTPUT_PATH    = _PROJECT_ROOT / "data" / "generated" / "synthetic_accounts.csv"


class AccountGenerator:
    """Generate synthetic bank accounts using SMOTE.

    Parameters
    ----------
    knowledge : dict
        Knowledge base (kept for API compatibility; not used by SMOTE).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, knowledge: dict, seed: int = 42) -> None:
        self.knowledge = knowledge
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
        """Generate *n* synthetic accounts using SMOTE.

        Parameters
        ----------
        n : int
            Number of synthetic accounts to generate.
        output_path : Path | str | None
            Where to save the output CSV.  Defaults to
            ``data/generated/synthetic_accounts.csv`` when
            ``merge_with_original=True``.  Pass an explicit path to
            override, or ``None`` + ``merge_with_original=False`` to
            skip saving entirely.
        merge_with_original : bool, default ``True``
            Prepend original seed rows to the synthetic output.
            A ``data_source`` column ('original' / 'synthetic') is added.
        original_data_path : Path | str | None
            Path to the seed CSV.  Defaults to
            ``data/original/accounts.csv``.

        Returns
        -------
        pd.DataFrame
        """
        from src.generation.synthesizers.smote_generator import SMOTEGenerator

        # ── Resolve seed path ──────────────────────────────────────────
        seed_path = Path(original_data_path) if original_data_path else _DEFAULT_SEED_PATH
        if not seed_path.exists():
            raise FileNotFoundError(
                f"Seed accounts file not found: {seed_path}\n"
                "Run phase2_cleaning.ipynb first to produce this file."
            )

        logger.info("Phase 4 · Loading seed accounts from: %s", seed_path)
        seed_df = pd.read_csv(seed_path)
        logger.info("Seed data: %d rows, %d columns", len(seed_df), len(seed_df.columns))

        # ── Fit + generate ─────────────────────────────────────────────
        gen = SMOTEGenerator(config={"k_neighbors": 5, "random_state": self.seed})
        gen.fit(seed_df)

        logger.info("Generating %d synthetic accounts via SMOTE...", n)
        synthetic_df = gen.generate(num_rows=n)
        logger.info("SMOTE generation complete: %d rows", len(synthetic_df))

        # ── Merge ──────────────────────────────────────────────────────
        if merge_with_original:
            seed_df = seed_df.copy()
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
            df.to_csv(out, index=False)
            logger.info("Saved accounts to: %s", out)

        return df
