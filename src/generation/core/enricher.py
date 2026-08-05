"""
src/generation/enricher.py
----------------------------
Phase 7 — Transaction Enrichment.

Deterministic enrichment: merges synthetic accounts into each transaction
row to attach sender/receiver metadata. No random generation occurs here —
everything is derived directly from account records.

Inputs
------
synthetic_accounts.csv   — output of Phase 4
synthetic_transactions   — merged core + fraud transactions (Phases 5 + 6)

Output
------
data/interim/transactions_enriched.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.nepal_context import CORRIDOR_RISK_SCORES, EXCHANGE_RATE_RANGES, REMITTANCE_CORRIDORS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field mappings
# ---------------------------------------------------------------------------

# Canonical account fields to attach for sender/receiver
_SENDER_FIELDS = {
    "institution":      "sender_institution",
    "branch":           "sender_branch",
    "city":             "sender_city",
    "risk_grade":       "sender_risk_grade",
    "account_type":     "sender_account_type",
    "pep_flag":         "sender_pep",
    "sanctions_hit":    "sender_sanctions",
    "kyc_verified":     "sender_kyc_verified",
    "opened":           "sender_opened",
}

_RECEIVER_FIELDS = {
    "institution":      "receiver_institution",
    "branch":           "receiver_branch",
    "city":             "receiver_city",
    "risk_grade":       "receiver_risk_grade",
    "account_type":     "receiver_account_type",
    "pep_flag":         "receiver_pep",
    "sanctions_hit":    "receiver_sanctions",
    "kyc_verified":     "receiver_kyc_verified",
    "opened":           "receiver_opened",
}


class TransactionEnricher:
    """Enrich synthetic transactions by joining account metadata.

    Parameters
    ----------
    accounts_df : pd.DataFrame
        Synthetic accounts (canonical schema from Phase 4).
    """

    def __init__(self, accounts_df: pd.DataFrame) -> None:
        self.accounts = accounts_df.copy()
        self.accounts["account_id"] = self.accounts["account_id"].astype(str)
        self._accounts_indexed = self.accounts.set_index("account_id")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Enrich transaction rows with sender/receiver account metadata.

        Parameters
        ----------
        transactions : pd.DataFrame
            Synthetic transactions (Phases 5 + 6 combined).

        Returns
        -------
        pd.DataFrame — fully enriched transactions.
        """
        logger.info("Phase 7: Enriching %d transactions...", len(transactions))
        df = transactions.copy()

        # Normalise sender/receiver column names
        df = self._normalise_account_cols(df)

        # Attach sender account fields
        df = self._attach_account_fields(df, "sender_account_id", _SENDER_FIELDS)

        # Attach receiver account fields
        df = self._attach_account_fields(df, "receiver_account_id", _RECEIVER_FIELDS)

        # Derive geographic fields from account cities
        df = self._derive_geographic_fields(df)

        # Derive account age at transaction date
        df = self._derive_account_ages(df)

        # Derive FX / NPR amount where missing
        df = self._derive_amounts(df)

        # Derive country risk scores
        df = self._derive_country_risk(df)

        logger.info("Enrichment complete: %d rows, %d cols", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_account_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure sender/receiver columns use canonical names."""
        rename = {}
        if "Sender_account" in df.columns and "sender_account_id" not in df.columns:
            rename["Sender_account"] = "sender_account_id"
        if "Receiver_account" in df.columns and "receiver_account_id" not in df.columns:
            rename["Receiver_account"] = "receiver_account_id"
        if rename:
            df = df.rename(columns=rename)
        df["sender_account_id"] = df["sender_account_id"].astype(str)
        df["receiver_account_id"] = df["receiver_account_id"].astype(str)
        return df

    def _attach_account_fields(
        self,
        df: pd.DataFrame,
        id_col: str,
        field_map: dict[str, str],
    ) -> pd.DataFrame:
        """Left-join account fields onto df by id_col."""
        available_fields = [f for f in field_map.keys() if f in self._accounts_indexed.columns]
        if not available_fields:
            logger.warning("No account fields available to attach for %s", id_col)
            return df

        subset = self._accounts_indexed[available_fields].rename(columns=field_map)
        df = df.merge(
            subset,
            left_on=id_col,
            right_index=True,
            how="left",
        )
        return df

    def _derive_geographic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive sender/receiver bank location and cross-border flag."""
        # Map city → "Nepal" (domestic); cross-border comes from the transaction itself
        if "sender_city" in df.columns and "Sender_bank_location" not in df.columns:
            df["Sender_bank_location"] = df["sender_city"].fillna("Nepal")
        if "receiver_city" in df.columns and "Receiver_bank_location" not in df.columns:
            df["Receiver_bank_location"] = df["receiver_city"].fillna("Nepal")

        # Override with remittance corridor data when is_cross_border = 1
        cb_col = "is_cross_border"
        if cb_col in df.columns and "remittance_corridor" in df.columns:
            mask = df[cb_col].fillna(0).astype(int) == 1
            corridors = df.loc[mask, "remittance_corridor"].fillna("")
            # Extract source country from corridor string e.g. "Qatar->Nepal"
            src_countries = corridors.str.split("->").str[0].replace("", "Unknown")
            df.loc[mask, "Sender_bank_location"] = src_countries
            df.loc[mask, "Receiver_bank_location"] = "Nepal"

        # Derive currency_mismatch where not already set
        if "currency_mismatch" not in df.columns:
            p_curr = df.get("Payment_currency", pd.Series(["NPR"] * len(df)))
            r_curr = df.get("Received_currency", pd.Series(["NPR"] * len(df)))
            df["currency_mismatch"] = (p_curr != r_curr).astype(int)

        return df

    def _derive_account_ages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sender/receiver account age in days at transaction date."""
        date_col = "Date" if "Date" in df.columns else "transaction_date"
        if date_col not in df.columns:
            return df

        tx_dates = pd.to_datetime(df[date_col], errors="coerce")

        if "sender_opened" in df.columns:
            sender_open = pd.to_datetime(df["sender_opened"], errors="coerce")
            df["sender_account_age_days"] = (tx_dates - sender_open).dt.days.clip(lower=0).fillna(0).astype(int)

        if "receiver_opened" in df.columns:
            recv_open = pd.to_datetime(df["receiver_opened"], errors="coerce")
            df["receiver_account_age_days"] = (tx_dates - recv_open).dt.days.clip(lower=0).fillna(0).astype(int)

        return df

    def _derive_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure amount_local_npr is present; fill from Amount × fx_rate."""
        if "amount_local_npr" not in df.columns or df["amount_local_npr"].isna().all():
            amount_col = "Amount" if "Amount" in df.columns else None
            fx_col = "fx_rate_to_npr" if "fx_rate_to_npr" in df.columns else None
            if amount_col and fx_col:
                df["amount_local_npr"] = (
                    pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
                    * pd.to_numeric(df[fx_col], errors="coerce").fillna(1.0)
                ).round(2)
            elif amount_col:
                df["amount_local_npr"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

        # Ensure fx_rate is populated for domestic (NPR → NPR)
        if "fx_rate_to_npr" not in df.columns:
            df["fx_rate_to_npr"] = 1.0
        else:
            df["fx_rate_to_npr"] = pd.to_numeric(df["fx_rate_to_npr"], errors="coerce").fillna(1.0)

        return df

    def _derive_country_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach sender_country_risk and receiver_country_risk."""
        # Build country → risk score from REMITTANCE_CORRIDORS
        country_risk: dict[str, float] = {"Nepal": 0.1}
        for corridor, tier in REMITTANCE_CORRIDORS.items():
            src = corridor.split("->")[0]
            country_risk[src] = CORRIDOR_RISK_SCORES[tier]

        if "Sender_bank_location" in df.columns:
            df["sender_country_risk"] = (
                df["Sender_bank_location"]
                .map(country_risk)
                .fillna(0.5)   # unknown countries → medium risk
            )

        if "Receiver_bank_location" in df.columns:
            df["receiver_country_risk"] = (
                df["Receiver_bank_location"]
                .map(country_risk)
                .fillna(0.5)
            )

        return df


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def enrich_transactions(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Enrich transactions and optionally save to CSV."""
    enricher = TransactionEnricher(accounts)
    df = enricher.enrich(transactions)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Enriched transactions saved to: %s", output_path)
    return df
