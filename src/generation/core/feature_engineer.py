"""
src/generation/feature_engineer.py
-------------------------------------
Phase 8 — Feature Engineering.

Computes all derived features from enriched transactions deterministically.
Every feature is a pure function of the enriched data — no random generation.

Features produced
-----------------
Temporal   : hour_of_day, day_of_week, month, is_weekend
Amount     : amount_local_npr, log_amount, amount_zscore, above_1M_NPR, above_10M_NPR
Geographic : sender_country_risk, receiver_country_risk, cross_border_flag, currency_mismatch
Velocity   : velocity_sum_10tx, tx_count_10, tx_count_30
Account    : sender_account_age_days, receiver_account_age_days
Encoding   : transmode_A/B/E/F/J/P/Z (one-hot from Payment_type / transmode_code)
Label      : is_suspicious_tx (alias of is_fraud)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: Payment_type / transaction_type → transmode code
# ---------------------------------------------------------------------------

_PAYMENT_TYPE_TO_TRANSMODE: dict[str, str] = {
    "deposit":               "A",
    "cash_out":              "A",
    "Cash Deposit":          "A",
    "Cash Withdrawal":       "A",
    "transfer":              "B",
    "Wire Transfer":         "B",
    "Internal":              "J",
    "remittance_inbound":    "F",
    "remittance_outbound":   "F",
    "Cross-border":          "F",
    "payment":               "P",
    "Payment":               "P",
    "POS":                   "P",
    "withdrawal":            "Z",
    "ATM":                   "Z",
}

# transmode codes used in the original dataset
_TRANSMODE_CODES = ["A", "B", "E", "F", "J", "P", "Z"]

_NPR_1M  = 1_000_000.0
_NPR_10M = 10_000_000.0


class FeatureEngineer:
    """Compute all derived ML features from enriched synthetic transactions.

    Parameters
    ----------
    global_amount_mean : float, optional
        Pre-computed global mean of amount_local_npr (for z-score).
        If None, computed from the input data.
    global_amount_std : float, optional
        Pre-computed global std of amount_local_npr (for z-score).
        If None, computed from the input data.
    """

    def __init__(
        self,
        global_amount_mean: float | None = None,
        global_amount_std: float | None = None,
    ) -> None:
        self._global_mean = global_amount_mean
        self._global_std = global_amount_std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for the enriched transaction DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched transactions (output of Phase 7).

        Returns
        -------
        pd.DataFrame with all features added.
        """
        logger.info("Phase 8: Feature engineering on %d rows...", len(df))
        df = df.copy()

        df = self._temporal_features(df)
        df = self._amount_features(df)
        df = self._velocity_features(df)
        df = self._account_age_features(df)
        df = self._transmode_encoding(df)
        df = self._label_column(df)

        logger.info("Feature engineering complete. Columns: %d", len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = "Date" if "Date" in df.columns else "transaction_date"
        time_col = "Time" if "Time" in df.columns else "transaction_time"

        if date_col in df.columns:
            dt_series = pd.to_datetime(df[date_col], errors="coerce")

            if "hour_of_day" not in df.columns or df["hour_of_day"].isna().all():
                if time_col in df.columns:
                    times = pd.to_datetime(df[time_col], format="%H:%M:%S", errors="coerce")
                    df["hour_of_day"] = times.dt.hour.fillna(12).astype(int)
                else:
                    df["hour_of_day"] = 12

            df["day_of_week"] = dt_series.dt.dayofweek.fillna(0).astype(int)
            df["month"]       = dt_series.dt.month.fillna(1).astype(int)
            df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

            # Combined datetime for velocity sort
            df["_datetime"] = dt_series

        return df

    # ------------------------------------------------------------------
    # Amount features
    # ------------------------------------------------------------------

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "amount_local_npr" in df.columns:
            amount_col = "amount_local_npr"
        elif "amount_npr" in df.columns:
            amount_col = "amount_npr"
        elif "Amount" in df.columns:
            amount_col = "Amount"
        else:
            logger.warning("No amount column found; skipping amount features")
            return df

        amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0).clip(lower=0.01)
        df["amount_local_npr"] = amounts

        df["log_amount"]  = np.log(amounts)

        # Global z-score
        mu  = self._global_mean if self._global_mean is not None else float(amounts.mean())
        sig = self._global_std  if self._global_std  is not None else float(amounts.std(ddof=0))
        if sig == 0:
            sig = 1.0
        self._global_mean = mu
        self._global_std  = sig
        df["amount_zscore"]  = ((amounts - mu) / sig).round(6)

        df["above_1M_NPR"]  = (amounts >= _NPR_1M).astype(int)
        df["above_10M_NPR"] = (amounts >= _NPR_10M).astype(int)

        return df

    # ------------------------------------------------------------------
    # Velocity features (computed chronologically per sender)
    # ------------------------------------------------------------------

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        sender_col = "sender_account_id" if "sender_account_id" in df.columns else "Sender_account"
        if sender_col not in df.columns or "_datetime" not in df.columns:
            df["velocity_sum_10tx"] = 0.0
            df["tx_count_10"]       = 0
            df["tx_count_30"]       = 0
            return df

        logger.info("  Computing velocity features (may take a moment)...")
        df = df.sort_values([sender_col, "_datetime"]).reset_index(drop=True)

        velocity_sum = np.zeros(len(df), dtype=np.float64)
        tx_count_10  = np.zeros(len(df), dtype=np.int32)
        tx_count_30  = np.zeros(len(df), dtype=np.int32)

        has_amount = "amount_local_npr" in df.columns

        for sender_id, grp in df.groupby(sender_col, sort=False):
            idx_list    = grp.index.tolist()
            dts         = grp["_datetime"].values
            amounts     = df.loc[idx_list, "amount_local_npr"].values if has_amount else np.zeros(len(idx_list))

            for i, (idx, ts, amt) in enumerate(zip(idx_list, dts, amounts)):
                # Look back over previous transactions
                look_back_10 = i   # last 10 transactions
                start_10 = max(0, i - 10)
                window_amounts = amounts[start_10:i]
                velocity_sum[idx]  = float(np.sum(window_amounts)) + amt
                tx_count_10[idx]   = len(window_amounts) + 1

                # Count within last 30 transactions
                start_30 = max(0, i - 30)
                tx_count_30[idx]   = (i - start_30) + 1

        df["velocity_sum_10tx"] = velocity_sum
        df["tx_count_10"]       = tx_count_10
        df["tx_count_30"]       = tx_count_30

        # Restore original order
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Account age
    # ------------------------------------------------------------------

    def _account_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # These may already be computed by the enricher
        if "sender_account_age_days" not in df.columns:
            df["sender_account_age_days"] = 0
        if "receiver_account_age_days" not in df.columns:
            df["receiver_account_age_days"] = 0
        return df

    # ------------------------------------------------------------------
    # Transmode one-hot encoding
    # ------------------------------------------------------------------

    def _transmode_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Payment_type → transmode_code then one-hot encode."""
        # Resolve transmode code
        if "transmode_code" in df.columns:
            codes = df["transmode_code"].fillna("E")
        elif "Payment_type" in df.columns:
            codes = df["Payment_type"].map(_PAYMENT_TYPE_TO_TRANSMODE).fillna("E")
        elif "transaction_type" in df.columns:
            codes = df["transaction_type"].map(_PAYMENT_TYPE_TO_TRANSMODE).fillna("E")
        else:
            codes = pd.Series(["E"] * len(df), index=df.index)

        df["transmode_code"] = codes

        # One-hot encode
        for code in _TRANSMODE_CODES:
            df[f"transmode_{code}"] = (codes == code).astype(int)

        return df

    # ------------------------------------------------------------------
    # Label column
    # ------------------------------------------------------------------

    def _label_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "is_fraud" in df.columns:
            df["is_suspicious_tx"] = df["is_fraud"].fillna(0).astype(int)
        else:
            df["is_suspicious_tx"] = 0
        return df


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    global_mean: float | None = None,
    global_std:  float | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Engineer all features and optionally save to CSV."""
    eng = FeatureEngineer(global_amount_mean=global_mean, global_amount_std=global_std)
    result = eng.engineer(df)
    # Drop internal helper columns
    result = result.drop(columns=["_datetime"], errors="ignore")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("Feature-engineered data saved to: %s", output_path)
    return result
