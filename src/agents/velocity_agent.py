"""Velocity and transaction-frequency anomaly detection agent."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover - minimal environments use rule fallback
    IsolationForest = None  # type: ignore[assignment,misc]

from src.agents.base_agent import BaseAgent
from src.utils.config import get_config


class VelocityAgent(BaseAgent):
    """Score unusual account activity using rolling features and Isolation Forest."""

    agent_name = "velocity"
    _feature_columns = (
        "amount_npr",
        "sender_count_1h",
        "sender_count_6h",
        "sender_count_24h",
        "sender_count_168h",
        "sender_sum_1h",
        "sender_sum_6h",
        "sender_sum_24h",
        "sender_sum_168h",
        "seconds_since_sender_txn",
        "amount_zscore",
    )

    def __init__(self, config: Any = None, logger: logging.Logger | None = None) -> None:
        """Initialize the agent and its unsupervised model settings."""
        super().__init__(config or get_config(), logger)
        self._model: IsolationForest | None = None
        self._score_low = 0.0
        self._score_high = 1.0
        self._last_features = pd.DataFrame()
        self._explanations: dict[str, str] = {}

    def _timestamp(self, data: pd.DataFrame) -> pd.Series:
        """Return timestamps from canonical or source-compatible columns."""
        if "timestamp" in data:
            return pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
        if {"transaction_date", "transaction_time"}.issubset(data.columns):
            return pd.to_datetime(
                data["transaction_date"].astype(str) + " " + data["transaction_time"].astype(str),
                errors="coerce", utc=True,
            )
        return pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")

    def _account_column(self, data: pd.DataFrame, prefix: str) -> str | None:
        """Find a canonical account column or the original dataset alias."""
        for candidate in (f"{prefix}_account_id", f"{prefix}_account"):
            if candidate in data.columns:
                return candidate
        return None

    def _features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build leakage-safe rolling activity features without mutating input."""
        frame = data.copy(deep=True)
        frame["_ts"] = self._timestamp(frame)
        missing_time = frame["_ts"].isna()
        if missing_time.any():
            fallback = pd.Timestamp("1970-01-01", tz="UTC")
            offsets = pd.to_timedelta(np.arange(len(frame)), unit="s")
            frame.loc[missing_time, "_ts"] = (fallback + offsets[missing_time.to_numpy()]).to_numpy()
        amount_source = frame.get("amount_npr", frame.get("amount"))
        frame["_amount"] = (
            pd.to_numeric(amount_source, errors="coerce").fillna(0.0)
            if amount_source is not None
            else pd.Series(0.0, index=frame.index)
        )
        sender = self._account_column(frame, "sender")
        frame["_sender"] = frame[sender].astype("string").fillna("__missing__") if sender else "__missing__"
        ordered = frame.sort_values(["_sender", "_ts"], kind="mergesort")
        grouped = ordered.groupby("_sender", sort=False)["_amount"]
        features = pd.DataFrame(index=ordered.index)
        features["amount_npr"] = ordered["_amount"]
        for hours in self.config.get("velocity_windows_hours", (1, 6, 24, 168)):
            window = f"{int(hours)}h"
            counts = ordered.set_index("_ts").groupby("_sender")["_amount"].rolling(window, min_periods=1).count()
            sums = ordered.set_index("_ts").groupby("_sender")["_amount"].rolling(window, min_periods=1).sum()
            features[f"sender_count_{int(hours)}h"] = counts.reset_index(level=0, drop=True).to_numpy()
            if int(hours) in (1, 6, 24, 168):
                features[f"sender_sum_{int(hours)}h"] = sums.reset_index(level=0, drop=True).to_numpy()
        features["seconds_since_sender_txn"] = ordered.groupby("_sender")["_ts"].diff().dt.total_seconds().fillna(86400.0)
        mean = grouped.transform("mean").replace(0, np.nan)
        std = grouped.transform("std").fillna(0).replace(0, np.nan)
        features["amount_zscore"] = ((ordered["_amount"] - mean) / std).fillna(0.0).abs()
        return features.reindex(data.index).fillna(0.0).astype(float)

    def fit(self, data: pd.DataFrame) -> "VelocityAgent":
        """Fit Isolation Forest on rolling velocity features."""
        self._last_features = self._features(data)
        if len(self._last_features) >= 2 and IsolationForest is not None:
            self._model = IsolationForest(
                n_estimators=int(self.config.get("velocity_isolation_estimators", 100)),
                contamination=float(self.config.get("velocity_isolation_contamination", 0.05)),
                random_state=int(self.config.get("random_seed", 42)),
            ).fit(self._last_features.loc[:, self._feature_columns])
            raw = -self._model.decision_function(self._last_features.loc[:, self._feature_columns])
            self._score_low, self._score_high = float(raw.min()), float(raw.max())
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return per-transaction velocity risk scores and reason codes."""
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        features = self._features(data)
        if not self.is_fitted:
            self.logger.warning("VelocityAgent.predict() called before fit(); call fit() first.")
            raise RuntimeError("VelocityAgent must be fitted before calling predict().")
        raw = -self._model.decision_function(features.loc[:, self._feature_columns]) if self._model else np.zeros(len(data))
        span = max(self._score_high - self._score_low, 1e-9)
        model_score = np.clip((raw - self._score_low) / span, 0.0, 1.0)
        one_hour = features["sender_count_1h"] >= int(self.config.get("velocity_spike_min_transactions", 3))
        scores = np.maximum(model_score, np.where(one_hour, np.minimum(features["sender_count_1h"] / 10.0, 1.0), 0.0))
        reasons = np.where(one_hour, "VELOCITY_SPIKE", np.where(model_score >= self.alert_threshold, "ISOLATION_ANOMALY", "NORMAL_VELOCITY"))
        self._explanations = dict(zip(data["transaction_id"].astype(str), reasons.astype(str)))
        result = self.build_predictions(data, scores, reason_code="VELOCITY_ANALYSIS")
        result["reason_code"] = reasons
        result["explanation"] = [f"Velocity analysis: {reason.replace('_', ' ').lower()}." for reason in reasons]
        return result

    def explain(self, transaction_id: str) -> str:
        """Explain a previously scored transaction."""
        reason = self._explanations.get(str(transaction_id), "no velocity score is available")
        return f"Transaction {transaction_id}: {reason.replace('_', ' ').lower()}."
