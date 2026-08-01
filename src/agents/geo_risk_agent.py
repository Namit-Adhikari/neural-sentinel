"""Geographic and cross-border risk agent backed by CatBoost when available."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError:  # pragma: no cover - deterministic fallback is intentional
    ColumnTransformer = SimpleImputer = LogisticRegression = Pipeline = OneHotEncoder = None  # type: ignore[assignment,misc]

from src.agents.base_agent import BaseAgent
from src.utils.config import get_config
from src.utils.nepal_context import CORRIDOR_RISK_SCORES, REMITTANCE_CORRIDORS

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - exercised in minimal environments
    CatBoostClassifier = None  # type: ignore[assignment,misc]


class GeoRiskAgent(BaseAgent):
    """Estimate geographic risk from corridors, countries, currency and IP signals."""

    agent_name = "geo_risk"
    _categorical = ("sender_country", "receiver_country", "remittance_corridor", "original_currency", "ip_country")
    _numeric = ("is_cross_border", "ip_is_vpn", "corridor_risk", "ip_country_mismatch")

    def __init__(self, config: Any = None, logger: logging.Logger | None = None) -> None:
        """Initialize the geographic model and feature metadata."""
        super().__init__(config or get_config(), logger)
        self._model: Any = None
        self._use_catboost = CatBoostClassifier is not None
        self._features = pd.DataFrame()
        self._explanations: dict[str, str] = {}

    def _features_for(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic geo features from canonical or source columns."""
        frame = pd.DataFrame(index=data.index)
        aliases = {"original_currency": ("original_currency", "currency"), "sender_country": ("sender_country", "country"), "receiver_country": ("receiver_country", "country"), "remittance_corridor": ("remittance_corridor",), "ip_country": ("ip_country", "country")}
        for name, candidates in aliases.items():
            source = next((col for col in candidates if col in data.columns), None)
            frame[name] = data[source].astype("string").fillna("unknown") if source else "unknown"
        for name in ("is_cross_border", "ip_is_vpn"):
            frame[name] = pd.to_numeric(data[name], errors="coerce").fillna(0.0) if name in data else 0.0
        frame["corridor_risk"] = frame["remittance_corridor"].map(REMITTANCE_CORRIDORS).map(CORRIDOR_RISK_SCORES).fillna(0.1)
        frame["ip_country_mismatch"] = ((frame["ip_country"] != frame["sender_country"]) & (frame["is_cross_border"] == 0)).astype(float)
        return frame

    def fit(self, data: pd.DataFrame) -> "GeoRiskAgent":
        """Fit CatBoost or a scikit-learn categorical fallback when labels exist."""
        self._features = self._features_for(data)
        if "is_fraud" not in data or data["is_fraud"].nunique(dropna=True) < 2:
            self.logger.warning("Geo-Risk agent has no two-class fraud label; using deterministic scores")
            self.is_fitted = True
            return self
        target = pd.to_numeric(data["is_fraud"], errors="coerce").fillna(0).astype(int)
        if self._use_catboost:
            self._model = CatBoostClassifier(iterations=int(self.config.get("geo_risk_iterations", 300)), learning_rate=float(self.config.get("geo_risk_learning_rate", 0.05)), depth=int(self.config.get("geo_risk_depth", 6)), loss_function="Logloss", verbose=False, random_seed=int(self.config.get("geo_risk_random_seed", 42)), allow_writing_files=False)
            self._model.fit(self._features, target, cat_features=list(self._categorical))
        elif all(item is not None for item in (ColumnTransformer, SimpleImputer, LogisticRegression, Pipeline, OneHotEncoder)):
            transformer = ColumnTransformer([("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(self._categorical)), ("numeric", SimpleImputer(strategy="median"), list(self._numeric))])
            self._model = Pipeline([("features", transformer), ("classifier", LogisticRegression(max_iter=300, class_weight="balanced"))]).fit(self._features, target)
        else:
            self.logger.warning("Geo-Risk ML dependencies unavailable; using deterministic scores")
        self.is_fitted = True
        return self

    def _deterministic_score(self, features: pd.DataFrame) -> np.ndarray:
        """Provide a transparent fallback score when supervised labels are absent."""
        return np.clip(0.45 * features["corridor_risk"].to_numpy() + 0.25 * features["ip_is_vpn"].to_numpy() + 0.20 * features["is_cross_border"].to_numpy() + 0.10 * features["ip_country_mismatch"].to_numpy(), 0.0, 1.0)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return per-transaction geographic risk scores and reason codes."""
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        features = self._features_for(data)
        if not self.is_fitted:
            self.fit(data)
        has_geo_signal = any(
            column in data.columns
            for column in (
                "sender_country", "receiver_country", "country", "remittance_corridor",
                "original_currency", "currency", "ip_country", "ip_is_vpn", "is_cross_border",
            )
        )
        if not has_geo_signal:
            scores = np.zeros(len(data))
        elif self._model is None:
            scores = self._deterministic_score(features)
        elif self._use_catboost:
            scores = self._model.predict_proba(features)[:, 1]
        else:
            scores = self._model.predict_proba(features)[:, 1]
        reasons = np.where(features["ip_is_vpn"] > 0, "VPN_OR_PROXY", np.where(features["ip_country_mismatch"] > 0, "IP_COUNTRY_MISMATCH", np.where(features["corridor_risk"] >= 0.85, "HIGH_RISK_CORRIDOR", np.where(features["is_cross_border"] > 0, "CROSS_BORDER", "DOMESTIC"))))
        self._explanations = dict(zip(data["transaction_id"].astype(str), reasons.astype(str)))
        result = self.build_predictions(data, scores, reason_code="GEO_RISK_ANALYSIS")
        result["reason_code"] = reasons
        result["explanation"] = [f"Geographic analysis: {reason.replace('_', ' ').lower()}." for reason in reasons]
        return result

    def explain(self, transaction_id: str) -> str:
        """Explain a previously scored transaction."""
        reason = self._explanations.get(str(transaction_id), "no geographic score is available")
        return f"Transaction {transaction_id}: {reason.replace('_', ' ').lower()}."
