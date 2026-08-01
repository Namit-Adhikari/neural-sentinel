"""Human-readable explanations for the multi-agent fraud-risk pipeline."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent


class ExplanationAgent(BaseAgent):
    """Compose transaction facts, agent reasons, and optional SHAP attributions.

    The agent is deliberately a deterministic template composer rather than a
    generative language model.  Banking analysts can therefore reproduce the
    explanation from the stored transaction and agent outputs, while optional
    SHAP values identify which meta-features moved the score.
    """

    agent_name = "explanation"
    default_alert_threshold = 0.5

    def __init__(
        self,
        config: Mapping[str, Any] | Any | None = None,
        logger: logging.Logger | None = None,
        meta_learner: Any | None = None,
        upstream_agents: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the composer with optional fitted model objects."""
        super().__init__(config, logger)
        self.meta_learner = meta_learner
        self.upstream_agents = dict(upstream_agents or {})
        self._context = pd.DataFrame()
        self._explanations: dict[str, str] = {}

    @staticmethod
    def _first(row: pd.Series, names: tuple[str, ...], default: str = "unknown") -> str:
        """Return the first non-empty value from a row."""
        for name in names:
            if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
                return str(row[name])
        return default

    def _shap_reasons(self, data: pd.DataFrame) -> dict[str, list[str]]:
        """Return positive SHAP feature labels when a fitted meta-model supports it."""
        if self.meta_learner is None:
            return {}
        try:
            import shap

            explainer = shap.Explainer(self.meta_learner, data)
            shap_values = explainer(data)
            columns = list(data.columns)
            if shap_values.values.ndim != 2 or shap_values.values.shape[0] != len(data) or shap_values.values.shape[1] != len(columns):
                return {}
            ids = data["transaction_id"].astype(str).tolist()
            return {
                tid: [columns[i].removesuffix("_score") for i in np.argsort(-shap_values.values[row])[:3] if shap_values.values[row, i] > 0]
                for row, tid in enumerate(ids)
            }
        except Exception as exc:  # pragma: no cover - optional SHAP/runtime failures
            self.logger.warning("SHAP attribution unavailable: %s", exc)
            return {}

    def _compose(self, row: pd.Series, shap_reasons: list[str] | None = None) -> str:
        """Compose one concise, auditable explanation from available fields."""
        tid = self._first(row, ("transaction_id",), "unknown")
        amount = pd.to_numeric(row.get("amount_npr", row.get("amount", np.nan)), errors="coerce")
        amount_text = f"NPR {amount:,.2f}" if pd.notna(amount) else "an unknown amount"
        action = self._first(row, ("transaction_type",), "transaction")
        channel = self._first(row, ("channel",), "unknown channel")
        corridor = self._first(row, ("remittance_corridor",), "domestic corridor")
        score = pd.to_numeric(row.get("meta_risk_score", row.get("risk_score", 0.0)), errors="coerce")
        score_text = f"{float(score):.3f}" if pd.notna(score) else "unknown"

        signals: list[str] = []
        for name in ("meta_reason_code", "reason_code"):
            if name in row.index and pd.notna(row[name]) and str(row[name]) not in {"META_NORMAL", "NORMAL"}:
                signals.append(str(row[name]).replace("_", " ").lower())
        for column in row.index:
            if column.endswith("_reason_code") and pd.notna(row[column]):
                value = str(row[column])
                if value not in {"NO_VIOLATIONS", "NORMAL_VELOCITY", "META_NORMAL"}:
                    signals.append(value.replace("_", " ").lower())
        signals.extend(reason.replace("_", " ").lower() for reason in shap_reasons or [])
        signals = list(dict.fromkeys(signals))[:4]
        signal_text = ", ".join(signals) if signals else "no dominant upstream reason was recorded"
        alert_text = "flagged for review" if float(score or 0.0) >= self.alert_threshold else "not flagged"
        return (
            f"Transaction {tid} ({action}, {amount_text}, via {channel}, {corridor}) is {alert_text} "
            f"with combined risk score {score_text}. Contributing signals: {signal_text}. "
            "This is a suspicious-activity triage explanation, not a finding of criminal conduct."
        )

    def fit(self, data: pd.DataFrame) -> "ExplanationAgent":
        """Store a defensive copy of the explanation context."""
        if not self.require_columns(data, ("transaction_id",)):
            self._context = pd.DataFrame()
        else:
            self._context = data.copy(deep=True)
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return canonical rows with deterministic human-readable explanations."""
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        frame = data.copy(deep=True)
        if "meta_risk_score" not in frame.columns and self.meta_learner is not None:
            try:
                meta = self.meta_learner.predict(frame)
                frame = frame.merge(meta[["transaction_id", "risk_score", "alert_flag", "reason_code"]], on="transaction_id", how="left")
                frame = frame.rename(columns={"risk_score": "meta_risk_score", "alert_flag": "meta_alert_flag", "reason_code": "meta_reason_code"})
            except Exception as exc:  # pragma: no cover - defensive optional model path
                self.logger.warning("Could not obtain meta-learner predictions: %s", exc)
        score_source = "meta_risk_score" if "meta_risk_score" in frame else "risk_score"
        scores = pd.to_numeric(frame.get(score_source, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
        shap_reasons = self._shap_reasons(frame)
        explanations = [self._compose(row, shap_reasons.get(str(row.transaction_id), [])) for _, row in frame.iterrows()]
        self._explanations = dict(zip(frame["transaction_id"].astype(str), explanations))
        result = self.build_predictions(frame, scores, reason_code="EXPLANATION_READY")
        result["reason_code"] = np.where(scores >= self.alert_threshold, "EXPLANATION_ALERT", "EXPLANATION_NORMAL")
        result["explanation"] = explanations
        return result

    def explain(self, transaction_id: str) -> str:
        """Return the cached explanation for one transaction."""
        return self._explanations.get(str(transaction_id), f"No explanation is available for transaction {transaction_id}.")
