"""Common contract for all Neural Sentinel detection agents.

Specialized agents should inherit :class:`BaseAgent` and implement ``fit``,
``predict``, and ``explain``.  The class deliberately keeps model-specific
dependencies out of the interface so that every agent remains usable in CPU-
only tests and Kaggle environments where optional accelerators may be absent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, ClassVar

import pandas as pd


class BaseAgent(ABC):
    """Abstract interface shared by every detection agent.

    Args:
        config: Mapping of agent settings.  A Pydantic configuration object is
            also accepted and is converted to a plain dictionary when it
            exposes ``model_dump``.
        logger: Logger used for decisions, thresholds, and abnormal inputs.

    Notes:
        Implementations must not mutate their input DataFrames.  Use
        :meth:`build_predictions` to normalize a model's raw scores into the
        canonical agent-output columns.
    """

    agent_name: ClassVar[str] = "base_agent"
    default_alert_threshold: ClassVar[float] = 0.5
    prediction_columns: ClassVar[tuple[str, ...]] = (
        "transaction_id",
        "agent_name",
        "risk_score",
        "alert_flag",
        "reason_code",
        "explanation",
        "timestamp",
    )

    def __init__(
        self,
        config: Mapping[str, Any] | Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize an agent with copied configuration and a logger."""
        self.config = self._config_to_dict(config)
        self.logger = logger or logging.getLogger(f"neural_sentinel.agents.{self.agent_name}")
        self.alert_threshold = self._resolve_alert_threshold()
        self.is_fitted = False

        if not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError("alert_threshold must be between 0.0 and 1.0")
        self.logger.info(
            "Initialized agent '%s' with alert threshold %.3f",
            self.agent_name,
            self.alert_threshold,
        )

    @staticmethod
    def _config_to_dict(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
        """Return a detached dictionary for mapping-like configuration input."""
        if config is None:
            return {}
        if hasattr(config, "model_dump"):
            return dict(config.model_dump())
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError("config must be a mapping, Pydantic model, or None")

    def _resolve_alert_threshold(self) -> float:
        """Resolve the agent-specific threshold, then generic fallback."""
        names = [self.agent_name]
        if self.agent_name.endswith("_agent"):
            names.append(self.agent_name.removesuffix("_agent"))
        candidates = tuple(f"{name}_alert_threshold" for name in names) + ("alert_threshold",)
        for key in candidates:
            if key in self.config:
                return float(self.config[key])
        return float(self.default_alert_threshold)

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseAgent":
        """Fit the agent and return ``self`` for fluent pipeline composition."""

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return canonical risk predictions without mutating ``data``."""

    @abstractmethod
    def explain(self, transaction_id: str) -> str:
        """Return a human-readable explanation for one transaction."""

    def build_predictions(
        self,
        data: pd.DataFrame,
        risk_scores: Any,
        reason_code: str = "AGENT_SCORE",
        explanation: str | None = None,
    ) -> pd.DataFrame:
        """Build and validate canonical output rows from raw risk scores.

        Missing ``transaction_id`` is handled gracefully by returning an empty
        result with the correct schema.  Scores are clipped to ``[0, 1]`` so
        imperfect model output cannot violate the data contract; a warning is
        logged when clipping is required.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if "transaction_id" not in data.columns:
            self.logger.warning("Missing required feature: transaction_id")
            return self.empty_predictions()

        frame = data.loc[:, ["transaction_id"]].copy()
        scores = pd.Series(risk_scores, index=frame.index, dtype="float64")
        if len(scores) != len(frame):
            raise ValueError("risk_scores must contain one value per input row")
        clipped = scores.clip(0.0, 1.0)
        if not scores.equals(clipped):
            self.logger.warning("Clipped out-of-range risk scores for '%s'", self.agent_name)

        frame["agent_name"] = self.agent_name
        frame["risk_score"] = clipped
        frame["alert_flag"] = (clipped >= self.alert_threshold).astype("int8")
        frame["reason_code"] = reason_code
        frame["explanation"] = explanation or "Risk score generated by the agent."
        frame["timestamp"] = datetime.now(timezone.utc)
        return frame.loc[:, self.prediction_columns]

    @classmethod
    def empty_predictions(cls) -> pd.DataFrame:
        """Return an empty DataFrame with the complete prediction schema."""
        return pd.DataFrame({column: pd.Series(dtype="object") for column in cls.prediction_columns})

    def require_columns(self, data: pd.DataFrame, columns: tuple[str, ...]) -> bool:
        """Check required input columns and log missing features without raising."""
        missing = sorted(set(columns).difference(data.columns))
        if missing:
            self.logger.warning("Agent '%s' missing features: %s", self.agent_name, missing)
            return False
        return True

    def __getstate__(self) -> dict[str, Any]:
        """Keep serialized agents portable by storing the logger name only."""
        state = self.__dict__.copy()
        state["logger_name"] = self.logger.name
        state["logger"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the logger after pickle/joblib deserialization."""
        logger_name = state.pop("logger_name", f"neural_sentinel.agents.{self.agent_name}")
        self.__dict__.update(state)
        self.logger = logging.getLogger(logger_name)
