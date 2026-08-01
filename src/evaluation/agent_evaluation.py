"""Per-agent and system-level evaluation for suspicious-transaction scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover - test/runtime environments install sklearn
    average_precision_score = brier_score_loss = confusion_matrix = f1_score = None  # type: ignore[assignment]
    precision_score = roc_auc_score = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AgentMetrics:
    """Serializable metric bundle for one score column."""

    agent_name: str
    auc_roc: float
    auc_pr: float
    f1_at_optimal_threshold: float
    optimal_threshold: float
    precision_at_k: float
    brier_score: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]

    def as_dict(self) -> dict[str, object]:
        """Return a tabular-friendly representation."""
        return {
            "agent_name": self.agent_name,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "f1_at_optimal_threshold": self.f1_at_optimal_threshold,
            "optimal_threshold": self.optimal_threshold,
            "precision_at_k": self.precision_at_k,
            "brier_score": self.brier_score,
            "confusion_matrix": self.confusion_matrix,
        }


def _safe_metric(function, default: float, *args, **kwargs) -> float:
    """Evaluate a sklearn metric while handling one-class fixtures."""
    try:
        return float(function(*args, **kwargs))
    except (ValueError, TypeError):
        return default


def _optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Select the threshold with maximum F1 using observed score cut points."""
    thresholds = np.unique(np.clip(scores, 0.0, 1.0))
    candidates = np.r_[0.0, thresholds, 1.0]
    values = [_safe_metric(f1_score, 0.0, y_true, scores >= threshold, zero_division=0) for threshold in candidates]
    index = int(np.argmax(values))
    return float(candidates[index]), float(values[index])


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Measure precision among the highest-risk k rows."""
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(len(y_true), int(np.ceil(len(y_true) * fraction))))
    order = np.argsort(-scores)[:k]
    return float(np.mean(y_true[order]))


class AgentEvaluator:
    """Compute comparable metrics for agent score columns.

    ``score_columns`` maps a display name to a numeric column in the supplied
    frame.  This keeps the evaluator independent of specific agent classes and
    supports both individual outputs and the meta-learner output.
    """

    def __init__(self, precision_at_k_fraction: float = 0.01) -> None:
        """Initialize the evaluator."""
        if not 0.0 < precision_at_k_fraction <= 1.0:
            raise ValueError("precision_at_k_fraction must be in (0, 1]")
        self.precision_at_k_fraction = precision_at_k_fraction

    def evaluate(self, data: pd.DataFrame, score_columns: dict[str, str], label_column: str = "is_fraud") -> pd.DataFrame:
        """Return one metric row per agent score column."""
        if label_column not in data.columns:
            raise ValueError(f"Missing required label column: {label_column}")
        y_true = pd.to_numeric(data[label_column], errors="coerce").fillna(0).astype(int).to_numpy()
        rows: list[dict[str, object]] = []
        for name, column in score_columns.items():
            if column not in data.columns:
                continue
            scores = pd.to_numeric(data[column], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
            threshold, f1 = _optimal_threshold(y_true, scores)
            matrix = confusion_matrix(y_true, scores >= threshold, labels=[0, 1]).tolist() if confusion_matrix else [[0, 0], [0, 0]]
            rows.append(AgentMetrics(
                name,
                _safe_metric(roc_auc_score, 0.5, y_true, scores) if roc_auc_score else 0.5,
                _safe_metric(average_precision_score, float(y_true.mean()), y_true, scores) if average_precision_score else float(y_true.mean()),
                f1,
                threshold,
                _precision_at_k(y_true, scores, self.precision_at_k_fraction),
                _safe_metric(brier_score_loss, 1.0, y_true, scores) if brier_score_loss else 1.0,
                (tuple(matrix[0]), tuple(matrix[1])),
            ).as_dict())
        return pd.DataFrame(rows)

    def compare_system(self, metrics: pd.DataFrame, meta_name: str = "meta_learner") -> dict[str, float]:
        """Summarize meta-learner improvement over the best individual agent."""
        if metrics.empty or "auc_roc" not in metrics.columns:
            return {"best_individual_auc_roc": 0.0, "meta_auc_roc": 0.0, "auc_roc_delta": 0.0}
        individual = metrics.loc[metrics["agent_name"] != meta_name, "auc_roc"]
        best = float(individual.max()) if not individual.empty else 0.0
        meta = metrics.loc[metrics["agent_name"] == meta_name, "auc_roc"]
        meta_value = float(meta.iloc[0]) if not meta.empty else 0.0
        return {"best_individual_auc_roc": best, "meta_auc_roc": meta_value, "auc_roc_delta": meta_value - best}

    def aml_detection_rates(self, data: pd.DataFrame, score_columns: dict[str, str], pattern_column: str = "fraud_type") -> pd.DataFrame:
        """Measure alert recall for each labeled AML pattern."""
        if pattern_column not in data.columns:
            return pd.DataFrame(columns=["agent_name", "pattern", "detection_rate", "count"])
        rows: list[dict[str, object]] = []
        for name, column in score_columns.items():
            if column not in data.columns:
                continue
            alerts = pd.to_numeric(data[column], errors="coerce").fillna(0.0) >= 0.5
            for pattern, group in data.groupby(pattern_column, dropna=True):
                rows.append({"agent_name": name, "pattern": str(pattern), "detection_rate": float(alerts.loc[group.index].mean()), "count": int(len(group))})
        return pd.DataFrame(rows)


def evaluate_agents(data: pd.DataFrame, score_columns: dict[str, str], label_column: str = "is_fraud") -> pd.DataFrame:
    """Convenience wrapper around :class:`AgentEvaluator`."""
    return AgentEvaluator().evaluate(data, score_columns, label_column)
