"""Meta-learner agent that combines all upstream agent scores into a single
calibrated fraud probability.

The meta-learner treats each upstream agent's ``risk_score`` output as a feature
and trains a second-level model to produce a final, well-calibrated probability.
This stacking approach consistently outperforms any single agent by exploiting
complementary detection signals (velocity, geography, behaviour, graph, KYC/AML).
Primary model  : Calibrated Random Forest (Platt scaling or isotonic regression).
Challenger model: XGBoost (GPU when available, CPU fallback).

Design notes
------------
- Input is the ``alert_scores`` table (AGENTS.md §5.3) pivoted so that each
  agent becomes one column.  Missing agent scores are imputed with 0.0.
- The meta-learner is trained on the same data the upstream agents were fitted
  on; the ground-truth ``is_fraud`` label must therefore be present in the
  joined DataFrame passed to ``fit()``.
- When fewer than two classes are present in the training data (e.g. a tiny
  synthetic test fixture) the agent falls back to averaging the upstream scores
  so it is never unfitted.
- ``predict()`` accepts either:
    (a) a pre-pivoted DataFrame with one ``<agent_name>_score`` column per agent, or
    (b) a raw transactions DataFrame joined with the pivoted agent scores.
  Both paths normalise to the same internal feature layout.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.utils.config import get_config

# ---------------------------------------------------------------------------
# Optional dependencies — fail gracefully in CPU-only test environments
# ---------------------------------------------------------------------------
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    CalibratedClassifierCV = None  # type: ignore[assignment,misc]
    RandomForestClassifier = None  # type: ignore[assignment,misc]
    SimpleImputer = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Known upstream agent names (determines canonical feature column order)
# ---------------------------------------------------------------------------
_AGENT_NAMES: tuple[str, ...] = (
    "velocity",
    "geo_risk",
    "behaviour",
    "kyc_aml",
    "graph",
)

# ---------------------------------------------------------------------------
# Training logic (defined in meta_learner_training.py to respect 500-line limit)
# ---------------------------------------------------------------------------
from src.agents.meta_learner_training import MetaLearnerTrainingMixin

class MetaLearnerAgent(MetaLearnerTrainingMixin, BaseAgent):
    """Combine upstream agent risk scores into a single calibrated prediction.

    The meta-learner stacks agent outputs by treating each agent's ``risk_score``
    as a numeric feature.  A calibrated Random Forest is the primary model;
    XGBoost is evaluated as a challenger.

    Args:
        config: Configuration mapping.  Recognised keys:

            - ``meta_learner_model_type``: ``"random_forest"`` (default) or ``"xgboost"``.
            - ``meta_learner_n_estimators``: Number of trees (default 200).
            - ``meta_learner_max_depth``: Maximum tree depth (default ``None`` for RF, 6 for XGB).
            - ``meta_learner_calibration_method``: ``"isotonic"`` (default) or ``"sigmoid"``.
            - ``meta_learner_calibration_cv``: CV folds for calibration (default 3).
            - ``meta_learner_alert_threshold``: Alert threshold (default 0.5).
            - ``meta_learner_xgb_use_gpu``: Use GPU for XGBoost when available (default True).
            - ``random_seed``: Global random seed (default 42).

        logger: Structured logger.
    """

    agent_name = "meta_learner"

    def __init__(self, config: Any = None, logger: logging.Logger | None = None) -> None:
        """Initialise model configuration and internal state."""
        super().__init__(config or get_config(), logger)
        self.model_type: str = str(
            self.config.get("meta_learner_model_type", "random_forest")
        ).lower()
        self.n_estimators: int = int(self.config.get("meta_learner_n_estimators", 200))
        self.max_depth: int | None = (
            int(self.config["meta_learner_max_depth"])
            if "meta_learner_max_depth" in self.config and self.config["meta_learner_max_depth"] is not None
            else None
        )
        self.calibration_method: str = str(
            self.config.get("meta_learner_calibration_method", "isotonic")
        )
        self.calibration_cv: int = int(self.config.get("meta_learner_calibration_cv", 3))
        self.use_gpu: bool = bool(self.config.get("meta_learner_xgb_use_gpu", True))
        self.random_seed: int = int(self.config.get("random_seed", 42))
        self.xgb_learning_rate: float = float(
            self.config.get("meta_learner_xgb_learning_rate", 0.05)
        )
        self.xgb_subsample: float = float(
            self.config.get("meta_learner_xgb_subsample", 0.8)
        )
        self.xgb_colsample_bytree: float = float(
            self.config.get("meta_learner_xgb_colsample_bytree", 0.8)
        )
        self.xgb_default_max_depth: int = int(
            self.config.get("meta_learner_xgb_default_max_depth", 6)
        )
        self.calibration_val_fraction: float = float(
            self.config.get("meta_learner_calibration_val_fraction", 0.2)
        )

        # Populated during fit
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._imputer: Any = None
        self._explanations: dict[str, str] = {}
        self._feature_importances: dict[str, float] = {}
        # Computed from training label distribution; used by XGBoost to handle imbalance.
        self._scale_pos_weight: float = 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_column(agent_name: str) -> str:
        """Return the canonical meta-feature column name for an agent.

        Args:
            agent_name: Name of the upstream agent.

        Returns:
            Column name string e.g. ``"velocity_score"``.
        """
        return f"{agent_name}_score"

    def _pivot_agent_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert a long-format alert_scores table to wide-format meta-features.

        If ``data`` already contains ``<agent>_score`` columns the method
        returns a copy without modification.

        Args:
            data: Either a wide DataFrame (one row per transaction, columns
                ``<agent>_score``) or a long DataFrame with columns
                ``transaction_id``, ``agent_name``, ``risk_score``.

        Returns:
            Wide-format DataFrame indexed by ``transaction_id`` with one column
            per agent score.
        """
        # Detect format: long format has an "agent_name" column
        if "agent_name" in data.columns and "risk_score" in data.columns:
            pivoted = (
                data.pivot_table(
                    index="transaction_id",
                    columns="agent_name",
                    values="risk_score",
                    aggfunc="first",
                )
                .reset_index()
            )
            # Rename columns to canonical <agent>_score format
            pivoted.columns = [
                self._score_column(c) if c != "transaction_id" else c
                for c in pivoted.columns
            ]
            return pivoted

        # Already wide format — return as-is
        return data.copy()

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and align the meta-feature matrix from ``data``.

        Missing agent columns are filled with 0.0.  The column order follows
        ``_AGENT_NAMES`` so that the fitted model always sees features in the
        same order regardless of which agents are present in the input.

        Args:
            data: Wide-format DataFrame (output of :meth:`_pivot_agent_scores`).

        Returns:
            Float64 DataFrame with exactly the columns in ``self._feature_columns``.
        """
        if not self._feature_columns:
            # During fit: build the canonical column list from what is present
            present = [
                self._score_column(name)
                for name in _AGENT_NAMES
                if self._score_column(name) in data.columns
            ]
            # Also catch any extra <agent>_score columns not in _AGENT_NAMES
            extra = [
                c for c in data.columns
                if c.endswith("_score") and c not in present and c != "transaction_id"
            ]
            self._feature_columns = present + extra

        frame = pd.DataFrame(index=data.index)
        for col in self._feature_columns:
            if col in data.columns:
                frame[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
            else:
                frame[col] = 0.0
        return frame.astype(np.float64)

    # ------------------------------------------------------------------
    # Training helpers are provided by MetaLearnerTrainingMixin
    # (src/agents/meta_learner_training.py) to keep this file under 500
    # lines.
    # ------------------------------------------------------------------

    def _average_scores(self, features: pd.DataFrame) -> np.ndarray:
        """Fallback: simple average of all available agent scores.

        Args:
            features: Wide-format float DataFrame.

        Returns:
            Float32 array of averaged scores.
        """
        return features.mean(axis=1).clip(0.0, 1.0).to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "MetaLearnerAgent":
        """Train the meta-learner on upstream agent scores.

        ``data`` must contain either:
        - A long-format alert_scores table (``agent_name`` + ``risk_score``
          columns) joined with a ground-truth ``is_fraud`` column, **or**
        - A wide-format table with ``<agent>_score`` columns and ``is_fraud``.

        If fewer than 2 classes are present in ``is_fraud`` the agent falls
        back to score-averaging mode and logs a warning.

        Args:
            data: Training DataFrame.

        Returns:
            self for fluent composition.
        """
        if not self.require_columns(data, ("transaction_id",)):
            self.is_fitted = True
            return self

        wide = self._pivot_agent_scores(data)
        features = self._extract_features(wide)

        if "is_fraud" not in data.columns:
            self.logger.warning(
                "MetaLearnerAgent.fit(): no 'is_fraud' column found; "
                "agent will use score-averaging fallback."
            )
            self.is_fitted = True
            return self

        labels = pd.to_numeric(data["is_fraud"], errors="coerce").fillna(0).astype(int)

        if labels.nunique(dropna=True) < 2:
            self.logger.warning(
                "MetaLearnerAgent.fit(): fewer than 2 classes in 'is_fraud'; "
                "using score-averaging fallback."
            )
            self.is_fitted = True
            return self

        if not _SKLEARN_AVAILABLE:
            self.logger.warning(
                "scikit-learn not available — MetaLearnerAgent using score-averaging fallback."
            )
            self.is_fitted = True
            return self

        self._imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        X = self._imputer.fit_transform(features.to_numpy())
        y = labels.to_numpy()

        if self.model_type == "xgboost" and _XGB_AVAILABLE:
            self._fit_xgb(X, y)
        else:
            if self.model_type == "xgboost" and not _XGB_AVAILABLE:
                self.logger.warning("XGBoost not available; falling back to Random Forest.")
            self._fit_rf(X, y, labels)

        self._store_feature_importances()
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return calibrated meta-learner risk scores.

        Args:
            data: Wide-format DataFrame with ``<agent>_score`` columns and
                ``transaction_id``, or a long-format alert_scores table.

        Returns:
            Canonical agent output DataFrame.
        """
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()

        if not self.is_fitted:
            self.logger.warning(
                "MetaLearnerAgent.predict() called before fit(); fitting now."
            )
            self.fit(data)

        wide = self._pivot_agent_scores(data)

        # Align transaction IDs — wide may have fewer rows if pivoted from long format
        if "transaction_id" in wide.columns:
            # Use wide's transaction_id order to build feature matrix; map back to original
            features = self._extract_features(wide)
            tid_series = wide["transaction_id"].astype(str)
        else:
            features = self._extract_features(wide)
            tid_series = data["transaction_id"].astype(str)

        if self._model is None or not _SKLEARN_AVAILABLE:
            # Fallback: average of available agent scores
            scores_arr = self._average_scores(features)
        else:
            # self._imputer is None only when sklearn was unavailable at fit time;
            # in that case self._model is also None so the fallback branch above
            # is taken first — this else arm is a belt-and-suspenders guard.
            X = self._imputer.transform(features.to_numpy()) if self._imputer else features.to_numpy()
            try:
                scores_arr = self._model.predict_proba(X)[:, 1].astype(np.float32)
            except Exception as exc:
                self.logger.warning(
                    "MetaLearnerAgent.predict() model inference failed (%s); "
                    "using score-averaging fallback.",
                    exc,
                )
                scores_arr = self._average_scores(features)

        reasons = np.where(
            scores_arr >= self.alert_threshold,
            "META_ALERT",
            "META_NORMAL",
        )

        self._explanations = dict(zip(tid_series, reasons.astype(str)))

        # Build output aligned to the *original* data's transaction_id order
        # We need to create a result DataFrame aligned to `data`
        if len(scores_arr) == len(data):
            result_data = data
            result_scores = scores_arr
            result_reasons = reasons
        else:
            # Scores computed on pivoted (wide) data — map back by transaction_id
            score_map = dict(zip(tid_series, scores_arr))
            reason_map = dict(zip(tid_series, reasons.astype(str)))
            result_scores = np.array(
                [score_map.get(str(tid), 0.0) for tid in data["transaction_id"].astype(str)],
                dtype=np.float32,
            )
            result_reasons = np.array(
                [reason_map.get(str(tid), "META_NORMAL") for tid in data["transaction_id"].astype(str)]
            )
            result_data = data

        result = self.build_predictions(result_data, result_scores, reason_code="META_LEARNER")
        result["reason_code"] = result_reasons
        result["explanation"] = [
            f"Meta-learner ({self.model_type.replace('_', ' ')}): "
            f"{r.replace('_', ' ').lower().replace(':', ' driven by ')}."
            for r in result_reasons
        ]
        return result

    def explain(self, transaction_id: str) -> str:
        """Return a human-readable meta-learner explanation for one transaction.

        Args:
            transaction_id: The transaction to explain.

        Returns:
            Explanation string including top contributing agent if available.
        """
        # TODO (Phase 5 — ExplanationAgent): replace this stub with per-transaction
        # SHAP attribution.  Call self.get_shap_values() to obtain the SHAP matrix,
        # index by transaction_id, and combine with each upstream agent's explain()
        # output to produce the full narrative required by AGENTS.md §8.2.
        reason = self._explanations.get(
            str(transaction_id),
            "no meta-learner score available for this transaction",
        )
        return (
            f"Transaction {transaction_id}: {reason.replace('_', ' ').lower()} "
            f"(meta-learner: {self.model_type.replace('_', ' ')})."
        )

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return the feature importances from the trained model.

        Returns:
            Mapping from agent score column name to importance value.
            Empty dict if the model has not been fitted or does not expose importances.
        """
        return dict(self._feature_importances)

    def get_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Return the aligned, imputed feature matrix for ``data``.

        The Explanation Agent (Phase 5) calls this to obtain the same numpy
        array that ``predict()`` passes to the model, so SHAP values are
        computed on exactly the same input representation.

        Args:
            data: Wide-format or long-format agent scores DataFrame (same
                input contract as :meth:`predict`).

        Returns:
            Float64 numpy array of shape ``(n_rows, n_features)``.  Returns
            an empty array if the agent is not yet fitted.
        """
        if not self.is_fitted or not self._feature_columns:
            return np.empty((0, 0), dtype=np.float64)
        wide = self._pivot_agent_scores(data)
        features = self._extract_features(wide)
        if self._imputer is not None:
            return self._imputer.transform(features.to_numpy())
        return features.to_numpy(dtype=np.float64)

    def get_shap_values(self, data: pd.DataFrame) -> np.ndarray:
        """Compute per-row SHAP values for the fitted model.

        Intended for use by the Explanation Agent (Phase 5).  Returns a
        matrix of shape ``(n_rows, n_features)`` where each cell is the SHAP
        contribution of one agent's score to the final risk prediction for
        that row.

        Requires ``shap`` to be installed.  Falls back gracefully to a zero
        matrix if SHAP is unavailable or the model type is not supported.

        Args:
            data: Same input contract as :meth:`predict`.

        Returns:
            Float64 numpy array of shape ``(n_rows, n_features)``.
        """
        X = self.get_feature_matrix(data)
        if X.size == 0 or self._model is None:
            return np.zeros((len(data), len(self._feature_columns)), dtype=np.float64)
        try:
            import shap  # optional — only required in Phase 5
            # CalibratedClassifierCV is not directly supported by TreeExplainer;
            # unwrap to the underlying base estimator for SHAP computation.
            raw_model = None
            if hasattr(self._model, "estimator"):
                raw_model = self._model.estimator
            elif hasattr(self._model, "calibrated_classifiers_"):
                cal = self._model.calibrated_classifiers_[0]
                raw_model = getattr(cal, "estimator", None)
            elif hasattr(self._model, "feature_importances_"):
                raw_model = self._model
            if raw_model is None:
                raise ValueError("Cannot unwrap base estimator for SHAP.")
            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(X)
            # For binary classifiers TreeExplainer returns a list [class0, class1];
            # we want the fraud class (index 1).
            if isinstance(shap_values, list) and len(shap_values) == 2:
                return np.array(shap_values[1], dtype=np.float64)
            return np.array(shap_values, dtype=np.float64)
        except Exception as exc:
            self.logger.warning(
                "get_shap_values() failed (%s); returning zero matrix.", exc
            )
            return np.zeros((X.shape[0], len(self._feature_columns)), dtype=np.float64)
