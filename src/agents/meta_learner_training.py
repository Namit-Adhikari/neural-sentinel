"""Training utilities for MetaLearnerAgent.

Extracted to ``meta_learner_training.py`` so that ``meta_learner.py`` stays
within the 500-line limit required by AGENTS.md §10.1.

The ``MetaLearnerTrainingMixin`` class provides all model-building, fitting,
and feature-importance logic.  ``MetaLearnerAgent`` inherits from it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — fail gracefully in CPU-only test environments
# ---------------------------------------------------------------------------
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    CalibratedClassifierCV = None  # type: ignore[assignment,misc]
    RandomForestClassifier = None  # type: ignore[assignment,misc]
    train_test_split = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False


class MetaLearnerTrainingMixin:
    """Mixin providing training logic for MetaLearnerAgent.

    Attributes expected to exist on the host class:
        - ``n_estimators``, ``max_depth``, ``calibration_method``,
          ``calibration_cv``, ``calibration_val_fraction``, ``random_seed``,
          ``use_gpu``, ``xgb_learning_rate``, ``xgb_subsample``,
          ``xgb_colsample_bytree``, ``xgb_default_max_depth``,
          ``_scale_pos_weight``, ``_model``, ``_feature_columns``,
          ``_feature_importances``, ``logger``.
    """

    def _build_rf(self) -> Any:
        """Construct the calibrated Random Forest pipeline.

        Returns:
            A ``CalibratedClassifierCV`` wrapping a ``RandomForestClassifier``.
        """
        base = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        return CalibratedClassifierCV(
            estimator=base,
            method=self.calibration_method,
            cv=self.calibration_cv,
        )

    def _build_xgb(self) -> Any:
        """Construct the XGBoost classifier with optional GPU support.

        Returns:
            An ``xgboost.XGBClassifier`` instance.
        """
        device = "cuda" if self.use_gpu else "cpu"
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth if self.max_depth is not None else self.xgb_default_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            scale_pos_weight=self._scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_seed,
            device=device,
            verbosity=0,
        )

    def _compute_scale_pos_weight(self, y: "np.ndarray") -> float:
        """Compute XGBoost scale_pos_weight from label distribution.

        Returns the ratio of negative to positive samples, clamped to [1, 99].
        A ratio of 1 means balanced classes; a large ratio upweights fraud detection.

        Args:
            y: Integer label array (0 = legitimate, 1 = fraud).

        Returns:
            Float ratio n_negative / n_positive, clamped to [1.0, 99.0].
        """
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0:
            return 1.0
        return float(np.clip(n_neg / n_pos, 1.0, 99.0))

    def _fit_xgb(self, X: "np.ndarray", y: "np.ndarray") -> None:
        """Train the XGBoost challenger model.

        Sets ``self._model`` and updates ``self._scale_pos_weight`` before building
        the classifier so the imbalance ratio is reflected in training.

        Args:
            X: Imputed feature matrix of shape (n_samples, n_features).
            y: Integer label array of shape (n_samples,).
        """
        self._scale_pos_weight = self._compute_scale_pos_weight(y)
        self.logger.info(
            "Training MetaLearnerAgent (XGBoost, scale_pos_weight=%.2f) "
            "on %d samples, %d features.",
            self._scale_pos_weight,
            len(X),
            X.shape[1],
        )
        self._model = self._build_xgb()
        self._model.fit(X, y)

    def _fit_rf(self, X: "np.ndarray", y: "np.ndarray", labels: "pd.Series") -> None:
        """Train the calibrated Random Forest primary model.

        Uses cross-validated calibration when enough per-class samples exist;
        falls back to a held-out validation split (``calibration_val_fraction``)
        to avoid calibrating on training data (prefit path).

        Args:
            X: Imputed feature matrix of shape (n_samples, n_features).
            y: Integer label array of shape (n_samples,).
            labels: Original label Series (used only for class-count guard).
        """
        n_splits = min(
            self.calibration_cv,
            int(labels.sum()),
            int((labels == 0).sum()),
        )
        if n_splits < 2:
            # Not enough per-class samples for k-fold — use a hold-out split
            # so the calibrator is never fit on the same rows as the base RF.
            self.logger.warning(
                "Not enough per-class samples for CV calibration (%d); "
                "using hold-out split (%.0f%%) for prefit calibration.",
                n_splits,
                self.calibration_val_fraction * 100,
            )
            split_size = max(1, int(len(X) * self.calibration_val_fraction))
            if split_size >= len(X):
                # Degenerate: dataset too small to split — train and calibrate on full set
                self.logger.warning(
                    "Dataset too small to hold out a calibration split; "
                    "calibrating on full training set."
                )
                X_train, X_cal, y_train, y_cal = X, X, y, y
            else:
                X_train, X_cal, y_train, y_cal = train_test_split(
                    X, y,
                    test_size=self.calibration_val_fraction,
                    random_state=self.random_seed,
                    stratify=y if len(np.unique(y)) > 1 else None,
                )
            base = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.random_seed,
                n_jobs=-1,
            )
            base.fit(X_train, y_train)
            self._model = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv="prefit",
            )
            self._model.fit(X_cal, y_cal)
        else:
            self.logger.info(
                "Training MetaLearnerAgent (Random Forest + %s calibration) "
                "on %d samples, %d features.",
                self.calibration_method,
                len(X),
                X.shape[1],
            )
            self._model = self._build_rf()
            self._model.fit(X, y)

    def _store_feature_importances(self) -> None:
        """Extract and cache feature importances from the trained model.

        Handles both ``CalibratedClassifierCV`` (RF path) and plain estimators
        (XGBoost path).  Silently skips if the model does not expose importances.
        """
        try:
            raw_model = None
            if hasattr(self._model, "estimator") and hasattr(
                self._model.estimator, "feature_importances_"
            ):
                raw_model = self._model.estimator
            elif hasattr(self._model, "calibrated_classifiers_"):
                cal = self._model.calibrated_classifiers_[0]
                raw_model = getattr(cal, "estimator", None)
            elif hasattr(self._model, "feature_importances_"):
                raw_model = self._model

            if raw_model is not None and hasattr(raw_model, "feature_importances_"):
                importances = raw_model.feature_importances_
                self._feature_importances = dict(
                    zip(self._feature_columns, importances.tolist())
                )
                top = sorted(
                    self._feature_importances.items(), key=lambda kv: kv[1], reverse=True
                )
                self.logger.info(
                    "MetaLearnerAgent feature importances: %s",
                    ", ".join(f"{k}={v:.3f}" for k, v in top),
                )
        except Exception as exc:
            self.logger.debug("Could not extract feature importances: %s", exc)
