"""
src/evaluation/ml_utility.py
-------------------------------
Machine Learning Utility metrics for synthetic data evaluation.

Implements the full model suite from Benchmark_pipeline.md (CTAB-GAN+ paper):

Classification models
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Linear SVM
  - MLP

Metrics per model
  - Accuracy
  - F1 (weighted)
  - ROC-AUC

Evaluation protocol
  - 80/20 stratified split on REAL data
  - Train each model on REAL train set  → evaluate on REAL test set
  - Train each model on SYNTHETIC set   → evaluate on REAL test set
  - Difference = abs(real_metric − synthetic_metric)
  - Average difference across all models

Usage
-----
    from src.evaluation.ml_utility import compute_ml_utility, compute_full_utility

    # Quick single-model (backwards compat)
    result = compute_ml_utility(real_df, synthetic_df, "is_suspicious_tx", feature_cols)

    # Full 5-model suite matching the paper
    result = compute_full_utility(real_df, synthetic_df, "is_suspicious_tx", feature_cols)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Original function (backwards-compatible — uses XGBoost only)
# ---------------------------------------------------------------------------

def compute_ml_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict:
    """Train XGBoost on synthetic data and evaluate AUC on real held-out test.

    This is the original single-model function, preserved for backwards
    compatibility. For the full CTAB-GAN+ paper suite, use ``compute_full_utility``.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not available; using RandomForest as fallback")
        XGBClassifier = None

    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df_proc = df[feature_cols + [target_col]].copy()
        for c in df_proc.columns:
            if df_proc[c].dtype == "object":
                le = LabelEncoder()
                df_proc[c] = le.fit_transform(df_proc[c].astype(str))
        return df_proc.apply(pd.to_numeric, errors="coerce").fillna(0)

    try:
        real_proc = preprocess(real_data)
        syn_proc  = preprocess(synthetic_data)
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        return {"auc_roc_diff": None, "real_auc": None, "synthetic_auc": None}

    X_real = real_proc.drop(columns=[target_col])
    y_real = real_proc[target_col]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    ModelClass = XGBClassifier if XGBClassifier else RandomForestClassifier
    kwargs = {"use_label_encoder": False, "eval_metric": "logloss"} if XGBClassifier else {}

    model_r = ModelClass(**kwargs)
    model_r.fit(X_train_r, y_train_r)
    preds_r = model_r.predict_proba(X_test_r)[:, 1]
    auc_r   = roc_auc_score(y_test_r, preds_r)

    X_syn = syn_proc.drop(columns=[target_col])
    y_syn = syn_proc[target_col]
    model_s = ModelClass(**kwargs)
    model_s.fit(X_syn, y_syn)
    preds_s = model_s.predict_proba(X_test_r)[:, 1]
    auc_s   = roc_auc_score(y_test_r, preds_s)

    return {
        "real_auc":     float(auc_r),
        "synthetic_auc": float(auc_s),
        "auc_roc_diff": float(abs(auc_r - auc_s)),
    }


# ---------------------------------------------------------------------------
# Full 5-model suite (CTAB-GAN+ paper protocol)
# ---------------------------------------------------------------------------

def _get_classifiers() -> list[tuple[str, object]]:
    """Return the 5 classifiers from the CTAB-GAN+ paper."""
    return [
        ("DecisionTree",    DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("RandomForest",    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ("LogisticReg",     LogisticRegression(max_iter=1000, random_state=42)),
        ("LinearSVM",       CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))),
        ("MLP",             MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)),
    ]


def _preprocess_for_utility(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    fit_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode categoricals and fill NaNs. Returns (processed_df, encoders)."""
    avail_features = [c for c in feature_cols if c in df.columns]
    proc = df[avail_features + [target_col]].copy()

    encoders: dict[str, LabelEncoder] = fit_encoders or {}
    for col in proc.columns:
        if proc[col].dtype == "object" or proc[col].dtype.name == "category":
            le = encoders.get(col)
            if le is None:
                le = LabelEncoder()
                le.fit(proc[col].astype(str))
                encoders[col] = le
            # Handle unseen labels
            known = set(le.classes_)
            proc[col] = proc[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            proc[col] = le.transform(proc[col])

    proc = proc.apply(pd.to_numeric, errors="coerce").fillna(0)
    return proc, encoders


def compute_full_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    seed: int = 42,
) -> dict:
    """Full 5-model classification utility suite (CTAB-GAN+ paper protocol).

    Protocol
    --------
    1. Stratified 80/20 split on REAL data → (real_train, real_test)
    2. For each of 5 classifiers:
       a. Train on real_train → evaluate on real_test  → real_metrics
       b. Train on synthetic  → evaluate on real_test  → syn_metrics
       c. diff = abs(real - synthetic) per metric

    Parameters
    ----------
    real_data : pd.DataFrame
        The original (real) dataset.
    synthetic_data : pd.DataFrame
        Synthetic data generated by a synthesizer. Must have same columns.
    target_col : str
        Binary classification target column.
    feature_cols : list[str]
        Feature columns to use (subset that exists in both DataFrames).
    seed : int
        Random seed for train/test split reproducibility.

    Returns
    -------
    dict with keys:
        per_model       : {model_name: {real_*, syn_*, diff_*}}
        avg_accuracy    : avg diff in Accuracy across models
        avg_f1          : avg diff in F1 across models
        avg_auc         : avg diff in AUC across models
        real_auc        : avg AUC of real-trained models (baseline)
        synthetic_auc   : avg AUC of synthetic-trained models
    """
    avail_features = [c for c in feature_cols if c in real_data.columns and c in synthetic_data.columns]

    try:
        real_proc, encoders = _preprocess_for_utility(real_data, avail_features, target_col)
        syn_proc, _         = _preprocess_for_utility(synthetic_data, avail_features, target_col, fit_encoders=encoders)
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        return {}

    X_real = real_proc.drop(columns=[target_col])
    y_real = real_proc[target_col].astype(int)
    X_syn  = syn_proc.drop(columns=[target_col])
    y_syn  = syn_proc[target_col].astype(int)

    # Stratified split on real data
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_real, y_real, test_size=0.2, random_state=seed,
        stratify=y_real if y_real.nunique() > 1 else None,
    )

    results: dict[str, dict] = {}
    classifiers = _get_classifiers()

    for name, clf in classifiers:
        try:
            # Real-trained model
            clf_r = clf.__class__(**clf.get_params()) if hasattr(clf, "get_params") else clf
            clf_r.fit(X_train_r, y_train_r)
            y_pred_r = clf_r.predict(X_test_r)
            y_prob_r = clf_r.predict_proba(X_test_r)[:, 1] if hasattr(clf_r, "predict_proba") else y_pred_r

            real_acc = accuracy_score(y_test_r, y_pred_r)
            real_f1  = f1_score(y_test_r, y_pred_r, average="weighted", zero_division=0)
            real_auc = roc_auc_score(y_test_r, y_prob_r) if y_test_r.nunique() > 1 else 0.5

            # Synthetic-trained model evaluated on real test
            clf_s = clf.__class__(**clf.get_params()) if hasattr(clf, "get_params") else clf
            clf_s.fit(X_syn, y_syn)
            y_pred_s = clf_s.predict(X_test_r)
            y_prob_s = clf_s.predict_proba(X_test_r)[:, 1] if hasattr(clf_s, "predict_proba") else y_pred_s

            syn_acc = accuracy_score(y_test_r, y_pred_s)
            syn_f1  = f1_score(y_test_r, y_pred_s, average="weighted", zero_division=0)
            syn_auc = roc_auc_score(y_test_r, y_prob_s) if y_test_r.nunique() > 1 else 0.5

            results[name] = {
                "real_accuracy":  round(real_acc * 100, 4),
                "syn_accuracy":   round(syn_acc * 100, 4),
                "diff_accuracy":  round(abs(real_acc - syn_acc) * 100, 4),
                "real_f1":        round(real_f1, 4),
                "syn_f1":         round(syn_f1, 4),
                "diff_f1":        round(abs(real_f1 - syn_f1), 4),
                "real_auc":       round(real_auc, 4),
                "syn_auc":        round(syn_auc, 4),
                "diff_auc":       round(abs(real_auc - syn_auc), 4),
            }
            logger.info(
                "  %-15s real_auc=%.4f syn_auc=%.4f diff=%.4f",
                name, real_auc, syn_auc, abs(real_auc - syn_auc),
            )

        except Exception as e:
            logger.warning("  %-15s FAILED: %s", name, e)
            results[name] = {}

    # Aggregate means
    def _avg(key: str) -> float:
        vals = [r[key] for r in results.values() if key in r]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    return {
        "per_model":     results,
        "avg_accuracy":  _avg("diff_accuracy"),
        "avg_f1":        _avg("diff_f1"),
        "avg_auc":       _avg("diff_auc"),
        "real_auc":      _avg("real_auc"),
        "synthetic_auc": _avg("syn_auc"),
    }
