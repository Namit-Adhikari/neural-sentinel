"""Centralized configuration for the Neural Sentinel project.

All paths, NRB regulatory thresholds, per-agent alert thresholds, KYC/AML rule
weights, generator hyperparameters, evaluation targets, and runtime flags live
here.  No magic numbers or hardcoded paths should appear in agent, generator, or
notebook code — always call ``get_config()`` and reference the returned object.

Usage::

    from src.utils.config import get_config
    cfg = get_config()
    print(cfg.data_raw_dir)          # pathlib.Path
    print(cfg.nrb_cash_reporting_threshold_npr)  # 1_000_000.0

The ``get_config()`` function returns a cached singleton so repeated calls are
free of object-construction overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from src.utils.nepal_context import NRB_CASH_REPORTING_THRESHOLD_NPR

logger = logging.getLogger(__name__)

# Project root is three levels above this file:
#   src/utils/config.py  →  src/utils/  →  src/  →  <project_root>/
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Config(BaseModel):
    """Typed configuration object for Neural Sentinel.

    All fields have sensible defaults so that the object can be instantiated
    without any external configuration file.  Path fields are resolved relative
    to the project root at construction time.

    Attributes:
        project_root: Absolute path to the repository root.
        data_original_dir: Directory containing original / source data files.
        data_interim_dir: Directory for cleaned, canonical-schema data.
        data_generated_dir: Directory for the 5 M-row generated dataset.
        models_dir: Directory for serialized model artefacts.
        nrb_cash_reporting_threshold_npr: NRB cash-reporting threshold (NPR).
        structuring_min_npr: Lower bound of the structuring detection window.
        structuring_max_npr: Upper bound of the structuring detection window.
        velocity_alert_threshold: Alert threshold for the Velocity agent.
        geo_risk_alert_threshold: Alert threshold for the Geo-Risk agent.
        kyc_aml_alert_threshold: Alert threshold for the KYC/AML agent.
        behaviour_alert_threshold: Alert threshold for the Behaviour agent.
        graph_alert_threshold: Alert threshold for the Graph agent.
        meta_learner_alert_threshold: Alert threshold for the Meta-Learner.
        weight_pep_flag: KYC/AML rule weight for PEP flag.
        weight_sanctions_match: KYC/AML rule weight for sanctions match.
        weight_kyc_unverified: KYC/AML rule weight for unverified KYC.
        weight_new_account: KYC/AML rule weight for new accounts (< 90 days).
        weight_high_risk_grade: KYC/AML rule weight for high-risk grade.
        weight_structuring_pattern: KYC/AML rule weight for structuring pattern.
        weight_layering_pattern: KYC/AML rule weight for layering pattern.
        generator_chunk_size: Number of rows to generate per batch.
        random_seed: Global random seed for reproducibility.
        target_ks_statistic: Maximum acceptable KS statistic for fidelity check.
        target_auc_roc_delta: Maximum acceptable AUC-ROC drop vs. real-trained model.
        kaggle_mode: Set True when running inside a Kaggle notebook environment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = _PROJECT_ROOT
    data_original_dir: Path = _PROJECT_ROOT / "data" / "original"
    data_interim_dir: Path = _PROJECT_ROOT / "data" / "interim"
    data_generated_dir: Path = _PROJECT_ROOT / "data" / "generated"
    models_dir: Path = _PROJECT_ROOT / "models"

    # ------------------------------------------------------------------
    # NRB regulatory thresholds
    # Default values are imported from nepal_context so there is a single
    # source of truth.
    # ------------------------------------------------------------------
    nrb_cash_reporting_threshold_npr: float = NRB_CASH_REPORTING_THRESHOLD_NPR
    structuring_min_npr: float = 900_000.0
    structuring_max_npr: float = 999_000.0

    # ------------------------------------------------------------------
    # Per-agent alert thresholds
    # ------------------------------------------------------------------
    velocity_alert_threshold: float = 0.7
    geo_risk_alert_threshold: float = 0.65
    velocity_windows_hours: tuple[int, ...] = (1, 6, 24, 168)
    velocity_isolation_contamination: float = 0.05
    velocity_isolation_estimators: int = 100
    velocity_spike_min_transactions: int = 3
    geo_risk_iterations: int = 300
    geo_risk_learning_rate: float = 0.05
    geo_risk_depth: int = 6
    geo_risk_random_seed: int = 42
    kyc_aml_alert_threshold: float = 0.6
    behaviour_alert_threshold: float = 0.7
    graph_alert_threshold: float = 0.65
    meta_learner_alert_threshold: float = 0.5

    # ------------------------------------------------------------------
    # KYC/AML rule weights
    # ------------------------------------------------------------------
    weight_pep_flag: float = 0.4
    weight_sanctions_match: float = 1.0
    weight_kyc_unverified: float = 0.5
    weight_new_account: float = 0.3
    weight_high_risk_grade: float = 0.3
    weight_structuring_pattern: float = 0.6
    weight_layering_pattern: float = 0.7

    # ------------------------------------------------------------------
    # Generator settings
    # ------------------------------------------------------------------
    generator_chunk_size: int = 500_000
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Evaluation targets
    # ------------------------------------------------------------------
    target_ks_statistic: float = 0.05
    target_auc_roc_delta: float = 0.05

    # ------------------------------------------------------------------
    # Runtime flags
    # ------------------------------------------------------------------
    kaggle_mode: bool = False


# Module-level singleton — populated on first call to get_config().
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the singleton :class:`Config` instance.

    On the first call the :class:`Config` object is constructed from defaults
    and each path field is checked for existence.  If a path does not exist a
    ``WARNING`` is logged but no exception is raised — paths are created lazily
    by pipeline code, not by the config module.

    Subsequent calls return the cached instance without reconstruction.

    Returns:
        The singleton :class:`Config` object.
    """
    global _config
    if _config is None:
        _config = Config()
        _check_paths(_config)
    return _config


def _check_paths(cfg: Config) -> None:
    """Log a warning for every path field that does not exist on disk.

    Args:
        cfg: The :class:`Config` instance whose path fields to inspect.
    """
    path_fields = [
        "project_root",
        "data_original_dir",
        "data_interim_dir",
        "data_generated_dir",
        "models_dir",
    ]
    for field_name in path_fields:
        path_value: Path = getattr(cfg, field_name)
        if not path_value.exists():
            logger.warning(
                "Config path does not exist (will be created by pipeline): %s = %s",
                field_name,
                path_value,
            )
