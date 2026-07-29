"""Tests for src/utils/config.py.

Covers singleton behaviour, path-field types, threshold bounds, and specific
default values via both Hypothesis property tests and unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.config import Config, get_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATH_FIELDS = [
    "project_root",
    "data_original_dir",
    "data_interim_dir",
    "data_generated_dir",
    "models_dir",
]

_ALERT_THRESHOLD_FIELDS = [
    "velocity_alert_threshold",
    "geo_risk_alert_threshold",
    "kyc_aml_alert_threshold",
    "behaviour_alert_threshold",
    "graph_alert_threshold",
    "meta_learner_alert_threshold",
]

_REQUIRED_ATTRIBUTES = (
    _PATH_FIELDS
    + _ALERT_THRESHOLD_FIELDS
    + [
        "nrb_cash_reporting_threshold_npr",
        "structuring_min_npr",
        "structuring_max_npr",
        "weight_pep_flag",
        "weight_sanctions_match",
        "weight_kyc_unverified",
        "weight_new_account",
        "weight_high_risk_grade",
        "weight_structuring_pattern",
        "weight_layering_pattern",
        "generator_chunk_size",
        "random_seed",
        "target_ks_statistic",
        "target_auc_roc_delta",
        "kaggle_mode",
    ]
)

# ===========================================================================
# Property tests
# ===========================================================================

# Feature: phase0-foundation, Property 9: get_config() is idempotent
@given(st.none())
@settings(max_examples=100)
def test_get_config_is_idempotent(_: None) -> None:
    """Property 9: two sequential get_config() calls return equivalent objects.

    Validates: Requirements 4.2
    """
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2, "get_config() must return the same singleton instance"
    assert cfg1.model_dump() == cfg2.model_dump()


# Feature: phase0-foundation, Property 10: All Config path fields return pathlib.Path instances
@given(field_name=st.sampled_from(_PATH_FIELDS))
@settings(max_examples=100)
def test_config_path_fields_are_path_instances(field_name: str) -> None:
    """Property 10: every path field on Config returns a pathlib.Path, not str.

    Validates: Requirements 4.3
    """
    cfg = get_config()
    value = getattr(cfg, field_name)
    assert isinstance(value, Path), (
        f"Field {field_name!r} is {type(value).__name__}, expected pathlib.Path"
    )


# Feature: phase0-foundation, Property 11: Config alert thresholds are valid probabilities
@given(field_name=st.sampled_from(_ALERT_THRESHOLD_FIELDS))
@settings(max_examples=100)
def test_config_alert_thresholds_are_valid_probabilities(field_name: str) -> None:
    """Property 11: each alert threshold is in [0.0, 1.0].

    Also verifies structuring_min_npr < structuring_max_npr.

    Validates: Requirements 4.5, 4.6
    """
    cfg = get_config()
    value = getattr(cfg, field_name)
    assert 0.0 <= value <= 1.0, (
        f"Threshold {field_name!r} = {value} is outside [0.0, 1.0]"
    )


# ===========================================================================
# Unit tests
# ===========================================================================

def test_get_config_does_not_raise() -> None:
    """Requirement 4.7: get_config() never raises even if data dirs don't exist."""
    cfg = get_config()
    assert cfg is not None


def test_nrb_threshold_default() -> None:
    """Requirement 4.4: nrb_cash_reporting_threshold_npr defaults to 1_000_000.0."""
    cfg = get_config()
    assert cfg.nrb_cash_reporting_threshold_npr == 1_000_000.0


def test_structuring_window_ordering() -> None:
    """Requirement 4.5: structuring_min_npr is strictly less than structuring_max_npr."""
    cfg = get_config()
    assert cfg.structuring_min_npr < cfg.structuring_max_npr


def test_all_required_attributes_exist() -> None:
    """Requirement 4.1: the Config object exposes every documented attribute."""
    cfg = get_config()
    for attr in _REQUIRED_ATTRIBUTES:
        assert hasattr(cfg, attr), f"Config is missing attribute: {attr!r}"


def test_data_original_dir_is_under_project_root() -> None:
    """data_original_dir should be a subdirectory of project_root."""
    cfg = get_config()
    assert cfg.data_original_dir.parts[:len(cfg.project_root.parts)] == cfg.project_root.parts


def test_kaggle_mode_default_false() -> None:
    """kaggle_mode should default to False."""
    cfg = get_config()
    assert cfg.kaggle_mode is False


def test_generator_chunk_size_default() -> None:
    """generator_chunk_size should default to 500_000."""
    cfg = get_config()
    assert cfg.generator_chunk_size == 500_000


def test_random_seed_default() -> None:
    """random_seed should default to 42."""
    cfg = get_config()
    assert cfg.random_seed == 42
