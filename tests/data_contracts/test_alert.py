"""Property and unit tests for the AlertScore Pydantic contract."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.data_contracts import AlertScore

from tests.data_contracts.conftest import base_alert_kwargs, valid_alert_score_st


@given(kwargs=valid_alert_score_st())
@settings(max_examples=100)
def test_alert_score_round_trip(kwargs: dict[str, Any]) -> None:
    score = AlertScore(**kwargs)
    data = score.model_dump()
    score2 = AlertScore(**data)
    assert score.model_dump() == score2.model_dump()


@given(
    bad_score=st.one_of(
        st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100)
def test_alert_score_out_of_range_raises(bad_score: float) -> None:
    with pytest.raises(ValidationError):
        AlertScore(**{**base_alert_kwargs(), "risk_score": bad_score})


def test_alert_score_zero_is_valid() -> None:
    kwargs = base_alert_kwargs()
    kwargs["risk_score"] = 0.0
    score = AlertScore(**kwargs)
    assert score.risk_score == 0.0


def test_alert_score_one_is_valid() -> None:
    kwargs = base_alert_kwargs()
    kwargs["risk_score"] = 1.0
    score = AlertScore(**kwargs)
    assert score.risk_score == 1.0


def test_alert_score_just_above_one_raises() -> None:
    kwargs = base_alert_kwargs()
    kwargs["risk_score"] = 1.0001
    with pytest.raises(ValidationError):
        AlertScore(**kwargs)


def test_alert_flag_invalid_raises() -> None:
    kwargs = base_alert_kwargs()
    kwargs["alert_flag"] = 2
    with pytest.raises(ValidationError):
        AlertScore(**kwargs)


def test_alert_score_json_serialization() -> None:
    score = AlertScore(**base_alert_kwargs())
    parsed = json.loads(score.model_dump_json())
    assert parsed["agent_name"] == "kyc_aml_agent"
