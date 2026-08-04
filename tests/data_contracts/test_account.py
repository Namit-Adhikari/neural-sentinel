"""Property and unit tests for the Account Pydantic contract."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.data_contracts import Account

from tests.data_contracts.conftest import base_account_kwargs, valid_account_st


_VALID_ACCOUNT_TYPES = ["savings", "current", "salary", "fixed_deposit"]
_VALID_KYC_GRADES = ["low", "medium", "high"]


@given(kwargs=valid_account_st())
@settings(max_examples=100)
def test_account_round_trip(kwargs: dict[str, Any]) -> None:
    acct = Account(**kwargs)
    data = acct.model_dump()
    acct2 = Account(**data)
    assert acct.model_dump() == acct2.model_dump()


@given(
    bad_value=st.text(min_size=1).filter(
        lambda s: s not in _VALID_ACCOUNT_TYPES and s not in _VALID_KYC_GRADES
    ),
    target_field=st.sampled_from(["account_type", "kyc_risk_grade"]),
)
@settings(max_examples=100)
def test_invalid_account_categoricals_raise(bad_value: str, target_field: str) -> None:
    base = base_account_kwargs()
    if target_field == "account_type" and bad_value in _VALID_ACCOUNT_TYPES:
        return
    if target_field == "kyc_risk_grade" and bad_value in _VALID_KYC_GRADES:
        return
    base[target_field] = bad_value
    with pytest.raises(ValidationError):
        Account(**base)


def test_account_age_zero_is_valid() -> None:
    kwargs = base_account_kwargs()
    kwargs["account_age_days"] = 0
    acct = Account(**kwargs)
    assert acct.account_age_days == 0


def test_account_negative_age_raises() -> None:
    kwargs = base_account_kwargs()
    kwargs["account_age_days"] = -1
    with pytest.raises(ValidationError):
        Account(**kwargs)


def test_account_city_optional_none() -> None:
    kwargs = base_account_kwargs()
    kwargs["city"] = None
    acct = Account(**kwargs)
    assert acct.city is None


def test_account_json_serialization() -> None:
    acct = Account(**base_account_kwargs())
    parsed = json.loads(acct.model_dump_json())
    assert parsed["account_id"] == "ACC-001"
