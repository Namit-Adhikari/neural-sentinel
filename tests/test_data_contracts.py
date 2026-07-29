"""Tests for src/data_contracts.py.

Covers all three Pydantic models (Transaction, Account, AlertScore) via
Hypothesis property tests for universal invariants and unit tests for specific
edge cases.
"""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.data_contracts import Account, AlertScore, Transaction

# ---------------------------------------------------------------------------
# Valid-value constants for strategies
# ---------------------------------------------------------------------------

_VALID_TRANSACTION_TYPES = [
    "transfer",
    "payment",
    "withdrawal",
    "deposit",
    "cash_out",
    "remittance_inbound",
    "remittance_outbound",
]

_VALID_CHANNELS = [
    "mobile_banking",
    "atm",
    "branch",
    "online_banking",
    "pos",
]

_VALID_ACCOUNT_TYPES = ["savings", "current", "salary", "fixed_deposit"]
_VALID_KYC_GRADES = ["low", "medium", "high"]

_VALID_FRAUD_TYPES = [
    "transaction_fraud",
    "aml_structuring",
    "aml_layering",
    "aml_mule_network",
    "identity_fraud",
]

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_date_st = st.dates(
    min_value=datetime.date(2020, 1, 1),
    max_value=datetime.date(2025, 12, 31),
)
_time_st = st.times()
_non_empty_text_st = st.text(min_size=1, max_size=50).filter(lambda s: s.strip())


def _base_transaction_kwargs() -> dict[str, Any]:
    """Return a minimal dict of valid Transaction field values for direct construction."""
    return dict(
        transaction_id="txn-001",
        transaction_date=datetime.date(2024, 1, 15),
        transaction_time=datetime.time(10, 30, 0),
        sender_account_id="ACC001",
        receiver_account_id="ACC002",
        transaction_type="transfer",
        amount_npr=5000.0,
        original_currency="NPR",
        exchange_rate=1.0,
        channel="mobile_banking",
        sender_country=None,
        receiver_country=None,
        is_cross_border=0,
        remittance_corridor=None,
        merchant_category=None,
        device_type="mobile",
        ip_address="192.168.1.1",
        ip_country=None,
        ip_is_vpn=0,
        is_fraud=0,
        fraud_type=None,
        aml_risk_indicator=0,
    )


@st.composite
def valid_transaction_st(draw: st.DrawFn) -> dict[str, Any]:
    """Hypothesis strategy producing dicts for valid Transaction construction."""
    is_cross_border = draw(st.integers(min_value=0, max_value=1))
    corridor = None
    if is_cross_border == 1:
        corridor = draw(st.sampled_from(["Qatar->Nepal", "India->Nepal", "USA->Nepal"]))

    return dict(
        transaction_id=draw(_non_empty_text_st),
        transaction_date=draw(_date_st),
        transaction_time=draw(_time_st),
        sender_account_id=draw(_non_empty_text_st),
        receiver_account_id=draw(_non_empty_text_st),
        transaction_type=draw(st.sampled_from(_VALID_TRANSACTION_TYPES)),
        amount_npr=draw(st.floats(min_value=0.01, max_value=10_000_000.0, allow_nan=False, allow_infinity=False)),
        original_currency=draw(_non_empty_text_st),
        exchange_rate=draw(st.floats(min_value=0.01, max_value=200.0, allow_nan=False, allow_infinity=False)),
        channel=draw(st.sampled_from(_VALID_CHANNELS)),
        sender_country=draw(st.none() | _non_empty_text_st),
        receiver_country=draw(st.none() | _non_empty_text_st),
        is_cross_border=is_cross_border,
        remittance_corridor=corridor,
        merchant_category=draw(st.none() | _non_empty_text_st),
        device_type=draw(_non_empty_text_st),
        ip_address=draw(_non_empty_text_st),
        ip_country=draw(st.none() | _non_empty_text_st),
        ip_is_vpn=draw(st.integers(min_value=0, max_value=1)),
        is_fraud=draw(st.integers(min_value=0, max_value=1)),
        fraud_type=draw(st.none() | st.sampled_from(_VALID_FRAUD_TYPES)),
        aml_risk_indicator=draw(st.integers(min_value=0, max_value=1)),
    )


@st.composite
def valid_account_st(draw: st.DrawFn) -> dict[str, Any]:
    """Hypothesis strategy producing dicts for valid Account construction."""
    return dict(
        account_id=draw(_non_empty_text_st),
        account_type=draw(st.sampled_from(_VALID_ACCOUNT_TYPES)),
        account_open_date=draw(_date_st),
        account_age_days=draw(st.integers(min_value=0, max_value=36500)),
        kyc_verified=draw(st.integers(min_value=0, max_value=1)),
        kyc_risk_grade=draw(st.sampled_from(_VALID_KYC_GRADES)),
        is_pep=draw(st.integers(min_value=0, max_value=1)),
        is_sanctioned=draw(st.integers(min_value=0, max_value=1)),
        average_monthly_volume=draw(st.floats(min_value=0.0, max_value=100_000_000.0, allow_nan=False, allow_infinity=False)),
        average_monthly_count=draw(st.integers(min_value=0, max_value=10000)),
        country=draw(_non_empty_text_st),
        city=draw(st.none() | _non_empty_text_st),
        is_mule=draw(st.integers(min_value=0, max_value=1)),
    )


@st.composite
def valid_alert_score_st(draw: st.DrawFn) -> dict[str, Any]:
    """Hypothesis strategy producing dicts for valid AlertScore construction."""
    return dict(
        transaction_id=draw(_non_empty_text_st),
        agent_name=draw(_non_empty_text_st),
        risk_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        alert_flag=draw(st.integers(min_value=0, max_value=1)),
        reason_code=draw(_non_empty_text_st),
        explanation=draw(_non_empty_text_st),
        timestamp=draw(
            st.datetimes(
                min_value=datetime.datetime(2020, 1, 1),
                max_value=datetime.datetime(2025, 12, 31),
            )
        ),
    )


# ===========================================================================
# Property tests — Transaction
# ===========================================================================

# Feature: phase0-foundation, Property 1: Transaction serialization round-trip
@given(kwargs=valid_transaction_st())
@settings(max_examples=100)
def test_transaction_round_trip(kwargs: dict[str, Any]) -> None:
    """Property 1: Transaction model_dump() round-trip preserves all field values.

    Validates: Requirements 1.1, 1.7
    """
    txn = Transaction(**kwargs)
    data = txn.model_dump()
    txn2 = Transaction(**data)
    assert txn.model_dump() == txn2.model_dump()


# Feature: phase0-foundation, Property 2: Invalid transaction_type always raises ValidationError
@given(
    bad_type=st.text(min_size=1).filter(lambda s: s not in _VALID_TRANSACTION_TYPES)
)
@settings(max_examples=100)
def test_invalid_transaction_type_raises(bad_type: str) -> None:
    """Property 2: any transaction_type not in the allowed set raises ValidationError.

    Validates: Requirements 1.2
    """
    kwargs = _base_transaction_kwargs()
    kwargs["transaction_type"] = bad_type
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


# Feature: phase0-foundation, Property 3: Non-positive amount_npr always raises ValidationError
@given(bad_amount=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_non_positive_amount_raises(bad_amount: float) -> None:
    """Property 3: amount_npr <= 0 always raises ValidationError.

    Validates: Requirements 1.3
    """
    kwargs = _base_transaction_kwargs()
    kwargs["amount_npr"] = bad_amount
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


# Feature: phase0-foundation, Property 4: Cross-border transaction requires remittance corridor
@given(is_domestic=st.booleans())
@settings(max_examples=100)
def test_cross_border_corridor_rule(is_domestic: bool) -> None:
    """Property 4: is_cross_border=1 with corridor=None raises; is_cross_border=0 is fine.

    Validates: Requirements 1.5
    """
    kwargs = _base_transaction_kwargs()
    if is_domestic:
        kwargs["is_cross_border"] = 0
        kwargs["remittance_corridor"] = None
        txn = Transaction(**kwargs)  # must not raise
        assert txn.is_cross_border == 0
    else:
        kwargs["is_cross_border"] = 1
        kwargs["remittance_corridor"] = None
        with pytest.raises(ValidationError):
            Transaction(**kwargs)


# Feature: phase0-foundation, Property 5: Invalid channel always raises ValidationError
@given(
    bad_channel=st.text(min_size=1).filter(lambda s: s not in _VALID_CHANNELS)
)
@settings(max_examples=100)
def test_invalid_channel_raises(bad_channel: str) -> None:
    """Property 5: any channel not in the allowed set raises ValidationError.

    Validates: Requirements 1.6
    """
    kwargs = _base_transaction_kwargs()
    kwargs["channel"] = bad_channel
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


# ===========================================================================
# Property tests — Account
# ===========================================================================

# Feature: phase0-foundation, Property 6: Account serialization round-trip
@given(kwargs=valid_account_st())
@settings(max_examples=100)
def test_account_round_trip(kwargs: dict[str, Any]) -> None:
    """Property 6: Account model_dump() round-trip preserves all field values.

    Validates: Requirements 2.1, 2.5
    """
    acct = Account(**kwargs)
    data = acct.model_dump()
    acct2 = Account(**data)
    assert acct.model_dump() == acct2.model_dump()


# Feature: phase0-foundation, Property 7: Invalid Account categorical fields raise ValidationError
@given(
    bad_value=st.text(min_size=1).filter(
        lambda s: s not in _VALID_ACCOUNT_TYPES and s not in _VALID_KYC_GRADES
    ),
    target_field=st.sampled_from(["account_type", "kyc_risk_grade"]),
)
@settings(max_examples=100)
def test_invalid_account_categoricals_raise(bad_value: str, target_field: str) -> None:
    """Property 7: invalid account_type or kyc_risk_grade raises ValidationError.

    Validates: Requirements 2.2, 2.3
    """
    base = dict(
        account_id="ACC-001",
        account_type="savings",
        account_open_date=datetime.date(2020, 1, 1),
        account_age_days=365,
        kyc_verified=1,
        kyc_risk_grade="low",
        is_pep=0,
        is_sanctioned=0,
        average_monthly_volume=10000.0,
        average_monthly_count=10,
        country="Nepal",
        city="Kathmandu",
        is_mule=0,
    )
    if target_field == "account_type" and bad_value in _VALID_ACCOUNT_TYPES:
        return
    if target_field == "kyc_risk_grade" and bad_value in _VALID_KYC_GRADES:
        return
    base[target_field] = bad_value
    with pytest.raises(ValidationError):
        Account(**base)


# ===========================================================================
# Property tests — AlertScore
# ===========================================================================

# Feature: phase0-foundation, Property 8: AlertScore serialization round-trip and risk_score bounds
@given(kwargs=valid_alert_score_st())
@settings(max_examples=100)
def test_alert_score_round_trip(kwargs: dict[str, Any]) -> None:
    """Property 8a: AlertScore model_dump() round-trip preserves all field values.

    Validates: Requirements 3.1, 3.4
    """
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
    """Property 8b: risk_score outside [0.0, 1.0] raises ValidationError.

    Validates: Requirements 3.2
    """
    with pytest.raises(ValidationError):
        AlertScore(
            transaction_id="txn-1",
            agent_name="velocity_agent",
            risk_score=bad_score,
            alert_flag=0,
            reason_code="TEST",
            explanation="test",
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )


# ===========================================================================
# Unit tests — Transaction edge cases
# ===========================================================================

def test_transaction_valid_construction() -> None:
    """A fully valid Transaction constructs without error."""
    txn = Transaction(**_base_transaction_kwargs())
    assert txn.amount_npr == 5000.0
    assert txn.is_cross_border == 0


def test_transaction_cross_border_with_corridor_valid() -> None:
    """is_cross_border=1 with a corridor string is valid."""
    kwargs = _base_transaction_kwargs()
    kwargs.update(is_cross_border=1, remittance_corridor="Qatar->Nepal")
    txn = Transaction(**kwargs)
    assert txn.remittance_corridor == "Qatar->Nepal"


def test_transaction_is_fraud_binary_constraint() -> None:
    """Requirement 1.4: is_fraud=2 raises ValidationError."""
    kwargs = _base_transaction_kwargs()
    kwargs["is_fraud"] = 2
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


def test_transaction_fraud_type_allowed_with_is_fraud_zero() -> None:
    """fraud_type may be set even when is_fraud=0 (no cross-field rule in Phase 0)."""
    kwargs = _base_transaction_kwargs()
    kwargs["is_fraud"] = 0
    kwargs["fraud_type"] = "transaction_fraud"
    txn = Transaction(**kwargs)
    assert txn.fraud_type == "transaction_fraud"


def test_transaction_str_strip_whitespace() -> None:
    """model_config str_strip_whitespace strips leading/trailing spaces."""
    kwargs = _base_transaction_kwargs()
    kwargs["transaction_id"] = "  txn-space  "
    txn = Transaction(**kwargs)
    assert txn.transaction_id == "txn-space"


def test_transaction_json_serialization() -> None:
    """Requirement 1.7: Transaction.model_dump_json() produces valid JSON."""
    import json
    txn = Transaction(**_base_transaction_kwargs())
    json_str = txn.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["transaction_id"] == "txn-001"


# ===========================================================================
# Unit tests — Account edge cases
# ===========================================================================

def _base_account_kwargs() -> dict[str, Any]:
    return dict(
        account_id="ACC-001",
        account_type="savings",
        account_open_date=datetime.date(2020, 6, 1),
        account_age_days=365,
        kyc_verified=1,
        kyc_risk_grade="low",
        is_pep=0,
        is_sanctioned=0,
        average_monthly_volume=50000.0,
        average_monthly_count=20,
        country="Nepal",
        city="Kathmandu",
        is_mule=0,
    )


def test_account_age_zero_is_valid() -> None:
    """Requirement 2.4: account_age_days=0 is valid (brand-new account)."""
    kwargs = _base_account_kwargs()
    kwargs["account_age_days"] = 0
    acct = Account(**kwargs)
    assert acct.account_age_days == 0


def test_account_negative_age_raises() -> None:
    """Requirement 2.4: account_age_days=-1 raises ValidationError."""
    kwargs = _base_account_kwargs()
    kwargs["account_age_days"] = -1
    with pytest.raises(ValidationError):
        Account(**kwargs)


def test_account_city_optional_none() -> None:
    """city may be None (optional field)."""
    kwargs = _base_account_kwargs()
    kwargs["city"] = None
    acct = Account(**kwargs)
    assert acct.city is None


def test_account_json_serialization() -> None:
    """Requirement 2.5: Account.model_dump_json() produces valid JSON."""
    import json
    acct = Account(**_base_account_kwargs())
    parsed = json.loads(acct.model_dump_json())
    assert parsed["account_id"] == "ACC-001"


# ===========================================================================
# Unit tests — AlertScore edge cases
# ===========================================================================

def _base_alert_kwargs() -> dict[str, Any]:
    return dict(
        transaction_id="txn-001",
        agent_name="kyc_aml_agent",
        risk_score=0.75,
        alert_flag=1,
        reason_code="HIGH_RISK_CORRIDOR",
        explanation="Transaction in high-risk corridor with unverified KYC.",
        timestamp=datetime.datetime(2024, 3, 15, 14, 30, 0),
    )


def test_alert_score_zero_is_valid() -> None:
    """Requirement 3.2: risk_score=0.0 is a valid lower boundary."""
    kwargs = _base_alert_kwargs()
    kwargs["risk_score"] = 0.0
    score = AlertScore(**kwargs)
    assert score.risk_score == 0.0


def test_alert_score_one_is_valid() -> None:
    """Requirement 3.2: risk_score=1.0 is a valid upper boundary."""
    kwargs = _base_alert_kwargs()
    kwargs["risk_score"] = 1.0
    score = AlertScore(**kwargs)
    assert score.risk_score == 1.0


def test_alert_score_just_above_one_raises() -> None:
    """Requirement 3.2: risk_score=1.0001 raises ValidationError."""
    kwargs = _base_alert_kwargs()
    kwargs["risk_score"] = 1.0001
    with pytest.raises(ValidationError):
        AlertScore(**kwargs)


def test_alert_flag_invalid_raises() -> None:
    """Requirement 3.3: alert_flag=2 raises ValidationError."""
    kwargs = _base_alert_kwargs()
    kwargs["alert_flag"] = 2
    with pytest.raises(ValidationError):
        AlertScore(**kwargs)


def test_alert_score_json_serialization() -> None:
    """Requirement 3.4: AlertScore.model_dump_json() produces valid JSON."""
    import json
    score = AlertScore(**_base_alert_kwargs())
    parsed = json.loads(score.model_dump_json())
    assert parsed["agent_name"] == "kyc_aml_agent"
