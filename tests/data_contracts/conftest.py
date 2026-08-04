"""Shared Hypothesis strategies and data-contract fixtures."""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from hypothesis import strategies as st

from src.data_contracts import Account, AlertScore, Transaction


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

_date_st = st.dates(
    min_value=datetime.date(2020, 1, 1),
    max_value=datetime.date(2025, 12, 31),
)
_time_st = st.times()
_non_empty_text_st = st.text(min_size=1, max_size=50).filter(lambda s: s.strip())


@pytest.fixture(scope="session")
def valid_transaction_types() -> list[str]:
    return list(_VALID_TRANSACTION_TYPES)


@pytest.fixture(scope="session")
def valid_channels() -> list[str]:
    return list(_VALID_CHANNELS)


@pytest.fixture(scope="session")
def valid_account_types() -> list[str]:
    return list(_VALID_ACCOUNT_TYPES)


@pytest.fixture(scope="session")
def valid_kyc_grades() -> list[str]:
    return list(_VALID_KYC_GRADES)


@pytest.fixture(scope="session")
def valid_fraud_types() -> list[str]:
    return list(_VALID_FRAUD_TYPES)


def base_transaction_kwargs() -> dict[str, Any]:
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


def base_account_kwargs() -> dict[str, Any]:
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


def base_alert_kwargs() -> dict[str, Any]:
    return dict(
        transaction_id="txn-001",
        agent_name="kyc_aml_agent",
        risk_score=0.75,
        alert_flag=1,
        reason_code="HIGH_RISK_CORRIDOR",
        explanation="Transaction in high-risk corridor with unverified KYC.",
        timestamp=datetime.datetime(2024, 3, 15, 14, 30, 0),
    )


@st.composite
def valid_transaction_st(draw: st.DrawFn) -> dict[str, Any]:
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
        amount_npr=draw(
            st.floats(
                min_value=0.01,
                max_value=10_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        original_currency=draw(_non_empty_text_st),
        exchange_rate=draw(
            st.floats(min_value=0.01, max_value=200.0, allow_nan=False, allow_infinity=False)
        ),
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
    return dict(
        account_id=draw(_non_empty_text_st),
        account_type=draw(st.sampled_from(_VALID_ACCOUNT_TYPES)),
        account_open_date=draw(_date_st),
        account_age_days=draw(st.integers(min_value=0, max_value=36500)),
        kyc_verified=draw(st.integers(min_value=0, max_value=1)),
        kyc_risk_grade=draw(st.sampled_from(_VALID_KYC_GRADES)),
        is_pep=draw(st.integers(min_value=0, max_value=1)),
        is_sanctioned=draw(st.integers(min_value=0, max_value=1)),
        average_monthly_volume=draw(
            st.floats(
                min_value=0.0,
                max_value=100_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        average_monthly_count=draw(st.integers(min_value=0, max_value=10000)),
        country=draw(_non_empty_text_st),
        city=draw(st.none() | _non_empty_text_st),
        is_mule=draw(st.integers(min_value=0, max_value=1)),
    )


@st.composite
def valid_alert_score_st(draw: st.DrawFn) -> dict[str, Any]:
    return dict(
        transaction_id=draw(_non_empty_text_st),
        agent_name=draw(_non_empty_text_st),
        risk_score=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
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
