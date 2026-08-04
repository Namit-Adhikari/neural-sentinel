"""Property and unit tests for the Transaction Pydantic contract."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.data_contracts import Transaction

from tests.data_contracts.conftest import (
    base_transaction_kwargs,
    valid_transaction_st,
)


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


@given(kwargs=valid_transaction_st())
@settings(max_examples=100)
def test_transaction_round_trip(kwargs: dict[str, Any]) -> None:
    txn = Transaction(**kwargs)
    data = txn.model_dump()
    txn2 = Transaction(**data)
    assert txn.model_dump() == txn2.model_dump()


@given(
    bad_type=st.text(min_size=1).filter(lambda s: s not in _VALID_TRANSACTION_TYPES)
)
@settings(max_examples=100)
def test_invalid_transaction_type_raises(bad_type: str) -> None:
    kwargs = base_transaction_kwargs()
    kwargs["transaction_type"] = bad_type
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


@given(bad_amount=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_non_positive_amount_raises(bad_amount: float) -> None:
    kwargs = base_transaction_kwargs()
    kwargs["amount_npr"] = bad_amount
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


@given(is_domestic=st.booleans())
@settings(max_examples=100)
def test_cross_border_corridor_rule(is_domestic: bool) -> None:
    kwargs = base_transaction_kwargs()
    if is_domestic:
        kwargs["is_cross_border"] = 0
        kwargs["remittance_corridor"] = None
        txn = Transaction(**kwargs)
        assert txn.is_cross_border == 0
    else:
        kwargs["is_cross_border"] = 1
        kwargs["remittance_corridor"] = None
        with pytest.raises(ValidationError):
            Transaction(**kwargs)


@given(
    bad_channel=st.text(min_size=1).filter(lambda s: s not in _VALID_CHANNELS)
)
@settings(max_examples=100)
def test_invalid_channel_raises(bad_channel: str) -> None:
    kwargs = base_transaction_kwargs()
    kwargs["channel"] = bad_channel
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


def test_transaction_valid_construction() -> None:
    txn = Transaction(**base_transaction_kwargs())
    assert txn.amount_npr == 5000.0
    assert txn.is_cross_border == 0


def test_transaction_cross_border_with_corridor_valid() -> None:
    kwargs = base_transaction_kwargs()
    kwargs.update(is_cross_border=1, remittance_corridor="Qatar->Nepal")
    txn = Transaction(**kwargs)
    assert txn.remittance_corridor == "Qatar->Nepal"


def test_transaction_is_fraud_binary_constraint() -> None:
    kwargs = base_transaction_kwargs()
    kwargs["is_fraud"] = 2
    with pytest.raises(ValidationError):
        Transaction(**kwargs)


def test_transaction_fraud_type_allowed_with_is_fraud_zero() -> None:
    kwargs = base_transaction_kwargs()
    kwargs["is_fraud"] = 0
    kwargs["fraud_type"] = "transaction_fraud"
    txn = Transaction(**kwargs)
    assert txn.fraud_type == "transaction_fraud"


def test_transaction_str_strip_whitespace() -> None:
    kwargs = base_transaction_kwargs()
    kwargs["transaction_id"] = "  txn-space  "
    txn = Transaction(**kwargs)
    assert txn.transaction_id == "txn-space"


def test_transaction_json_serialization() -> None:
    txn = Transaction(**base_transaction_kwargs())
    parsed = json.loads(txn.model_dump_json())
    assert parsed["transaction_id"] == "txn-001"
