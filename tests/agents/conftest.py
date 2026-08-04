"""Conftest for the agent test subpackage."""

from __future__ import annotations

import logging
import pickle

import pandas as pd
import pytest

from src.agents import BaseAgent


class StubAgent(BaseAgent):
    """Minimal concrete implementation used to exercise the base contract."""

    agent_name = "velocity"

    def fit(self, data: pd.DataFrame) -> "StubAgent":
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        return self.build_predictions(
            data, [0.2] * len(data), reason_code="STUB_SCORE"
        )

    def explain(self, transaction_id: str) -> str:
        return f"Stub explanation for transaction {transaction_id}."


@pytest.fixture()
def stub_agent() -> StubAgent:
    return StubAgent()


@pytest.fixture()
def basic_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["txn-1", "txn-2", "txn-3", "txn-4"],
        "timestamp": pd.to_datetime(
            [
                "2025-01-01 00:00",
                "2025-01-01 00:10",
                "2025-01-01 04:00",
                "2025-01-02 00:00",
            ],
            utc=True,
        ),
        "sender_account_id": ["a", "a", "b", "c"],
        "receiver_account_id": ["b", "c", "a", "a"],
        "amount_npr": [100.0, 200.0, 500.0, 1_000_000.0],
        "sender_country": ["Nepal", "Nepal", "Qatar", "Nepal"],
        "receiver_country": ["Nepal", "Nepal", "Nepal", "Nepal"],
        "is_cross_border": [0, 0, 1, 0],
        "remittance_corridor": [None, None, "Qatar->Nepal", None],
        "original_currency": ["NPR", "NPR", "QAR", "NPR"],
        "ip_country": ["Nepal", "Nepal", "Qatar", "Nepal"],
        "ip_is_vpn": [0, 0, 1, 0],
        "is_fraud": [0, 0, 1, 0],
    })
