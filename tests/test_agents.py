"""Tests for the shared Phase 1 agent interface."""

from __future__ import annotations

import logging
import pickle

import pandas as pd
import pytest

from src.agents import BaseAgent
from src.agents.geo_risk_agent import GeoRiskAgent
from src.agents.velocity_agent import VelocityAgent


class StubAgent(BaseAgent):
    """Minimal concrete implementation used to exercise the base contract."""

    agent_name = "velocity"

    def fit(self, data: pd.DataFrame) -> "StubAgent":
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        return self.build_predictions(data, [0.2] * len(data), reason_code="STUB_SCORE")

    def explain(self, transaction_id: str) -> str:
        return f"Stub explanation for transaction {transaction_id}."


def test_agent_initializes_with_agent_specific_threshold() -> None:
    agent = StubAgent(config={"velocity_alert_threshold": 0.7})

    assert agent.alert_threshold == 0.7
    assert agent.agent_name == "velocity"
    assert isinstance(agent.logger, logging.Logger)


def test_predict_returns_complete_canonical_output_without_mutating_input() -> None:
    data = pd.DataFrame({"transaction_id": ["txn-1", "txn-2"], "amount_npr": [10.0, 20.0]})
    original = data.copy(deep=True)
    result = StubAgent(config={"alert_threshold": 0.1}).fit(data).predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert result["alert_flag"].tolist() == [1, 1]
    pd.testing.assert_frame_equal(data, original)


def test_missing_transaction_id_returns_schema_safe_empty_output() -> None:
    result = StubAgent().predict(pd.DataFrame({"amount_npr": [100.0]}))

    assert result.empty
    assert list(result.columns) == list(BaseAgent.prediction_columns)


def test_explain_returns_human_readable_text() -> None:
    explanation = StubAgent().explain("txn-1")

    assert "txn-1" in explanation


def test_agent_is_pickle_serializable() -> None:
    restored = pickle.loads(pickle.dumps(StubAgent(config={"alert_threshold": 0.8})))

    assert isinstance(restored, StubAgent)
    assert restored.alert_threshold == 0.8
    assert restored.logger.name == "neural_sentinel.agents.velocity"


def _transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["txn-1", "txn-2", "txn-3", "txn-4"],
        "timestamp": pd.to_datetime(
            ["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 04:00", "2025-01-02 00:00"],
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


def test_velocity_agent_returns_schema_and_does_not_mutate() -> None:
    data = _transactions()
    original = data.copy(deep=True)
    result = VelocityAgent(config={"velocity_alert_threshold": 0.5}).fit(data).predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result.risk_score.between(0, 1).all()
    pd.testing.assert_frame_equal(data, original)


def test_velocity_agent_handles_missing_features() -> None:
    result = VelocityAgent().fit(pd.DataFrame({"transaction_id": ["txn-1"]})).predict(pd.DataFrame({"transaction_id": ["txn-1"]}))

    assert len(result) == 1
    assert result.risk_score.iloc[0] == 0.0


def test_velocity_agent_raises_if_predict_called_before_fit() -> None:
    data = _transactions()
    with pytest.raises(RuntimeError, match="must be fitted"):
        VelocityAgent().predict(data)


def test_geo_risk_agent_detects_vpn_and_cross_border() -> None:
    data = _transactions()
    result = GeoRiskAgent().fit(data).predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert result.loc[result.transaction_id == "txn-3", "reason_code"].item() == "VPN_OR_PROXY"
    assert result.risk_score.between(0, 1).all()


def test_geo_risk_agent_handles_missing_features() -> None:
    result = GeoRiskAgent().predict(pd.DataFrame({"transaction_id": ["txn-1"]}))

    assert len(result) == 1
    assert result.risk_score.iloc[0] == 0.0
