"""Tests for the shared Phase 1 agent interface."""

from __future__ import annotations

import logging
import pickle

import pandas as pd

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
