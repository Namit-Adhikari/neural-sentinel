"""Tests for BehaviourAgent (uses heuristics if torch unavailable)."""

from __future__ import annotations

import pandas as pd

from src.agents import BaseAgent
from src.agents.behaviour import BehaviourAgent


def _behaviour_transactions(n: int = 60) -> pd.DataFrame:
    accounts = [f"ACC{(i % 5):02d}" for i in range(n)]
    return pd.DataFrame({
        "transaction_id": [f"BT{i:04d}" for i in range(n)],
        "sender_account_id": accounts,
        "amount_npr": [float(10_000 + (i * 1_000)) for i in range(n)],
        "transaction_type": [
            ["transfer", "payment", "deposit", "withdrawal"][i % 4] for i in range(n)
        ],
        "channel": [
            ["mobile_banking", "atm", "branch"][i % 3] for i in range(n)
        ],
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="2h"),
        "is_fraud": [int(i % 10 == 0) for i in range(n)],
    })


def test_behaviour_agent_initializes_correctly() -> None:
    agent = BehaviourAgent()

    assert agent.agent_name == "behaviour"
    assert not agent.is_fitted
    assert agent.model_type in ("gru", "lstm")


def test_behaviour_agent_predict_returns_correct_schema() -> None:
    data = _behaviour_transactions()
    agent = BehaviourAgent(
        config={"behaviour_epochs": 1, "behaviour_batch_size": 16}
    )
    agent.fit(data)
    result = agent.predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_behaviour_agent_handles_missing_transaction_id() -> None:
    result = BehaviourAgent().predict(
        pd.DataFrame({"amount_npr": [1000.0, 2000.0]})
    )

    assert result.empty
    assert list(result.columns) == list(BaseAgent.prediction_columns)


def test_behaviour_agent_lstm_variant() -> None:
    data = _behaviour_transactions()
    agent = BehaviourAgent(
        config={
            "behaviour_model_type": "lstm",
            "behaviour_epochs": 1,
            "behaviour_batch_size": 16,
        }
    )
    agent.fit(data)
    result = agent.predict(data)

    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_behaviour_agent_does_not_mutate_input() -> None:
    data = _behaviour_transactions()
    original = data.copy(deep=True)
    agent = BehaviourAgent(
        config={"behaviour_epochs": 1, "behaviour_batch_size": 16}
    )
    agent.fit(data)
    agent.predict(data)
    pd.testing.assert_frame_equal(data, original)


def test_behaviour_agent_explain() -> None:
    data = _behaviour_transactions()
    agent = BehaviourAgent(
        config={"behaviour_epochs": 1, "behaviour_batch_size": 16}
    )
    agent.fit(data)
    agent.predict(data)

    explanation = agent.explain("BT0000")
    assert "BT0000" in explanation


def test_behaviour_agent_handles_missing_account_column() -> None:
    data = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "amount_npr": [1000.0, 2000.0],
    })
    agent = BehaviourAgent(config={"behaviour_epochs": 1})
    agent.fit(data)
    result = agent.predict(data)
    assert len(result) == 2
