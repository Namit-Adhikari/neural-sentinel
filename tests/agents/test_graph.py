"""Tests for GraphAgent."""

from __future__ import annotations

import pickle

import pandas as pd

from src.agents import BaseAgent
from src.agents.graph_agent import GraphAgent


def _graph_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": [f"G{i:03d}" for i in range(12)],
        "sender_account_id": [
            "ACC_A", "ACC_B", "ACC_C", "ACC_D",
            "ACC_A", "ACC_A", "ACC_A",
            "ACC_E", "ACC_F", "ACC_G",
            "ACC_E", "ACC_F",
        ],
        "receiver_account_id": [
            "ACC_B", "ACC_C", "ACC_D", "ACC_E",
            "ACC_C", "ACC_D", "ACC_E",
            "ACC_H", "ACC_H", "ACC_H",
            "ACC_H", "ACC_H",
        ],
        "amount_npr": [
            500_000.0, 200_000.0, 150_000.0, 100_000.0,
            300_000.0, 250_000.0, 180_000.0,
            950_000.0, 970_000.0, 930_000.0,
            960_000.0, 940_000.0,
        ],
        "is_cross_border": [0] * 7 + [1] * 5,
        "is_fraud": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })


def test_graph_agent_initializes_correctly() -> None:
    agent = GraphAgent()

    assert agent.agent_name == "graph"
    assert not agent.is_fitted
    assert agent.model_type in ("graphsage", "gat")


def test_graph_agent_predict_returns_correct_schema() -> None:
    data = _graph_transactions()
    agent = GraphAgent(config={"graph_alert_threshold": 0.3})
    agent.fit(data)
    result = agent.predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_graph_agent_handles_missing_transaction_id() -> None:
    result = GraphAgent().predict(pd.DataFrame({"amount_npr": [1000.0]}))

    assert result.empty
    assert list(result.columns) == list(BaseAgent.prediction_columns)


def test_graph_agent_handles_missing_account_columns() -> None:
    data = pd.DataFrame({
        "transaction_id": ["G001", "G002"],
        "amount_npr": [100_000.0, 200_000.0],
        "is_fraud": [0, 1],
    })
    agent = GraphAgent()
    agent.fit(data)
    result = agent.predict(data)

    assert len(result) == 2
    assert result["risk_score"].between(0, 1).all()


def test_graph_agent_does_not_mutate_input() -> None:
    data = _graph_transactions()
    original = data.copy(deep=True)
    GraphAgent().fit(data).predict(data)
    pd.testing.assert_frame_equal(data, original)


def test_graph_agent_explain() -> None:
    data = _graph_transactions()
    agent = GraphAgent(config={"graph_alert_threshold": 0.3})
    agent.fit(data)
    agent.predict(data)

    explanation = agent.explain("G000")
    assert "G000" in explanation


def test_graph_agent_gat_variant() -> None:
    data = _graph_transactions()
    agent = GraphAgent(
        config={"graph_model_type": "gat", "graph_alert_threshold": 0.3}
    )
    agent.fit(data)
    result = agent.predict(data)

    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_graph_agent_is_pickle_serializable() -> None:
    data = _graph_transactions()
    agent = GraphAgent()
    agent.fit(data)
    restored = pickle.loads(pickle.dumps(agent))

    assert isinstance(restored, GraphAgent)
    assert restored.is_fitted
