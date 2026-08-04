"""Tests for MetaLearnerAgent."""

from __future__ import annotations

import pickle

import pandas as pd

from src.agents import BaseAgent
from src.agents.meta_learner import MetaLearnerAgent


def _meta_wide_data() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": [f"M{i:03d}" for i in range(40)],
        "velocity_score":   [float(i % 10) / 10.0 for i in range(40)],
        "geo_risk_score":   [float((i + 2) % 10) / 10.0 for i in range(40)],
        "behaviour_score":  [float((i + 4) % 10) / 10.0 for i in range(40)],
        "kyc_aml_score":    [float((i + 6) % 10) / 10.0 for i in range(40)],
        "graph_score":      [float((i + 8) % 10) / 10.0 for i in range(40)],
        "is_fraud": [int(i >= 30) for i in range(40)],
    })


def _meta_long_data() -> pd.DataFrame:
    agents = ["velocity", "geo_risk", "behaviour", "kyc_aml", "graph"]
    rows = []
    for i in range(20):
        for j, agent in enumerate(agents):
            rows.append({
                "transaction_id": f"ML{i:03d}",
                "agent_name": agent,
                "risk_score": float((i + j) % 10) / 10.0,
                "is_fraud": int(i >= 15),
            })
    return pd.DataFrame(rows)


def test_meta_learner_agent_initializes_correctly() -> None:
    agent = MetaLearnerAgent()

    assert agent.agent_name == "meta_learner"
    assert not agent.is_fitted
    assert agent.model_type in ("random_forest", "xgboost")


def test_meta_learner_agent_predict_wide_returns_correct_schema() -> None:
    data = _meta_wide_data()
    agent = MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    )
    agent.fit(data)
    result = agent.predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_meta_learner_agent_predict_long_format() -> None:
    long_data = _meta_long_data()
    agent = MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    )
    wide = _meta_wide_data()
    agent.fit(wide)
    result = agent.predict(long_data)

    assert len(result) > 0
    assert result["risk_score"].between(0, 1).all()


def test_meta_learner_agent_handles_missing_transaction_id() -> None:
    result = MetaLearnerAgent().predict(pd.DataFrame({"velocity_score": [0.8]}))

    assert result.empty
    assert list(result.columns) == list(BaseAgent.prediction_columns)


def test_meta_learner_agent_fallback_without_fraud_labels() -> None:
    data = _meta_wide_data().drop(columns=["is_fraud"])
    agent = MetaLearnerAgent()
    agent.fit(data)
    result = agent.predict(data)

    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_meta_learner_agent_does_not_mutate_input() -> None:
    data = _meta_wide_data()
    original = data.copy(deep=True)
    MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    ).fit(data).predict(data)
    pd.testing.assert_frame_equal(data, original)


def test_meta_learner_agent_explain() -> None:
    data = _meta_wide_data()
    agent = MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    )
    agent.fit(data)
    agent.predict(data)

    explanation = agent.explain("M000")
    assert "M000" in explanation


def test_meta_learner_agent_is_pickle_serializable() -> None:
    data = _meta_wide_data()
    agent = MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    )
    agent.fit(data)
    restored = pickle.loads(pickle.dumps(agent))

    assert isinstance(restored, MetaLearnerAgent)
    assert restored.is_fitted


def test_meta_learner_agent_missing_agent_columns_imputed() -> None:
    train = _meta_wide_data()
    agent = MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    )
    agent.fit(train)

    partial = pd.DataFrame({
        "transaction_id": ["M000", "M001"],
        "velocity_score": [0.8, 0.2],
        "geo_risk_score": [0.6, 0.1],
    })
    result = agent.predict(partial)

    assert len(result) == 2
    assert result["risk_score"].between(0, 1).all()


def test_meta_learner_feature_importances_populated_after_fit() -> None:
    data = _meta_wide_data()
    agent = MetaLearnerAgent(
        config={
            "meta_learner_n_estimators": 10,
            "meta_learner_calibration_cv": 2,
        }
    )
    agent.fit(data)

    importances = agent.feature_importances
    assert len(importances) > 0
    assert all(isinstance(v, float) for v in importances.values())
