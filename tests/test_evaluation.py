"""Tests for the explanation and evaluation components."""

from __future__ import annotations

import pandas as pd

from src.agents.explanation_agent import ExplanationAgent
from src.evaluation.agent_evaluation import AgentEvaluator


def _context() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "transaction_type": ["transfer"] * 4,
        "amount_npr": [950_000.0, 100.0, 1_500_000.0, 250.0],
        "channel": ["mobile_banking"] * 4,
        "remittance_corridor": ["Qatar->Nepal", "domestic", "India->Nepal", "domestic"],
        "is_fraud": [1, 0, 1, 0],
        "meta_risk_score": [0.91, 0.08, 0.74, 0.12],
        "kyc_aml_reason_code": ["STRUCTURING", "NO_VIOLATIONS", "CROSS_BORDER", "NO_VIOLATIONS"],
    })


def test_explanation_agent_returns_actionable_canonical_output() -> None:
    data = _context()
    result = ExplanationAgent().fit(data).predict(data)

    assert len(result) == len(data)
    assert result.loc[0, "alert_flag"] == 1
    assert "STRUCTURING".lower() in result.loc[0, "explanation"]
    assert "not a finding of criminal conduct" in result.loc[0, "explanation"]


def test_explanation_agent_initializes_correctly() -> None:
    agent = ExplanationAgent()
    assert agent.agent_name == "explanation"
    assert agent.alert_threshold == 0.5
    assert not agent.is_fitted


def test_explanation_agent_explain_returns_string() -> None:
    data = _context()
    agent = ExplanationAgent().fit(data).predict(data)

    explanation = agent.explain("T1")
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_explanation_agent_explain_missing_returns_fallback() -> None:
    agent = ExplanationAgent()
    result = agent.explain("nonexistent")

    assert "nonexistent" in result
    assert "No explanation" in result


def test_explanation_agent_handles_missing_context() -> None:
    result = ExplanationAgent().predict(pd.DataFrame({"amount_npr": [100.0]}))

    assert result.empty


def test_evaluator_reports_imbalance_aware_and_calibration_metrics() -> None:
    data = _context()
    metrics = AgentEvaluator(precision_at_k_fraction=0.5).evaluate(data, {"meta_learner": "meta_risk_score"})

    assert {"auc_roc", "auc_pr", "f1_at_optimal_threshold", "precision_at_k", "brier_score"}.issubset(metrics.columns)
    assert metrics.loc[0, "auc_roc"] == 1.0
    assert metrics.loc[0, "auc_pr"] == 1.0


def test_evaluator_compares_meta_learner_and_individual() -> None:
    metrics = pd.DataFrame({"agent_name": ["velocity", "meta_learner"], "auc_roc": [0.7, 0.8]})
    summary = AgentEvaluator().compare_system(metrics)

    assert summary["auc_roc_delta"] == 0.1
