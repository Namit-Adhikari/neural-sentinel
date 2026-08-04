"""Tests for KYCAMLAgent."""

from __future__ import annotations

import pandas as pd

from src.agents import BaseAgent
from src.agents.kyc_aml_agent import KYCAMLAgent


def _kyc_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["T001", "T002", "T003", "T004", "T005", "T006"],
        "sender_account_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "amount_npr": [
            950_000.0,
            500.0,
            999_000.0,
            50_000.0,
            200_000.0,
            1_500_000.0,
        ],
        "is_cross_border": [0, 0, 0, 0, 0, 0],
        "remittance_corridor": [None, None, None, None, None, None],
        "is_pep": [0, 0, 0, 1, 0, 0],
        "is_sanctioned": [0, 0, 0, 0, 1, 0],
        "kyc_verified": [1, 1, 1, 1, 1, 0],
        "kyc_risk_grade": ["low", "low", "low", "high", "medium", "high"],
        "account_age_days": [500, 365, 200, 50, 730, 10],
    })


def test_kyc_aml_agent_initializes_correctly() -> None:
    agent = KYCAMLAgent()

    assert agent.agent_name == "kyc_aml"
    assert not agent.is_fitted
    assert 0.0 < agent.alert_threshold <= 1.0


def test_kyc_aml_agent_predict_returns_correct_schema() -> None:
    data = _kyc_transactions()
    agent = KYCAMLAgent()
    agent.fit(data)
    result = agent.predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_kyc_aml_agent_flags_sanctioned_accounts() -> None:
    data = _kyc_transactions()
    result = KYCAMLAgent().fit(data).predict(data)

    sanctioned = result.loc[result["transaction_id"] == "T005"]
    assert "SANCTIONS" in sanctioned["reason_code"].values[0]
    assert sanctioned["risk_score"].values[0] > 0.0


def test_kyc_aml_agent_detects_structuring_pattern() -> None:
    data = _kyc_transactions()
    result = KYCAMLAgent().fit(data).predict(data)

    for txn_id in ["T001", "T003"]:
        row = result.loc[result["transaction_id"] == txn_id]
        assert "STRUCTURING" in row["reason_code"].values[0]
        assert row["risk_score"].values[0] > 0.0


def test_kyc_aml_agent_handles_missing_account_features() -> None:
    data = pd.DataFrame({
        "transaction_id": ["T001", "T002"],
        "sender_account_id": ["A1", "A2"],
        "amount_npr": [950_000.0, 100.0],
    })
    result = KYCAMLAgent().predict(data)

    assert len(result) == 2
    assert result["risk_score"].between(0, 1).all()


def test_kyc_aml_agent_does_not_mutate_input() -> None:
    data = _kyc_transactions()
    original = data.copy(deep=True)
    KYCAMLAgent().fit(data).predict(data)
    pd.testing.assert_frame_equal(data, original)


def test_kyc_aml_agent_explain() -> None:
    data = _kyc_transactions()
    agent = KYCAMLAgent()
    agent.fit(data)
    agent.predict(data)

    explanation = agent.explain("T001")
    assert "T001" in explanation


def test_kyc_aml_agent_normal_transaction_has_low_score() -> None:
    data = pd.DataFrame({
        "transaction_id": ["T_NORMAL"],
        "sender_account_id": ["A_CLEAN"],
        "amount_npr": [500.0],
        "is_cross_border": [0],
        "remittance_corridor": [None],
        "is_pep": [0],
        "is_sanctioned": [0],
        "kyc_verified": [1],
        "kyc_risk_grade": ["low"],
        "account_age_days": [365],
    })
    result = KYCAMLAgent().predict(data)

    assert result["risk_score"].values[0] == 0.0
    assert result["alert_flag"].values[0] == 0
    assert result["reason_code"].values[0] == "NO_VIOLATIONS"
