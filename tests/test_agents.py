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


# ---------------------------------------------------------------------------
# KYC/AML Agent tests
# ---------------------------------------------------------------------------

from src.agents.kyc_aml_agent import KYCAMLAgent


def _kyc_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["T001", "T002", "T003", "T004", "T005", "T006"],
        "sender_account_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "amount_npr": [
            950_000.0,   # structuring (just below 1M threshold)
            500.0,        # normal
            999_000.0,   # structuring (just below 1M threshold)
            50_000.0,     # normal — PEP account
            200_000.0,   # normal — sanctioned account
            1_500_000.0, # above threshold — no structuring flag
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

    # T005 is the sanctioned account
    sanctioned_row = result.loc[result["transaction_id"] == "T005"]
    assert "SANCTIONS" in sanctioned_row["reason_code"].values[0]
    assert sanctioned_row["risk_score"].values[0] > 0.0


def test_kyc_aml_agent_detects_structuring_pattern() -> None:
    data = _kyc_transactions()
    result = KYCAMLAgent().fit(data).predict(data)

    # T001 and T003 are structuring transactions (900K–999K NPR)
    for txn_id in ["T001", "T003"]:
        row = result.loc[result["transaction_id"] == txn_id]
        assert "STRUCTURING" in row["reason_code"].values[0], f"Expected STRUCTURING in {txn_id}"
        assert row["risk_score"].values[0] > 0.0


def test_kyc_aml_agent_handles_missing_account_features() -> None:
    # Only transaction-level columns — no account features
    data = pd.DataFrame({
        "transaction_id": ["T001", "T002"],
        "sender_account_id": ["A1", "A2"],
        "amount_npr": [950_000.0, 100.0],
    })
    agent = KYCAMLAgent()
    result = agent.predict(data)

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

    # Normal transaction should have zero risk score
    assert result["risk_score"].values[0] == 0.0
    assert result["alert_flag"].values[0] == 0
    assert result["reason_code"].values[0] == "NO_VIOLATIONS"


# ---------------------------------------------------------------------------
# Behaviour Agent tests
# ---------------------------------------------------------------------------

from src.agents.behaviour_agent import BehaviourAgent


def _behaviour_transactions(n: int = 60) -> pd.DataFrame:
    rng = pd.array([i for i in range(n)])
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
        "is_fraud": [int(i % 10 == 0) for i in range(n)],  # 10% fraud
    })


def test_behaviour_agent_initializes_correctly() -> None:
    agent = BehaviourAgent()

    assert agent.agent_name == "behaviour"
    assert not agent.is_fitted
    assert agent.model_type in ("gru", "lstm")


def test_behaviour_agent_predict_returns_correct_schema() -> None:
    data = _behaviour_transactions()
    agent = BehaviourAgent(config={"behaviour_epochs": 1, "behaviour_batch_size": 16})
    agent.fit(data)
    result = agent.predict(data)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_behaviour_agent_handles_missing_transaction_id() -> None:
    data = pd.DataFrame({"amount_npr": [1000.0, 2000.0]})
    result = BehaviourAgent().predict(data)

    assert result.empty
    assert list(result.columns) == list(BaseAgent.prediction_columns)


def test_behaviour_agent_lstm_variant() -> None:
    data = _behaviour_transactions()
    agent = BehaviourAgent(
        config={"behaviour_model_type": "lstm", "behaviour_epochs": 1, "behaviour_batch_size": 16}
    )
    agent.fit(data)
    result = agent.predict(data)

    assert len(result) == len(data)
    assert result["risk_score"].between(0, 1).all()


def test_behaviour_agent_does_not_mutate_input() -> None:
    data = _behaviour_transactions()
    original = data.copy(deep=True)
    agent = BehaviourAgent(config={"behaviour_epochs": 1, "behaviour_batch_size": 16})
    agent.fit(data)
    agent.predict(data)
    pd.testing.assert_frame_equal(data, original)


def test_behaviour_agent_explain() -> None:
    data = _behaviour_transactions()
    agent = BehaviourAgent(config={"behaviour_epochs": 1, "behaviour_batch_size": 16})
    agent.fit(data)
    agent.predict(data)

    explanation = agent.explain("BT0000")
    assert "BT0000" in explanation


def test_behaviour_agent_handles_missing_account_column() -> None:
    # No sender_account_id at all — agent should raise ValueError internally
    # and fall back gracefully (the catch in predict() handles it)
    data = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "amount_npr": [1000.0, 2000.0],
    })
    agent = BehaviourAgent(config={"behaviour_epochs": 1})
    agent.fit(data)
    # Without account col, _build_sequences raises; predict catches and falls back
    result = agent.predict(data)
    assert len(result) == 2
