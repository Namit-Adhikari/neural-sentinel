"""Tests for VelocityAgent and GeoRiskAgent."""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents import BaseAgent
from src.agents.geo_risk_agent import GeoRiskAgent
from src.agents.velocity_agent import VelocityAgent


# ---------------------------------------------------------------------------
# Velocity agent
# ---------------------------------------------------------------------------


def test_velocity_agent_returns_schema_and_does_not_mutate(
    basic_transactions: pd.DataFrame,
) -> None:
    data = basic_transactions
    original = data.copy(deep=True)
    result = (
        VelocityAgent(config={"velocity_alert_threshold": 0.5})
        .fit(data)
        .predict(data)
    )

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    assert len(result) == len(data)
    assert result.risk_score.between(0, 1).all()
    pd.testing.assert_frame_equal(data, original)


def test_velocity_agent_handles_missing_features() -> None:
    agent = VelocityAgent()
    tiny = pd.DataFrame({"transaction_id": ["txn-1"]})
    result = agent.fit(tiny).predict(tiny)

    assert len(result) == 1
    assert result.risk_score.iloc[0] == 0.0


def test_velocity_agent_raises_if_predict_called_before_fit(
    basic_transactions: pd.DataFrame,
) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        VelocityAgent().predict(basic_transactions)


# ---------------------------------------------------------------------------
# Geo-risk agent
# ---------------------------------------------------------------------------


def test_geo_risk_agent_detects_vpn_and_cross_border(
    basic_transactions: pd.DataFrame,
) -> None:
    result = GeoRiskAgent().fit(basic_transactions).predict(basic_transactions)

    assert list(result.columns) == list(BaseAgent.prediction_columns)
    vpn_row = result.loc[result.transaction_id == "txn-3"]
    assert vpn_row["reason_code"].item() == "VPN_OR_PROXY"
    assert result.risk_score.between(0, 1).all()


def test_geo_risk_agent_handles_missing_features() -> None:
    result = GeoRiskAgent().predict(pd.DataFrame({"transaction_id": ["txn-1"]}))

    assert len(result) == 1
    assert result.risk_score.iloc[0] == 0.0
