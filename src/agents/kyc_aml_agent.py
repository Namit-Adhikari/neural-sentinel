"""KYC/AML compliance rules agent for account-level risk scoring.

This agent implements a deterministic, rules-based scoring system that evaluates
account-level KYC metadata, regulatory compliance indicators, and transaction
patterns against Nepal Rastra Bank (NRB) regulations and AML best practices.

The agent does not require model training — all rules are pre-defined with
configurable weights stored in the centralized configuration module.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.utils.config import get_config


class KYCAMLAgent(BaseAgent):
    """Rules-based AML and KYC compliance scoring agent.

    This agent evaluates transactions against a set of weighted compliance rules:
    
    1. **PEP flag** — Politically Exposed Person accounts receive elevated scores
    2. **Sanctions match** — Accounts on sanctions lists trigger critical alerts
    3. **KYC unverified** — Unverified KYC status elevates risk
    4. **New accounts** — Accounts < 90 days old receive elevated scores
    5. **High risk grade** — Accounts with high KYC risk grade are flagged
    6. **Structuring patterns** — Transactions near NPR 1M threshold (smurfing)
    7. **Layering patterns** — Rapid cross-border transfers to same corridor
    
    All rule weights are configurable via the centralized config module and the
    final risk score is normalized to [0, 1] based on maximum possible weight sum.
    
    Args:
        config: Configuration mapping with rule weights and thresholds.
        logger: Logger for rule evaluations and missing features.
    
    Notes:
        This agent requires both transaction and account data to be joined before
        calling ``predict()``. If account features are missing, deterministic
        fallback scores are provided based on transaction-level features only.
    """

    agent_name = "kyc_aml"
    
    # Required account-level features for full rule evaluation
    _account_features = (
        "is_pep",
        "is_sanctioned",
        "kyc_verified",
        "account_age_days",
        "kyc_risk_grade",
    )
    
    # Transaction-level features used in pattern detection
    _transaction_features = (
        "amount_npr",
        "is_cross_border",
        "remittance_corridor",
        "sender_account_id",
    )

    def __init__(
        self,
        config: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize rule weights and NRB regulatory thresholds."""
        super().__init__(config or get_config(), logger)
        
        # Load rule weights from config
        self.weight_pep = float(self.config.get("weight_pep_flag", 0.4))
        self.weight_sanctions = float(self.config.get("weight_sanctions_match", 1.0))
        self.weight_kyc_unverified = float(self.config.get("weight_kyc_unverified", 0.5))
        self.weight_new_account = float(self.config.get("weight_new_account", 0.3))
        self.weight_high_risk = float(self.config.get("weight_high_risk_grade", 0.3))
        self.weight_structuring = float(self.config.get("weight_structuring_pattern", 0.6))
        self.weight_layering = float(self.config.get("weight_layering_pattern", 0.7))
        
        # NRB regulatory thresholds
        self.nrb_threshold = float(
            self.config.get("nrb_cash_reporting_threshold_npr", 1_000_000.0)
        )
        self.structuring_min = float(self.config.get("structuring_min_npr", 900_000.0))
        self.structuring_max = float(self.config.get("structuring_max_npr", 999_000.0))
        self.new_account_threshold_days = int(
            self.config.get("new_account_threshold_days", 90)
        )
        
        # Compute maximum possible weight for normalization
        self.max_weight = (
            self.weight_pep +
            self.weight_sanctions +
            self.weight_kyc_unverified +
            self.weight_new_account +
            self.weight_high_risk +
            self.weight_structuring +
            self.weight_layering
        )
        
        # Store explanations per transaction
        self._explanations: dict[str, str] = {}
        
        self.logger.info(
            "KYC/AML agent initialized with max_weight=%.2f, NRB threshold=%.0f NPR",
            self.max_weight,
            self.nrb_threshold,
        )

    def fit(self, data: pd.DataFrame) -> "KYCAMLAgent":
        """Mark agent as fitted without training.
        
        This is a deterministic rules-based agent that does not require training.
        The method exists only to satisfy the BaseAgent contract.
        
        Args:
            data: Transaction DataFrame (unused).
        
        Returns:
            self for fluent pipeline composition.
        """
        self.is_fitted = True
        self.logger.info("KYC/AML agent marked as fitted (no training required)")
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate compliance rules and return per-transaction risk scores.
        
        Args:
            data: DataFrame containing transactions, ideally joined with account
                features. If account features are missing, transaction-only rules
                are evaluated.
        
        Returns:
            DataFrame with canonical agent output columns including risk scores,
            alert flags, reason codes, and explanations.
        """
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        
        # Mark as fitted if predict is called before explicit fit()
        if not self.is_fitted:
            self.is_fitted = True
        
        # Initialize score accumulator
        scores = np.zeros(len(data), dtype=float)
        reason_parts = [[] for _ in range(len(data))]
        
        # Account-level rules (if features are present)
        has_account_features = all(col in data.columns for col in self._account_features)
        
        if has_account_features:
            # Rule 1: PEP flag
            is_pep = pd.to_numeric(data["is_pep"], errors="coerce").fillna(0).astype(int)
            pep_score = is_pep * self.weight_pep
            scores += pep_score
            for idx in data.index[is_pep == 1]:
                reason_parts[idx].append("PEP")
            
            # Rule 2: Sanctions match (critical — highest weight)
            is_sanctioned = pd.to_numeric(data["is_sanctioned"], errors="coerce").fillna(0).astype(int)
            sanctions_score = is_sanctioned * self.weight_sanctions
            scores += sanctions_score
            for idx in data.index[is_sanctioned == 1]:
                reason_parts[idx].append("SANCTIONS")
            
            # Rule 3: KYC unverified
            kyc_verified = pd.to_numeric(data["kyc_verified"], errors="coerce").fillna(1).astype(int)
            kyc_unverified = (kyc_verified == 0).astype(int)
            kyc_score = kyc_unverified * self.weight_kyc_unverified
            scores += kyc_score
            for idx in data.index[kyc_unverified == 1]:
                reason_parts[idx].append("KYC_UNVERIFIED")
            
            # Rule 4: New account (< 90 days)
            account_age = pd.to_numeric(
                data["account_age_days"], errors="coerce"
            ).fillna(9999)
            is_new = (account_age < self.new_account_threshold_days).astype(int)
            new_account_score = is_new * self.weight_new_account
            scores += new_account_score
            for idx in data.index[is_new == 1]:
                reason_parts[idx].append("NEW_ACCOUNT")
            
            # Rule 5: High risk grade
            risk_grade = data["kyc_risk_grade"].astype(str).str.lower().fillna("low")
            is_high_risk = (risk_grade == "high").astype(int)
            risk_score = is_high_risk * self.weight_high_risk
            scores += risk_score
            for idx in data.index[is_high_risk == 1]:
                reason_parts[idx].append("HIGH_RISK_GRADE")
        else:
            self.logger.warning(
                "KYC/AML agent missing account features; using transaction-only rules"
            )
        
        # Transaction-level rules (always evaluated)
        
        # Rule 6: Structuring pattern (transactions just below NPR 1M threshold)
        if "amount_npr" in data.columns:
            amount_npr = pd.to_numeric(data["amount_npr"], errors="coerce").fillna(0.0)
            is_structuring = (
                (amount_npr >= self.structuring_min) &
                (amount_npr <= self.structuring_max)
            ).astype(int)
            structuring_score = is_structuring * self.weight_structuring
            scores += structuring_score
            for idx in data.index[is_structuring == 1]:
                reason_parts[idx].append("STRUCTURING")
        
        # Rule 7: Layering pattern (multiple cross-border transfers to same corridor)
        # This is a simplified version; full layering detection requires temporal analysis
        if all(col in data.columns for col in ("is_cross_border", "remittance_corridor")):
            is_cross_border = pd.to_numeric(
                data["is_cross_border"], errors="coerce"
            ).fillna(0).astype(int)
            corridor = data["remittance_corridor"].astype(str).fillna("")
            
            # Count cross-border transactions per corridor per sender
            if "sender_account_id" in data.columns:
                sender_corridor = (
                    data["sender_account_id"].astype(str) + ":" + corridor
                )
                corridor_counts = sender_corridor.map(
                    sender_corridor.value_counts()
                ).fillna(0)
                # Flag if sender has 2+ cross-border transactions to same corridor
                is_layering = (
                    (is_cross_border == 1) & (corridor_counts >= 2)
                ).astype(int)
                layering_score = is_layering * self.weight_layering
                scores += layering_score
                for idx in data.index[is_layering == 1]:
                    reason_parts[idx].append("LAYERING")
        
        # Normalize scores to [0, 1]
        normalized_scores = np.clip(scores / max(self.max_weight, 1e-9), 0.0, 1.0)
        
        # Build reason codes
        reason_codes = np.array([
            "|".join(parts) if parts else "NO_VIOLATIONS"
            for parts in reason_parts
        ])
        
        # Build human-readable explanations
        explanations = []
        for idx, parts in enumerate(reason_parts):
            if not parts:
                explanation = "No KYC/AML violations detected."
            else:
                violation_text = ", ".join(part.lower().replace("_", " ") for part in parts)
                explanation = f"KYC/AML violations: {violation_text}."
            explanations.append(explanation)
        
        # Store explanations for later retrieval
        self._explanations = dict(
            zip(data["transaction_id"].astype(str), explanations)
        )
        
        # Build canonical output
        result = self.build_predictions(
            data,
            normalized_scores,
            reason_code="KYC_AML_ANALYSIS"
        )
        result["reason_code"] = reason_codes
        result["explanation"] = explanations
        
        return result

    def explain(self, transaction_id: str) -> str:
        """Return human-readable explanation for a previously scored transaction.
        
        Args:
            transaction_id: Transaction identifier to explain.
        
        Returns:
            Human-readable explanation string.
        """
        explanation = self._explanations.get(
            str(transaction_id),
            "No KYC/AML score is available for this transaction."
        )
        return f"Transaction {transaction_id}: {explanation}"
