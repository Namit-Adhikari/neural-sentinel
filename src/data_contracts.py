"""Pydantic data-contract models for the Neural Sentinel canonical schema.

Three models are defined here — :class:`Transaction`, :class:`Account`, and
:class:`AlertScore` — corresponding to the three tables described in
AGENTS.md Sections 5.1–5.3.  These models are the single source of truth for
all data flowing between agents, generators, and notebooks.

Usage::

    from src.data_contracts import Transaction, Account, AlertScore

All models use Pydantic v2 syntax (``model_validator``, ``field_validator``,
``Annotated``, ``Field``, ``ConfigDict``).
"""

from __future__ import annotations

import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """Canonical representation of a single financial transaction row.

    Corresponds to the ``transactions`` table defined in AGENTS.md §5.1.

    Attributes:
        transaction_id: Unique identifier (UUID format, not validated for performance).
        transaction_date: Calendar date of the transaction.
        transaction_time: Wall-clock time of the transaction.
        sender_account_id: FK to the ``accounts`` table for the sending account.
        receiver_account_id: FK to the ``accounts`` table for the receiving account.
        transaction_type: Transaction category; one of the seven allowed values.
        amount_npr: Transaction amount in Nepali Rupees; must be strictly positive.
        original_currency: ISO-4217 currency code of the original transaction.
        exchange_rate: NPR exchange rate applied; must be strictly positive.
        channel: Banking channel through which the transaction was initiated.
        sender_country: Country of the sender; ``None`` for domestic transactions.
        receiver_country: Country of the receiver; ``None`` for domestic transactions.
        is_cross_border: 1 if sender and receiver countries differ, else 0.
        remittance_corridor: Corridor string (e.g. "Qatar->Nepal"); required when
            ``is_cross_border`` is 1.
        merchant_category: Merchant or business category; optional.
        device_type: Device used to initiate the transaction.
        ip_address: Client IP address (IPv4 format, not validated for performance).
        ip_country: Geo-located country from the IP address; optional.
        ip_is_vpn: 1 if the IP is a known VPN/proxy, else 0.
        is_fraud: 1 if the transaction is flagged as suspicious, else 0.
        fraud_type: Category of fraud; ``None`` when ``is_fraud`` is 0.
        aml_risk_indicator: 1 if the transaction is part of an AML pattern, else 0.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str
    transaction_date: datetime.date
    transaction_time: datetime.time
    sender_account_id: str
    receiver_account_id: str
    transaction_type: Literal[
        "transfer",
        "payment",
        "withdrawal",
        "deposit",
        "cash_out",
        "remittance_inbound",
        "remittance_outbound",
    ]
    amount_npr: float
    original_currency: str
    exchange_rate: float
    channel: Literal[
        "mobile_banking",
        "atm",
        "branch",
        "online_banking",
        "pos",
    ]
    sender_country: Optional[str] = None
    receiver_country: Optional[str] = None
    is_cross_border: Annotated[int, Field(ge=0, le=1)]
    remittance_corridor: Optional[str] = None
    merchant_category: Optional[str] = None
    device_type: str
    ip_address: str
    ip_country: Optional[str] = None
    ip_is_vpn: Annotated[int, Field(ge=0, le=1)]
    is_fraud: Annotated[int, Field(ge=0, le=1)]
    fraud_type: Optional[
        Literal[
            "transaction_fraud",
            "aml_structuring",
            "aml_layering",
            "aml_mule_network",
            "identity_fraud",
        ]
    ] = None
    aml_risk_indicator: Annotated[int, Field(ge=0, le=1)]

    @field_validator("amount_npr")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        """Validate that amount_npr is strictly greater than zero.

        Args:
            v: The candidate value for ``amount_npr``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If ``v`` is less than or equal to zero.
        """
        if v <= 0.0:
            raise ValueError(f"amount_npr must be > 0, got {v}")
        return v

    @field_validator("exchange_rate")
    @classmethod
    def exchange_rate_must_be_positive(cls, v: float) -> float:
        """Validate that exchange_rate is strictly greater than zero.

        Args:
            v: The candidate value for ``exchange_rate``.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If ``v`` is less than or equal to zero.
        """
        if v <= 0.0:
            raise ValueError(f"exchange_rate must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def cross_border_requires_corridor(self) -> "Transaction":
        """Enforce that cross-border transactions declare their remittance corridor.

        Raises:
            ValueError: If ``is_cross_border`` is 1 and ``remittance_corridor``
                is ``None``.
        """
        if self.is_cross_border == 1 and self.remittance_corridor is None:
            raise ValueError(
                "remittance_corridor must be set when is_cross_border=1 "
                "(both fields are in conflict: is_cross_border=1, remittance_corridor=None)"
            )
        return self


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class Account(BaseModel):
    """Canonical representation of a bank account record.

    Corresponds to the ``accounts`` table defined in AGENTS.md §5.2.

    Attributes:
        account_id: Primary key for the account.
        account_type: Category of account; one of the four allowed values.
        account_open_date: Date the account was opened.
        account_age_days: Days since account opening at the reference date; >= 0.
        kyc_verified: 1 if KYC has been completed, else 0.
        kyc_risk_grade: KYC risk classification: ``low``, ``medium``, or ``high``.
        is_pep: 1 if the account holder is a Politically Exposed Person, else 0.
        is_sanctioned: 1 if the account holder appears on a sanctions list, else 0.
        average_monthly_volume: Mean monthly transaction volume in NPR; >= 0.
        average_monthly_count: Mean monthly transaction count; >= 0.
        country: Country of the account holder.
        city: City of the account holder; optional (may be ``None``).
        is_mule: 1 if this is a synthetically generated mule account, else 0.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    account_id: str
    account_type: Literal["savings", "current", "salary", "fixed_deposit"]
    account_open_date: datetime.date
    account_age_days: Annotated[int, Field(ge=0)]
    kyc_verified: Annotated[int, Field(ge=0, le=1)]
    kyc_risk_grade: Literal["low", "medium", "high"]
    is_pep: Annotated[int, Field(ge=0, le=1)]
    is_sanctioned: Annotated[int, Field(ge=0, le=1)]
    average_monthly_volume: Annotated[float, Field(ge=0.0)]
    average_monthly_count: Annotated[int, Field(ge=0)]
    country: str
    city: Optional[str] = None
    is_mule: Annotated[int, Field(ge=0, le=1)]


# ---------------------------------------------------------------------------
# AlertScore
# ---------------------------------------------------------------------------

class AlertScore(BaseModel):
    """Risk-score output produced by a single agent for a single transaction.

    Corresponds to the ``alert_scores`` table defined in AGENTS.md §5.3.

    Attributes:
        transaction_id: FK to the ``transactions`` table.
        agent_name: Name of the agent that produced this score.
        risk_score: Calibrated probability / risk score in [0.0, 1.0].
        alert_flag: 1 if the score exceeds the agent's alert threshold, else 0.
        reason_code: Machine-readable code identifying the reason for the score.
        explanation: Human-readable explanation of the scoring decision.
        timestamp: Datetime at which the score was computed.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str
    agent_name: str
    risk_score: Annotated[float, Field(ge=0.0, le=1.0)]
    alert_flag: Annotated[int, Field(ge=0, le=1)]
    reason_code: str
    explanation: str
    timestamp: datetime.datetime


# ---------------------------------------------------------------------------
# Generator Input Schema
# ---------------------------------------------------------------------------

# The exact subset of columns from the canonical schema that should be fed
# into synthetic data generators. Deterministically derived columns (e.g., 
# is_cross_border, remittance_corridor) are EXCLUDED here. They should be 
# re-derived in a post-processing step to maintain strict logical consistency.
GENERATOR_BASE_COLUMNS = [
    "transaction_id",
    "transaction_date",
    "transaction_time",
    "sender_account_id",
    "receiver_account_id",
    "transaction_type",
    "amount_npr",
    "original_currency",
    "exchange_rate",
    "channel",
    "sender_country",
    "receiver_country",
    "merchant_category",
    "device_type",
    "ip_address",
    "ip_country",
    "ip_is_vpn",
    "is_fraud",
    "fraud_type",
    "aml_risk_indicator"
]
