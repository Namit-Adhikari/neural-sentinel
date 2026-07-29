"""Tests for src/utils/nepal_context.py.

Covers all module-level constants via both unit tests (specific values, counts)
and Hypothesis property tests (structural invariants).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.nepal_context import (
    CHANNEL_MIX,
    CORRIDOR_CURRENCIES,
    EXCHANGE_RATE_RANGES,
    NEPALI_CITIES,
    NRB_CASH_REPORTING_THRESHOLD_NPR,
    REMITTANCE_CORRIDORS,
)

# ---------------------------------------------------------------------------
# Expected reference data
# ---------------------------------------------------------------------------

_REQUIRED_CORRIDORS = [
    "Qatar->Nepal",
    "UAE->Nepal",
    "Saudi Arabia->Nepal",
    "Bahrain->Nepal",
    "Kuwait->Nepal",
    "Oman->Nepal",
    "India->Nepal",
    "Malaysia->Nepal",
    "South Korea->Nepal",
    "Thailand->Nepal",
    "USA->Nepal",
    "UK->Nepal",
    "Australia->Nepal",
    "Japan->Nepal",
]

_VALID_TIERS = {"high", "medium", "low"}

_REQUIRED_CHANNELS = {
    "mobile_banking",
    "branch",
    "atm",
    "online_banking",
    "pos",
}

_REQUIRED_CURRENCIES = {"USD", "EUR", "GBP", "INR", "QAR", "SAR", "AED", "MYR"}

_REQUIRED_CITIES = [
    "Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Biratnagar",
    "Bharatpur", "Birganj", "Dharan", "Butwal", "Hetauda",
    "Nepalgunj", "Bhadrapur", "Itahari", "Dhangadhi", "Mahendranagar",
]

# ===========================================================================
# Property tests
# ===========================================================================

# Feature: phase0-foundation, Property 12: REMITTANCE_CORRIDORS covers all required corridors with valid tiers
@given(corridor=st.sampled_from(_REQUIRED_CORRIDORS))
@settings(max_examples=100)
def test_remittance_corridors_all_required_with_valid_tiers(corridor: str) -> None:
    """Property 12: every required corridor exists and has a valid risk tier.

    Validates: Requirements 5.1
    """
    assert corridor in REMITTANCE_CORRIDORS, (
        f"Missing corridor: {corridor!r}"
    )
    assert REMITTANCE_CORRIDORS[corridor] in _VALID_TIERS, (
        f"Invalid tier {REMITTANCE_CORRIDORS[corridor]!r} for {corridor!r}"
    )


# Feature: phase0-foundation, Property 13: CHANNEL_MIX weights form a valid probability distribution
@given(channel=st.sampled_from(sorted(_REQUIRED_CHANNELS)))
@settings(max_examples=100)
def test_channel_mix_valid_probability_distribution(channel: str) -> None:
    """Property 13: each channel weight is in (0, 1) and the total is 1.0.

    Validates: Requirements 5.2
    """
    assert channel in CHANNEL_MIX, f"Channel {channel!r} missing from CHANNEL_MIX"
    weight = CHANNEL_MIX[channel]
    assert 0.0 < weight < 1.0, f"Weight {weight} for {channel!r} not in (0, 1)"
    total = sum(CHANNEL_MIX.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), (
        f"CHANNEL_MIX weights sum to {total}, expected 1.0"
    )


# Feature: phase0-foundation, Property 14: EXCHANGE_RATE_RANGES entries have valid bounds
@given(currency=st.sampled_from(sorted(_REQUIRED_CURRENCIES)))
@settings(max_examples=100)
def test_exchange_rate_ranges_valid_bounds(currency: str) -> None:
    """Property 14: min rate > 0 and min rate < max rate for every currency.

    Validates: Requirements 5.3
    """
    assert currency in EXCHANGE_RATE_RANGES, (
        f"Currency {currency!r} missing from EXCHANGE_RATE_RANGES"
    )
    min_rate, max_rate = EXCHANGE_RATE_RANGES[currency]
    assert min_rate > 0.0, f"{currency}: min_rate {min_rate} must be > 0"
    assert min_rate < max_rate, (
        f"{currency}: min_rate {min_rate} must be < max_rate {max_rate}"
    )


# Feature: phase0-foundation, Property 15: CORRIDOR_CURRENCIES maps every corridor to a non-empty currency string
@given(corridor=st.sampled_from(_REQUIRED_CORRIDORS))
@settings(max_examples=100)
def test_corridor_currencies_non_empty_and_complete(corridor: str) -> None:
    """Property 15: every REMITTANCE_CORRIDORS key maps to a non-empty currency.

    Validates: Requirements 5.6
    """
    assert corridor in CORRIDOR_CURRENCIES, (
        f"Corridor {corridor!r} missing from CORRIDOR_CURRENCIES"
    )
    currency = CORRIDOR_CURRENCIES[corridor]
    assert isinstance(currency, str) and len(currency) > 0, (
        f"Empty/non-string currency for corridor {corridor!r}: {currency!r}"
    )


# ===========================================================================
# Unit tests
# ===========================================================================

def test_module_imports_without_exception() -> None:
    """Requirement 5.7: nepal_context imports cleanly with no I/O."""
    import src.utils.nepal_context  # noqa: F401 — import is the test


def test_nrb_threshold_value() -> None:
    """Requirement 5.5: NRB_CASH_REPORTING_THRESHOLD_NPR equals 1_000_000.0."""
    assert NRB_CASH_REPORTING_THRESHOLD_NPR == 1_000_000.0


def test_nepali_cities_count() -> None:
    """Requirement 5.4: NEPALI_CITIES contains exactly 15 entries."""
    assert len(NEPALI_CITIES) == 15


def test_nepali_cities_all_present() -> None:
    """Requirement 5.4: all 15 expected cities are in NEPALI_CITIES."""
    for city in _REQUIRED_CITIES:
        assert city in NEPALI_CITIES, f"City {city!r} missing from NEPALI_CITIES"


def test_remittance_corridors_count() -> None:
    """REMITTANCE_CORRIDORS has exactly 14 entries."""
    assert len(REMITTANCE_CORRIDORS) == 14


def test_channel_mix_keys_match_channels() -> None:
    """CHANNEL_MIX contains exactly the five expected channel keys."""
    assert set(CHANNEL_MIX.keys()) == _REQUIRED_CHANNELS


def test_exchange_rate_ranges_count() -> None:
    """EXCHANGE_RATE_RANGES has exactly 8 currency entries."""
    assert len(EXCHANGE_RATE_RANGES) == 8


def test_corridor_currencies_count() -> None:
    """CORRIDOR_CURRENCIES has exactly 14 entries."""
    assert len(CORRIDOR_CURRENCIES) == 14


def test_corridor_currencies_keys_match_corridors() -> None:
    """Every key in REMITTANCE_CORRIDORS is present in CORRIDOR_CURRENCIES."""
    for corridor in REMITTANCE_CORRIDORS:
        assert corridor in CORRIDOR_CURRENCIES, (
            f"Corridor {corridor!r} missing from CORRIDOR_CURRENCIES"
        )


def test_channel_mix_sums_to_one() -> None:
    """CHANNEL_MIX weights sum to exactly 1.0 (floating-point tolerance 1e-9)."""
    assert math.isclose(sum(CHANNEL_MIX.values()), 1.0, abs_tol=1e-9)
