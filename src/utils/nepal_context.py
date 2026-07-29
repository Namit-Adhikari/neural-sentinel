"""Nepal-specific reference constants for the Neural Sentinel project.

This module contains all Nepal-specific domain knowledge encoded as pure Python
literals: remittance corridors with risk tiers, banking channel mix weights,
NPR exchange-rate ranges, major Nepali cities, NRB regulatory thresholds, and
the primary currency for each remittance corridor.

All data is embedded as Python dict/list literals.  No I/O is performed and no
external dependencies are required at import time.
"""

# ---------------------------------------------------------------------------
# Remittance corridors  (AGENTS.md §12.1)
# Keys follow the "{source_country}->Nepal" convention.
# Values are risk-tier labels used by the KYC/AML and Geo-Risk agents.
# ---------------------------------------------------------------------------
REMITTANCE_CORRIDORS: dict[str, str] = {
    # Gulf countries – high volume, medium risk
    "Qatar->Nepal":        "medium",
    "UAE->Nepal":          "medium",
    "Saudi Arabia->Nepal": "medium",
    "Bahrain->Nepal":      "medium",
    "Kuwait->Nepal":       "medium",
    "Oman->Nepal":         "medium",
    # South / Southeast Asia
    "India->Nepal":        "low",
    "Malaysia->Nepal":     "medium",
    "South Korea->Nepal":  "medium",
    "Thailand->Nepal":     "low",
    # Developed markets – lower volume, higher per-transaction value, high risk
    "USA->Nepal":          "high",
    "UK->Nepal":           "high",
    "Australia->Nepal":    "high",
    "Japan->Nepal":        "high",
}

# ---------------------------------------------------------------------------
# Banking channel probability weights  (AGENTS.md §12.2)
# Weights sum to exactly 1.0 and reflect approximate Nepali banking channel mix.
# ---------------------------------------------------------------------------
CHANNEL_MIX: dict[str, float] = {
    "mobile_banking": 0.45,
    "branch":         0.225,
    "atm":            0.125,
    "online_banking": 0.125,
    "pos":            0.075,
}

# ---------------------------------------------------------------------------
# NPR exchange-rate ranges  (AGENTS.md §12.3)
# Maps ISO-4217 currency codes to (min_rate, max_rate) tuples expressed as
# units of NPR per 1 unit of foreign currency.
# ---------------------------------------------------------------------------
EXCHANGE_RATE_RANGES: dict[str, tuple[float, float]] = {
    "USD": (110.0, 135.0),
    "EUR": (120.0, 150.0),
    "GBP": (140.0, 170.0),
    "INR": (1.58,  1.65),
    "QAR": (30.0,  38.0),
    "SAR": (29.0,  36.0),
    "AED": (30.0,  37.0),
    "MYR": (24.0,  30.0),
}

# ---------------------------------------------------------------------------
# Major Nepali cities  (AGENTS.md §12.4)
# Used to assign realistic city values to generated account records.
# ---------------------------------------------------------------------------
NEPALI_CITIES: list[str] = [
    "Kathmandu",
    "Lalitpur",
    "Bhaktapur",
    "Pokhara",
    "Biratnagar",
    "Bharatpur",
    "Birganj",
    "Dharan",
    "Butwal",
    "Hetauda",
    "Nepalgunj",
    "Bhadrapur",
    "Itahari",
    "Dhangadhi",
    "Mahendranagar",
]

# ---------------------------------------------------------------------------
# NRB cash-reporting threshold  (AGENTS.md §4.4)
# Transactions at or above this amount in NPR must be reported to Nepal Rastra
# Bank.  Used as the reference value for structuring-detection logic.
# ---------------------------------------------------------------------------
NRB_CASH_REPORTING_THRESHOLD_NPR: float = 1_000_000.0

# ---------------------------------------------------------------------------
# Primary currency per corridor  (design doc §Components)
# Maps corridor strings to the ISO-4217 currency code most commonly used for
# remittances through that corridor.  Generators use this to assign realistic
# original_currency values to cross-border transactions.
# ---------------------------------------------------------------------------
CORRIDOR_CURRENCIES: dict[str, str] = {
    "Qatar->Nepal":        "QAR",
    "UAE->Nepal":          "AED",
    "Saudi Arabia->Nepal": "SAR",
    "Bahrain->Nepal":      "SAR",
    "Kuwait->Nepal":       "SAR",
    "Oman->Nepal":         "SAR",
    "India->Nepal":        "INR",
    "Malaysia->Nepal":     "MYR",
    "South Korea->Nepal":  "USD",
    "Thailand->Nepal":     "USD",
    "USA->Nepal":          "USD",
    "UK->Nepal":           "GBP",
    "Australia->Nepal":    "AUD",
    "Japan->Nepal":        "JPY",
}
