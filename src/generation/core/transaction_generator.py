"""
src/generation/transaction_generator.py
-----------------------------------------
Phase 5 — Core Transaction Generation.

Generates realistic transaction events from scratch using the knowledge
base extracted in Phase 3. Uses graph-aware receiver selection and
learned temporal / amount distributions.

Outputs
-------
data/interim/synthetic_transactions_core.csv
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.nepal_context import (
    CHANNEL_MIX,
    CORRIDOR_CURRENCIES,
    EXCHANGE_RATE_RANGES,
    REMITTANCE_CORRIDORS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_PAYMENT_TYPES = [
    "transfer", "payment", "withdrawal", "deposit",
    "cash_out", "remittance_inbound", "remittance_outbound",
]

# Transaction period: matches original dataset
_TX_START = pd.Timestamp("2022-10-07")
_TX_END   = pd.Timestamp("2023-10-06")

# Receiver selection strategy probabilities
_RECEIVER_STRATEGY = {
    "frequent":   0.50,   # same account seen before (repeat transfer)
    "community":  0.30,   # account in same "community" (same institution)
    "new":        0.20,   # completely random new account
}

# NPR-only exchange rate (domestic)
_NPR_RATE = 1.0


class TransactionGenerator:
    """Generate core transaction events using the knowledge base.

    Parameters
    ----------
    knowledge : dict
        Loaded knowledge base from ``load_knowledge_base()``.
    accounts_df : pd.DataFrame
        Synthetic accounts (output of Phase 4 AccountGenerator).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        knowledge: dict,
        accounts_df: pd.DataFrame,
        seed: int = 42,
    ) -> None:
        self.knowledge = knowledge
        self.accounts = accounts_df.copy()
        self.rng = np.random.default_rng(seed)

        # Build lookup structures from accounts
        self._account_ids: list[str] = self.accounts["account_id"].astype(str).tolist()
        self._n_accounts = len(self._account_ids)

        # Index accounts by institution for community-based receiver selection
        if "institution" in self.accounts.columns:
            self._inst_to_accounts: dict[str, list[str]] = {}
            for _, row in self.accounts.iterrows():
                inst = str(row["institution"])
                self._inst_to_accounts.setdefault(inst, []).append(str(row["account_id"]))
        else:
            self._inst_to_accounts = {}

        # Institution map per account_id
        if "institution" in self.accounts.columns:
            self._account_institution: dict[str, str] = dict(
                zip(self.accounts["account_id"].astype(str), self.accounts["institution"].astype(str))
            )
        else:
            self._account_institution = {}

        # Account open date per account_id
        if "opened" in self.accounts.columns:
            self._account_open: dict[str, pd.Timestamp] = {
                str(row["account_id"]): pd.Timestamp(row["opened"])
                for _, row in self.accounts.iterrows()
            }
        else:
            self._account_open = {}

        # Per-sender receiver history (built during generation)
        self._sender_receiver_history: dict[str, list[str]] = {}

        # Parse distributions
        self._amount_dist = knowledge.get("amount_distribution", {})
        self._temporal_dist = knowledge.get("temporal_distribution", {})
        self._currency_dist = knowledge.get("currency_distribution", {})
        self._payment_type_dist = knowledge.get("payment_type_distribution", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n: int) -> pd.DataFrame:
        """Generate *n* synthetic core transactions.

        Parameters
        ----------
        n : int
            Number of transactions to generate.

        Returns
        -------
        pd.DataFrame with core transaction columns.
        """
        logger.info("Generating %d synthetic transactions...", n)
        records: list[dict] = []

        for i in range(n):
            record = self._generate_one()
            records.append(record)
            if (i + 1) % 20_000 == 0:
                logger.info("  Generated %d / %d transactions", i + 1, n)

        df = pd.DataFrame(records)
        logger.info("Transaction generation complete: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Single transaction generation
    # ------------------------------------------------------------------

    def _generate_one(self) -> dict[str, Any]:
        # 1. Sample sender
        sender_idx = int(self.rng.integers(self._n_accounts))
        sender_id = self._account_ids[sender_idx]

        # 2. Sample receiver (graph-aware)
        receiver_id = self._sample_receiver(sender_id)

        # 3. Generate timestamp
        tx_date, tx_time, tx_hour = self._sample_timestamp(sender_id, receiver_id)

        # 4. Generate amount (log-normal with heavy tail)
        amount = self._sample_amount()

        # 5. Payment type
        payment_type = self._sample_payment_type()

        # 6. Currency selection (domestic vs cross-border)
        is_cross_border, payment_currency, received_currency, corridor = \
            self._sample_currency_and_corridor(sender_id, receiver_id)

        # 7. FX rate
        fx_rate = self._get_fx_rate(payment_currency)

        # Update receiver history for this sender
        self._sender_receiver_history.setdefault(sender_id, []).append(receiver_id)

        return {
            "Sender_account": sender_id,
            "Receiver_account": receiver_id,
            "Date": tx_date,
            "Time": tx_time,
            "Amount": round(amount, 2),
            "Payment_currency": payment_currency,
            "Received_currency": received_currency,
            "Payment_type": payment_type,
            "is_cross_border": is_cross_border,
            "remittance_corridor": corridor,
            "fx_rate_to_npr": fx_rate,
            "amount_local_npr": round(amount * fx_rate, 2),
            "hour_of_day": tx_hour,
            "is_fraud": 0,
            "fraud_type": None,
            "aml_risk_indicator": 0,
        }

    # ------------------------------------------------------------------
    # Receiver selection (graph-aware)
    # ------------------------------------------------------------------

    def _sample_receiver(self, sender_id: str) -> str:
        """Select receiver using graph-aware strategy."""
        strategy_r = self.rng.random()
        cumulative = 0.0

        for strategy, prob in _RECEIVER_STRATEGY.items():
            cumulative += prob
            if strategy_r <= cumulative:
                chosen_strategy = strategy
                break
        else:
            chosen_strategy = "new"

        receiver_id = None

        if chosen_strategy == "frequent":
            history = self._sender_receiver_history.get(sender_id, [])
            if history:
                receiver_id = history[int(self.rng.integers(len(history)))]

        if chosen_strategy == "community" or receiver_id is None:
            inst = self._account_institution.get(sender_id)
            if inst and inst in self._inst_to_accounts:
                candidates = [a for a in self._inst_to_accounts[inst] if a != sender_id]
                if candidates:
                    receiver_id = candidates[int(self.rng.integers(len(candidates)))]

        if receiver_id is None:
            # Random new receiver — ensure different from sender
            while True:
                idx = int(self.rng.integers(self._n_accounts))
                rid = self._account_ids[idx]
                if rid != sender_id:
                    receiver_id = rid
                    break

        return receiver_id

    # ------------------------------------------------------------------
    # Timestamp generation
    # ------------------------------------------------------------------

    def _sample_timestamp(
        self, sender_id: str, receiver_id: str
    ) -> tuple[str, str, int]:
        """Generate realistic timestamp, ensuring it's after both accounts opened."""
        # Determine the earliest valid transaction date
        sender_open = self._account_open.get(sender_id, _TX_START)
        recv_open = self._account_open.get(receiver_id, _TX_START)
        earliest = max(max(sender_open, recv_open), _TX_START)
        latest = _TX_END

        if earliest >= latest:
            earliest = _TX_START

        # Sample date
        delta_days = max((latest - earliest).days, 1)
        offset = int(self.rng.integers(0, delta_days))
        date = earliest + timedelta(days=offset)

        # Sample hour from learned distribution
        hour_weights = self._temporal_dist.get("hour_weights", {})
        if hour_weights:
            hours = [int(h) for h in hour_weights.keys()]
            probs = list(hour_weights.values())
            total = sum(probs)
            probs = [p / total for p in probs]
            hour = hours[int(self.rng.choice(len(hours), p=probs))]
        else:
            # Default: business hours peak with some night activity
            all_hours = list(range(24))
            base_weights = [1.0] * 24
            for h in range(9, 18):
                base_weights[h] = 4.0   # business hours
            for h in range(0, 4):
                base_weights[h] = 0.3   # early morning low
            total = sum(base_weights)
            probs = [w / total for w in base_weights]
            hour = int(self.rng.choice(24, p=probs))

        minute = int(self.rng.integers(0, 60))
        second = int(self.rng.integers(0, 60))

        date_str = date.strftime("%Y-%m-%d")
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        return date_str, time_str, hour

    # ------------------------------------------------------------------
    # Amount generation
    # ------------------------------------------------------------------

    def _sample_amount(self) -> float:
        """Generate realistic transaction amount using log-normal distribution."""
        log_mean = self._amount_dist.get("log_mean", 11.0)  # ~NPR 60K
        log_std  = self._amount_dist.get("log_std", 2.5)
        p99      = self._amount_dist.get("p99", 5_000_000.0)
        p_min    = self._amount_dist.get("min", 10.0)

        # Sample from log-normal
        log_val = self.rng.normal(log_mean, log_std)
        amount = float(np.exp(log_val))

        # Clip to realistic range
        amount = float(np.clip(amount, p_min, p99 * 2))
        return max(amount, 1.0)

    # ------------------------------------------------------------------
    # Payment type
    # ------------------------------------------------------------------

    def _sample_payment_type(self) -> str:
        if self._payment_type_dist:
            types = list(self._payment_type_dist.keys())
            probs = list(self._payment_type_dist.values())
            total = sum(probs)
            probs = [p / total for p in probs]
            idx = int(self.rng.choice(len(types), p=probs))
            return types[idx]
        # Default distribution
        types = _VALID_PAYMENT_TYPES
        weights = [0.35, 0.20, 0.10, 0.10, 0.10, 0.08, 0.07]
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = int(self.rng.choice(len(types), p=probs))
        return types[idx]

    # ------------------------------------------------------------------
    # Currency and corridor
    # ------------------------------------------------------------------

    def _sample_currency_and_corridor(
        self, sender_id: str, receiver_id: str
    ) -> tuple[int, str, str, str | None]:
        """Determine currencies and cross-border status."""
        cross_border_rate = self.knowledge.get("country_mapping", {}).get(
            "cross_border_rate", 0.15
        )

        is_cross_border = int(self.rng.random() < cross_border_rate)

        if not is_cross_border:
            return 0, "NPR", "NPR", None

        # Sample a remittance corridor
        corridors = list(CORRIDOR_CURRENCIES.keys())
        corridor = corridors[int(self.rng.integers(len(corridors)))]
        payment_currency = CORRIDOR_CURRENCIES.get(corridor, "USD")
        received_currency = "NPR"

        return 1, payment_currency, received_currency, corridor

    def _get_fx_rate(self, currency: str) -> float:
        """Return a realistic FX rate for the given currency."""
        if currency == "NPR":
            return 1.0
        rate_range = EXCHANGE_RATE_RANGES.get(currency, (1.0, 1.0))
        lo, hi = rate_range
        return float(self.rng.uniform(lo, hi))


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def generate_transactions(
    knowledge: dict,
    accounts_df: pd.DataFrame,
    n: int = 50_000,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate *n* synthetic core transactions and optionally save to CSV."""
    gen = TransactionGenerator(knowledge, accounts_df, seed=seed)
    df = gen.generate(n)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Core transactions saved to: %s", output_path)
    return df


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.generation.core.knowledge_extractor import load_knowledge_base
    kb = load_knowledge_base(root)
    acc = pd.read_csv(root / "data" / "interim" / "synthetic_accounts.csv")
    df = generate_transactions(
        kb, acc, n=10_000,
        output_path=root / "data" / "interim" / "synthetic_transactions_core.csv",
    )
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Cross-border rate: {df['is_cross_border'].mean():.1%}")
    print(f"Fraud (pre-injection): {df['is_fraud'].mean():.1%}")
