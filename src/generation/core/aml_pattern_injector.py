"""
src/generation/aml_pattern_injector.py
----------------------------------------
Phase 6 — Fraud Scenario Injection.

Injects all 11 fraud scenarios defined in Complete Pipeline.md into
the synthetic transaction dataset. Each transaction gets is_fraud=1
and a fraud_type label.

Scenarios
---------
1.  Large Amount Fraud            — extremely high amount, new receiver, unusual hour
2.  Smurfing / Structuring        — multiple txns just below NPR 1M threshold
3.  Velocity Fraud                — many rapid txns in a very short window
4.  Money Mule                    — victim → mule → beneficiary chain
5.  Fan-Out                       — one sender → many receivers simultaneously
6.  Fan-In                        — many senders → one receiver simultaneously
7.  Layering (chain)              — A → B → C → D sequential transfers
8.  Circular Transactions         — A → B → C → A cycles
9.  Cross-Border Fraud            — high-risk destination, currency mismatch
10. Dormant Account Activation    — large txn after 180+ day inactivity
11. PEP / High-Risk Customer      — transactions involving PEP/sanctioned accounts
"""

import pandas as pd
import numpy as np
import uuid
from datetime import timedelta


class AMLPatternInjector:
    """
    Injects specific AML (Anti-Money Laundering) patterns into synthetic
    transaction data. Covers all 11 fraud scenarios from Complete Pipeline.md.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.npr_threshold = self.config.get("npr_threshold", 1_000_000)
        self.num_injections = self.config.get("num_injections", 100)
        self._rng = np.random.default_rng(self.config.get("seed", 42))

    # ------------------------------------------------------------------
    # Master injection method
    # ------------------------------------------------------------------

    def inject_all_patterns(
        self, transactions: pd.DataFrame, accounts: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply all 11 AML fraud patterns to the dataset.

        Parameters
        ----------
        transactions : pd.DataFrame
            Core synthetic transactions (output of Phase 5).
        accounts : pd.DataFrame
            Synthetic accounts (output of Phase 4).

        Returns
        -------
        pd.DataFrame with fraud rows appended and labelled.
        """
        df = transactions.copy()
        df = self.inject_large_amount_fraud(df, accounts)       # Scenario 1
        df = self.inject_structuring(df, accounts)              # Scenario 2
        df = self.inject_velocity_fraud(df, accounts)           # Scenario 3
        df = self.inject_money_mule(df, accounts)               # Scenario 4
        df = self.inject_fan_in_out(df, accounts)               # Scenario 5 & 6
        df = self.inject_layering_chains(df, accounts)          # Scenario 7
        df = self.inject_cycles(df, accounts)                   # Scenario 8
        df = self.inject_cross_border_fraud(df, accounts)       # Scenario 9
        df = self.inject_dormant_account_activation(df, accounts)  # Scenario 10
        df = self.inject_pep_high_risk(df, accounts)            # Scenario 11
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Scenario 1 — Large Amount Fraud
    # ------------------------------------------------------------------

    def inject_large_amount_fraud(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extremely high amount, new receiver, unusual hour (2–4 AM)."""
        new_txs = []
        for _ in range(self.num_injections):
            sender, receiver = self._get_random_accounts(accounts, 2)
            # Amount well above reporting threshold: 2M–50M NPR
            amount = float(self._rng.uniform(2_000_000, 50_000_000))
            # Unusual hour: 2–4 AM
            hour = int(self._rng.integers(2, 5))
            minute = int(self._rng.integers(0, 60))
            second = int(self._rng.integers(0, 60))
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            tx_time = base_time.replace(hour=hour, minute=minute, second=second)
            tx = self._create_base_tx()
            tx.update(
                {
                    "Sender_account": sender,
                    "Receiver_account": receiver,
                    "amount_local_npr": amount,
                    "Amount": amount,
                    "Date": tx_time.strftime("%Y-%m-%d"),
                    "Time": tx_time.strftime("%H:%M:%S"),
                    "hour_of_day": hour,
                    "fraud_type": "transaction_fraud",
                    "aml_risk_indicator": 0,
                }
            )
            new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 2 — Smurfing / Structuring
    # ------------------------------------------------------------------

    def inject_structuring(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Multiple transactions just below the NPR 1M reporting threshold."""
        new_txs = []
        for _ in range(self.num_injections):
            sender, receiver = self._get_random_accounts(accounts, 2)
            num_tx = int(self._rng.integers(3, 7))
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            for i in range(num_tx):
                amount = float(
                    self._rng.uniform(
                        self.npr_threshold * 0.90, self.npr_threshold * 0.99
                    )
                )
                tx_time = base_time + timedelta(hours=i)
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "aml_structuring",
                        "aml_risk_indicator": 1,
                    }
                )
                new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 3 — Velocity Fraud
    # ------------------------------------------------------------------

    def inject_velocity_fraud(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Many rapid transactions from one sender within minutes."""
        new_txs = []
        for _ in range(self.num_injections):
            sender = self._get_random_accounts(accounts, 1)[0]
            num_tx = int(self._rng.integers(20, 51))  # 20–50 transactions
            # Small amount per transaction
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            for i in range(num_tx):
                receiver = self._get_random_accounts(accounts, 1)[0]
                amount = float(self._rng.uniform(1_000, 50_000))
                # Transactions seconds apart
                tx_time = base_time + timedelta(seconds=int(i * self._rng.integers(5, 30)))
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "transaction_fraud",
                        "aml_risk_indicator": 0,
                    }
                )
                new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 4 — Money Mule (Victim → Mule → Beneficiary)
    # ------------------------------------------------------------------

    def inject_money_mule(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Victim → Mule → Beneficiary chain with realistic timing."""
        new_txs = []
        for _ in range(self.num_injections):
            victim, mule, beneficiary = self._get_random_accounts(accounts, 3)
            amount = float(self._rng.uniform(100_000, 2_000_000))
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )

            # Leg 1: Victim → Mule (within hours)
            t1 = base_time
            tx1 = self._create_base_tx()
            tx1.update(
                {
                    "Sender_account": victim,
                    "Receiver_account": mule,
                    "amount_local_npr": amount,
                    "Amount": amount,
                    "Date": t1.strftime("%Y-%m-%d"),
                    "Time": t1.strftime("%H:%M:%S"),
                    "hour_of_day": t1.hour,
                    "fraud_type": "aml_mule_network",
                    "aml_risk_indicator": 1,
                }
            )
            new_txs.append(tx1)

            # Leg 2: Mule → Beneficiary (30 min to 6 hours later)
            t2 = t1 + timedelta(minutes=int(self._rng.integers(30, 360)))
            mule_amount = amount * float(self._rng.uniform(0.90, 0.98))  # subtract "fee"
            tx2 = self._create_base_tx()
            tx2.update(
                {
                    "Sender_account": mule,
                    "Receiver_account": beneficiary,
                    "amount_local_npr": mule_amount,
                    "Amount": mule_amount,
                    "Date": t2.strftime("%Y-%m-%d"),
                    "Time": t2.strftime("%H:%M:%S"),
                    "hour_of_day": t2.hour,
                    "fraud_type": "aml_mule_network",
                    "aml_risk_indicator": 1,
                }
            )
            new_txs.append(tx2)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 5 & 6 — Fan-Out and Fan-In
    # ------------------------------------------------------------------

    def inject_fan_in_out(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Fan-In (N→1) and Fan-Out (1→N) patterns."""
        new_txs = []

        # Fan-in
        for _ in range(self.num_injections // 2):
            num_nodes = int(self._rng.integers(5, 15))
            nodes = self._get_random_accounts(accounts, num_nodes)
            receiver = nodes[0]
            senders = nodes[1:]
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            for sender in senders:
                tx_time = base_time + timedelta(minutes=int(self._rng.integers(1, 120)))
                amount = float(self._rng.uniform(50_000, 500_000))
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "aml_mule_network",
                        "aml_risk_indicator": 1,
                    }
                )
                new_txs.append(tx)

        # Fan-out
        for _ in range(self.num_injections // 2):
            num_nodes = int(self._rng.integers(5, 15))
            nodes = self._get_random_accounts(accounts, num_nodes)
            sender = nodes[0]
            receivers = nodes[1:]
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            for receiver in receivers:
                tx_time = base_time + timedelta(minutes=int(self._rng.integers(1, 120)))
                amount = float(self._rng.uniform(50_000, 500_000))
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "aml_mule_network",
                        "aml_risk_indicator": 1,
                    }
                )
                new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 7 — Layering Chains (A→B→C→D)
    # ------------------------------------------------------------------

    def inject_layering_chains(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """A → B → C → D rapid transaction chains with peeling amounts."""
        new_txs = []
        for _ in range(self.num_injections):
            chain_length = int(self._rng.integers(3, 6))
            chain_accounts = self._get_random_accounts(accounts, chain_length)
            base_amount = float(self._rng.uniform(100_000, 5_000_000))
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            for i in range(chain_length - 1):
                sender = chain_accounts[i]
                receiver = chain_accounts[i + 1]
                amount = base_amount * float(self._rng.uniform(0.95, 0.99))
                base_amount = amount
                tx_time = base_time + timedelta(
                    minutes=int(self._rng.integers(5, 60)) * i
                )
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "aml_layering",
                        "aml_risk_indicator": 1,
                    }
                )
                new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 8 — Circular Transactions (A→B→C→A)
    # ------------------------------------------------------------------

    def inject_cycles(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Circular trading pattern: A → B → C → A."""
        new_txs = []
        for _ in range(self.num_injections):
            cycle_length = int(self._rng.integers(3, 6))
            chain_accounts = self._get_random_accounts(accounts, cycle_length)
            base_amount = float(self._rng.uniform(500_000, 2_000_000))
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            for i in range(cycle_length):
                sender = chain_accounts[i]
                receiver = chain_accounts[(i + 1) % cycle_length]
                amount = base_amount * float(self._rng.uniform(0.95, 1.05))
                tx_time = base_time + timedelta(
                    hours=int(i * self._rng.integers(1, 5))
                )
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "aml_layering",
                        "aml_risk_indicator": 1,
                    }
                )
                new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 9 — Cross-Border Fraud
    # ------------------------------------------------------------------

    def inject_cross_border_fraud(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """High-risk corridor, currency mismatch, foreign transfer."""
        from src.utils.nepal_context import CORRIDOR_CURRENCIES

        # High-risk corridors
        high_risk_corridors = ["USA->Nepal", "UK->Nepal", "Australia->Nepal", "Japan->Nepal"]
        new_txs = []

        for _ in range(self.num_injections):
            sender, receiver = self._get_random_accounts(accounts, 2)
            corridor = high_risk_corridors[int(self._rng.integers(len(high_risk_corridors)))]
            currency = CORRIDOR_CURRENCIES.get(corridor.replace("->", "->"), "USD")
            amount_foreign = float(self._rng.uniform(5_000, 100_000))
            fx_rate = float(self._rng.uniform(110.0, 170.0))
            amount_npr = amount_foreign * fx_rate

            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )
            tx = self._create_base_tx()
            tx.update(
                {
                    "Sender_account": sender,
                    "Receiver_account": receiver,
                    "amount_local_npr": amount_npr,
                    "Amount": amount_foreign,
                    "Payment_currency": currency,
                    "Received_currency": "NPR",
                    "Date": base_time.strftime("%Y-%m-%d"),
                    "Time": base_time.strftime("%H:%M:%S"),
                    "hour_of_day": base_time.hour,
                    "is_cross_border": 1,
                    "remittance_corridor": corridor,
                    "fx_rate_to_npr": fx_rate,
                    "currency_mismatch": 1,
                    "fraud_type": "aml_layering",
                    "aml_risk_indicator": 1,
                }
            )
            new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 10 — Dormant Account Activation
    # ------------------------------------------------------------------

    def inject_dormant_account_activation(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Large transactions immediately after 180+ day inactivity period."""
        new_txs = []

        for _ in range(self.num_injections):
            sender, receiver = self._get_random_accounts(accounts, 2)
            # Simulate the dormancy gap then a burst of large transactions
            # Gap: 180–730 days
            gap_days = int(self._rng.integers(180, 731))
            base_time = pd.Timestamp("2022-10-07") + pd.Timedelta(days=gap_days)
            if base_time > pd.Timestamp("2023-10-06"):
                base_time = pd.Timestamp("2023-10-01")

            # 2–4 large transactions in quick succession
            num_burst = int(self._rng.integers(2, 5))
            for j in range(num_burst):
                amount = float(self._rng.uniform(500_000, 10_000_000))
                tx_time = base_time + timedelta(hours=j * int(self._rng.integers(1, 6)))
                tx = self._create_base_tx()
                tx.update(
                    {
                        "Sender_account": sender,
                        "Receiver_account": receiver,
                        "amount_local_npr": amount,
                        "Amount": amount,
                        "Date": tx_time.strftime("%Y-%m-%d"),
                        "Time": tx_time.strftime("%H:%M:%S"),
                        "hour_of_day": tx_time.hour,
                        "fraud_type": "transaction_fraud",
                        "aml_risk_indicator": 0,
                    }
                )
                new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Scenario 11 — PEP / High-Risk Customer
    # ------------------------------------------------------------------

    def inject_pep_high_risk(
        self, df: pd.DataFrame, accounts: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Transactions involving PEP-flagged or sanctioned accounts."""
        new_txs = []

        # Try to find PEP accounts in the accounts dataframe
        pep_accounts: list[str] = []
        if accounts is not None and "pep_flag" in accounts.columns:
            pep_mask = accounts["pep_flag"].astype(int) == 1
            pep_accounts = accounts.loc[pep_mask, "account_id"].astype(str).tolist()
        if accounts is not None and "sanctions_hit" in accounts.columns:
            sanc_mask = accounts["sanctions_hit"].astype(int) == 1
            pep_accounts += accounts.loc[sanc_mask, "account_id"].astype(str).tolist()

        for _ in range(self.num_injections):
            # Use a known PEP account if available, else random
            if pep_accounts:
                sender = pep_accounts[int(self._rng.integers(len(pep_accounts)))]
            else:
                sender = self._get_random_accounts(accounts, 1)[0]

            receiver = self._get_random_accounts(accounts, 1)[0]
            amount = float(self._rng.uniform(200_000, 5_000_000))
            base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(self._rng.integers(0, 365))
            )

            tx = self._create_base_tx()
            tx.update(
                {
                    "Sender_account": sender,
                    "Receiver_account": receiver,
                    "amount_local_npr": amount,
                    "Amount": amount,
                    "Date": base_time.strftime("%Y-%m-%d"),
                    "Time": base_time.strftime("%H:%M:%S"),
                    "hour_of_day": base_time.hour,
                    "fraud_type": "identity_fraud",
                    "aml_risk_indicator": 1,
                }
            )
            new_txs.append(tx)

        return self._append(df, new_txs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_random_accounts(
        self, accounts: pd.DataFrame, n: int = 2
    ) -> list[str]:
        if accounts is not None and "account_id" in accounts.columns:
            ids = accounts["account_id"].astype(str).tolist()
            chosen = self._rng.choice(len(ids), size=min(n, len(ids)), replace=False)
            return [ids[i] for i in chosen]
        return [str(uuid.uuid4())[:8] for _ in range(n)]

    def _create_base_tx(self) -> dict:
        return {
            "transaction_id": str(uuid.uuid4()),
            "Payment_currency": "NPR",
            "Received_currency": "NPR",
            "Payment_type": "transfer",
            "transaction_type": "transfer",
            "channel": "mobile_banking",
            "fx_rate_to_npr": 1.0,
            "is_cross_border": 0,
            "remittance_corridor": None,
            "currency_mismatch": 0,
            "is_fraud": 1,
        }

    def _append(self, df: pd.DataFrame, new_txs: list[dict]) -> pd.DataFrame:
        if not new_txs:
            return df
        return pd.concat([df, pd.DataFrame(new_txs)], ignore_index=True)
