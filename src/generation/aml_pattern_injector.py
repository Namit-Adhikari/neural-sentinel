import pandas as pd
import numpy as np
import uuid
from datetime import timedelta

class AMLPatternInjector:
    """
    Injects specific AML (Anti-Money Laundering) patterns into synthetic transaction data.
    These patterns are crucial for graph/network-based detection models and include:
    - Structuring (Smurfing)
    - Layering (Chains)
    - Fan-in / Fan-out
    - Cycle / Circular trading
    - Mule networks
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.npr_threshold = self.config.get('npr_threshold', 1000000)
        self.num_injections = self.config.get('num_injections', 100)
        
    def inject_all_patterns(self, transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
        """Applies all configured AML patterns to the dataset."""
        df = transactions.copy()
        df = self.inject_structuring(df, accounts)
        df = self.inject_layering_chains(df, accounts)
        df = self.inject_fan_in_out(df, accounts)
        df = self.inject_cycles(df, accounts)
        return df

    def _get_random_accounts(self, accounts: pd.DataFrame, n=2):
        if accounts is not None and 'account_id' in accounts.columns:
            return np.random.choice(accounts['account_id'], size=n, replace=False)
        else:
            return [str(uuid.uuid4())[:8] for _ in range(n)]

    def _create_base_tx(self):
        # Create a dictionary of default transaction fields
        return {
            'transaction_id': str(uuid.uuid4()),
            'transaction_type': 'transfer',
            'channel': 'mobile_banking',
            'is_fraud': 1
        }

    def inject_structuring(self, df: pd.DataFrame, accounts: pd.DataFrame = None) -> pd.DataFrame:
        """Injects multiple small transactions just below reporting thresholds."""
        new_txs = []
        for _ in range(self.num_injections):
            sender, receiver = self._get_random_accounts(accounts, 2)
            # 3 to 6 transactions just below the threshold
            num_tx = np.random.randint(3, 7)
            base_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            
            for i in range(num_tx):
                amount = np.random.uniform(self.npr_threshold * 0.90, self.npr_threshold * 0.99)
                tx_time = base_time + timedelta(hours=i)
                tx = self._create_base_tx()
                tx.update({
                    'sender_account_id': sender,
                    'receiver_account_id': receiver,
                    'amount_npr': amount,
                    'transaction_date': tx_time.date(),
                    'transaction_time': tx_time.time(),
                    'fraud_type': 'aml_structuring',
                    'aml_risk_indicator': 1
                })
                new_txs.append(tx)
                
        if new_txs:
            df = pd.concat([df, pd.DataFrame(new_txs)], ignore_index=True)
        return df
        
    def inject_layering_chains(self, df: pd.DataFrame, accounts: pd.DataFrame = None) -> pd.DataFrame:
        """Injects A -> B -> C -> D rapid transaction chains."""
        new_txs = []
        for _ in range(self.num_injections):
            chain_length = np.random.randint(3, 6)
            chain_accounts = self._get_random_accounts(accounts, chain_length)
            base_amount = np.random.uniform(100000, 5000000)
            base_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            
            for i in range(chain_length - 1):
                sender = chain_accounts[i]
                receiver = chain_accounts[i+1]
                # Slightly decrease amount to simulate fees/layering splits
                amount = base_amount * np.random.uniform(0.95, 0.99)
                base_amount = amount
                tx_time = base_time + timedelta(minutes=np.random.randint(5, 60) * i)
                
                tx = self._create_base_tx()
                tx.update({
                    'sender_account_id': sender,
                    'receiver_account_id': receiver,
                    'amount_npr': amount,
                    'transaction_date': tx_time.date(),
                    'transaction_time': tx_time.time(),
                    'fraud_type': 'aml_layering',
                    'aml_risk_indicator': 1
                })
                new_txs.append(tx)
                
        if new_txs:
            df = pd.concat([df, pd.DataFrame(new_txs)], ignore_index=True)
        return df
        
    def inject_fan_in_out(self, df: pd.DataFrame, accounts: pd.DataFrame = None) -> pd.DataFrame:
        """Injects N -> 1 (Fan-in) and 1 -> N (Fan-out) patterns."""
        new_txs = []
        # Fan-in
        for _ in range(self.num_injections // 2):
            num_nodes = np.random.randint(5, 15)
            nodes = self._get_random_accounts(accounts, num_nodes)
            receiver = nodes[0]
            senders = nodes[1:]
            base_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            
            for i, sender in enumerate(senders):
                tx_time = base_time + timedelta(minutes=np.random.randint(1, 120))
                amount = np.random.uniform(50000, 500000)
                tx = self._create_base_tx()
                tx.update({
                    'sender_account_id': sender,
                    'receiver_account_id': receiver,
                    'amount_npr': amount,
                    'transaction_date': tx_time.date(),
                    'transaction_time': tx_time.time(),
                    'fraud_type': 'aml_mule_network',
                    'aml_risk_indicator': 1
                })
                new_txs.append(tx)
                
        # Fan-out
        for _ in range(self.num_injections // 2):
            num_nodes = np.random.randint(5, 15)
            nodes = self._get_random_accounts(accounts, num_nodes)
            sender = nodes[0]
            receivers = nodes[1:]
            base_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            
            for i, receiver in enumerate(receivers):
                tx_time = base_time + timedelta(minutes=np.random.randint(1, 120))
                amount = np.random.uniform(50000, 500000)
                tx = self._create_base_tx()
                tx.update({
                    'sender_account_id': sender,
                    'receiver_account_id': receiver,
                    'amount_npr': amount,
                    'transaction_date': tx_time.date(),
                    'transaction_time': tx_time.time(),
                    'fraud_type': 'aml_mule_network',
                    'aml_risk_indicator': 1
                })
                new_txs.append(tx)
                
        if new_txs:
            df = pd.concat([df, pd.DataFrame(new_txs)], ignore_index=True)
        return df
        
    def inject_cycles(self, df: pd.DataFrame, accounts: pd.DataFrame = None) -> pd.DataFrame:
        """Injects circular trading patterns (e.g., A -> B -> C -> A)."""
        new_txs = []
        for _ in range(self.num_injections):
            cycle_length = np.random.randint(3, 6)
            chain_accounts = self._get_random_accounts(accounts, cycle_length)
            base_amount = np.random.uniform(500000, 2000000)
            base_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            
            for i in range(cycle_length):
                sender = chain_accounts[i]
                receiver = chain_accounts[(i + 1) % cycle_length] # Loop back to start
                amount = base_amount * np.random.uniform(0.95, 1.05)
                tx_time = base_time + timedelta(hours=i * np.random.randint(1, 5))
                
                tx = self._create_base_tx()
                tx.update({
                    'sender_account_id': sender,
                    'receiver_account_id': receiver,
                    'amount_npr': amount,
                    'transaction_date': tx_time.date(),
                    'transaction_time': tx_time.time(),
                    'fraud_type': 'aml_layering',
                    'aml_risk_indicator': 1
                })
                new_txs.append(tx)
                
        if new_txs:
            df = pd.concat([df, pd.DataFrame(new_txs)], ignore_index=True)
        return df
