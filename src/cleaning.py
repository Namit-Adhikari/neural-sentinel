import os
import uuid
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path to ensure it runs from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.data_contracts import Transaction, Account
from src.utils.nepal_context import CHANNEL_MIX

def clean_data():
    data_dir = os.path.join(PROJECT_ROOT, "data", "original")
    interim_dir = os.path.join(PROJECT_ROOT, "data", "interim")
    os.makedirs(interim_dir, exist_ok=True)
    
    print("Loading raw data...")
    tx_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    acc_df = pd.read_csv(os.path.join(data_dir, "accounts.csv"))
    ml_df = pd.read_csv(os.path.join(data_dir, "ml_features.csv"))
    
    # ---------------------------------------------------------
    # ACCOUNTS
    # ---------------------------------------------------------
    print("Processing accounts...")
    
    # Map account types
    acct_type_map = {
        "FIXED": "fixed_deposit",
        "SAVINGS": "savings",
        "CURRENT": "current",
        "NOSTRO": "current",
        "SALARY": "salary"
    }
    
    # Map risk grades
    risk_map = {
        "RISK-LOW": "low",
        "RISK-MED": "medium",
        "RISK-MEDIUM": "medium",
        "RISK-HIGH": "high"
    }
    
    cleaned_acc = pd.DataFrame()
    cleaned_acc['account_id'] = acc_df['account_id'].astype(str)
    cleaned_acc['account_type'] = acc_df['acct_type'].map(acct_type_map).fillna("current")
    cleaned_acc['account_open_date'] = pd.to_datetime(acc_df['opened'])
    
    reference_date = pd.to_datetime("2024-01-01")
    cleaned_acc['account_age_days'] = (reference_date - cleaned_acc['account_open_date']).dt.days
    cleaned_acc['account_age_days'] = cleaned_acc['account_age_days'].clip(lower=0)
    cleaned_acc['account_open_date'] = cleaned_acc['account_open_date'].dt.date
    
    cleaned_acc['kyc_verified'] = 1  # Assume all verified in this dataset
    cleaned_acc['kyc_risk_grade'] = acc_df['risk_grade'].map(risk_map).fillna("medium")
    cleaned_acc['is_pep'] = acc_df['pep_flag'].astype(int)
    cleaned_acc['is_sanctioned'] = acc_df['sanctions_hit'].astype(int)
    
    # Deriving volume/count from transactions
    tx_volume = tx_df.groupby('Sender_account')['amount_local_npr'].sum().reset_index()
    tx_volume.rename(columns={'Sender_account': 'account_id', 'amount_local_npr': 'total_vol'}, inplace=True)
    tx_volume['account_id'] = tx_volume['account_id'].astype(str)
    
    tx_count = tx_df.groupby('Sender_account').size().reset_index(name='total_count')
    tx_count.rename(columns={'Sender_account': 'account_id'}, inplace=True)
    tx_count['account_id'] = tx_count['account_id'].astype(str)
    
    cleaned_acc = cleaned_acc.merge(tx_volume, on='account_id', how='left')
    cleaned_acc = cleaned_acc.merge(tx_count, on='account_id', how='left')
    
    # Assume 12 months for average
    cleaned_acc['average_monthly_volume'] = (cleaned_acc['total_vol'].fillna(0) / 12.0).round(2)
    cleaned_acc['average_monthly_count'] = (cleaned_acc['total_count'].fillna(0) / 12.0).astype(int)
    
    cleaned_acc['country'] = "Nepal"  # Default
    cleaned_acc['city'] = acc_df['city'].fillna("Kathmandu")
    cleaned_acc['is_mule'] = 0
    
    # Validate via Pydantic model
    validated_accounts = []
    for row in cleaned_acc.to_dict('records'):
        try:
            validated_accounts.append(Account(**row).model_dump())
        except Exception as e:
            # Fallback for minor errors
            row['average_monthly_volume'] = max(0.0, row['average_monthly_volume'])
            row['average_monthly_count'] = max(0, row['average_monthly_count'])
            validated_accounts.append(Account(**row).model_dump())
            
    final_acc_df = pd.DataFrame(validated_accounts)
    
    # ---------------------------------------------------------
    # TRANSACTIONS
    # ---------------------------------------------------------
    print("Processing transactions...")
    cleaned_tx = pd.DataFrame()
    cleaned_tx['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(tx_df))]
    cleaned_tx['transaction_date'] = pd.to_datetime(tx_df['Date']).dt.date
    cleaned_tx['transaction_time'] = pd.to_datetime(tx_df['Time']).dt.time
    cleaned_tx['sender_account_id'] = tx_df['Sender_account'].astype(str)
    cleaned_tx['receiver_account_id'] = tx_df['Receiver_account'].astype(str)
    
    # Map transaction types
    def map_tx_type(ptype, sender_loc, recv_loc):
        if ptype == "Cash Deposit": return "deposit"
        if ptype == "Cross-border":
            return "remittance_inbound" if recv_loc == "Nepal" else "remittance_outbound"
        return "transfer"
    
    cleaned_tx['transaction_type'] = tx_df.apply(
        lambda r: map_tx_type(r['Payment_type'], r['Sender_bank_location'], r['Receiver_bank_location']), axis=1
    )
    
    # Ensure amount and exchange rate are positive
    cleaned_tx['amount_npr'] = tx_df['amount_local_npr'].clip(lower=0.01)
    
    curr_map = {"UK pounds": "GBP", "US Dollar": "USD", "Euro": "EUR", "Qatari Rial": "QAR"}
    cleaned_tx['original_currency'] = tx_df['Payment_currency'].map(curr_map).fillna("NPR")
    
    cleaned_tx['exchange_rate'] = tx_df['fx_rate_to_npr'].clip(lower=0.01)
    
    channels = list(CHANNEL_MIX.keys())
    probs = list(CHANNEL_MIX.values())
    cleaned_tx['channel'] = np.random.choice(channels, size=len(tx_df), p=probs)
    
    cleaned_tx['sender_country'] = tx_df['Sender_bank_location']
    cleaned_tx['receiver_country'] = tx_df['Receiver_bank_location']
    
    cleaned_tx['is_cross_border'] = tx_df['cross_border_flag'].astype(int)
    
    def get_corridor(sender, recv, is_cb):
        if is_cb:
            s = sender if sender != "Nepal" else "Nepal"
            r = recv if recv != "Nepal" else "Nepal"
            if s == "Nepal" and r == "Nepal": return "India->Nepal" # Fallback
            return f"{s}->{r}"
        return None
        
    cleaned_tx['remittance_corridor'] = tx_df.apply(
        lambda r: get_corridor(r['Sender_bank_location'], r['Receiver_bank_location'], r['cross_border_flag']), axis=1
    )
    
    cleaned_tx['merchant_category'] = None
    cleaned_tx['device_type'] = "mobile"
    cleaned_tx['ip_address'] = "192.168.1.1"
    cleaned_tx['ip_country'] = cleaned_tx['sender_country']
    cleaned_tx['ip_is_vpn'] = 0
    
    cleaned_tx['is_fraud'] = ml_df['is_suspicious_tx'].astype(int)
    
    def assign_fraud_type(is_f):
        if not is_f: return None
        return np.random.choice(["transaction_fraud", "aml_structuring", "aml_layering", "aml_mule_network"])
        
    cleaned_tx['fraud_type'] = cleaned_tx['is_fraud'].apply(assign_fraud_type)
    cleaned_tx['aml_risk_indicator'] = cleaned_tx['is_fraud']
    
    # Validate via Pydantic
    validated_tx = []
    for row in cleaned_tx.to_dict('records'):
        # Fix missing corridor for cross border
        if row['is_cross_border'] == 1 and not row['remittance_corridor']:
            row['remittance_corridor'] = f"{row['sender_country']}->{row['receiver_country']}"
            
        validated_tx.append(Transaction(**row).model_dump())
        
    final_tx_df = pd.DataFrame(validated_tx)
    
    print(f"Writing {len(final_acc_df)} accounts to Parquet...")
    final_acc_df.to_parquet(os.path.join(interim_dir, "accounts.parquet"), index=False)
    
    print(f"Writing {len(final_tx_df)} transactions to Parquet...")
    final_tx_df.to_parquet(os.path.join(interim_dir, "transactions.parquet"), index=False)
    print("Done!")

if __name__ == "__main__":
    clean_data()
