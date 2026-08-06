# Data Dictionary for Original CSV Files

This document provides a breakdown of each CSV file present in this directory and describes their contents and specific fields.

## 1. `accounts.csv`
Contains the profile and demographic data of all bank accounts in the dataset.

**Fields:**
- `account_id`: Unique identifier for the account.
- `account_number`: Formal bank account number format.
- `institution`: Bank or financial institution holding the account (e.g., HBL, NCC).
- `branch`: Specific branch identifier.
- `acct_type`: Type of the account (e.g., SAVINGS, FIXED, CURRENT).
- `risk_grade`: Initial KYC risk assessment (e.g., RISK-LOW, RISK-HIGH).
- `is_person`: Boolean flag indicating if the account belongs to an individual (True) or a corporate entity (False).
- `name`: Account holder's name.
- `tax_number`: Tax identification number (PAN).
- `pep_flag`: Indicates if the account holder is a Politically Exposed Person (0=No, 1=Yes).
- `sanctions_hit`: Indicates if the account holder matches any sanctions list (0=No, 1=Yes).
- `city`: Account holder's city of residence.
- `opened`: The date when the account was opened.

## 2. `transactions.csv`
Contains comprehensive details of financial transactions occurring between accounts. Includes both raw details and some enriched or joined metadata from the sender/receiver.

**Key Fields:**
- **Identifiers & Temporal**: `row_index`, `Date`, `Time`, `date_transaction`, `hour_of_day`, `day_of_week`, `is_weekend`, `month`
- **Parties involved**: `Sender_account`, `Receiver_account`
- **Amount Details**: `Amount`, `Payment_currency`, `Received_currency`, `fx_rate_to_npr`, `amount_local_npr`, `log_amount`, `amount_zscore`, `above_1M_NPR`, `above_10M_NPR`
- **Location & Risk**: `Sender_bank_location`, `Receiver_bank_location`, `sender_country_risk`, `receiver_country_risk`, `cross_border_flag`, `currency_mismatch`
- **Transaction Types**: `Payment_type`, `transmode_code`, along with one-hot encoded flags like `transmode_A`, `transmode_B`, etc.
- **Velocity Features**: `velocity_sum_10tx`, `tx_count_10`, `tx_count_30`
- **Sender/Receiver Enriched Data**: Includes branches, institutions, account types, risk grades, PEP flags, sanctions flags, and account age in days (e.g. `sender_account_age_days`, `receiver_pep`).

## 3. `graph_edges.csv`
A lightweight, stripped-down file meant specifically for graph/network analysis (e.g., node-edge representations).

**Fields:**
- `row_index`: Refers to the transaction sequence/identifier.
- `Sender_account`: The source node (account initiating the transfer).
- `Receiver_account`: The target node (account receiving the transfer).
- `amount_local_npr`: The weight of the edge (transaction volume in NPR).
- `Date` / `Time`: Temporal markers for when the edge was created.

## 4. `ml_features.csv`
Contains the preprocessed, numerical, and engineered feature set ready to be ingested directly into Machine Learning models (like Random Forest, XGBoost, or Neural Networks).

**Fields:**
- Contains a subset of `transactions.csv` but focuses only on features useful for modeling.
- **Core Identifiers**: `Date`, `Time`, `Sender_account`, `Receiver_account`.
- **Amounts & Scaling**: `amount_local_npr`, `log_amount`, `amount_zscore`, `above_1M_NPR`, `above_10M_NPR`.
- **Temporal & Velocity**: `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `velocity_sum_10tx`, `tx_count_10`, `tx_count_30`.
- **Risk Indicators**: `sender_country_risk`, `receiver_country_risk`, `cross_border_flag`, `currency_mismatch`.
- **KYC & Account Info**: `sender_account_age_days`, `receiver_account_age_days`, `sender_is_person`, `sender_pep`, `sender_sanctions`, `receiver_pep`, `receiver_sanctions`.
- **Categorical Encodings**: `transmode_A`, `transmode_B`, `transmode_E`, `transmode_F`, `transmode_J`, `transmode_P`, `transmode_Z`.
- **Target Variable**: `is_suspicious_tx` (The label used for supervised training: 1 for suspicious/fraudulent, 0 for legitimate).
