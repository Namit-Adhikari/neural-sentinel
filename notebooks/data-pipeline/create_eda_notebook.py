import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Add a title and introduction
title_md = """# Phase 1: Exploratory Data Analysis (EDA) on Real Data
This notebook follows the structured framework outlined in `EDA.md` to establish a defensible, data-driven baseline of the real financial transactions dataset. This baseline is critical for evaluating synthetic data generators later.

### Goals:
1. Understand the structure, missingness, and distributions of the real data.
2. Characterize the extreme class imbalance and target-conditioned behaviors.
3. Validate logical consistency of engineered features.
4. Establish network/graph-level baselines (e.g., fan-in/fan-out degree distributions).
"""
nb.cells.append(nbf.v4.new_markdown_cell(title_md))

# Setup
setup_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import zscore

warnings.filterwarnings('ignore')

# Set aesthetic parameters for seaborn to make plots look 'nice'
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 120})
"""
nb.cells.append(nbf.v4.new_code_cell(setup_code))

# Data Loading
load_md = """## Data Loading
We load the original `transactions.csv` and `ml_features.csv` (which contains the target variable `is_suspicious_tx` and engineered features)."""
nb.cells.append(nbf.v4.new_markdown_cell(load_md))

load_code = """# Define paths (compatible with local and Kaggle if mounted)
import os

if os.path.exists('../../data/original/transactions.csv'):
    tx_path = '../../data/original/transactions.csv'
    ml_path = '../../data/original/ml_features.csv'
else:
    # Kaggle fallback (update as needed)
    tx_path = '/kaggle/input/neural-sentinel-data/transactions.csv'
    ml_path = '/kaggle/input/neural-sentinel-data/ml_features.csv'

print(f"Loading transactions from {tx_path}")
df_tx = pd.read_csv(tx_path)

print(f"Loading ml_features from {ml_path}")
df_ml = pd.read_csv(ml_path)

# Merge if needed, but for now we'll combine them to have a unified view
# Drop overlapping columns before merge except join keys if we wanted to join.
# Since ml_features has same rows, we can just concat horizontally if they align, 
# or merge on Date, Time, Sender_account, Receiver_account
df = pd.merge(df_tx, df_ml[['Date', 'Time', 'Sender_account', 'Receiver_account', 'is_suspicious_tx']], 
              on=['Date', 'Time', 'Sender_account', 'Receiver_account'], 
              how='left')

# If is_suspicious_tx is mostly null due to merge issues, we'll just use df_ml directly for target analysis.
display(df.head(3))
"""
nb.cells.append(nbf.v4.new_code_cell(load_code))

# 1.1 Structure
md_1_1 = """## 1.1 Structure
* Shape, dtypes, memory size
* Duplicate rows / duplicate transaction keys
* Confirm label column exists
* Class imbalance ratio"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_1))

code_1_1 = """print("--- Data Structure ---")
print(f"Shape: {df.shape}")
print(f"Memory Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Identify label column
label_col = 'is_suspicious_tx'
if label_col in df.columns:
    print(f"\\nLabel column found: {label_col}")
    imbalance = df[label_col].value_counts(normalize=True) * 100
    print("\\nClass Imbalance Ratio (%):")
    print(imbalance)
else:
    print(f"\\nLabel column '{label_col}' NOT FOUND!")
    
# Dtypes summary
print("\\nData Types Summary:")
print(df.dtypes.value_counts())
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_1))

# 1.2 Missing values
md_1_2 = """## 1.2 Missing values & quality
* Missingness % per column
* Inconsistent categories"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_2))

code_1_2 = """missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_cols = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False)

if not missing_cols.empty:
    display(missing_cols)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing_cols.index, y='Missing %', data=missing_cols, palette='Reds_r')
    plt.xticks(rotation=45, ha='right')
    plt.title('Missing Values Percentage')
    plt.show()
else:
    print("No missing values found.")
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_2))

# 1.3 Univariate
md_1_3 = """## 1.3 Univariate Analysis
* Numeric distributions: Skew, Kurtosis
* Categorical distributions: Frequency
* Binary/Datetime distributions"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_3))

code_1_3 = """# Key Numeric Columns
numeric_cols = ['Amount', 'amount_local_npr', 'velocity_sum_10tx', 'tx_count_10', 'sender_account_age_days']
numeric_cols = [c for c in numeric_cols if c in df.columns]

df[numeric_cols].hist(bins=50, figsize=(15, 10), color='teal', edgecolor='black')
plt.suptitle('Histograms of Key Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# Categorical Frequencies
cat_cols = ['Payment_type', 'transmode_code', 'sender_risk_grade']
cat_cols = [c for c in cat_cols if c in df.columns]

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='viridis')
    plt.title(f'Frequency of {col}')
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_3))

# 1.4 Target-conditioned
md_1_4 = """## 1.4 Target-conditioned analysis
* Feature distributions split by fraud vs non-fraud
* Fraud rate by categorical variables"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_4))

code_1_4 = """if label_col in df.columns:
    # Amount by Fraud Status
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=label_col, y='log_amount', data=df, palette='Set2')
    plt.title('Log Amount by Suspicious Flag')
    plt.show()
    
    # Fraud Rate by Payment Type
    if 'Payment_type' in df.columns:
        fraud_rate = df.groupby('Payment_type')[label_col].mean().sort_values(ascending=False) * 100
        plt.figure(figsize=(8, 5))
        sns.barplot(x=fraud_rate.values, y=fraud_rate.index, palette='magma')
        plt.title('Suspicious Rate (%) by Payment Type')
        plt.xlabel('Rate (%)')
        plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_4))

# 1.5 Bivariate / multivariate
md_1_5 = """## 1.5 Bivariate / Multivariate
* Correlation matrix for numeric features (checking for multicollinearity)"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_5))

code_1_5 = """plt.figure(figsize=(12, 10))
num_df = df.select_dtypes(include=[np.number])
# Select a subset to avoid giant unreadable heatmaps
corr_cols = [c for c in ['Amount', 'amount_local_npr', 'log_amount', 'velocity_sum_10tx', 'tx_count_10', 'sender_account_age_days', 'fx_rate_to_npr', 'amount_zscore'] if c in num_df.columns]

if corr_cols:
    corr = num_df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=.5)
    plt.title('Correlation Matrix of Selected Numeric Features')
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_5))

# 1.6 Consistency
md_1_6 = """## 1.6 Consistency of engineered columns
* Ensure `log_amount` ≈ log(Amount)
* `amount_local_npr` ≈ Amount * fx_rate_to_npr
* Check binary flag logic"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_6))

code_1_6 = """print("--- Logical Consistency Checks ---")

if all(c in df.columns for c in ['Amount', 'log_amount']):
    # using log1p just in case of 0s
    calc_log = np.log1p(df['Amount'])
    # comparing with a small tolerance
    log_diff = np.abs(calc_log - df['log_amount']).mean()
    print(f"Mean absolute difference between calc log1p(Amount) and log_amount: {log_diff:.6f}")

if all(c in df.columns for c in ['Amount', 'fx_rate_to_npr', 'amount_local_npr']):
    calc_local = df['Amount'] * df['fx_rate_to_npr']
    local_diff = np.abs(calc_local - df['amount_local_npr']).mean()
    print(f"Mean absolute difference between (Amount * fx) and amount_local_npr: {local_diff:.6f}")

if all(c in df.columns for c in ['cross_border_flag', 'Sender_bank_location', 'Receiver_bank_location']):
    calc_cross = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
    cross_diff = (calc_cross != df['cross_border_flag']).sum()
    print(f"Mismatches in cross_border_flag logic: {cross_diff}")
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_6))

# 1.7 Network structure
md_1_7 = """## 1.7 Network structure
* Degree distributions (transaction count per sender/receiver account)
* Important for identifying fan-in/fan-out patterns common in money laundering"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_7))

code_1_7 = """if all(c in df.columns for c in ['sender_account_id', 'receiver_account_id']):
    sender_counts = df['sender_account_id'].value_counts()
    receiver_counts = df['receiver_account_id'].value_counts()
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(sender_counts, bins=50, log_scale=(False, True), ax=ax[0], color='coral')
    ax[0].set_title('Sender Degree Distribution (Log Scale Y)')
    ax[0].set_xlabel('Number of transactions per Sender')
    
    sns.histplot(receiver_counts, bins=50, log_scale=(False, True), ax=ax[1], color='purple')
    ax[1].set_title('Receiver Degree Distribution (Log Scale Y)')
    ax[1].set_xlabel('Number of transactions per Receiver')
    
    plt.tight_layout()
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_7))

# 1.8 Outliers
md_1_8 = """## 1.8 Outliers
* Inspect `amount_zscore` extremes
* Check if outliers cluster in the fraud class"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_8))

code_1_8 = """if 'amount_zscore' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['amount_zscore'], bins=100, kde=True, color='darkred')
    plt.title('Distribution of Amount Z-Scores')
    plt.show()
    
    # Check outlier fraud rate (Z > 3)
    outliers = df[df['amount_zscore'] > 3]
    if label_col in outliers.columns and not outliers.empty:
        outlier_fraud_rate = outliers[label_col].mean() * 100
        overall_fraud_rate = df[label_col].mean() * 100
        print(f"Overall Suspicious Rate: {overall_fraud_rate:.2f}%")
        print(f"Suspicious Rate among Amount Z > 3 Outliers: {outlier_fraud_rate:.2f}%")
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_8))

# Expanded 1.4
md_1_4_exp = """## 1.4 Expanded: Temporal Deep Dive
* Hourly heatmap (hour_of_day vs day_of_week)"""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_4_exp))

code_1_4_exp = """if all(c in df.columns for c in ['hour_of_day', 'day_of_week', label_col]):
    heatmap_data = df.pivot_table(index='day_of_week', columns='hour_of_day', values=label_col, aggfunc='mean') * 100
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap='Reds', annot=False)
    plt.title('Suspicious Rate (%) Heatmap: Day of Week vs Hour of Day')
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_4_exp))

# 1.9 Benford's Law
md_1_9 = """## 1.9 Benford's Law Analysis
* Check if transaction amounts follow the expected first-digit distribution.
* Synthetic data must preserve this distribution."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_9))

code_1_9 = """import math

if 'amount_local_npr' in df.columns:
    # Extract first digit
    df['first_digit'] = df['amount_local_npr'].astype(str).str.extract(r'([1-9])').astype(float)
    
    # Calculate observed frequencies
    observed_counts = df['first_digit'].value_counts(normalize=True).sort_index()
    
    # Calculate expected Benford frequencies
    digits = np.arange(1, 10)
    expected_freq = [math.log10(1 + 1/d) for d in digits]
    
    plt.figure(figsize=(10, 5))
    plt.bar(digits, observed_counts, alpha=0.7, label='Observed (amount_local_npr)')
    plt.plot(digits, expected_freq, color='red', marker='o', linestyle='-', linewidth=2, label='Benford Expected')
    plt.xticks(digits)
    plt.title("Benford's Law: First Digit Distribution")
    plt.xlabel("First Digit")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    mad = np.mean(np.abs(observed_counts - expected_freq))
    print(f"Mean Absolute Deviation (MAD) from Benford: {mad:.5f}")
    if mad < 0.006:
        print("Conclusion: Close conformity")
    elif mad < 0.012:
        print("Conclusion: Acceptable conformity")
    else:
        print("Conclusion: Non-conformity")
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_9))

# 1.10 Structuring
md_1_10 = """## 1.10 Structuring / Threshold Proximity
* Check distribution of amounts near the NRB NPR 1,000,000 threshold."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_10))

code_1_10 = """if 'amount_local_npr' in df.columns:
    # Filter for transactions near 1,000,000 (800k to 1.2M)
    structuring_band = df[(df['amount_local_npr'] >= 800000) & (df['amount_local_npr'] <= 1200000)]
    
    if not structuring_band.empty:
        plt.figure(figsize=(12, 5))
        sns.histplot(structuring_band['amount_local_npr'], bins=100, color='darkorange')
        plt.axvline(1000000, color='red', linestyle='--', label='1,000,000 Threshold')
        plt.title("Transaction Amounts Near 1M NPR Reporting Threshold")
        plt.xlabel("Amount (NPR)")
        plt.legend()
        plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_10))

# 1.11 Dimensionality Reduction
md_1_11 = """## 1.11 Dimensionality Reduction & Class Separability
* Using PCA to visualize class separability."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_11))

code_1_11 = """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

num_df = df.select_dtypes(include=[np.number]).dropna()
if not num_df.empty and label_col in num_df.columns:
    features = [c for c in num_df.columns if c != label_col]
    X = num_df[features]
    y = num_df[label_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
    plt.title('PCA of Numeric Features (Colored by Suspicious Flag)')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.colorbar(scatter, label=label_col)
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_11))


# 1.12 Feature Importance
md_1_12 = """## 1.12 Feature Importance Baseline
* Train XGBoost classifier to find the most discriminative features."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_12))

code_1_12 = """from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if label_col in df.columns:
    # Prepare data (simple fillna and label encoding for baseline)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Handle categoricals
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
    X = X.fillna(0) # Simple imputation for baseline
    
    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title('Top 20 XGBoost Feature Importances (Gain)')
    plt.xlabel('Importance')
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_12))

# 1.13 Account Profiles
md_1_13 = """## 1.13 Account-level Behavioral Profiles
* Fraud is an account-level phenomenon. Generators must preserve account distributions."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_13))

code_1_13 = """if all(c in df.columns for c in ['Sender_account', 'amount_local_npr', label_col]):
    acct_profile = df.groupby('Sender_account').agg({
        'amount_local_npr': ['count', 'sum', 'mean'],
        label_col: 'max' # 1 if any tx is suspicious
    })
    acct_profile.columns = ['tx_count', 'total_volume', 'avg_tx_size', 'is_suspicious_acct']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='is_suspicious_acct', y='tx_count', data=acct_profile)
    plt.yscale('log')
    plt.title('Tx Count by Suspicious Account')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='is_suspicious_acct', y='total_volume', data=acct_profile)
    plt.yscale('log')
    plt.title('Total Volume by Suspicious Account')
    
    plt.tight_layout()
    plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_13))

# 1.14 Corridor Analysis
md_1_14 = """## 1.14 Cross-border & Remittance Corridor Analysis
* Analyze specific cross-border risks."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_14))

code_1_14 = """if all(c in df.columns for c in ['Sender_bank_location', 'Receiver_bank_location', label_col]):
    # Only cross border
    cb_df = df[df['Sender_bank_location'] != df['Receiver_bank_location']]
    
    if not cb_df.empty:
        corridor_rates = cb_df.groupby(['Sender_bank_location', 'Receiver_bank_location'])[label_col].agg(['mean', 'count'])
        corridor_rates = corridor_rates[corridor_rates['count'] > 10] # Filter small corridors
        corridor_rates = corridor_rates.sort_values(by='mean', ascending=False).head(10)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=corridor_rates['mean']*100, y=[f"{idx[0]} -> {idx[1]}" for idx in corridor_rates.index], palette='Reds_r')
        plt.title('Top 10 Riskiest Cross-Border Corridors (Min 10 txs)')
        plt.xlabel('Suspicious Rate (%)')
        plt.show()
    else:
        print("No cross-border transactions found.")
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_14))

# 1.15 Leakage Audit
md_1_15 = """## 1.15 Data Leakage & Generator Input Schema
* Generators should ONLY be trained on base columns.
* Derived columns (log_amount, cross_border_flag, etc.) must be EXCLUDED from generator input and reconstructed post-generation."""
nb.cells.append(nbf.v4.new_markdown_cell(md_1_15))

code_1_15 = """print("Generator Input Schema (Base Columns to use for training):")
base_columns = [
    "transaction_id", "transaction_date", "transaction_time", "sender_account_id", 
    "receiver_account_id", "transaction_type", "amount_npr", "original_currency", 
    "exchange_rate", "channel", "sender_country", "receiver_country", "merchant_category", 
    "device_type", "ip_address", "ip_country", "ip_is_vpn", "is_fraud", "fraud_type", "aml_risk_indicator"
]
print(f"Total Base Columns: {len(base_columns)}")
for c in base_columns:
    print(f" - {c}")
"""
nb.cells.append(nbf.v4.new_code_cell(code_1_15))

with open('c:/Users/sauga/Desktop/neural-sentinel/notebooks/data-pipeline/phase1_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
