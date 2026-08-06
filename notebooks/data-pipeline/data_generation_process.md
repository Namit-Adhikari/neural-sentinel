# Synthetic Banking Data Generation Pipeline

The Neural Sentinel project uses a **10-phase layered generation strategy** to create a highly realistic 5-million row synthetic dataset. Instead of using a single tabular ML generator (like CTGAN) which struggles with relational consistency and graph structures, the pipeline learns the distributions of the real data and generates a logical, rule-bound synthetic world from the ground up.

This approach ensures the generated data preserves statistical fidelity, banking business rules, temporal consistency, and the complex network (graph) structures required for multi-agent fraud detection.

---

## The 10-Phase Pipeline

### Phase 1: Schema Validation
Before any generation begins, the original dataset is inspected to understand data types, missing values, duplicates, and distributions. This produces a `schema_report.json` that acts as the baseline ground truth.

### Phase 2: Data Cleaning
The raw data is normalized and cleaned. This involves:
- Removing invalid accounts, negative amounts, and duplicate transactions.
- Normalizing categorical fields like institution names, branch names, and currencies.
- Validating PEP (Politically Exposed Person) flags, sanction hits, and account open dates.

### Phase 3: Banking Knowledge Extraction & Benchmarking

**Knowledge Extraction (`KnowledgeExtractor` class)**

The system learns the behavior of the real banking environment by analyzing cleaned data and extracting structured knowledge into `data/interim/knowledge_base/`:

1. **Institution Knowledge**: Maps institutions → branches → cities with frequency weights for realistic sampling
2. **Geographic Knowledge**: Country/city frequencies, cross-border transaction rates, country risk scores
3. **Transaction Distributions**: 
   - Amount: Log-normal parameters (mean, std), quantiles (P25, P50, P75, P90, P95, P99)
   - Temporal: Hour-of-day weights, day-of-week weights, month weights
   - Currency: Currency distribution frequencies
   - Payment types: Transaction type distribution frequencies
4. **Customer Behavior Profiles** (per-account):
   - Transaction counts, amount statistics (mean, median, std, min, max)
   - Preferred hours, days of week, payment types
   - Cross-border frequency, daily transaction frequency
   - Average interval between transactions
5. **Graph Knowledge**: Builds transaction network using NetworkX
   - Node degree distributions (in-degree, out-degree)
   - Top fan-out accounts (1 → many) and fan-in accounts (many → 1)
   - PageRank scores for influential nodes
   - Community detection via connected components
   - Graph statistics: avg degrees, max degrees, percentiles

**Generator Benchmarking (`BenchmarkRunner` class)**

During this phase, multiple ML-based generators are benchmarked against the cleaned data:
- **Generators tested**: SMOTE, CTGAN, TVAE, CopulaGAN (SMOTE uses imbalanced-learn, others use SDV library)
- **Evaluation metrics**:
  - ML Utility: 5 classifiers (RF, XGB, LR, MLP, NB) evaluated on Accuracy/F1/AUC
  - Avg JSD: Jensen-Shannon divergence for categorical features
  - Avg WD: Wasserstein distance for continuous features
  - Diff Corr: Frobenius norm of correlation matrix difference
- **Process per generator**:
  1. 80/20 stratified train/test split
  2. Fit generator on training set
  3. Generate synthetic set (same size as training)
  4. Evaluate against test set across 3 random seeds
  5. Report mean ± std for all metrics

*Note: While generators are benchmarked for validation, the final 5M dataset uses the extracted knowledge base for deterministic, graph-aware generation to ensure relational consistency.*

### Phase 4: Synthetic Account Generation

**Account Generator (`AccountGenerator` class) - SMOTE-based**

Generates synthetic bank accounts using SMOTE (Synthetic Minority Over-sampling Technique). SMOTE learns from original account data and creates new synthetic samples through k-nearest neighbor interpolation.

**Why SMOTE?**
- **Best performer**: Outperformed CTGAN, TVAE, and CopulaGAN in benchmarking
- **Fast**: No training required, instant generation
- **Preserves structure**: k-NN interpolation maintains local data patterns
- **Handles imbalance**: Designed for imbalanced datasets (fraud detection)

**Generation Process**:
1. **Load Original Data**: Loads seed accounts from `data/original/accounts.csv`
2. **Fit SMOTE**: Learns k-nearest neighbors structure (k=5) from original data
3. **Generate Samples**: Creates n synthetic samples by interpolating between neighbors
4. **Merge (default)**: Combines synthetic + original accounts
5. **Save**: Auto-saves to `data/generated/synthetic_accounts.csv`

**SMOTE Configuration**:
- `k_neighbors`: 5 (number of nearest neighbors for interpolation)
- `random_state`: From generator seed (for reproducibility)
- Automatic handling of mixed data types (categorical + continuous)

**Output Fields** (inherited from original schema):
- `account_id`, `account_number`, `institution`, `branch`
- `account_type`, `risk_grade`, `is_person`, `name`, `tax_number`
- `pep_flag`, `sanctions_hit`, `city`, `opened`, `kyc_verified`
- `account_age_days`, `is_mule`
- `data_source`: 'original' or 'synthetic' (tracking column)

### Phase 5: Core Transaction Generation

**Transaction Generator (`TransactionGenerator` class) - SMOTE-based**

Creates transaction events using SMOTE to learn from and expand upon original transaction data.

**Why SMOTE for Transactions?**
- Preserves complex transaction patterns (amounts, times, relationships)
- Maintains realistic distributions through neighbor interpolation
- Fast generation (no iterative training)
- Handles high-dimensional transaction features effectively

**Generation Process**:

1. **Load Original Data**: Loads seed transactions from `data/interim/transactions.parquet`

2. **Fit SMOTE**: 
   - Automatically detects categorical features (payment types, currencies, etc.)
   - Uses SMOTENC for mixed data types
   - Learns k=5 nearest neighbors structure

3. **Generate Samples**: Creates n synthetic transactions by:
   - Finding k-nearest neighbors for each seed transaction
   - Interpolating feature values between neighbors
   - Preserving categorical features exactly
   - Maintaining realistic numerical ranges

4. **Merge (default)**: Combines synthetic + original transactions

5. **Save**: Auto-saves to `data/generated/synthetic_transactions.csv`

**Features Preserved**:
- **Account IDs**: Sender/receiver account relationships
- **Amounts**: Transaction amounts with realistic distributions
- **Timestamps**: Date/time patterns (business hours, weekends)
- **Payment Types**: transfer, payment, withdrawal, deposit, remittance, etc.
- **Currencies**: NPR, USD, EUR, GBP, QAR, etc.
- **Geographic**: Cross-border flags, remittance corridors
- **Risk Indicators**: Fraud labels, AML indicators

**SMOTE Advantages Over Knowledge-Based**:
- ✅ Learns complex multivariate dependencies automatically
- ✅ Preserves rare patterns and edge cases
- ✅ No need for explicit distribution modeling
- ✅ Handles high-dimensional feature spaces naturally
- ✅ Maintains local data structure
- ✅ Fast: seconds vs minutes for large datasets

### Phase 6: Fraud Scenario Injection (AML Patterns)

**AML Pattern Injector (`AMLPatternInjector` class)**

Injects 11 specific fraud scenarios to simulate real-world Anti-Money Laundering (AML) threats. Each scenario adds synthetic fraudulent transactions to the dataset.

**Configuration**:
- `npr_threshold`: NPR 1,000,000 (cash reporting threshold)
- `num_injections`: ~100 per scenario (configurable)
- All injected transactions have `is_fraud = 1` and appropriate `fraud_type` label

**The 11 Fraud Scenarios**:

1. **Large Amount Fraud** (`fraud_type: transaction_fraud`):
   - Amount: NPR 2M-50M (well above reporting threshold)
   - Hour: 2-4 AM (unusual hours)
   - New receiver (no prior relationship)

2. **Smurfing / Structuring** (`fraud_type: aml_structuring`):
   - 3-7 transactions per sequence
   - Amount: 90-99% of NPR 1M threshold (e.g., NPR 950K)
   - Same sender → same receiver
   - Transactions hours apart

3. **Velocity Fraud** (`fraud_type: transaction_fraud`):
   - 20-50 rapid transactions from one sender
   - Small amounts (NPR 1K-50K each)
   - Different receivers
   - Transactions seconds apart (5-30 second intervals)

4. **Money Mule Network** (`fraud_type: aml_mule_network`):
   - Two-leg chain: Victim → Mule → Beneficiary
   - Leg 1: Large transfer (NPR 100K-2M)
   - Leg 2: 30min-6hr later, 90-98% of original amount (mule "fee")

5. **Fan-Out** (`fraud_type: aml_mule_network`):
   - One sender → 5-15 receivers
   - Amount: NPR 50K-500K per receiver
   - All transactions within 1-2 hour window

6. **Fan-In** (`fraud_type: aml_mule_network`):
   - 5-15 senders → one receiver
   - Amount: NPR 50K-500K per sender
   - All transactions within 1-2 hour window

7. **Layering Chains** (`fraud_type: aml_layering`):
   - Chain: A → B → C → D (3-6 hops)
   - Amount decreases slightly each hop (95-99% of previous)
   - Transactions 5-60 minutes apart

8. **Circular Transactions** (`fraud_type: aml_layering`):
   - Cycle: A → B → C → A (3-6 nodes)
   - Amount fluctuates slightly (95-105%)
   - Completes cycle within hours

9. **Cross-Border Fraud** (`fraud_type: aml_layering`):
   - High-risk corridors (Qatar, UAE, etc.)
   - Amount: NPR 500K-10M
   - `is_cross_border = 1`, `currency_mismatch = 1`
   - FX rate from corridor ranges

10. **Dormant Account Activation** (`fraud_type: transaction_fraud`):
    - Simulates 180-730 day inactivity gap
    - Then 2-4 large transactions (NPR 500K-10M)
    - Transactions hours apart

11. **PEP / High-Risk Customer** (`fraud_type: identity_fraud`):
    - Involves PEP-flagged or sanctioned accounts
    - Amount: NPR 200K-5M
    - Sender is PEP/sanctioned account from accounts table

**Injection Strategy**:
- Scenarios are applied sequentially to the core transaction dataset
- Each scenario appends new fraudulent transactions
- Maintains temporal consistency (all dates within 2022-10-07 to 2023-10-06)
- Preserves account constraints (uses existing account IDs)

### Phase 7: Transaction Enrichment

**Transaction Enricher (`TransactionEnricher` class)**

Deterministically joins synthetic transactions with account metadata. No random generation occurs here — purely relational merging.

**Process**:
1. **Normalize Column Names**: Map `Sender_account` → `sender_account_id`, `Receiver_account` → `receiver_account_id`

2. **Attach Sender Account Fields** (left join on `sender_account_id`):
   - `sender_institution`, `sender_branch`, `sender_city`
   - `sender_risk_grade`, `sender_account_type`
   - `sender_pep`, `sender_sanctions`, `sender_kyc_verified`
   - `sender_opened` (account opening date)

3. **Attach Receiver Account Fields** (left join on `receiver_account_id`):
   - `receiver_institution`, `receiver_branch`, `receiver_city`
   - `receiver_risk_grade`, `receiver_account_type`
   - `receiver_pep`, `receiver_sanctions`, `receiver_kyc_verified`
   - `receiver_opened` (account opening date)

4. **Derive Geographic Fields**:
   - `Sender_bank_location`: Defaults to sender_city, overridden for cross-border (from corridor)
   - `Receiver_bank_location`: Defaults to receiver_city, always "Nepal" for cross-border
   - `currency_mismatch`: 1 if Payment_currency ≠ Received_currency, else 0

5. **Derive Account Ages**:
   - `sender_account_age_days`: (transaction_date - sender_opened) in days, clipped ≥ 0
   - `receiver_account_age_days`: (transaction_date - receiver_opened) in days, clipped ≥ 0

6. **Derive Amounts**:
   - `amount_local_npr`: Amount × fx_rate_to_npr (if not already present)
   - `fx_rate_to_npr`: Defaults to 1.0 for domestic transactions

7. **Derive Country Risk Scores**:
   - Maps sender/receiver countries to risk scores (from `CORRIDOR_RISK_SCORES`)
   - Nepal: 0.1 (low), High-risk corridors: up to 0.8

**Output**: Fully enriched transactions with 50+ columns ready for feature engineering

### Phase 8: Feature Engineering

**Feature Engineer (`FeatureEngineer` class)**

Computes all derived features deterministically from enriched transactions. Every feature is a pure function of the enriched data.

**Feature Categories**:

1. **Temporal Features**:
   - `hour_of_day`: Extracted from Time field (0-23)
   - `day_of_week`: Extracted from Date (0=Monday, 6=Sunday)
   - `month`: Extracted from Date (1-12)
   - `is_weekend`: 1 if day_of_week ≥ 5, else 0

2. **Amount Features**:
   - `amount_local_npr`: Primary amount in NPR (from enrichment)
   - `log_amount`: log(amount_local_npr)
   - `amount_zscore`: (amount - global_mean) / global_std
   - `above_1M_NPR`: 1 if amount ≥ NPR 1,000,000, else 0
   - `above_10M_NPR`: 1 if amount ≥ NPR 10,000,000, else 0

3. **Geographic Features**:
   - `sender_country_risk`: Risk score [0-1] based on sender country
   - `receiver_country_risk`: Risk score [0-1] based on receiver country
   - `cross_border_flag`: 1 if is_cross_border, else 0
   - `currency_mismatch`: 1 if Payment_currency ≠ Received_currency, else 0

4. **Velocity Features** (chronological per sender):
   - Sort transactions by sender + timestamp
   - `velocity_sum_10tx`: Sum of amounts in last 10 transactions (including current)
   - `tx_count_10`: Count of transactions in last 10 (including current)
   - `tx_count_30`: Count of transactions in last 30 (including current)
   - Computed using rolling windows, efficiently

5. **Account Age Features**:
   - `sender_account_age_days`: Already computed in enrichment
   - `receiver_account_age_days`: Already computed in enrichment

6. **Transmode Encoding** (One-Hot):
   - Maps Payment_type/transaction_type → transmode_code (A/B/E/F/J/P/Z)
   - Creates binary columns: `transmode_A`, `transmode_B`, ..., `transmode_Z`
   - Codes: A=Cash, B=Branch, E=Electronic, F=SWIFT, J=Journal, P=POS, Z=ATM

7. **Label Column**:
   - `is_suspicious_tx`: Alias of `is_fraud` (for ML training)

**Output**: Transaction dataset with 70+ engineered features ready for validation

### Phase 9: Constraint Validation

**Constraint Validator (`ConstraintValidator` class)**

Ensures every generated row passes all logical and business constraints. Rows violating constraints are flagged and optionally dropped.

**Account-Level Constraints**:
1. **Unique account_id**: No duplicate account IDs within accounts table
2. **Valid institution**: Institution must exist in knowledge base
3. **Valid branch**: Branch must belong to declared institution
4. **Valid city**: City must be a known Nepali city (from `NEPALI_CITIES`)
5. **Valid opening date**: Not null, not in future

**Transaction-Level Constraints**:
1. **Valid sender**: `sender_account_id` must exist in accounts table
2. **Valid receiver**: `receiver_account_id` must exist in accounts table
3. **Positive amount**: Amount must be > 0
4. **Valid timestamp**: Date not null, not in future
5. **Temporal consistency**: 
   - Transaction date ≥ sender account opening date
   - Transaction date ≥ receiver account opening date
6. **Valid currency**: Payment_currency must be known ISO code (NPR, USD, EUR, GBP, etc.)

**Feature-Level Constraints**:
1. **day_of_week correctness**: Must match actual day from Date field
2. **is_weekend correctness**: Must equal (day_of_week ≥ 5)
3. **above_1M_NPR correctness**: Must equal (amount ≥ 1,000,000)
4. **above_10M_NPR correctness**: Must equal (amount ≥ 10,000,000)
5. **Transmode encoding**: One-hot columns (transmode_A...Z) must sum to exactly 1
6. **Account age non-negative**: sender/receiver_account_age_days ≥ 0

**Output**:
- Adds `_validation_passed` column (boolean)
- Adds `_violation_codes` column (pipe-separated violation codes, e.g., "INVALID_SENDER|NON_POSITIVE_AMOUNT")
- Generates `validation_report.json` with violation statistics
- If `drop_invalid=True`, removes invalid rows

**Typical Results**:
- Accounts: 99.9%+ pass rate (duplicates rare due to unique ID generation)
- Transactions: 99%+ pass rate (mainly catches edge cases in timestamp logic)

### Phase 10: Final Dataset Assembly

**Dataset Builder (`DatasetBuilder` class)**

Assembles and saves all final output files from validated accounts and transactions.

**Output Files**:

1. **`synthetic_accounts.csv`**:
   - Clean accounts table with internal columns removed
   - Columns: account_id, account_number, institution, branch, account_type, risk_grade, is_person, name, tax_number, pep_flag, sanctions_hit, city, opened, kyc_verified, account_age_days, is_mule

2. **`synthetic_transactions.csv`**:
   - Full enriched and feature-engineered transactions
   - Internal columns removed (_validation_passed, _violation_codes, _datetime)
   - 70+ columns including all features from Phase 8

3. **`graph_edges.csv`**:
   - Lightweight edge list for graph-based agents
   - Columns: row_index, Sender_account, Receiver_account, amount_local_npr, Date, Time
   - Optimized for graph construction (NetworkX, PyTorch Geometric)

4. **`ml_features.csv`**:
   - ML-ready feature set with label
   - Selected columns (40+): Date, Time, accounts, amounts, temporal features, geographic features, velocity features, account ages, encoded transmodes, `is_suspicious_tx` (label)
   - Matches original ml_features.csv schema for compatibility

5. **`schema_report.json`** (if provided):
   - Phase 1 schema statistics
   - Data types, missing values, duplicates, distributions

6. **`validation_report.json`** (if provided):
   - Phase 9 constraint violation statistics
   - Breakdown by violation type, pass/fail rates

7. **`generation_log.json`**:
   - Complete pipeline metadata:
     - Start/end timestamps
     - Elapsed time
     - Row counts (accounts, transactions)
     - Fraud statistics (total fraud %, breakdown by fraud_type)
     - Phase timings

**Fraud Statistics Computed**:
- Total fraud rate (% of transactions with is_fraud=1)
- Breakdown by fraud_type:
  - transaction_fraud
  - aml_structuring
  - aml_mule_network
  - aml_layering
  - identity_fraud

**Final Output**:
All files saved to `data/generated/` (or configured output directory). These files constitute the complete 5M-row synthetic banking dataset ready for multi-agent fraud detection training.


---

## Complete Workflow: From Original Data to 5M Synthetic Dataset

### Step-by-Step Execution

The pipeline is orchestrated through Jupyter notebooks located in `notebooks/data-pipeline/`:

**1. Initial Setup**
```python
# Configuration
from src.utils.config import get_config
cfg = get_config()

# Random seed for reproducibility
RANDOM_SEED = 42
```

**2. Phase 1-2: Data Cleaning** (`phase2_cleaning.ipynb`)
```python
from src.cleaning import run_full_cleaning_pipeline

# Load original data
transactions_raw = pd.read_csv(cfg.data_original_dir / "ml_features.csv")
accounts_raw = pd.read_csv(cfg.data_original_dir / "accounts.csv")

# Run cleaning pipeline
transactions_clean, accounts_clean = run_full_cleaning_pipeline(
    transactions_raw, accounts_raw, 
    output_dir=cfg.data_interim_dir
)

# Outputs:
# - data/interim/transactions.parquet (canonical schema)
# - data/interim/accounts.parquet
# - data/interim/schema2_location.parquet (for benchmarking)
# - data/interim/schema3_label_encoded.parquet
# - data/interim/schema4_quantile_amount.parquet
# - data/interim/schema5_cbrt_amount.parquet
```

**3. Phase 3a: Generator Benchmarking** (`phase3_benchmark_generators.ipynb`)
```python
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.generation.synthesizers import (
    CTGANGenerator, TVAEGenerator, 
    CopulaGANGenerator, SMOTEGenerator
)

# Load cleaned data
real_data = pd.read_parquet(cfg.data_interim_dir / "transactions.parquet")

# Initialize generators
generators = {
    "SMOTE": SMOTEGenerator(k_neighbors=5, random_state=42),
    "CTGAN": CTGANGenerator(epochs=30),
    "TVAE": TVAEGenerator(epochs=30),
    "CopulaGAN": CopulaGANGenerator(epochs=30)
}

# Run benchmark across 3 seeds
runner = BenchmarkRunner(
    real_data=real_data,
    target_col="is_suspicious_tx",
    seeds=[42, 7, 123],
    test_size=0.2,
    output_dir=cfg.data_generated_dir / "benchmark"
)

results = runner.run(generators)

# Outputs:
# - data/generated/benchmark/results.csv (aggregated metrics)
# - data/generated/benchmark/{generator}/synthetic_seed{seed}.csv
# - data/generated/benchmark/benchmark_heatmap.png
# - data/generated/benchmark/benchmark_radar.png
```

**4. Phase 3b: Knowledge Extraction** (`phase4_knowledge_extraction.ipynb`)
```python
from src.generation.core.knowledge_extractor import KnowledgeExtractor

# Initialize extractor
extractor = KnowledgeExtractor(project_root=cfg.project_root)

# Extract knowledge from cleaned data
extractor.extract(
    transactions=transactions_clean,
    accounts=accounts_clean
)

# Save knowledge base
extractor.save()

# Outputs:
# - data/interim/knowledge_base/behavior_profiles.parquet
# - data/interim/knowledge_base/institution_mapping.json
# - data/interim/knowledge_base/branch_mapping.json
# - data/interim/knowledge_base/country_mapping.json
# - data/interim/knowledge_base/graph_statistics.json
# - data/interim/knowledge_base/amount_distribution.json
# - data/interim/knowledge_base/temporal_distribution.json
# - data/interim/knowledge_base/currency_distribution.json
# - data/interim/knowledge_base/payment_type_distribution.json
```

**5. Phase 4-10: Full 5M Generation with Optional SMOTE Enhancement** (`phase5_generate_5m.ipynb`)
```python
from src.generation.core.account_generator import AccountGenerator
from src.generation.core.transaction_generator import TransactionGenerator
from src.generation.core.aml_pattern_injector import AMLPatternInjector
from src.generation.core.enricher import TransactionEnricher
from src.generation.core.feature_engineer import FeatureEngineer
from src.generation.core.validator import ConstraintValidator
from src.generation.core.dataset_builder import DatasetBuilder
from src.generation.core.knowledge_extractor import load_knowledge_base

# Load knowledge base
knowledge = load_knowledge_base(cfg.project_root)

# Phase 4: Generate accounts using SMOTE (learns from original data)
account_gen = AccountGenerator(knowledge, seed=RANDOM_SEED)

# Generate 100K synthetic accounts using SMOTE
# This will:
# 1. Load original accounts from data/original/accounts.csv
# 2. Fit SMOTE generator on original data
# 3. Generate 100K new synthetic samples via k-NN interpolation
# 4. Merge with original data (default behavior)
# 5. Save to data/generated/synthetic_accounts.csv
synthetic_accounts = account_gen.generate(
    n=100_000,
    merge_with_original=True  # Default: merges with original, saves to data/generated/
)

print(f"Generated {len(synthetic_accounts)} accounts")
print(f"  - Original: {(synthetic_accounts['data_source'] == 'original').sum()}")
print(f"  - Synthetic: {(synthetic_accounts['data_source'] == 'synthetic').sum()}")

# Phase 5: Generate transactions using SMOTE
tx_gen = TransactionGenerator(knowledge, synthetic_accounts, seed=RANDOM_SEED)

# Generate 4.8M synthetic transactions using SMOTE
# This will:
# 1. Load original transactions from data/interim/transactions.parquet
# 2. Fit SMOTE generator on original data
# 3. Generate 4.8M new synthetic samples
# 4. Merge with original data
# 5. Save to data/generated/synthetic_transactions.csv
core_transactions = tx_gen.generate(
    n=4_800_000,
    merge_with_original=True
)

print(f"Generated {len(core_transactions)} transactions")

# Phase 6: Inject fraud scenarios (continues as before)
injector = AMLPatternInjector(config={"num_injections": 200, "seed": RANDOM_SEED})
fraud_transactions = injector.inject_all_patterns(
    transactions=core_transactions,
    accounts=synthetic_accounts
)
print(f"After fraud injection: {len(fraud_transactions)} transactions")
print(f"Fraud rate: {fraud_transactions['is_fraud'].mean():.2%}")

# Phase 7: Enrich transactions
enricher = TransactionEnricher(synthetic_accounts)
enriched_transactions = enricher.enrich(fraud_transactions)
print(f"Enriched: {enriched_transactions.shape}")

# Phase 8: Engineer features
engineer = FeatureEngineer()
featured_transactions = engineer.engineer(enriched_transactions)
print(f"Feature-engineered: {featured_transactions.shape}")

# Phase 9: Validate constraints
validator = ConstraintValidator(
    synthetic_accounts, 
    knowledge=knowledge, 
    drop_invalid=True
)
validated_accounts, acc_report = validator.validate_accounts()
validated_transactions, tx_report = validator.validate_transactions(featured_transactions)

print(f"Validation: {tx_report['rows_passed']} / {tx_report['total_rows']} passed")

# Phase 10: Build final dataset
builder = DatasetBuilder(output_dir=cfg.data_generated_dir)
output_paths = builder.build(
    accounts=validated_accounts,
    transactions=validated_transactions,
    validation_report={"accounts": acc_report, "transactions": tx_report}
)

print("Final outputs:")
for name, path in output_paths.items():
    print(f"  {name}: {path}")

# Outputs:
# - data/generated/synthetic_accounts.csv (~100K rows)
# - data/generated/synthetic_transactions.csv (~5M rows)
# - data/generated/graph_edges.csv (~5M edges)
# - data/generated/ml_features.csv (~5M rows, selected columns)
# - data/generated/validation_report.json
# - data/generated/generation_log.json
```

---

## Generator Selection Strategy

### Why SMOTE Was Selected

The benchmarking in Phase 3 evaluated SMOTE, CTGAN, TVAE, and CopulaGAN generators. **SMOTE emerged as the clear winner** and is now the production generator for Phases 4-5.

**SMOTE Advantages**:
- ✅ **Best Performance**: Highest ML utility scores (accuracy, F1, AUC)
- ✅ **Best Statistical Fidelity**: Lowest JSD and Wasserstein distances
- ✅ **Fastest**: No training required, instant generation (seconds vs hours)
- ✅ **Memory Efficient**: Minimal memory footprint
- ✅ **Handles Imbalance**: Designed specifically for imbalanced datasets
- ✅ **Interpretable**: k-NN interpolation is transparent and explainable
- ✅ **Production Ready**: Proven, stable, no training convergence issues

**Why Other Generators Were Not Selected**:

1. **CTGAN/TVAE (GAN-based)**:
   - ❌ Training instability (mode collapse, convergence issues)
   - ❌ Hours of training time (30+ epochs)
   - ❌ High memory requirements
   - ❌ Lower ML utility scores than SMOTE
   - ❌ No special handling for imbalanced data

2. **CopulaGAN**:
   - ❌ Slower than SMOTE
   - ❌ Medium performance on benchmarks
   - ❌ Complex copula modeling overhead
   - ❌ No clear advantage over SMOTE

3. **Relational Considerations**:
   - SMOTE preserves relationships present in original data through interpolation
   - Fraud patterns are injected in Phase 6 (AML Pattern Injector)
   - Validation in Phase 9 ensures consistency
   - Feature engineering in Phase 8 computes derived fields deterministically

**How SMOTE Handles Complex Banking Data**:
- **Accounts/Transactions**: Learns from original data structure
- **Graph Relationships**: Preserved through k-NN interpolation of account IDs
- **Temporal Patterns**: Naturally preserved in interpolated timestamps
- **Fraud Injection**: Done explicitly in Phase 6 (not during generation)
- **Validation**: Phase 9 catches any edge cases

**Production Pipeline**:
```
Phase 4: SMOTE generates accounts from original data
Phase 5: SMOTE generates transactions from original data  
Phase 6: AML patterns explicitly injected
Phase 7: Enrichment (deterministic joins)
Phase 8: Feature engineering (deterministic)
Phase 9: Validation (constraint checking)
Phase 10: Final assembly
```

### Why SMOTE is the Primary Generator

**Our Approach**: Use SMOTE as the **primary production generator** based on benchmark results:

1. **Best Performance** (Phase 3 benchmarking):
   - Superior statistical similarity (JSD, Wasserstein)
   - Best ML utility scores
   - Fastest generation time

2. **Production Benefits**:
   - No training overhead (instant generation)
   - Learns patterns automatically from original data
   - Handles mixed data types natively
   - Preserves local structure through k-NN interpolation

3. **Automatic Merging**:
   - Combines synthetic + original data by default
   - Tracks provenance with `data_source` column
   - Auto-saves to `data/generated/` folder

### Generator Comparison Summary

| Aspect | CTGAN/TVAE | CopulaGAN | SMOTE (Our Choice) |
|--------|------------|-----------|-------------------|
| Statistical fidelity | ⚠️ Good but slow | ⚠️ Medium | ✅ Best |
| Training time | ❌ Hours | ⚠️ Minutes | ✅ Seconds |
| Generation time | ⚠️ Minutes | ⚠️ Minutes | ✅ Seconds |
| Memory usage | ❌ High | ⚠️ Medium | ✅ Low |
| Handles imbalance | ❌ No | ❌ No | ✅ Yes |
| Mixed data types | ⚠️ Via encoding | ⚠️ Via encoding | ✅ Native |
| Interpretability | ❌ Black box | ❌ Black box | ✅ k-NN interpolation |
| ML utility | ⚠️ Medium | ⚠️ Medium | ✅ Best |

**Conclusion**: SMOTE outperformed all other generators in benchmarking and is now the primary generator for Phases 4-5. It combines speed, simplicity, and superior performance for fraud detection datasets.

---

## Configuration and Customization

### Key Configuration Parameters

All generation parameters are centralized in `src/utils/config.py`:

```python
class Config(BaseModel):
    # Paths
    data_original_dir: Path = "data/original"
    data_interim_dir: Path = "data/interim"
    data_generated_dir: Path = "data/generated"
    
    # Generation settings
    generator_chunk_size: int = 500_000  # Batch size for memory efficiency
    random_seed: int = 42
    
    # NRB regulatory thresholds
    nrb_cash_reporting_threshold_npr: float = 1_000_000.0
    structuring_min_npr: float = 900_000.0
    structuring_max_npr: float = 999_000.0
    
    # Fraud injection rates (configurable per scenario)
    # Set in AMLPatternInjector: num_injections per scenario
```

### Customizing the Pipeline

**1. Adjust Number of Accounts/Transactions**:
```python
# Generate more/fewer accounts
synthetic_accounts = account_gen.generate(n=200_000)  # Double accounts

# Generate more/fewer transactions
core_transactions = tx_gen.generate(n=10_000_000)  # 10M transactions
```

**2. Adjust Fraud Rates**:
```python
# More aggressive fraud injection
injector = AMLPatternInjector(config={
    "num_injections": 500,  # 5x more fraud per scenario
    "seed": RANDOM_SEED
})
```

**3. Custom Fraud Scenarios**:
```python
# Add custom scenario to AMLPatternInjector class
def inject_custom_pattern(self, df, accounts):
    # Your custom fraud logic here
    return df
```

**4. Different Random Seeds**:
```python
# Generate multiple independent datasets
for seed in [42, 123, 456]:
    account_gen = AccountGenerator(knowledge, seed=seed)
    # ... continue pipeline with new seed
```

**5. Regional Customization**:
```python
# Modify src/utils/nepal_context.py for different country
COUNTRY_NAME = "India"
CURRENCY = "INR"
CASH_REPORTING_THRESHOLD = 1_000_000  # INR
MAJOR_CITIES = ["Mumbai", "Delhi", "Bangalore", ...]
REMITTANCE_CORRIDORS = {...}  # Update corridors
```

---

## Quality Assurance

### Statistical Fidelity Checks

After generation, verify statistical similarity to original data:

```python
from src.evaluation.statistical_metrics import (
    compute_avg_jsd,
    compute_avg_wasserstein,
    compute_correlation_distance
)

# Compare distributions
continuous_cols = ["amount_local_npr", "hour_of_day", "day_of_week"]
categorical_cols = ["Payment_type", "transaction_type", "Payment_currency"]

jsd = compute_avg_jsd(
    real_data[categorical_cols],
    synthetic_data[categorical_cols],
    categorical_cols
)
print(f"Avg JSD (categorical): {jsd:.4f}")  # Target: < 0.05

wd = compute_avg_wasserstein(
    real_data[continuous_cols],
    synthetic_data[continuous_cols],
    continuous_cols
)
print(f"Avg Wasserstein (continuous): {wd:.4f}")  # Target: < 0.10

corr_dist = compute_correlation_distance(
    real_data[continuous_cols],
    synthetic_data[continuous_cols]
)
print(f"Correlation distance: {corr_dist:.4f}")  # Target: < 0.10
```

### Validation Report Review

```python
import json

# Check validation report
with open(cfg.data_generated_dir / "validation_report.json") as f:
    report = json.load(f)

print("Accounts validation:")
print(f"  Pass rate: {report['accounts']['total_rows'] - report['accounts']['total_violations']} / {report['accounts']['total_rows']}")

print("\nTransactions validation:")
print(f"  Pass rate: {report['transactions']['rows_passed']} / {report['transactions']['total_rows']}")
print(f"  Violations: {report['transactions']['total_violations']}")
```

### Graph Structure Validation

```python
import networkx as nx

# Build transaction graph
edges = pd.read_csv(cfg.data_generated_dir / "graph_edges.csv")
G = nx.from_pandas_edgelist(
    edges, 
    source="Sender_account", 
    target="Receiver_account",
    create_using=nx.DiGraph()
)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Avg out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")
print(f"Max out-degree: {max(dict(G.out_degree()).values())}")

# Compare to knowledge base expectations
with open(cfg.data_interim_dir / "knowledge_base" / "graph_statistics.json") as f:
    expected = json.load(f)
print(f"\nExpected avg out-degree: {expected['avg_out_degree']:.2f}")
print(f"Expected max fan-out: {expected['max_out_degree']}")
```

---

## Performance Considerations

### Memory Optimization

**Batch Generation**: For very large datasets (10M+ rows), generate in chunks:

```python
# Generate in 500K batches
batch_size = 500_000
total_transactions = 5_000_000
num_batches = total_transactions // batch_size

all_transactions = []
for i in range(num_batches):
    print(f"Generating batch {i+1}/{num_batches}...")
    batch = tx_gen.generate(n=batch_size)
    all_transactions.append(batch)

core_transactions = pd.concat(all_transactions, ignore_index=True)
```

**Parquet vs CSV**: Use Parquet for intermediate files (faster I/O, compression):

```python
# Save intermediate results as Parquet
enriched_transactions.to_parquet(
    cfg.data_interim_dir / "transactions_enriched.parquet",
    engine="pyarrow",
    compression="snappy"
)
```

### Parallelization

Generate multiple independent datasets in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

def generate_dataset(seed):
    """Generate one complete dataset with given seed."""
    # Run full pipeline with seed
    # Return output path
    pass

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(generate_dataset, seed) for seed in [42, 123, 456, 789]]
    results = [f.result() for f in futures]
```

### Timing Estimates

Typical generation times on modern hardware (16 CPU cores, 32GB RAM):

| Phase | 100K accounts, 5M transactions | Time |
|-------|-------------------------------|------|
| Phase 1-2: Cleaning | N/A (one-time) | ~5 min |
| Phase 3: Knowledge extraction + Benchmarking | N/A (one-time) | ~10 min |
| Phase 4: Account generation (SMOTE) | 100K rows | **~5 sec** |
| Phase 5: Transaction generation (SMOTE) | 5M rows | **~30 sec** |
| Phase 6: Fraud injection | +200K rows | ~5 min |
| Phase 7: Enrichment | 5.2M rows | ~8 min |
| Phase 8: Feature engineering | 5.2M rows | ~12 min |
| Phase 9: Validation | 5.2M rows | ~5 min |
| Phase 10: Assembly & save | 5.2M rows | ~3 min |
| **Total** | | **~50 min** |

**Note**: SMOTE generation is extremely fast (seconds) compared to previous approaches (minutes) or GAN-based methods (hours).

---

## Troubleshooting

### Common Issues

**1. "Account not found" errors in validation**:
- Cause: Fraud injection created transactions with non-existent accounts
- Fix: Ensure `AMLPatternInjector._get_random_accounts()` only samples from existing accounts

**2. Memory errors during generation**:
- Cause: Generating too many rows at once
- Fix: Use batch generation (see Performance section)

**3. Temporal constraint violations**:
- Cause: Transaction dates before account opening
- Fix: Check `TransactionGenerator._sample_timestamp()` logic

**4. Low fraud rate after injection**:
- Cause: `num_injections` too low relative to core transaction count
- Fix: Increase `num_injections` or reduce core transaction count

**5. Graph structure doesn't match knowledge base**:
- Cause: Receiver selection strategy weights incorrect
- Fix: Verify `_RECEIVER_STRATEGY` probabilities in `TransactionGenerator`

---

## Summary

The Neural Sentinel data generation pipeline is a sophisticated, 10-phase process that creates highly realistic synthetic banking data with embedded fraud scenarios. Key innovations include:

1. **SMOTE-based generation** (Phases 4-5) provides the best performance based on benchmarking results
2. **Automatic learning** from original data preserves statistical distributions and patterns
3. **Fast generation** (seconds instead of minutes/hours) enables rapid iteration
4. **Explicit fraud scenario injection** (Phase 6) provides precise control over AML pattern representation
5. **Comprehensive validation** (Phase 9) ensures every row passes business logic constraints
6. **Benchmarking framework** (Phase 3) validates SMOTE as the superior choice among all generators

This pipeline generates the foundation for training the multi-agent fraud detection system, ensuring agents learn from realistic, complex, and structurally sound synthetic data.

**Why SMOTE?**
- ✅ Best benchmark performance (statistical similarity + ML utility)
- ✅ Fastest generation (no training overhead)
- ✅ Handles imbalanced datasets (critical for fraud detection)
- ✅ Automatic merging with original data
- ✅ Production-ready and stable
