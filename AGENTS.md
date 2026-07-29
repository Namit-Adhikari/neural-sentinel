# AGENTS.md — Neural Sentinel: Multi-Agent Financial Fraud Detection System

> **Purpose**: This file provides full project context, constraints, architectural decisions, and execution instructions for any AI agent (Cursor, Copilot, Codex, ChatGPT, etc.) contributing code or artifacts to this repository. It contains **no code**. It tells you **what to build, why, and how** — not what to type.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | Neural Sentinel |
| **Domain** | Financial fraud detection — Nepali banking channel |
| **Nature** | Prototype / Proof-of-Concept (not production) |
| **Target Audience** | Nepali commercial banks (pitch deck + working demo) |
| **Team Size** | 2 undergraduate B.Tech AI students |
| **Public Repository** | Yes — GitHub, open-source, for community reuse |
| **Runtime Platform** | Kaggle Notebooks (GPU/TPU when beneficial, CPU fallback to conserve compute budget) |
| **Primary Language** | English (code, docs, comments); Nepali context in data/domain only |

---

## 2. Problem Statement

Nepali banks face rising financial fraud including transaction fraud, money laundering (AML), and cross-border remittance exploitation. Existing rule-based systems produce excessive false positives and miss layered, network-based fraud schemes. This project builds a **multi-agent detection system** where specialized agents score transactions on different risk dimensions, and a meta-learner combines them into a single actionable alert with human-readable explanations.

The target variable (`is_fraud`) is treated as **confirmed criminal fraud** — not merely suspicious. AML-patterned transactions (structuring, layering, mule-account networks) are considered a subset of fraud because GNN-based graph analysis substantially benefits from including them.

---

## 3. Reference Paper

A research paper from a Nepali team (TechRxiv preprint, focused on Global IME Bank data) has been provided as **guidance and reference only**. Key takeaways from that paper that inform this project:

- The paper employed synthetic data generation to overcome real-data scarcity in Nepali banking.
- They benchmarked multiple generators (CTGAN, TVAE, CopulaGAN, etc.) and used statistical fidelity metrics (KS test, column distribution similarity, correlation matrix distance) alongside ML-efficacy metrics (train/test AUC-ROC on a downstream classifier).
- Their canonical schema included: transaction identifiers, sender/receiver account numbers, timestamps, amounts, transaction types, channel, currency, location fields, device/IP metadata, and a binary fraud label.
- They recommended CTGAN or TVAE for tabular financial data with mixed categorical and continuous features.
- Evaluation emphasized: (a) statistical similarity between real and synthetic distributions, (b) downstream ML utility (classifier trained on synthetic, tested on real), and (c) privacy via distance-to-closest-record (DCR).

**What we adopt from the paper**: The evaluation methodology (statistical + ML-utility + privacy), the generator benchmarking approach, and the canonical schema philosophy.

**What we do differently**: We scale to 5M rows, we add AML-specific patterns (mule accounts, fan-in/fan-out, layering), we build a multi-agent system (the paper used a single model), and we target Nepali banking context specifically (NPR currency, NRB regulations, remittance corridors).

---

## 4. Source Data Analysis (Current State)

### 4.1 Dataset Overview

The uploaded dataset (`synthetic_financial_data.csv`, 1,048,575 rows) is an existing synthetic financial transaction dataset with the following characteristics discovered during analysis:

| Attribute | Detail |
|---|---|
| **Row Count** | 1,048,575 |
| **Format** | Single CSV file |
| **Fraud Rate** | ~10.0% (high — designed for ML training, not realistic banking prevalence) |
| **Fraud Count** | ~104,857 fraudulent rows |

### 4.2 Schema (Observed Columns)

| Column | Type | Description | Notes |
|---|---|---|---|
| `transaction_id` | string (UUID) | Unique transaction identifier | Format: UUID v4 |
| `sender_account` | string (16-digit) | Sender bank account number | Padded numeric string |
| `receiver_account` | string (16-digit) | Receiver bank account number | Padded numeric string |
| `transaction_type` | categorical | Type of transaction | Values: `transfer`, `payment`, `withdrawal`, `deposit`, `cash_out` |
| `amount` | float | Transaction amount in NPR | Range: 0.01 to ~10M NPR; right-skewed |
| `timestamp` | datetime (ISO 8601) | Transaction timestamp | Format: `YYYY-MM-DD HH:MM:SS`; single-day span |
| `channel` | categorical | Banking channel used | Values: `mobile_banking`, `atm`, `branch`, `online_banking`, `pos` |
| `currency` | string | Transaction currency | Values: `NPR`, `USD`, `EUR`, `INR`, `GBP`, `AUD`, `JPY`, `CNY`, `SGD` |
| `country` | string | Country of transaction origin/destination | 10+ countries incl. Nepal, India, UAE, Qatar, Saudi Arabia, Malaysia, USA, UK, Australia, Japan, China, Singapore, South Korea, Thailand, Bahrain, Kuwait, Oman |
| `merchant_category` | string | Merchant/business category | Values: `retail`, `grocery`, `restaurant`, `utility`, `travel`, `electronics`, `healthcare`, `education`, `entertainment`, `other` |
| `device_type` | categorical | Device used for transaction | Values: `mobile`, `desktop`, `tablet`, `atm`, `pos_terminal` |
| `ip_address` | string (IPv4) | Client IP address | Standard IPv4 format |
| `is_fraud` | binary (int) | Target variable: 0 = legitimate, 1 = fraud | Treated as **confirmed criminal fraud** |

### 4.3 Key Observations from EDA

1. **Temporal limitation**: All transactions fall within a single 24-hour window. The generator must expand this to a realistic multi-month or multi-year temporal range to support temporal-feature engineering (rolling windows, velocity features, day-of-week patterns).

2. **Amount distribution**: Right-skewed with a long tail. Fraud transactions tend to have higher mean amounts but significant overlap with legitimate transactions. The generator must preserve this distributional shape while extending the range realistically for a Nepali banking context (daily limits, NRB thresholds).

3. **Currency diversity**: Multiple currencies present, indicating cross-border/remittance transactions. NPR is dominant. This is valuable for the Geo-risk agent. The generator must maintain realistic NPR-dominant ratios with corridor-specific currency patterns (e.g., INR for Indian corridor, USD/QAR/SAR for Gulf corridors).

4. **Channel distribution**: `mobile_banking` is most frequent, reflecting Nepal's digital banking growth. `atm` and `branch` have lower counts. The generator should calibrate these proportions to approximate real Nepali banking channel mix.

5. **Country distribution**: Nepal is dominant. India, UAE, Qatar, Saudi Arabia, Malaysia appear frequently — consistent with Nepali remittance corridors (Gulf countries + India + Malaysia). The generator must preserve and enhance these corridor patterns.

6. **No account-level features**: The dataset lacks account-level metadata (account age, KYC status, risk grade, PEP flag, account balance, average monthly volume). These are critical for the KYC/AML rules agent and must be **synthetically derived** during generation by creating an auxiliary `accounts` table.

7. **No sequential linking**: Transactions are independent rows. There is no explicit linking of transactions belonging to the same account over time. The generator must ensure that each account's transaction sequence is coherent (realistic inter-arrival times, consistent channel preferences, plausible balance evolution).

8. **No AML-specific patterns**: The dataset does not contain explicit money-laundering typologies (structuring/smurfing below reporting thresholds, rapid fund movement through chains of mule accounts, round-tripping). These must be **injected** during generation to support the Graph/synthesis agent and AML rules agent.

9. **No network/graph structure**: While sender/receiver account pairs exist, the dataset lacks multi-hop transaction chains, fan-in/fan-out patterns, and cyclical flows characteristic of mule-account networks. The generator must deliberately construct such subgraphs.

10. **IP address as proxy**: IP addresses are present but not geolocated. For the Geo-risk agent, IP-to-geolocation mapping (country, city, ISP) should be derived or assigned consistently during generation.

### 4.4 NRB (Nepal Rastra Bank) Regulatory Context

The generator and all agents must be aware of these Nepal-specific regulatory thresholds (agents should reference these in rules and the generator should produce data that tests against them):

| Regulation | Threshold | Relevance |
|---|---|---|
| Cash transaction reporting | Transactions >= NPR 1,000,000 (10 lakh) | Structuring detection |
| International remittance reporting | All cross-border transfers | AML monitoring |
| PEP (Politically Exposed Person) screening | All accounts | KYC/AML agent |
| Sanctions screening | OFAC, UN, EU sanctions lists | KYC/AML agent |
| Account opening KYC | Mandatory for all accounts | KYC/AML agent |
| Suspicious Transaction Reporting (STR) | Any qualifying transaction | All agents contribute |

---

## 5. Target Schema (Canonical Data Contract)

The canonical schema is the **single source of truth** for all data flowing between agents. Every agent reads from this schema and writes scores/alerts back referencing transaction IDs from this schema.

### 5.1 `transactions` Table (5M rows target)

| Column | Type | Description | Source |
|---|---|---|---|
| `transaction_id` | string (UUID) | Unique identifier | From source / generated |
| `transaction_date` | date | Date of transaction (expanded from single-day) | Generated |
| `transaction_time` | time | Time of transaction | Generated |
| `sender_account_id` | string | FK to `accounts` table | From source / generated |
| `receiver_account_id` | string | FK to `accounts` table | From source / generated |
| `transaction_type` | categorical | transfer / payment / withdrawal / deposit / cash_out / remittance_inbound / remittance_outbound | Source + AML types added |
| `amount_npr` | float | Amount in NPR (converted if foreign currency) | From source / generated |
| `original_currency` | string | Original transaction currency | From source |
| `exchange_rate` | float | NPR exchange rate at transaction time | Generated (realistic rates) |
| `channel` | categorical | mobile_banking / atm / branch / online_banking / pos | From source |
| `sender_country` | string | Sender country (null for domestic) | From source / derived |
| `receiver_country` | string | Receiver country (null for domestic) | From source / derived |
| `is_cross_border` | binary | 1 if sender and receiver countries differ | Derived |
| `remittance_corridor` | string | e.g., "Qatar->Nepal", "India->Nepal" | Derived for cross-border |
| `merchant_category` | string | Merchant category | From source |
| `device_type` | categorical | Device used | From source |
| `ip_address` | string (IPv4) | Client IP | From source / generated |
| `ip_country` | string | Geo-located country from IP | Derived/generated |
| `ip_is_vpn` | binary | Whether IP is a known VPN/proxy | Generated for fraud scenarios |
| `is_fraud` | binary | Target: confirmed criminal fraud | From source / injected AML patterns |
| `fraud_type` | categorical | fraud_type for fraud rows | Generated: "transaction_fraud" / "aml_structuring" / "aml_layering" / "aml_mule_network" / "identity_fraud" |
| `aml_risk_indicator` | binary | 1 if transaction is part of AML pattern | Generated |

### 5.2 `accounts` Table (Derived/Generated)

| Column | Type | Description |
|---|---|---|
| `account_id` | string | Primary key |
| `account_type` | categorical | savings / current / salary / fixed_deposit |
| `account_open_date` | date | Account opening date |
| `account_age_days` | int | Days since opening (at reference date) |
| `kyc_verified` | binary | KYC completion status |
| `kyc_risk_grade` | categorical | low / medium / high |
| `is_pep` | binary | Politically Exposed Person flag |
| `is_sanctioned` | binary | Sanctions list match |
| `average_monthly_volume` | float | Average monthly transaction volume |
| `average_monthly_count` | int | Average monthly transaction count |
| `country` | string | Account holder's country |
| `city` | string | Account holder's city (Nepal: major cities) |
| `is_mule` | binary | Whether this is a generated mule account |

### 5.3 `alert_scores` Table (Agent Output)

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | FK to transactions |
| `agent_name` | string | Which agent produced this score |
| `risk_score` | float (0-1) | Calibrated probability/risk score |
| `alert_flag` | binary | 1 if score exceeds agent's threshold |
| `reason_code` | string | Machine-readable reason code |
| `explanation` | string | Human-readable explanation |
| `timestamp` | datetime | When the score was computed |

---

## 6. Repository Structure (Mandatory Layout)

All code must follow this directory structure. Deviating from this structure without explicit approval from the team causes merge conflicts and breaks the build pipeline.

```
neural-sentinel/
├── AGENTS.md                  # THIS FILE — context for AI agents
├── README.md                  # Project overview, setup, usage (public-facing)
├── .gitignore                 # Standard Python + Kaggle + data ignores
├── .python-version            # Pins Python 3.12.13 for uv (committed to repo)
├── requirements.txt           # Pinned dependencies (pip + uv)
├── environment.yml            # Conda environment spec (secondary — conda users)
├── pyproject.toml             # Modern Python project config (optional)
│
├── data/
│   ├── original/              # Original uploaded data (git-ignored, .gitkeep only)
│   │   └── synthetic_financial_data.csv
│   ├── interim/               # Cleaned, canonical-schema data (git-ignored)
│   │   ├── transactions.parquet
│   │   └── accounts.parquet
│   └── generated/             # 5M synthetic dataset output (git-ignored)
│       ├── transactions_5m.parquet
│       └── accounts_5m.parquet
│
├── docs/
│   ├── eda/                   # EDA notebooks and outputs
│   │   ├── 01_data_overview.ipynb
│   │   ├── 02_univariate_analysis.ipynb
│   │   ├── 03_bivariate_analysis.ipynb
│   │   ├── 04_temporal_analysis.ipynb
│   │   ├── 05_geographic_analysis.ipynb
│   │   └── 06_fraud_pattern_analysis.ipynb
│   ├── generator_benchmark/   # Generator comparison notebooks
│   │   └── benchmark_results.ipynb
│   └── architecture/          # System architecture diagrams, design docs
│       └── system_architecture.md
│
├── src/
│   ├── __init__.py
│   ├── data_contracts.py      # Pydantic models for canonical schema
│   ├── data_quality.py        # Data-quality agent (Dev 1 ownership)
│   ├── cleaning.py            # Data cleaning and canonical transformation
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── base_generator.py      # Abstract base class for generators
│   │   ├── ctgan_generator.py     # CTGAN implementation wrapper
│   │   ├── tvae_generator.py      # TVAE implementation wrapper
│   │   ├── copulagan_generator.py # CopulaGAN implementation wrapper
│   │   ├── gaussian_copula_generator.py  # Gaussian Copula wrapper
│   │   └── aml_pattern_injector.py     # Injects AML patterns into generated data
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Abstract base class for all agents
│   │   ├── data_quality_agent.py  # Schema, missing values, consistency
│   │   ├── velocity_agent.py      # Velocity, rolling windows, Isolation Forest
│   │   ├── behaviour_agent.py     # GRU/LSTM sequence deviation detection
│   │   ├── geo_risk_agent.py      # CatBoost/LightGBM geographic risk
│   │   ├── graph_agent.py         # GraphSAGE/GAT network analysis
│   │   ├── kyc_aml_agent.py       # Rules-based KYC/AML scoring
│   │   ├── meta_learner.py        # Random Forest / XGBoost ensemble
│   │   └── explanation_agent.py   # SHAP-based explanations
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── statistical_metrics.py   # KS test, correlation distance, etc.
│   │   ├── ml_utility.py           # Downstream classifier AUC-ROC
│   │   ├── privacy_metrics.py      # DCR, nearest-neighbor distance
│   │   └── agent_evaluation.py     # Per-agent and system-level metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Centralized configuration (paths, thresholds, hyperparameters)
│       ├── logger.py               # Structured logging setup
│       ├── device_utils.py         # GPU/TPU detection and fallback logic
│       └── nepal_context.py        # NRB thresholds, remittance corridors, city lists, etc.
│
├── notebooks/
│   ├── dev1/                   # Developer 1 notebooks (Kaggle-ready)
│   │   ├── phase1_eda.ipynb
│   │   ├── phase2_cleaning.ipynb
│   │   ├── phase3_benchmark_generators.ipynb
│   │   └── phase4_generate_5m.ipynb
│   └── dev2/                   # Developer 2 notebooks (Kaggle-ready)
│       ├── phase1_agent_interfaces.ipynb
│       ├── phase2_velocity_geo_agents.ipynb
│       ├── phase3_behaviour_gnn_agents.ipynb
│       ├── phase4_meta_learner.ipynb
│       └── phase5_evaluation_explainability.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data_contracts.py
│   ├── test_cleaning.py
│   ├── test_generators.py
│   ├── test_agents.py
│   └── test_evaluation.py
│
└── scripts/
    ├── setup_kaggle.sh           # Kaggle environment setup script
    └── validate_generated_data.py  # Post-generation validation script
```

---

## 7. Development Pipeline & Ownership Split

### 7.1 Parallel Workstreams (Merge-Conflict-Free)

The two developers work on **separate directories and files** at all times. The only shared files are `src/data_contracts.py` (canonical schema), `src/utils/config.py`, and `src/utils/nepal_context.py` — these must be **finalized first** before parallel work begins.

| Phase | Developer 1 (Data/Generation) | Developer 2 (Detection/Research) |
|---|---|---|
| **Phase 0** | Joint: Finalize `data_contracts.py`, `config.py`, `nepal_context.py` | Joint: Finalize `data_contracts.py`, `config.py`, `nepal_context.py` |
| **Phase 1** | EDA notebooks (`docs/eda/`), data profiling | Agent base class (`src/agents/base_agent.py`), agent interface design |
| **Phase 2** | Data cleaning (`src/cleaning.py`), canonical schema transformation | Velocity agent, Geo-risk agent (`src/agents/velocity_agent.py`, `geo_risk_agent.py`) |
| **Phase 3** | Generator benchmarking (`docs/generator_benchmark/`, `src/generation/`) | Behaviour agent (GRU/LSTM), KYC/AML rules agent |
| **Phase 4** | 5M row generation, AML pattern injection, validation | Graph agent (GraphSAGE/GAT), Meta-learner |
| **Phase 5** | Data-quality agent, generation report | Explanation agent, system evaluation, research documentation |

### 7.2 Branching Strategy

- `main` — stable, always runnable
- `dev/data-pipeline` — Developer 1's branch
- `dev/detection-agents` — Developer 2's branch
- Feature branches off respective dev branches: `feat/cta-benchmark`, `feat/velocity-agent`, etc.
- Pull requests merge into `main` only after both developers review

### 7.3 Kaggle-Specific Constraints

- Every notebook must be **self-contained and runnable on Kaggle** (free tier T4 GPU or CPU).
- Use `!pip install` only for packages not pre-installed on Kaggle. Prefer Kaggle's pre-installed environment.
- Use Kaggle's `input/` directory for reading datasets (the user uploads data via Kaggle Dataset).
- Save large outputs to Kaggle's `working/` directory; for persistence, use Kaggle Output.
- Always detect GPU availability and fall back to CPU. Never assume GPU exists. Use the utility in `src/utils/device_utils.py`.
- Use `%%time` and `%%memit` magic commands to profile expensive operations.
- For 5M row generation, process in **chunks** (e.g., 500K rows per batch) to avoid Kaggle memory limits (~13GB RAM on free tier, ~30GB on GPU tier).
- Save intermediate results as Parquet files (not CSV) for I/O efficiency with large datasets.

---

## 8. Agent Architecture & Design Principles

### 8.1 Agent Contract (All Agents Must Follow)

Every agent must:
1. Inherit from `BaseAgent` (defined in `src/agents/base_agent.py`).
2. Accept a `config` dict and a `logger` instance at initialization.
3. Implement `fit(data) -> self` for training/learning phase.
4. Implement `predict(data) -> pd.DataFrame` returning at minimum: `transaction_id`, `risk_score`, `alert_flag`, `reason_code`.
5. Implement `explain(transaction_id) -> str` returning a human-readable explanation for a single transaction.
6. Be **serializable** (pickle/joblib) so trained agents can be saved and loaded without refitting.
7. Log all decisions, thresholds, and abnormal conditions via the structured logger.
8. Never mutate input data — always return new DataFrames.
9. Handle missing features gracefully (log a warning, impute or skip, never crash).

### 8.2 Per-Agent Specifications

#### Data-Quality Agent (Deterministic)
- **Input**: Raw transaction rows
- **Output**: Pass/fail per row with violation codes
- **Logic**: Schema validation (types, ranges, allowed values), temporal consistency (timestamps not in future, no duplicates), account consistency (sender != receiver, accounts exist in accounts table), amount plausibility (positive, within channel limits)
- **No model training required** — purely rule-based
- **Owner**: Developer 1

#### Velocity Agent (Rules + Isolation Forest)
- **Input**: Transactions with temporal ordering, grouped by account
- **Features**: Rolling transaction count (1h, 6h, 24h, 7d), rolling sum of amounts, inter-transaction time, deviation from account's historical mean
- **Model**: Isolation Forest (unsupervised) for anomaly detection on velocity features
- **GPU**: Not needed — CPU is sufficient
- **Owner**: Developer 2

#### Behaviour Agent (GRU primary, LSTM as comparison)
- **Input**: Per-account ordered transaction sequences
- **Features**: Encoded sequence of (amount, type, channel, time_delta) per transaction
- **Model**: GRU (primary) — unidirectional LSTM as research comparison
- **Architecture**: Embedding layer for categoricals → concatenation → GRU/LSTM → Dense → sigmoid
- **GPU**: Required for training (use Kaggle T4 GPU)
- **Training strategy**: Train on legitimate sequences; flag sequences with high reconstruction error or anomaly score
- **Owner**: Developer 2

#### Geo-Risk Agent (CatBoost or LightGBM)
- **Input**: Transaction-level geographic features (sender_country, receiver_country, corridor, currency, is_cross_border, ip_country, ip_is_vpn)
- **Model**: CatBoost (primary, handles categoricals natively) or LightGBM
- **Features**: Country risk index (Nepal + corridor risk scores), currency deviation from account's norm, cross-border flag, IP geolocation mismatch, VPN/proxy flag, corridor-specific fraud rate
- **GPU**: CatBoost supports GPU training — use if available, else CPU
- **Owner**: Developer 2

#### Graph/Synthesis Agent (GraphSAGE primary, GAT as comparison)
- **Input**: Transaction graph — nodes are accounts, edges are transactions (weighted by amount, directed sender→receiver)
- **Features per node**: Account-level aggregates from the accounts table + graph metrics (degree, PageRank, community ID)
- **Model**: GraphSAGE (inductive, scales well) — GAT as explainability comparison (attention weights explain which neighbors contribute to fraud score)
- **Library**: PyTorch Geometric (PyG) — use Kaggle GPU
- **Subgraph patterns to detect**: Fan-in (many accounts sending to one), fan-out (one account sending to many), chains/layering (A→B→C→D), cycles (A→B→C→A), dense communities (mule networks)
- **GPU**: Required for GNN training and inference
- **Owner**: Developer 2

#### KYC/AML Rules Agent (Expert Rules + Calibrated Score)
- **Input**: Account-level KYC data joined with transactions
- **Rules**: PEP flag → elevated score, Sanctions match → critical alert, Account age < 90 days → elevated, KYC not verified → critical, Risk grade = high → elevated, Transaction amount near NPR 10 lakh threshold → structuring flag, Rapid successive transactions just below threshold → structuring flag, Multiple cross-border transfers to same corridor → layering flag
- **Output**: Calibrated risk score (0-1) based on weighted rule violations
- **No model training** — rules-based with configurable weights in `config.py`
- **Owner**: Developer 2

#### Meta-Learner Agent (Calibrated Random Forest, XGBoost as challenger)
- **Input**: All agent risk scores joined on `transaction_id`
- **Model**: Calibrated Random Forest (primary, good with heterogeneous features, interpretable) — XGBoost as challenger model
- **Calibration**: Use Platt scaling or isotonic regression on a held-out validation set
- **GPU**: XGBoost supports GPU — use if available; Random Forest is CPU-only
- **Evaluation**: Compare meta-learner AUC-ROC against individual agent AUC-ROCs
- **Owner**: Developer 2

#### Explanation Agent (SHAP + Templates)
- **Input**: Meta-learner predictions + feature attributions from each agent
- **Method**: SHAP values for model-based agents, rule-trace for rule-based agents, template-based natural language generation
- **Output**: Human-readable explanation string per alert, e.g., *"Transaction flagged: NPR 980,000 transfer to a newly opened account (< 30 days) in a high-risk corridor (Qatar→Nepal) via mobile banking at 02:43 AM. Account has 15 transactions in the past hour (velocity anomaly, p<0.01). Graph analysis shows this receiver is a fan-in node with 23 incoming edges from 18 distinct accounts in the past 7 days."*
- **Owner**: Developer 2

---

## 9. Data Generation Strategy

### 9.1 Generator Selection Process

The generator must be selected through a **structured benchmark** (not intuition). The benchmark evaluates candidates on three axes:

1. **Statistical Fidelity** — How well does the synthetic data reproduce the real data's marginal and joint distributions?
   - Kolmogorov-Smirnov (KS) test per continuous column (target: KS statistic < 0.05)
   - Column distribution similarity (Wasserstein distance or KL divergence)
   - Correlation matrix distance between real and synthetic
   - Categorical column value frequency comparison (chi-squared or total variation distance)

2. **ML Utility** — Can a model trained on synthetic data detect fraud in real data?
   - Train a classifier (e.g., XGBoost) on synthetic data, evaluate AUC-ROC on held-out real data
   - Target: AUC-ROC within 0.05 of a model trained on real data

3. **Privacy** — Does the synthetic data leak individual records?
   - Distance to Closest Record (DCR): minimum distance between any synthetic row and its nearest real row
   - Target: DCR significantly above the 5th percentile of inter-real-record distances

### 9.2 Candidate Generators

| Generator | Library | Strengths | Weaknesses |
|---|---|---|---|
| CTGAN | SDV | Handles mixed data types well; proven on tabular | Slow training on large data; memory-intensive |
| TVAE | SDV | Fast training; good for high-dimensional data | May oversmooth rare categories |
| CopulaGAN | SDV | Captures correlations via copulas + GAN | Complex; may not scale to 5M |
| Gaussian Copula | SDV | Very fast; lightweight | Limited capacity for complex joint distributions |
| WGAN-GP | Custom (PyTorch) | Stable GAN training; gradient penalty | Requires careful tuning |

**Recommended starting point**: Benchmark CTGAN, TVAE, and Gaussian Copula first. Add CopulaGAN and WGAN-GP if needed. The paper's findings suggest CTGAN or TVAE as likely winners.

### 9.3 Generation Pipeline (5M Rows)

The 5M row generation must follow this sequence:

1. **Preprocess source data** from `data/original/` into canonical schema (`src/cleaning.py`).
2. **Train selected generator** on preprocessed source data.
3. **Generate base synthetic data** in chunks (500K per chunk) to avoid OOM.
4. **Inject AML patterns** (`src/generation/aml_pattern_injector.py`):
   - Create mule account networks (fan-in, fan-out, chain structures)
   - Inject structuring patterns (transactions just below NPR 10 lakh threshold)
   - Inject layering patterns (multi-hop fund movements)
   - Inject round-tripping patterns (circular fund flows)
5. **Generate auxiliary `accounts` table** with account-level features for all unique accounts.
6. **Validate generated data** against canonical schema and data contracts.
7. **Run quality metrics** (statistical fidelity, ML utility, privacy) on generated data.
8. **Save as Parquet** in `data/generated/`.

### 9.4 AML Pattern Injection Specifications

When injecting AML patterns, the generator must create the following realistic typologies:

**Structuring (Smurfing)**:
- 3-10 transactions from the same account, each NPR 900,000–999,000 (just below the NPR 1M reporting threshold), within a 24-48 hour window.
- Same receiver account or a small set of receiver accounts.
- At least 50 such structuring sequences in the 5M dataset.

**Layering**:
- Chains of 3-6 accounts: A→B→C→D, where each transaction is a large transfer (NPR 500K–5M).
- Time gaps between hops: 1-72 hours (realistic settlement time).
- At least 30 such chains in the 5M dataset.

**Mule Account Networks**:
- Fan-in: 10-50 accounts sending money to a single collection account within 7 days.
- Fan-out: A single account distributing funds to 10-50 accounts within 7 days.
- The collection/distribution account then makes a large outbound transfer (or the reverse for fan-out).
- At least 20 such networks in the 5M dataset.

**Round-Tripping**:
- A→B→C→A cycles where the amounts are similar (allowing for fees/exchange rate differences).
- Cross-border round-tripping with currency conversion.
- At least 10 such cycles in the 5M dataset.

---

## 10. Coding Standards & Best Practices

### 10.1 Python Conventions
- Python 3.12.13 (pinned — local dev uses uv; Kaggle uses system Python 3.12)
- Type hints on all function signatures
- Docstrings on all public classes and functions (Google style)
- Maximum function length: 50 lines (refactor if longer)
- Maximum file length: 500 lines (split modules if longer)
- Use `pathlib.Path` for all file paths, never raw strings
- Use `logging` module (via `src/utils/logger.py`), never `print()` in library code

### 10.2 Dependencies Management
- Pin all dependencies with exact versions in `requirements.txt`
- Python version pinned to **3.12.13** via `.python-version` (uv) and `environment.yml` (conda)
- Pinned versions (current):

| Package | Version | Notes |
|---|---|---|
| `pandas` | 2.2.3 | Last 2.x LTS; stable for Python 3.12 |
| `numpy` | 2.2.6 | numpy 2.x series; required for catboost 1.2.10+ compat |
| `scikit-learn` | 1.6.1 | Latest stable |
| `sdv` | 1.17.2 | Pure Python; pip-only (no conda wheel) |
| `torch` | 2.5.1 | Python 3.12 supported; CPU wheel on PyPI |
| `torch-geometric` | 2.6.1 | pip-only; depends on torch 2.5.x |
| `catboost` | 1.2.10 | First release with numpy 2.x support |
| `lightgbm` | 4.6.0 | Latest stable |
| `xgboost` | 2.1.4 | Latest stable in 2.x series |
| `shap` | 0.46.0 | Compatible with numpy ≤ 2.2 (numba constraint) |
| `pydantic` | 2.12.5 | Latest stable |
| `pyarrow` | 20.0.0 | Latest stable; Python 3.12 supported |
| `pytest` | 8.3.5 | Latest stable |
| `hypothesis` | 6.131.15 | Latest stable |
| `tqdm` | 4.67.1 | Latest stable |
| `joblib` | 1.4.2 | Latest stable |

- Avoid heavyweight dependencies unless justified (e.g., no TensorFlow — use PyTorch only)
- When adding a new dependency, update both `requirements.txt` and `environment.yml`

### 10.3 Environment Setup

**uv is the primary package manager for local development.** pip and conda are supported as secondary options. Kaggle manages its own environment independently.

**Primary — uv:**
```bash
uv venv
uv pip install -r requirements.txt
uv run python <script>
```

**Secondary — pip:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

**Secondary — conda:**
```bash
conda env create -f environment.yml
conda activate neural-sentinel
```

**Kaggle:** Uses its own pre-installed environment. Manage missing packages via `!pip install -q` at the top of each notebook. Do not assume local venv is available on Kaggle.

The `.python-version` file at the project root pins `3.12.13` and is picked up automatically by uv. Do not delete or modify it.

### 10.4 Kaggle-Specific Practices
- At the top of every notebook, include a "Setup" cell that:
  - Detects and logs GPU/TPU availability
  - Installs any missing packages via `!pip install -q`
  - Sets random seeds for reproducibility (`numpy`, `torch`, `random`)
  - Configures logging
- Use `tqdm` for progress bars on long operations
- Profile memory usage before and after loading/generating large DataFrames
- For GNN training, batch graph construction — do not load the entire 5M-row graph into memory at once; use `torch_geometric.loader.NeighborLoader` or similar

### 10.5 Git Hygiene
- `.gitignore` must exclude: `data/original/`, `data/interim/`, `data/generated/`, `.ipynb_checkpoints/`, `__pycache__/`, `.env`, `*.pyc`, `models/` (saved model files)
- Commit messages: conventional commits format (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`); must be brief, precise, and one line (e.g. `feat: add velocity agent rolling window features`)
- Never commit large files (> 5MB) — use Git LFS or store externally
- Never commit API keys, credentials, or personal data

### 10.6 Testing
- Use `pytest` for all `src/` module tests
- Tests must be runnable on CPU without GPU
- Each agent must have at least: (1) a test that it initializes correctly, (2) a test that `predict()` returns correctly shaped output, (3) a test that it handles missing data gracefully
- Data contract tests must validate schema compliance (correct types, allowed values, no nulls in required fields)

### 10.7 Configuration Management
- All magic numbers, thresholds, file paths, and hyperparameters live in `src/utils/config.py`
- Use a dataclass or Pydantic `BaseModel` for typed configuration
- No hardcoded values in agent or generator code — always reference `config`

---

## 11. Evaluation Framework

### 11.1 Generator Evaluation
- **Statistical metrics**: KS test, Wasserstein distance, correlation matrix distance, categorical TV distance
- **ML utility**: Downstream classifier AUC-ROC (synthetic train → real test vs. real train → real test)
- **Privacy**: DCR (Distance to Closest Record)
- **Report**: Comparative table of all generators across all metrics; recommendation with justification

### 11.2 Agent Evaluation
- **Per-agent metrics**: AUC-ROC, AUC-PR, F1-score (at optimal threshold), Precision@k (top-k alerts), confusion matrix
- **Calibration**: Brier score, reliability diagram — risk scores must be well-calibrated probabilities
- **System-level metrics**: Meta-learner AUC-ROC vs. best individual agent AUC-ROC; improvement delta
- **Explainability quality**: Sample explanations reviewed for coherence, specificity, and actionability

### 11.3 AML-Specific Evaluation
- **Structuring detection rate**: What fraction of injected structuring patterns are flagged by the KYC/AML agent and the meta-learner?
- **Network detection rate**: What fraction of injected mule networks are identified by the Graph agent?
- **False positive rate**: How many legitimate transactions are incorrectly flagged? (Target: < 5% FPR at operational threshold)

---

## 12. Nepal-Specific Context (`src/utils/nepal_context.py`)

This module must contain all Nepal-specific reference data and constants. AI agents writing code should import from this module rather than hardcoding.

### 12.1 Remittance Corridors (by volume, approximate)
- **Gulf countries**: Qatar, UAE, Saudi Arabia, Bahrain, Kuwait, Oman — largest remittance sources
- **South/Southeast Asia**: India, Malaysia, South Korea, Thailand
- **Developed markets**: USA, UK, Australia, Japan — smaller volume but higher per-transaction value
- **Corridor risk tiers**: Gulf corridors = high volume / medium risk; India = high volume / low risk; Western corridors = low volume / high risk

### 12.2 Nepali Banking Channel Mix (approximate)
- Mobile banking: ~40-50% (growing rapidly)
- Branch/counter: ~20-25%
- ATM: ~10-15%
- Online banking: ~10-15%
- POS: ~5-10%

### 12.3 NPR Exchange Rate Ranges (reference for generator)
- USD/NPR: 110–135
- EUR/NPR: 120–150
- GBP/NPR: 140–170
- INR/NPR: 1.58–1.65 (fixed peg, minor fluctuation)
- QAR/NPR: 30–38
- SAR/NPR: 29–36
- AED/NPR: 30–37
- MYR/NPR: 24–30

### 12.4 Major Nepali Cities (for account holder location)
Kathmandu, Lalitpur, Bhaktapur, Pokhara, Biratnagar, Bharatpur, Birganj, Dharan, Butwal, Hetauda, Nepalgunj, Bhadrapur, Itahari, Dhangadhi, Mahendranagar

---

## 13. Open Questions & Clarifications Needed

Before proceeding to code generation, the following questions should be confirmed with the team:

1. **Time range for generated data**: Resolved — 1 year. The 5M rows will span 1 year of transaction history. This supports velocity feature windows (1h, 6h, 24h, 7d), seasonal patterns (monthly, quarterly), and day-of-week behavioral patterns.

2. **Fraud rate in 5M dataset**: Deferred — to be decided after EDA. The source data has ~10% fraud rate (unrealistically high). The target rate will be determined after EDA reveals the true distribution and after evaluating the impact on downstream model training.

3. **IP geolocation**: Resolved — no real GeoIP dependency (e.g., MaxMind) will be used in this prototype. Both fields are handled as deterministic post-processing steps after generation:
   - `ip_country`: for domestic transactions, copy directly from the `country` column; for cross-border transactions, assign consistent with the sender's country.
   - `ip_is_vpn`: synthetic binary flag with a higher probability assigned to fraud rows and a low baseline probability for legitimate rows.
   Real GeoIP enrichment is a production concern and out of scope for this prototype.

4. **Number of mule networks**: Deferred — to be decided after EDA and testing. The minimum counts specified in Section 9.4 are provisional. Final numbers will be calibrated based on EDA findings and GNN training requirements.

5. **Graph construction**: Resolved — nodes are accounts and directed edges are transactions (sender account → receiver account). All further graph construction decisions (sampling strategy, time windowing, edge weighting, etc.) will be determined after EDA on the generated data.

---

## 14. Prohibited Actions

AI agents contributing to this repository must **NOT**:

- Write code that accesses real banking data or calls live banking APIs
- Hardcode API keys, credentials, or personal information
- Commit generated data (5M rows) or model artifacts to the git repository
- Use TensorFlow — this project is PyTorch-only (to minimize dependency bloat on Kaggle)
- Install packages not listed in `requirements.txt` without updating that file
- Modify files owned by the other developer without explicit coordination
- Generate synthetic data that could be mistaken for real financial records
- Skip validation steps (schema checks, quality metrics) to save time
- Use `print()` for logging — always use the structured logger
- Write monolithic files (>500 lines) — split into focused modules
- Assume GPU availability — always check and fallback gracefully
- Inject placeholder or dummy data into production-quality modules

---

## 15. Success Criteria

The project is considered successful if:

1. EDA is thorough, well-documented, and leads to a justified generator choice.
2. The selected generator produces 5M rows that pass all statistical fidelity, ML utility, and privacy checks.
3. AML patterns are realistically injected and detectable by the relevant agents.
4. All 8 agents are implemented, tested, and produce correctly formatted output.
5. The meta-learner outperforms the best individual agent by at least 2% AUC-ROC.
6. Explanations are coherent, specific, and actionable for a banking analyst.
7. The codebase is clean, well-documented, and merge-conflict-free between developers.
8. The system can be demonstrated as a pitch to Nepali banks with a working Kaggle notebook.