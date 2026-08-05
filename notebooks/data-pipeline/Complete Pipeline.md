# Synthetic Banking Data Generation Pipeline

## Technical Design Specification (Data Generation Module)

**Version:** 1.0

---

# 1. Objective

The objective of this module is to generate a realistic synthetic banking dataset that preserves:

* Statistical properties of the original data
* Relationships between entities
* Banking business rules
* Fraud patterns
* Temporal consistency
* Network (graph) structure

The generated dataset will later be used to train and evaluate fraud detection models and downstream intelligent agents.

Unlike traditional tabular generation approaches, this pipeline does **not** generate every column independently. Instead, it generates a synthetic banking world in stages, where later stages depend on earlier ones.

---

# 2. Input Datasets

## Accounts

```
accounts.csv
```

Contains

* Account information
* Customer profile
* Institution
* Branch
* Risk information
* KYC information

---

## Transactions

```
transactions.csv
```

Contains

* Transaction events
* Sender
* Receiver
* Amount
* Currency
* Time
* Payment method

along with engineered features that will be reconstructed later.

---

# 3. Overall Pipeline

```
Original Dataset
        │
        ▼
Schema Validation
        │
        ▼
Data Cleaning
        │
        ▼
Banking Knowledge Extraction
        │
        ▼
Synthetic Account Generation
        │
        ▼
Synthetic Transaction Generation
        │
        ▼
Fraud Scenario Injection
        │
        ▼
Transaction Enrichment
        │
        ▼
Feature Engineering
        │
        ▼
Constraint Validation
        │
        ▼
Synthetic Dataset
```

---

# Phase 1 — Schema Validation

## Goal

Understand the dataset before generation.

---

## Tasks

Inspect every dataset and identify

### Data Types

* Numerical
* Categorical
* Boolean
* Datetime
* Identifier

---

### Missing Values

Calculate

* Missing percentage
* Null count
* Invalid values

---

### Duplicate Detection

Check

* Duplicate rows
* Duplicate account IDs
* Duplicate account numbers

---

### Distribution Analysis

For every feature compute

* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* Quantiles

---

### Categorical Analysis

For every categorical feature compute

* Unique values
* Frequency distribution

---

### Output

```
schema_report.json
```

---

# Phase 2 — Data Cleaning

## 2.1 Account Cleaning

Remove

* Duplicate accounts
* Invalid account IDs

Normalize

* Institution names
* Branch names
* City names
* Account type
* Risk grades

Validate

* PEP flag
* Sanction flag
* Opened date

Output

```
clean_accounts.csv
```

---

## 2.2 Transaction Cleaning

Remove

* Duplicate transactions
* Negative amounts
* Invalid timestamps
* Invalid sender IDs
* Invalid receiver IDs

Normalize

* Currency
* Payment mode
* Timestamp

Validate

* Sender exists
* Receiver exists
* Amount > 0

Output

```
clean_transactions.csv
```

---

# Phase 3 — Banking Knowledge Extraction

This phase learns how the banking system behaves.

No synthetic data is generated yet.

---

## Customer Profile Learning

For every account calculate

* Average transaction amount
* Median amount
* Standard deviation
* Daily transaction frequency
* Monthly transaction frequency
* Preferred transaction hours
* Preferred weekdays
* Preferred payment modes
* Currency usage
* Typical receivers
* Typical senders
* Cross-border frequency
* Average transaction interval

---

## Institution Knowledge

Learn

* Institution frequencies
* Branch hierarchy
* Branch-city relationship
* Institution-city relationship

---

## Geographic Knowledge

Learn

* Country frequency
* City frequency
* Branch locations
* Country risk mapping

---

## Transaction Knowledge

Learn distributions for

* Amount
* Time
* Currency
* Payment type
* Cross-border transfers

---

## Graph Knowledge

Construct transaction graph

```
Account
      │
      ▼
Account
```

Compute

* Degree
* Fan-in
* Fan-out
* Communities
* Centrality
* Common transfer paths

---

### Output

```
knowledge_base/
```

containing

```
behavior_profiles.parquet

institution_mapping.json

branch_mapping.json

country_mapping.json

graph_statistics.json
```

---

# Phase 4 — Synthetic Account Generation

## Goal

Generate realistic synthetic bank accounts.

Each generated account represents a new customer entering the banking system.

---

## Fields to Generate

Generate

```
account_id

account_number

institution

branch

account_type

risk_grade

is_person

name

tax_number

pep_flag

sanctions_hit

city

opened
```

---

## Generation Constraints

Institution must exist.

Branch must belong to institution.

City must match branch.

Risk grade must be valid.

Account number must be unique.

Opened date must precede every future transaction.

PEP and sanctions values must follow realistic frequencies.

---

## Validation

Reject generated accounts if

* Duplicate ID
* Duplicate account number
* Invalid branch
* Invalid institution
* Future opening date

---

### Output

```
synthetic_accounts.csv
```

---

# Phase 5 — Core Transaction Generation

## Goal

Generate realistic transaction events.

This stage generates only the fundamental transaction information.

---

## Generate

```
Sender_account

Receiver_account

Transaction_Date

Transaction_Time

Amount

Payment_currency

Received_currency

Payment_type
```

---

## Transaction Constraints

Sender must exist.

Receiver must exist.

Sender cannot equal receiver unless explicitly allowed.

Currencies must be valid.

Amount must be positive.

Timestamp must occur after both accounts were opened.

Payment type must be valid.

---

## Temporal Generation

Generate realistic

* Business hours
* Night transactions
* Weekend activity
* Holiday activity

using the learned temporal distributions.

---

## Amount Generation

Generate

* Small transfers
* Medium transfers
* Large transfers

while preserving

* Distribution
* Quantiles
* Heavy-tail behaviour

---

## Receiver Selection

Receivers should be selected according to learned graph behaviour.

Examples

* Frequent receiver

* New receiver

* Community receiver

* Cross-community receiver

---

### Output

```
synthetic_transactions_core.csv
```

---

# Phase 6 — Fraud Scenario Injection

After normal transactions are generated, inject fraudulent behaviour.

Each synthetic fraud transaction must belong to a defined fraud scenario.

---

## Scenario 1

Large Amount Fraud

Characteristics

* Extremely high amount
* New receiver
* Unusual hour

---

## Scenario 2

Smurfing

Generate

Multiple transactions

```
980000

990000

995000

999000
```

within a short period.

---

## Scenario 3

Velocity Fraud

Generate

```
Many transactions

↓

Very small time interval
```

---

## Scenario 4

Money Mule

```
Victim

↓

Mule

↓

Beneficiary
```

---

## Scenario 5

Fan-Out

```
One sender

↓

Many receivers
```

---

## Scenario 6

Fan-In

```
Many senders

↓

One receiver
```

---

## Scenario 7

Layering

```
A

↓

B

↓

C

↓

D
```

---

## Scenario 8

Circular Transactions

```
A

↓

B

↓

C

↓

A
```

---

## Scenario 9

Cross-Border Fraud

Generate

* High-risk destination
* Currency mismatch
* Foreign transfer

---

## Scenario 10

Dormant Account Activation

Generate

Large transactions immediately after long inactivity.

---

## Scenario 11

PEP or High-Risk Customer

Generate transactions involving

* PEP accounts
* High-risk accounts
* Sanctioned accounts

---

### Output

```
synthetic_fraud_transactions.csv
```

---

# Phase 7 — Transaction Enrichment

Merge

```
synthetic_accounts.csv

+

synthetic_transactions.csv
```

Attach

* Institution
* Branch
* City
* Risk grade
* PEP
* Sanctions
* Account type

to every sender and receiver.

No random generation occurs during this phase.

Everything is obtained from account information.

---

# Phase 8 — Feature Engineering

Compute all engineered features from the enriched transactions.

---

## Temporal Features

Generate

```
hour_of_day

day_of_week

month

is_weekend
```

---

## Amount Features

Compute

```
amount_local_npr

log_amount

amount_zscore

above_1M_NPR

above_10M_NPR
```

---

## Geographic Features

Compute

```
sender_country_risk

receiver_country_risk

cross_border_flag

currency_mismatch
```

---

## Velocity Features

Calculate

```
velocity_sum_10tx

tx_count_10

tx_count_30
```

using chronological transaction history.

---

## Account Features

Compute

```
sender_account_age_days

receiver_account_age_days
```

---

## Transaction Mode Encoding

Convert

```
Payment_type
```

into

```
transmode_A

transmode_B

transmode_E

transmode_F

transmode_J

transmode_P

transmode_Z
```

---

# Phase 9 — Constraint Validation

Every generated row must pass validation.

---

## Account Constraints

* Unique account
* Valid institution
* Valid branch
* Valid city

---

## Transaction Constraints

* Valid sender
* Valid receiver
* Positive amount
* Valid timestamp
* Timestamp after account opening
* Valid currency

---

## Feature Constraints

Validate

* Correct weekday
* Correct weekend
* Correct amount conversion
* Correct threshold flags
* Correct one-hot encoding
* Valid account age

Rows violating constraints are rejected or regenerated.

---

# Phase 10 — Final Dataset Construction

Produce

```
synthetic_accounts.csv

synthetic_transactions.csv

graph_edges.csv

ml_features.csv
```

These datasets become the input for

* Statistical benchmarking
* Privacy benchmarking
* Utility benchmarking
* Fraud detection models
* Behaviour Agent
* Velocity Agent
* Geo-Risk Agent
* KYC/Rules Agent
* Explainability Agent
* Meta-Learner

---

# Deliverables

The data generation pipeline should output

```
outputs/

├── clean_accounts.csv
├── clean_transactions.csv
├── synthetic_accounts.csv
├── synthetic_transactions_core.csv
├── synthetic_fraud_transactions.csv
├── transactions_enriched.csv
├── graph_edges.csv
├── ml_features.csv
├── schema_report.json
├── validation_report.json
├── behavior_profiles.parquet
└── generation_log.json
```

---

# Design Principle

The pipeline follows a layered generation strategy.

Instead of generating every column independently, it generates the **causal entities first** (accounts and transaction events), then **injects realistic fraud scenarios**, and finally **reconstructs all derived features** through deterministic feature engineering. This preserves statistical fidelity while maintaining logical consistency across the synthetic banking dataset, making it suitable for downstream fraud detection, explainability, and multi-agent risk analysis.
