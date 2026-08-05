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
The system learns the behavior of the real banking environment. It extracts distributions and statistics into a **Knowledge Base**:
- **Customer Profiles**: Average transaction sizes, frequencies, preferred hours/weekdays, cross-border frequency.
- **Geographic & Institution Data**: City frequencies, branch locations, and country risk mappings.
- **Graph Knowledge**: Constructs a transaction graph to compute node degrees, fan-in/fan-out rates, communities, and common transfer paths.

*(Note: During this phase, various ML generators like CTGAN, TVAE, and CopulaGAN are benchmarked to evaluate statistical similarity and ML utility, but the final 5M generation uses the extracted knowledge for deterministic graph-aware generation).*

### Phase 4: Synthetic Account Generation
New synthetic customers (accounts) are generated based on the learned distributions. 
- Fields generated include `account_id`, `institution`, `risk_grade`, `pep_flag`, `opened_date`, etc.
- **Constraints**: Risk grades must be valid, institution/branch combinations must exist, and opening dates must precede any future transactions.

### Phase 5: Core Transaction Generation
Using the `TransactionGenerator`, fundamental transaction events are created between the synthetic accounts.
- **Graph-Aware Receiver Selection**: Receivers are selected based on the graph knowledge (e.g., frequent previous receivers, community receivers, or completely new receivers) to recreate realistic network density.
- **Amounts and Times**: Amounts are sampled from heavy-tailed log-normal distributions. Timestamps are sampled according to learned business hour and weekend activity distributions.

### Phase 6: Fraud Scenario Injection (AML Patterns)
Once the baseline legitimate transactions are generated, the `AMLPatternInjector` introduces fraudulent behavior. There are 11 specific scenarios injected to simulate real-world Anti-Money Laundering (AML) threats:
1. **Large Amount Fraud**: Anomalously large transfers to new receivers.
2. **Smurfing / Structuring**: Multiple transactions just below reporting thresholds.
3. **Velocity Fraud**: High frequency of transactions in a very small time window.
4. **Money Mules**: A→B→C patterns where 'B' acts as a pass-through account.
5. **Fan-Out**: One sender quickly dispersing funds to many receivers.
6. **Fan-In**: Many senders funneling money into a single receiver.
7. **Layering**: Deep chains of transfers (A→B→C→D) to obfuscate origins.
8. **Circular Transactions**: Funds returning to the origin (A→B→C→A).
9. **Cross-Border Fraud**: High-risk destination transfers with currency mismatches.
10. **Dormant Account Activation**: Sudden large activity on an old, inactive account.
11. **High-Risk/PEP Activity**: Suspicious activity involving sanctioned or politically exposed persons.

### Phase 7: Transaction Enrichment
The generated transactions are joined with the synthetic account metadata. Account details (institution, risk grade, PEP status, etc.) are attached to every sender and receiver in the transaction log. No random generation occurs here; it is purely a relational join.

### Phase 8: Feature Engineering
All derived features required by the downstream detection agents (Velocity Agent, Geo-Risk Agent, Behaviour Agent, etc.) are deterministically computed:
- **Temporal**: `hour_of_day`, `is_weekend`
- **Amount**: `log_amount`, `amount_zscore`, `amount_local_npr`
- **Geographic**: `sender_country_risk`, `cross_border_flag`
- **Velocity**: Rolling window counts (e.g., `velocity_sum_10tx`, `tx_count_30`)
- **Account**: `sender_account_age_days`

### Phase 9: Constraint Validation
A final pass ensures every generated row is logically sound.
- Transactions cannot occur before an account is opened.
- Senders and receivers must exist and cannot be the same entity (unless explicitly allowed).
- Amounts must be positive and currency conversions must be mathematically accurate.
*(Rows violating these constraints are either rejected or regenerated).*

### Phase 10: Final Dataset Assembly
The pipeline outputs the finalized 5-million row datasets to disk as Parquet/CSV files (e.g., `synthetic_accounts.parquet`, `synthetic_transactions.parquet`, `ml_features.csv`, `graph_edges.csv`). 

These final files act as the foundational input for training the multi-agent detection system, ensuring the agents learn from complex, relational, and highly realistic synthetic fraud scenarios.
