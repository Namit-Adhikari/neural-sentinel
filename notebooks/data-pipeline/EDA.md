# Choosing the Best Synthetic Data Generator for Fraud Detection
### CTGAN vs TVAE vs CopulaGAN vs Gaussian Copula vs WGAN-GP vs CTAB-GAN+ vs TabDDPM

> This guide covers all seven generators. Follow it top to bottom and you'll end with a defensible, data-driven choice.

This is a full checklist, not just theory — follow it top to bottom and you'll end with a defensible, data-driven choice.

---

## 0. Ground rule for fraud/banking data

Fraud data has properties that make generator choice *not* a style preference — it directly affects whether your downstream fraud model is trustworthy:

- Extreme **class imbalance** (fraud is usually <1–5% of rows)
- Mixed types: continuous (`Amount`, `fx_rate_to_npr`), categorical (`Payment_type`, `sender_risk_grade`), binary flags (`sender_pep`, `cross_border_flag`), datetime
- **Derived/engineered columns** that must stay logically consistent with each other (e.g. `log_amount` = log(`Amount`), `above_1M_NPR` derived from `amount_local_npr`)
- **Relational/graph structure** between `Sender_account` and `Receiver_account` (repeated pairs, fan-in/fan-out patterns = fraud signal)
- Heavy-tailed, skewed numeric distributions (transaction amounts)
- **Benford's Law compliance** — natural financial transaction amounts follow a predictable first-digit distribution; generators must reproduce this
- **Threshold-proximity patterns** — AML structuring concentrates transactions just below reporting thresholds (NPR 1M for NRB)

Whichever generator best preserves these things wins — not whichever "looks smoothest."

---

## 1. Baseline EDA on the REAL dataset (do this first, always)

You need this as the ground truth to compare every generator against later.

### 1.1 Structure
- [x] Shape, dtypes, memory size
- [x] Duplicate rows / duplicate transaction keys
- [x] Confirm label column exists (fraud/laundering flag) and its exact name
- [x] Class imbalance ratio (fraud % vs non-fraud %)

### 1.2 Missing values & quality
- [x] Missingness % per column, and whether it's structural or random
- [x] Inconsistent categories / typos in categorical fields

### 1.3 Univariate
- [x] Numeric: `Amount`, `fx_rate_to_npr`, `amount_local_npr`, `sender_account_age_days`, `receiver_account_age_days`, `velocity_sum_10tx`, `tx_count_10`, `tx_count_30`, `amount_zscore` → histogram, skew, kurtosis, percentiles
- [x] Categorical: `Payment_currency`, `Payment_type`, `transmode_code`, bank locations, `sender_risk_grade` → frequency + cardinality
- [x] Binary flags: `cross_border_flag`, `currency_mismatch`, `sender_pep`, `sender_sanctions`, `receiver_pep`, `receiver_sanctions`, `is_weekend` → proportion of 1s
- [x] Datetime: `Date`, `Time`, `hour_of_day`, `day_of_week`, `month` → range, gaps

### 1.4 Target-conditioned analysis
- [x] Feature distributions split by fraud vs non-fraud (especially `Amount`, `velocity_sum_10tx`, `tx_count_10/30`)
- [x] Fraud rate by `Payment_type`, `transmode_code`, bank location, account type
- [x] Fraud rate over time (hour/day/month patterns)
- [ ] **Hourly × day-of-week heatmap**: fraud rate by `hour_of_day` × `day_of_week` (2D heatmap — reveals overnight/weekend fraud clusters)
- [ ] **Burst detection**: inter-transaction time (ITT) per account — distribution of time gaps between consecutive transactions, split by fraud/non-fraud
- [ ] **Velocity profile by account type**: different account types (savings vs current vs salary vs fixed_deposit) have different normal velocities — profile each and check where fraud clusters
- [ ] **Weekend/holiday effect by channel**: fraud rate on weekends vs weekdays, broken down by channel (mobile_banking / atm / branch / online_banking / pos)
- [ ] **Monthly seasonality**: month-end salary-day spike pattern, seasonal trends, fraud rate by month
- [ ] **Transaction chain timing**: for repeated sender→receiver pairs, time between successive transfers (short gaps = potential structuring)

### 1.5 Bivariate / multivariate
- [x] Correlation matrix (numeric) — expect `Amount`, `log_amount`, `amount_zscore`, `amount_local_npr` to be near-collinear
- [x] Cramér's V for categorical-categorical associations
- [x] VIF for multicollinearity

### 1.6 Consistency of engineered columns
- [x] `log_amount` ≈ log(`Amount`)
- [x] `amount_local_npr` ≈ `Amount` × `fx_rate_to_npr`
- [x] `above_1M_NPR` / `above_10M_NPR` match thresholds on `amount_local_npr`
- [x] `hour_of_day`/`day_of_week`/`month`/`is_weekend` match `Date`/`Time`
- [x] `transmode_A/B/E/F/J/P/Z` one-hot columns sum to 1 and match `transmode_code`
- [x] `currency_mismatch` = (`Payment_currency` ≠ `Received_currency`)
- [x] `cross_border_flag` = (`Sender_bank_location` ≠ `Receiver_bank_location`)

### 1.7 Network structure
- [x] Transaction count per sender/receiver account (degree distribution)
- [x] Repeated sender→receiver pairs (structuring/smurfing signal)
- [x] Verify `velocity_sum_10tx`/`tx_count_10`/`tx_count_30` are correctly rolling-windowed per account

### 1.8 Outliers
- [x] `amount_zscore` extremes, IQR outliers on `Amount`
- [x] Check whether outliers cluster in the fraud class

### 1.9 Benford's Law analysis

> **Why this matters for generators:** Benford's Law is a standard forensic accounting test used by auditors worldwide. Natural financial transaction amounts follow a predictable first-digit distribution (~30.1% for digit 1, ~4.6% for digit 9). If a generator produces amounts that violate Benford's Law, the synthetic data is immediately detectable as fake — and any AML/fraud model trained on it will learn the wrong amount distribution.

- [ ] First-digit frequency analysis on `Amount` and `amount_local_npr`
- [ ] Chi-square goodness-of-fit test against Benford's expected distribution
- [ ] Visualize: observed first-digit distribution vs theoretical Benford curve (bar chart overlay)
- [ ] Second-digit analysis (more sensitive to manipulation and structuring near thresholds)
- [ ] First-two-digit analysis for extra sensitivity
- [ ] Compare Benford compliance: fraud subset vs non-fraud subset (fraud transactions often cluster near specific "round" amounts or threshold amounts, breaking Benford)
- [ ] Compute MAD (Mean Absolute Deviation) from Benford's expected proportions — benchmark: MAD < 0.006 = close conformity, 0.006–0.012 = acceptable, > 0.015 = non-conformity

### 1.10 Structuring / threshold proximity analysis

> **Why this matters:** AML structuring ("smurfing") is the deliberate splitting of transactions to stay below reporting thresholds. For Nepali banking, the NRB cash-reporting threshold is NPR 10 lakh (1,000,000). This is a primary fraud signal and a pattern that generators must reproduce.

- [ ] Distribution of `amount_local_npr` near the NPR 1M threshold — histogram with fine bins in the 800K–1.2M band
- [ ] Count and proportion of transactions in the 90–100% range of the threshold (NPR 900K–1M)
- [ ] Fraud rate comparison: transactions just below threshold (900K–999K) vs just above (1M–1.1M) vs rest
- [ ] Per-account analysis: identify accounts with ≥3 transactions in the 800K–999K range within any 30-day window (classic structuring indicator)
- [ ] Same analysis for the NPR 10M threshold (`above_10M_NPR`)
- [ ] Distribution of `Amount` in original currency near common round numbers (1000, 5000, 10000) — fraud often involves round amounts
- [ ] "Round number ratio": percentage of transactions with amounts ending in 00, 000, or 0000 — split by fraud/non-fraud

### 1.11 Dimensionality reduction & class separability

> **Why this matters:** This becomes the baseline projection that every synthetic dataset gets compared against in §4.3. If real data shows clear fraud clusters in reduced dimensions, a generator that smooths them away is useless.

- [ ] PCA (2D) of numeric features, colored by `is_suspicious_tx` — check if fraud transactions cluster in specific principal component space regions
- [ ] Explained variance ratio: how many PCs needed to capture 90% / 95% variance? (informs dimensionality of the generation task)
- [ ] UMAP (2D) of all features (numerics + encoded categoricals), colored by fraud label — UMAP preserves local structure better than PCA for manifold-shaped data
- [ ] t-SNE (2D) for comparison — t-SNE is better at revealing local clusters but loses global structure
- [ ] Quantify class separability: silhouette score, Davies-Bouldin index on the fraud vs non-fraud clusters
- [ ] Overlay: PCA loadings plot — which features contribute most to the principal components that separate fraud from non-fraud?

### 1.12 Feature importance baseline

> **Why this matters:** Before training generators, you need to know which features *actually matter* for fraud detection. A generator that perfectly preserves irrelevant features but destroys discriminative ones is useless. This creates the "critical features list" — generator evaluation (§4, §5) should weight these features more heavily.

- [ ] Train a quick XGBoost classifier on the real data with `is_suspicious_tx` as target (5-fold CV, minimal tuning)
- [ ] Extract feature importance (gain-based) and rank top-20 features
- [ ] Compute SHAP values (TreeExplainer) — SHAP summary plot showing feature contributions
- [ ] Mutual Information (MI) between each feature and the target — identifies both linear and non-linear dependencies
- [ ] Compare MI ranking vs XGBoost importance ranking — discrepancies reveal features where the relationship is non-linear
- [ ] Point-biserial correlation for numeric features vs binary target
- [ ] Document the "Top-10 Critical Features" list — generators MUST preserve the distributions and inter-correlations of these features above all else

### 1.13 Account-level behavioral profiles

> **Why this matters:** Fraud is fundamentally an account-level phenomenon — transaction-level features alone miss the behavioral context. Generators that only model row-level distributions can't capture account-level fraud patterns (e.g., a previously dormant account suddenly making rapid transfers).

- [ ] Aggregate per-account statistics from `transactions.csv`:
  - Total transaction count
  - Total volume (sum of `amount_local_npr`)
  - Distinct counterparties (unique receivers for senders, unique senders for receivers)
  - Mean and standard deviation of `Amount`
  - Active days (number of distinct transaction dates)
  - Unique channels used
  - Date range of activity (first tx → last tx)
- [ ] Compare account-level profiles: fraud-involved accounts vs clean accounts (boxplots or violin plots for each metric)
- [ ] Distribution of "account age at first suspicious tx" — do new accounts get flagged more? (using `sender_account_age_days` or computing from `sender_opened`)
- [ ] Identify accounts involved as both sender and receiver in flagged transactions — potential mule account indicators
- [ ] Cross-reference with `accounts.csv`:
  - Fraud involvement rate by `risk_grade` (RISK-LOW / RISK-MEDIUM / RISK-HIGH)
  - Fraud involvement rate by `acct_type` (FIXED / SAVINGS / CURRENT / NOSTRO / SALARY)
  - PEP flag (`pep_flag`) vs fraud involvement rate
  - Sanctions hit (`sanctions_hit`) vs fraud involvement rate
  - `is_person` (individual vs corporate) vs fraud involvement rate
- [ ] Account degree vs fraud rate: do high-degree accounts (many transactions) have higher or lower fraud rates?

### 1.14 Cross-border & remittance corridor analysis

> **Why this matters:** This project specifically targets Nepali banking fraud, including cross-border remittance exploitation (AGENTS.md §2). The corridor analysis is essential domain-specific EDA that directly informs the Geo-Risk Agent (§7.2) and must be preserved by generators.

- [ ] Fraud rate: cross-border (`cross_border_flag=1`) vs domestic (`cross_border_flag=0`)
- [ ] Fraud rate by `Sender_bank_location` × `Receiver_bank_location` — heatmap (which country pairs are riskiest?)
- [ ] Derive remittance corridor from location pairs, map to `nepal_context.py` corridors — fraud rate by corridor
- [ ] `currency_mismatch` analysis: fraud rate when currencies match vs mismatch
- [ ] Country risk score (`sender_country_risk`, `receiver_country_risk`) distribution for fraud vs non-fraud
- [ ] Amount distribution by corridor — different corridors have different typical remittance amounts (e.g., Gulf worker remittances vs UK/US transfers)
- [ ] Cross-border transactions: channel distribution (do cross-border frauds prefer specific channels?)
- [ ] Institution analysis: fraud rate by `sender_institution` (HBL, CITIZENS, etc.)

### 1.15 Data leakage & column redundancy audit

> **Why this matters:** Before feeding data to generators, we need to know which columns are purely derived from others. Generators should NOT independently generate derived columns — they should be re-derived post-generation to maintain logical consistency. Training on redundant columns wastes generator capacity and creates inconsistencies.

- [ ] Identify perfectly correlated column pairs (|r| = 1.0 or very close)
- [ ] Formalize all deterministic derivations:
  - `log_amount` = log1p(`Amount`) — **derived, do not generate**
  - `amount_local_npr` = `Amount` × `fx_rate_to_npr` — **derived, do not generate**
  - `above_1M_NPR` = (`amount_local_npr` >= 1,000,000) — **derived, do not generate**
  - `above_10M_NPR` = (`amount_local_npr` >= 10,000,000) — **derived, do not generate**
  - `hour_of_day`, `day_of_week`, `month`, `is_weekend` = f(`Date`, `Time`) — **derived, do not generate**
  - `transmode_A`, `transmode_B`, `transmode_E`, `transmode_F`, `transmode_J`, `transmode_P`, `transmode_Z` = one-hot of `transmode_code` — **derived, do not generate**
  - `cross_border_flag` = (`Sender_bank_location` ≠ `Receiver_bank_location`) — **derived, do not generate**
  - `currency_mismatch` = (`Payment_currency` ≠ `Received_currency`) — **derived, do not generate**
  - `amount_zscore` = z-score of `Amount` — **derived, do not generate**
  - `date_transaction` = `Date` + `Time` combined — **derived, do not generate**
- [ ] Count: how many of the ~55 columns are derivable? (Expected: ~18–20 derived columns)
- [ ] Define the **Generator Input Schema** — the minimal set of base columns generators should train on (~25–30 columns). The official definition should reside in `src/data_contracts.py`.
- [ ] Recommendation: generators train only on base columns; derived columns are recomputed after generation via a post-processing pipeline.

**Tools:** `ydata-profiling`, `sweetviz`, `pandas-profiling`, manual `pandas`/`matplotlib`/`seaborn`, `scipy.stats` (for Benford's chi-square), `sklearn` (PCA, silhouette), `umap-learn`, `xgboost`, `shap`

---

## 2. Know what each generator is built for

| Generator | Type | Strengths | Weak points | Good fit for fraud data? |
|---|---|---|---|---|
| **Gaussian Copula** | Statistical (copula) | Fast, preserves marginal distributions + linear/rank correlations exactly, very stable, interpretable | Can't capture complex non-linear or conditional dependencies; struggles with highly skewed rare-event patterns | Good baseline, often surprisingly strong for tabular financial data |
| **CTGAN** | GAN (conditional, mode-specific normalization) | Handles mixed continuous/categorical well, models multi-modal numeric distributions (e.g. bimodal amounts), conditional vector helps with imbalanced categories | Training instability, mode collapse risk, can struggle to preserve rare classes without tuning (needs conditional sampling on fraud label), slower | Strong candidate — designed exactly for mixed-type tabular data |
| **TVAE** | VAE | More stable training than GANs, good likelihood-based fit, decent with mixed types | Tends to blur rare/extreme values (VAEs regress to the mean), may under-represent fraud outliers | Risk: may "smooth away" the rare fraud signal you actually need |
| **CopulaGAN** | Hybrid (GAN + copula transform) | Combines copula's marginal fidelity with GAN's ability to model complex joint structure | Newer/less battle-tested, still inherits GAN instability | Good middle ground worth testing |
| **WGAN-GP** | GAN (Wasserstein + gradient penalty) | More stable GAN training than vanilla GAN, better gradient behavior, less mode collapse than CTGAN in some cases | Not natively built for mixed categorical/continuous tabular data (originally image-domain) — needs careful preprocessing/embedding of categoricals, more engineering effort | Powerful but highest implementation risk for tabular fraud data |
| **CTAB-GAN+** | GAN (conditional + auxiliary classifier) | Built-in minority class oversampling, incorporates ML utility loss during training (not just distributional fidelity), Gaussian mixture + log-transform for long-tailed amounts, native mixed-type handling, specifically designed for imbalanced tabular data | Slower training than SDV models, newer library with less community usage | Strong candidate — purpose-built for imbalanced fraud/financial data. Includes a downstream classifier loss that directly optimizes for ML utility |
| **TabDDPM** | Diffusion | State-of-the-art distributional fidelity, highly stable training (avoids GAN mode collapse), naturally models non-normal distributions | Computationally expensive (requires GPU), slower sampling time than GANs | Ultimate benchmark for fidelity — current standard in ML research |

---

## 3. Train each generator (same protocol for fairness)

- [ ] Same train/test split of real data for all 7 generators (hold out a real test set — never train generators on it)
- [ ] Same preprocessing baseline (only what each model requires natively — SDV models auto-handle types)
- [ ] Same random seed / multiple seeds for variance check
- [ ] Same synthetic sample size (usually = size of real training set)
- [ ] Log training time and compute cost per generator
- [ ] For GAN-based models (CTGAN, CopulaGAN, WGAN-GP, CTAB-GAN+): check loss curves for convergence / mode collapse before trusting output
- [ ] Explicitly condition on the fraud label during generation if the library supports it (important given imbalance)
- [ ] For CTAB-GAN+: configure the auxiliary classifier loss weight and minority class oversampling ratio
- [ ] For TabDDPM: ensure adequate noise scheduling steps and time-embeddings for categorical encoding

---

## 4. Fidelity evaluation (real vs synthetic) — do this per generator

### 4.1 Marginal distributions
- [ ] Overlay histograms/KDEs per numeric column (real vs synthetic)
- [ ] KS-test or Jensen-Shannon divergence per numeric column
- [ ] Frequency table comparison per categorical column
- [ ] Total Variation Distance / Chi-square per categorical column

### 4.2 Rare-event preservation (critical for fraud)
- [ ] Does synthetic data preserve the fraud rate (%) close to real?
- [ ] Does it preserve rare flags: `sender_pep=1`, `sender_sanctions=1`, `receiver_sanctions=1`?
- [ ] Does it preserve extreme `Amount` values / tail behavior?

### 4.3 Dependency structure
- [ ] Compare real vs synthetic correlation matrices (numeric) — heatmap diff
- [ ] Compare categorical association matrices (Cramér's V) — heatmap diff
- [ ] PCA / UMAP / t-SNE projection, colored by real vs synthetic — check overlap
- [ ] Compare against the baseline projections from §1.11

### 4.4 Business-rule / logical consistency
- [ ] Verify that all 7 generators ONLY output the *Generator Input Schema* base columns
- [ ] Apply the post-processing pipeline to reconstruct derived columns
- [ ] Re-run all checks from section 1.6 on the *synthetic* data — do derived columns still hold together logically?
- [ ] Check `cross_border_flag`, `currency_mismatch` logic still holds
- [ ] Check transmode one-hot columns still sum to 1 and match `transmode_code`

### 4.5 Discriminator test
- [ ] Train a simple classifier (e.g. logistic regression or XGBoost) to distinguish real vs synthetic rows
- [ ] If it can't beat ~55–60% accuracy, that generator preserved structure well; near-100% accuracy = poor fidelity

### 4.6 Benford's Law preservation
- [ ] Re-run the Benford analysis from §1.9 on each generator's synthetic data
- [ ] Compare first-digit distributions: real vs synthetic (chi-square test, overlay bar chart)
- [ ] Compare MAD scores: real vs synthetic — synthetic should have MAD close to real, not closer to random/uniform

### 4.7 Structuring pattern preservation
- [ ] Re-run the threshold proximity analysis from §1.10 on synthetic data
- [ ] Does synthetic data reproduce the cluster of transactions near NPR 1M threshold?
- [ ] Does it preserve the "round number ratio" seen in real data?

### 4.8 Feature importance preservation
- [ ] Re-train the XGBoost classifier from §1.12 on synthetic data, extract feature importances
- [ ] Compare top-20 feature ranking: real vs synthetic (Spearman rank correlation of importance scores)
- [ ] Features in the "Top-10 Critical Features" list must have similar importance in synthetic data

**Tools:** `sdmetrics` (SDV's evaluation suite), `table-evaluator`, `scipy.stats` for KS-test

---

## 5. Utility evaluation — the most important test for your use case

This tells you whether synthetic data is actually *useful* for training a fraud model, not just statistically similar.

- [ ] **TRTR baseline**: Train fraud classifier on real train set → evaluate on real held-out test set → record AUC, PR-AUC, Recall@fraud, F1
- [ ] **TSTR per generator**: Train the same classifier on each generator's synthetic data → evaluate on the SAME real held-out test set
- [ ] Compare TSTR metrics to TRTR baseline — the generator with the smallest gap wins
- [ ] Pay special attention to **Recall and PR-AUC on the fraud class** (not just accuracy — accuracy is meaningless under imbalance)
- [ ] Optionally: Train on real+synthetic combined (augmentation) and see if it beats TRTR alone — useful if data volume is a concern

---

## 6. Privacy / leakage check

- [ ] Nearest-neighbor distance between each synthetic record and its closest real record — make sure the generator isn't memorizing/copying rows (especially with fields like account numbers)
- [ ] Membership inference risk check if this matters for your compliance context

---

## 7. Practical / operational factors (don't skip these)

- [ ] Training time and compute resources required per generator
- [ ] Stability across multiple runs/seeds (does quality vary a lot run to run? GANs often do)
- [ ] Ease of conditioning generation on the fraud label (needed to control class balance in synthetic output)
- [ ] Maintainability / library support (SDV supports Gaussian Copula, CTGAN, TVAE, CopulaGAN natively; WGAN-GP and TabDDPM need custom PyTorch code; CTAB-GAN+ uses the `ctabganplus` package)

---

## 8. Final decision — scoring matrix

Score each generator 1–5 on each criterion, weight by importance to your project, and sum.

| Criterion | Weight | Gaussian Copula | CTGAN | TVAE | CopulaGAN | WGAN-GP | CTAB-GAN+ | TabDDPM |
|---|---|---|---|---|---|---|---|---|
| Marginal fidelity (KS/JS divergence) | | | | | | | | |
| Correlation/dependency preservation | | | | | | | | |
| Rare-event (fraud class) preservation | | | | | | | | |
| Benford's Law compliance | | | | | | | | |
| Structuring pattern preservation | | | | | | | | |
| Business-rule consistency of derived cols | | | | | | | | |
| Discriminator test (lower detectability = better) | | | | | | | | |
| Feature importance ranking preservation | | | | | | | | |
| TSTR utility (closeness to TRTR, esp. fraud recall) | | | | | | | | |
| Privacy (no near-duplicate real rows) | | | | | | | | |
| Training stability across seeds | | | | | | | | |
| Training time / compute cost | | | | | | | | |
| Implementation effort / library maturity | | | | | | | | |
| **Total** | | | | | | | | |

Pick the highest-scoring generator — but if two are close, prefer the one with the **best fraud-class recall in TSTR**, since that's the metric that matters most for a bank-grade fraud system.

---

## 9. Tooling summary

- `ydata-profiling` / `sweetviz` — fast single-dataset EDA (step 1)
- `sdv` (Synthetic Data Vault) — has built-in Gaussian Copula, CTGAN, TVAE, CopulaGAN implementations, all in one consistent API
- `ctabganplus` — CTAB-GAN+ implementation, install via `pip install ctabganplus`
- `sdmetrics` — automated fidelity + utility scoring between real and synthetic data
- `table-evaluator` — quick visual real-vs-synthetic comparison plots
- `scikit-learn` / `xgboost` — for the discriminator test and TSTR utility test
- `shap` — feature importance explainability (§1.12, §4.8)
- `umap-learn` — dimensionality reduction for class separability (§1.11, §4.3)
- `scipy.stats` — Benford's Law chi-square, KS tests, point-biserial correlation
- Custom implementations (e.g. PyTorch) required for WGAN-GP and TabDDPM on tabular data.

---

## 10. Suggested order of operations

1. Run full EDA on real dataset (Section 1 — all 15 subsections)
2. Define Generator Input Schema based on §1.15 audit (base columns only) in `src/data_contracts.py`
3. Train all 7 generators with identical protocol (Section 3) on the base columns only
4. Reconstruct derived columns for synthetic datasets
5. Run fidelity tests on all 7, including Benford's and structuring checks (Section 4)
6. Run TSTR utility tests on all 7 (Section 5)
7. Run privacy check (Section 6)
8. Fill in scoring matrix (Section 8)
9. Select winner, document reasoning, keep runner-up as backup in case of production issues