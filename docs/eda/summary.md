# EDA Summary and Findings

## 1. Resolution of Working Assumptions
*   **A1 (Source file):** Replaced. The source is actually 4 CSV files (`transactions.csv`, `accounts.csv`, `ml_features.csv`, `graph_edges.csv`) + XML reports, not a single file.
*   **A2 (Row count):** Confirmed as 100,223 rows across the main CSVs.
*   **A3 (Suspicious rate):** Target is explicitly labeled in `ml_features.csv` as `is_suspicious_tx`. Rate is evaluated in the notebooks.
*   **A4 (Column candidates):** Partially true, but actual schema has 57 columns including engineered features and one-hot transmodes.
*   **A5 (Temporal span):** The `Date` and `Time` columns show a specific temporal duration evaluated in notebook 04.
*   **A6-A10:** Assumed absence of AML typologies and networks is false. Graph edges are provided explicitly.

## 2. Schema Gap Analysis
The original data lacks the canonical schema columns precisely. We must write a mapping in `src/cleaning.py` to translate `Sender_account`, `amount_local_npr`, `Payment_currency`, `Sender_bank_location` to `sender_account_id`, `amount_npr`, `original_currency`, `sender_country`. 

## 3. Next Steps (Generator Strategy)
Since the dataset is 100K rows, our generator will learn from it to produce 5M rows following the canonical schema.
