# Phase 1 EDA summary

## Scope and provenance

This is the Developer 1 Phase 1 profile of the supplied, fully synthetic
hackathon bundle in `data/original/original_data/`. It is intentionally a
source-data assessment, not a canonical transformation or generator benchmark;
those activities belong to later phases.

## Files inspected

| File | Rows | Role in Phase 1 |
|---|---:|---|
| `transactions.csv` | 100,222 | Primary transaction source and enriched operational fields |
| `accounts.csv` | 65,339 | Account/KYC reference records |
| `graph_edges.csv` | 100,222 | Directed sender-to-receiver edge list |
| `ml_features.csv` | 100,222 | Model-ready features and the supplied `is_suspicious_tx` label |
| `reports/*.xml` | 276 files | Sample transaction-report artefacts; not used for tabular EDA |

The dataset README explicitly identifies the records as synthetic. No claim in
this document should be interpreted as describing real customers or banking
activity.

## Material findings

- The source schema does **not** match the canonical contract. For example, it
  uses `Sender_account`, `Receiver_account`, `Date`, `Time`, `Payment_type`,
  and `amount_local_npr`; Phase 2 must map rather than rename in place.
- The transaction source has 100,222 rows, no empty cells, and 55 columns. It
  includes substantial engineered fields (`log_amount`, threshold flags,
  velocity fields, and one-hot transaction-mode columns) spread across the
  source files. These must be kept out of the canonical raw transaction
  contract unless deliberately derived downstream.
- `date_transaction` spans 2022-10-07 10:35:19 through 2022-11-06 21:04:35,
  but only 10 distinct calendar dates occur. Time coverage and date coverage
  are therefore inconsistent and must be validated before synthesizing a
  one-year history.
- Amounts are in NPR after conversion (`amount_local_npr`): median NPR
  1,238,979.56, mean NPR 1,769,234.80, 99th percentile NPR 9,221,958.84, and
  maximum NPR 552,796,380.22. The strong right tail warrants log-scale plots
  and robust summaries.
- The label appears only in `ml_features.csv`: 336 / 100,222 rows (0.3353%)
  are flagged `is_suspicious_tx=1`. This is suitable for stratified reporting,
  but is a suspicious-activity label, not confirmed fraud.
- Flagged rows have a higher mean amount (NPR 5.49M vs NPR 1.76M), greater
  cross-border rate (16.37% vs 10.10%), and greater currency-mismatch rate
  (14.29% vs 11.69%) than unflagged rows. These are descriptive associations,
  not causal evidence.
- The graph has 22,310 sender accounts, 46,586 receiver accounts, 50,586
  distinct directed pairs, 8,440 repeated pairs, maximum sender activity of
  265 transactions, and maximum receiver activity of 241 transactions. This
  supports later graph analysis but does not itself establish AML typologies.
- Geography is dominated by UK-located source fields rather than Nepal. It is
  therefore a useful structural seed but not a direct representation of Nepali
  banking geography. The Phase 2 canonical transformation must document its
  Nepal-context enrichment separately.

## Phase 2 hand-off

1. Preserve originals; create new canonical tables in `data/interim/`.
2. Join the label from `ml_features.csv` by stable row identity only after
   verifying `row_index` alignment and uniqueness.
3. Map source values through an explicit, reviewable mapping table. Do not
   infer unsupported Nepal-specific fields from UK-heavy geography.
4. Recompute temporal and velocity-derived fields after producing a consistent
   timestamp timeline; do not carry source engineered features as ground truth.
5. Validate source-to-canonical row counts, key uniqueness, amount positivity,
   categorical mappings, and cross-border consistency before generator work.

## Reproducibility

The six notebooks in this directory reproduce the profile against a Kaggle
input copy of the source bundle. They locate the bundle recursively under
`/kaggle/input` (or `data/original` locally) and provide a deliberately small
fallback only when no input is available. The fallback is for notebook smoke
testing, never for reporting results.
