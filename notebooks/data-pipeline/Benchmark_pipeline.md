# Generic Synthetic Tabular Data Evaluation Pipeline

## Objective

Build a reusable pipeline that evaluates the quality of synthetic
tabular data for **one local dataset** following the evaluation
methodology used in the CTAB-GAN+ paper. The pipeline must be
synthesizer-agnostic.

## Workflow

``` text
Dataset
  │
  ├── Load & Validate
  ├── Detect Feature Types
  ├── Train/Test Split
  ├── Train Synthesizer
  ├── Generate Synthetic Training Set
  ├── Evaluate ML Utility
  ├── Evaluate Statistical Similarity
  ├── Evaluate Correlation Preservation
  ├── Repeat N Seeds
  └── Produce Final Results Table
```

## Dataset Requirements

-   CSV input
-   Configurable target column
-   Automatic task detection:
    -   Classification
    -   Regression
-   Detect:
    -   Continuous
    -   Categorical
    -   Mixed (optional)

## Directory Structure

``` text
project/
├── config/
├── data/
├── synthesizers/
│   ├── base.py           # BaseSynthesizer interface
│   ├── ctabgan_plus.py
│   ├── ctgan.py
│   ├── tvae.py
│   ├── tab_ddpm.py       # TabDDPM wrapper
│   └── tabsyn.py
├── evaluation/
│   ├── ml_utility.py
│   ├── statistics.py
│   ├── correlation.py
│   └── report.py
├── metrics/
├── outputs/
│   └── <synthesizer_name>/   # one subdirectory per synthesizer
│       ├── synthetic.csv
│       ├── metrics.json
│       └── results.csv
├── logs/
└── run_pipeline.py
```

## Train/Test Split

Classification: - 80/20 - stratified split

Regression: - 80/20 random split

Fix random seed.

## Synthesizer Interface

``` python
class BaseSynthesizer:
    def fit(self, train_df): ...
    def sample(self, n_rows): ...
```

Examples: - CTAB-GAN+ - CTGAN - TVAE - TabDDPM - TabSyn

### TabDDPM Notes

TabDDPM uses a multinomial diffusion process for categorical columns
and Gaussian diffusion for continuous columns. It requires an additional
dependency (`tab_ddpm` or the original repo implementation). The
synthesizer wrapper must handle its MLP-based denoising network
internally; the `fit`/`sample` interface is identical to other synthesizers.

Key hyperparameters to expose in config:
- `num_timesteps` (default: 1000)
- `lr` (default: 0.002)
- `weight_decay` (default: 0.0)
- `batch_size` (default: 4096)
- `num_mlp_layers` (default: 3)
- `d_layers` (default: [256, 256])
- `gaussian_loss_type` (default: `mse`)

Synthetic size must equal training size.

## Machine Learning Utility

### Classification

Train on: - Real train - Synthetic train

Test on: - Same real test set

Models: - Decision Tree - Random Forest - Logistic Regression - Linear
SVM - MLP

Metrics: - Accuracy - F1 - ROC-AUC

Difference:

    abs(real_metric - synthetic_metric)

Average over all models.

### Regression

Models: - Linear Regression - Ridge - Lasso - Bayesian Ridge

Metrics: - MAPE - Explained Variance (EVS) - R²

Difference:

    abs(real_metric - synthetic_metric)

Average over all regressors.

## Statistical Similarity

### Average JSD

For each categorical column: - Compute probability distributions -
Jensen-Shannon Divergence - Average over categorical columns

### Average Wasserstein Distance

For each continuous column: - Fit MinMaxScaler on real data - Apply same
scaler to synthetic - Compute Wasserstein Distance - Average

### Correlation Difference

Continuous: - Pearson

Categorical: - Theil's U

Mixed: - Correlation Ratio

Compute:

    norm(C_real - C_syn)

Lower is better.

## Repeated Experiments

Repeat: - 3 or 5 seeds

Store every metric.

Compute: - Mean - Standard deviation

Format:

    5.23 ± 1.49

## Final Output Tables

### Classification

  ------------------------------------------------------------------------------------
  Method      Accuracy (%)   F1-score   AUC        Avg JSD    Avg WD     Diff. Corr.
  ----------- -------------- ---------- ---------- ---------- ---------- -------------
  CTAB-GAN+   mean±std       mean±std   mean±std   mean±std   mean±std   mean±std
  CTGAN       mean±std       mean±std   mean±std   mean±std   mean±std   mean±std
  TVAE        mean±std       mean±std   mean±std   mean±std   mean±std   mean±std
  TabDDPM     mean±std       mean±std   mean±std   mean±std   mean±std   mean±std
  TabSyn      mean±std       mean±std   mean±std   mean±std   mean±std   mean±std

  ------------------------------------------------------------------------------------

### Regression

  -------------------------------------------------------------------------------------
  Method      MAPE       EVS        R²         Avg JSD     Avg WD     Diff. Corr.
  ----------- ---------- ---------- ---------- ----------- ---------- -----------------
  CTAB-GAN+   mean±std   mean±std   mean±std   mean±std    mean±std   mean±std
  CTGAN       mean±std   mean±std   mean±std   mean±std    mean±std   mean±std
  TVAE        mean±std   mean±std   mean±std   mean±std    mean±std   mean±std
  TabDDPM     mean±std   mean±std   mean±std   mean±std    mean±std   mean±std
  TabSyn      mean±std   mean±std   mean±std   mean±std    mean±std   mean±std

  -------------------------------------------------------------------------------------

## Output Files

-   synthetic.csv
-   metrics.json
-   results.csv
-   final_table.csv
-   final_table.md

## Configuration Example

``` yaml
dataset: data/dataset.csv
target: target
task: auto
test_size: 0.2
random_seed: 42
runs: 5
synthesizers:
  - name: CTAB-GAN+
  - name: CTGAN
  - name: TVAE
  - name: TabDDPM
    params:
      num_timesteps: 1000
      lr: 0.002
      batch_size: 4096
      num_mlp_layers: 3
      d_layers: [256, 256]
      gaussian_loss_type: mse
  - name: TabSyn
```

## Logging

Record: - preprocessing - training - sampling - evaluation - runtime -
failures

## Acceptance Criteria

-   Same real test set used for all comparisons.
-   Synthetic size equals training size.
-   Metrics averaged across ML models.
-   Results averaged across random seeds.
-   Lower values indicate better performance.
-   Final tables match the layout of Tables 3/4 in the CTAB-GAN+ paper.