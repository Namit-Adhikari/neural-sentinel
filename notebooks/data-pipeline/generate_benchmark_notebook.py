#!/usr/bin/env python3
"""Generate the Phase 2 Benchmark Generators notebook (.ipynb).

Run:  python generate_benchmark_notebook.py
Output: notebooks/data-pipeline/phase2_benchmark_generators.ipynb
"""

import json
import os

def md(source: str) -> dict:
    """Create a markdown cell."""
    lines = source.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

def code(source: str) -> dict:
    """Create a code cell."""
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

cells = []

# ============================================================================
# Cell 1: Title & Methodology
# ============================================================================
cells.append(md("""\
# Phase 2: Generator Benchmark — Synthetic Data Quality Evaluation

**Neural Sentinel** · Data Pipeline · Phase 2

---

## Objective

Benchmark all available synthetic tabular-data generators against the cleaned
transactions dataset.  We follow the evaluation methodology from the
**CTAB-GAN+** paper (Tables 3 / 4), measuring three axes of quality:

| Axis | Metrics |
|---|---|
| **ML Utility** | Accuracy (%), F1-score, ROC-AUC — averaged across 5 classifiers |
| **Statistical Similarity** | Average JSD (categorical), Average Wasserstein Distance (continuous), Correlation Difference |
| **Privacy** | Distance to Closest Record (DCR) |

Each synthesizer is trained on the *real training set*, generates a synthetic
training set of equal size, and the synthetic set is evaluated against the
*same held-out real test set*.  Experiments are repeated across **3 random
seeds** and results are reported as **mean ± std**.

### Synthesizers Benchmarked

| # | Name | Library | Type |
|---|------|---------|------|
| 1 | Gaussian Copula | SDV | Statistical |
| 2 | CTGAN | SDV | GAN |
| 3 | TVAE | SDV | VAE |
| 4 | CopulaGAN | SDV | GAN + Copula |
| 5 | CTAB-GAN+ | ctabganplus | GAN + Classifier |
| 6 | WGAN-GP | Custom (PyTorch) | GAN |
| 7 | TabDDPM | Synthcity | Diffusion |"""))

# ============================================================================
# Cell 2: Imports & Environment Setup
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 2 · Imports & Environment Setup
# ──────────────────────────────────────────────────────────────────────────────
import sys, os, time, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# ── Resolve project root ─────────────────────────────────────────────────────
# Works both locally (notebook lives in notebooks/data-pipeline/) and on Kaggle
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    PROJECT_ROOT = Path("/kaggle/working/neural-sentinel")
    KAGGLE_MODE = True
else:
    PROJECT_ROOT = Path(os.getcwd()).resolve()
    # Walk upward until we find AGENTS.md (project root marker)
    while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
        PROJECT_ROOT = PROJECT_ROOT.parent
    KAGGLE_MODE = False

sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root : {PROJECT_ROOT}")
print(f"Kaggle mode  : {KAGGLE_MODE}")

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
    if GPU_AVAILABLE:
        print(f"GPU detected  : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU detected  : None — using CPU")
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = "cpu"
    print("PyTorch not installed — GPU features disabled")

print("\\n✓ Imports complete")\
"""))

# ============================================================================
# Cell 3: Configuration
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 3 · Configuration
# ──────────────────────────────────────────────────────────────────────────────

# ── Benchmark settings ────────────────────────────────────────────────────────
TARGET_COL     = "is_fraud"          # Binary classification target
TASK           = "classification"    # Auto-detected as classification
TEST_SIZE      = 0.20               # 80 / 20 split
RANDOM_SEEDS   = [42, 123, 7]       # 3 seeds for repeated experiments
PRIMARY_SEED   = 42                 # For the train/test split itself

# ── Columns to DROP before feeding to generators ──────────────────────────────
# High-cardinality identifiers / free-text fields that generators cannot learn
DROP_COLUMNS = [
    "transaction_id",
    "sender_account_id",
    "receiver_account_id",
    "ip_address",
    "transaction_date",
    "transaction_time",
    "ip_country",
    "remittance_corridor",
]

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthesizer registry ─────────────────────────────────────────────────────
# Each entry: (display_name, generator_class_name, kwargs)
# We import lazily inside the benchmark loop so missing libraries skip cleanly.
SYNTHESIZERS = [
    ("Gaussian Copula", "GaussianCopulaGenerator",  {}),
    ("CTGAN",           "CTGANGenerator",            {"epochs": 30}),
    ("TVAE",            "TVAEGenerator",             {"epochs": 30}),
    ("CopulaGAN",       "CopulaGANGenerator",        {"epochs": 30}),
    ("CTAB-GAN+",       "CTABGANPlusGenerator",      {"epochs": 150}),
    ("WGAN-GP",         "WGANGPGenerator",           {"epochs": 30}),
    ("TabDDPM",         "TabDDPMGenerator",          {"n_iter": 1000, "device": DEVICE}),
    # ── Add TabSyn here when a wrapper is available ──
    # ("TabSyn", "TabSynGenerator", {}),
]

print(f"Target column : {TARGET_COL}")
print(f"Test size     : {TEST_SIZE}")
print(f"Random seeds  : {RANDOM_SEEDS}")
print(f"Synthesizers  : {len(SYNTHESIZERS)}")
print(f"Output dir    : {OUTPUT_DIR}")
print("\\n✓ Configuration set")\
"""))

# ============================================================================
# Cell 4: Load Data & Detect Feature Types
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 4 · Load Data & Detect Feature Types
# ──────────────────────────────────────────────────────────────────────────────
%%time

DATA_PATH = PROJECT_ROOT / "data" / "interim" / "transactions.parquet"
if not DATA_PATH.exists():
    # Fallback to CSV in original/
    DATA_PATH = PROJECT_ROOT / "data" / "original" / "transactions.csv"
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded CSV: {DATA_PATH}")
else:
    df_raw = pd.read_parquet(DATA_PATH)
    print(f"Loaded Parquet: {DATA_PATH}")

print(f"Raw shape: {df_raw.shape}")

# ── Drop identifier / high-cardinality columns ───────────────────────────────
existing_drop = [c for c in DROP_COLUMNS if c in df_raw.columns]
df = df_raw.drop(columns=existing_drop, errors="ignore").copy()
print(f"After dropping {len(existing_drop)} ID columns: {df.shape}")

# ── Detect feature types ─────────────────────────────────────────────────────
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
continuous_cols  = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

# Remove target from feature lists (it's categorical but handled separately)
if TARGET_COL in categorical_cols:
    categorical_cols.remove(TARGET_COL)
if TARGET_COL in continuous_cols:
    continuous_cols.remove(TARGET_COL)

print(f"\\nFeature types detected:")
print(f"  Categorical : {len(categorical_cols)} — {categorical_cols}")
print(f"  Continuous  : {len(continuous_cols)} — {continuous_cols}")
print(f"  Target      : {TARGET_COL} (unique values: {df[TARGET_COL].nunique()})")

# ── Class balance ─────────────────────────────────────────────────────────────
print(f"\\nClass distribution:")
print(df[TARGET_COL].value_counts())
print(f"Fraud rate: {df[TARGET_COL].mean():.4%}")\
"""))

# ============================================================================
# Cell 5: Train/Test Split
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 5 · Train / Test Split
# ──────────────────────────────────────────────────────────────────────────────
# Stratified split to preserve class balance — same split used for ALL comparisons

df_train, df_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=PRIMARY_SEED,
    stratify=df[TARGET_COL],
)

print(f"Training set : {df_train.shape}")
print(f"Test set     : {df_test.shape}")
print(f"Train fraud% : {df_train[TARGET_COL].mean():.4%}")
print(f"Test fraud%  : {df_test[TARGET_COL].mean():.4%}")
print(f"\\nSynthetic size will equal training size: {len(df_train)} rows")\
"""))

# ============================================================================
# Cell 6: ML Utility Helpers
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 6 · ML Utility — Classification Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _encode_for_ml(df_in, target, cat_cols, cont_cols):
    \"\"\"Label-encode categoricals + keep continuous columns as-is.
    Returns (X, y) with all-numeric features.
    \"\"\"
    df_proc = df_in.copy()

    # Label-encode every categorical column
    encoders = {}
    for col in cat_cols:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            encoders[col] = le

    feature_cols = [c for c in cat_cols + cont_cols if c in df_proc.columns]
    X = df_proc[feature_cols].fillna(0).values.astype(np.float32)
    y = df_proc[target].values.astype(int)
    return X, y, encoders


def evaluate_ml_utility(train_data, test_data, target, cat_cols, cont_cols):
    \"\"\"Train 5 classifiers on *train_data*, evaluate on *test_data*.

    Returns dict with per-model and averaged metrics:
      accuracy, f1, roc_auc
    \"\"\"
    X_train, y_train, _ = _encode_for_ml(train_data, target, cat_cols, cont_cols)
    X_test,  y_test,  _ = _encode_for_ml(test_data,  target, cat_cols, cont_cols)

    models = {
        "DecisionTree":       DecisionTreeClassifier(random_state=42, max_depth=10),
        "RandomForest":       RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs"),
        "LinearSVM":          CalibratedClassifierCV(
                                  LinearSVC(random_state=42, max_iter=2000, dual="auto"),
                                  cv=3
                              ),
        "MLP":                MLPClassifier(random_state=42, hidden_layer_sizes=(128, 64),
                                           max_iter=300, early_stopping=True),
    }

    results_per_model = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred) * 100   # as percentage
            f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

            results_per_model[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc}
        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")
            results_per_model[name] = {"accuracy": np.nan, "f1": np.nan, "roc_auc": np.nan}

    # Average across all models
    avg = {
        "accuracy": np.nanmean([r["accuracy"] for r in results_per_model.values()]),
        "f1":       np.nanmean([r["f1"]       for r in results_per_model.values()]),
        "roc_auc":  np.nanmean([r["roc_auc"]  for r in results_per_model.values()]),
    }
    return {"per_model": results_per_model, "avg": avg}


print("✓ ML utility helpers defined")\
"""))

# ============================================================================
# Cell 7: Statistical Similarity Helpers
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 7 · Statistical Similarity — JSD, Wasserstein, Correlation
# ──────────────────────────────────────────────────────────────────────────────

# ── Average Jensen-Shannon Divergence (categorical columns) ───────────────────
def compute_avg_jsd(real, syn, cat_cols):
    \"\"\"Average JSD across categorical columns.  Lower is better.\"\"\"
    jsd_vals = []
    for col in cat_cols:
        if col not in real.columns or col not in syn.columns:
            continue
        # Align categories
        all_cats = sorted(set(real[col].dropna().unique()) | set(syn[col].dropna().unique()))
        real_freq = real[col].value_counts(normalize=True).reindex(all_cats, fill_value=0).values
        syn_freq  = syn[col].value_counts(normalize=True).reindex(all_cats, fill_value=0).values
        jsd_vals.append(jensenshannon(real_freq, syn_freq) ** 2)  # squared JSD
    return float(np.mean(jsd_vals)) if jsd_vals else np.nan


# ── Average Wasserstein Distance (continuous columns, MinMax-scaled) ──────────
def compute_avg_wasserstein(real, syn, cont_cols):
    \"\"\"Average Wasserstein-1 distance after MinMax scaling on real data.  Lower is better.\"\"\"
    wd_vals = []
    for col in cont_cols:
        if col not in real.columns or col not in syn.columns:
            continue
        r = real[col].dropna().values.reshape(-1, 1)
        s = syn[col].dropna().values.reshape(-1, 1)
        if len(r) == 0 or len(s) == 0:
            continue
        scaler = MinMaxScaler().fit(r)
        r_scaled = scaler.transform(r).ravel()
        s_scaled = scaler.transform(s).ravel()
        wd_vals.append(wasserstein_distance(r_scaled, s_scaled))
    return float(np.mean(wd_vals)) if wd_vals else np.nan


# ── Theil's U (asymmetric categorical association) ────────────────────────────
def _theils_u(x, y):
    \"\"\"Theil's U: uncertainty coefficient U(X|Y).\"\"\"
    ct = pd.crosstab(x, y)
    # H(X)
    px = ct.sum(axis=1) / ct.sum().sum()
    hx = entropy(px, base=2)
    if hx == 0:
        return 0.0
    # H(X|Y)
    hx_given_y = 0.0
    for j in range(ct.shape[1]):
        col = ct.iloc[:, j]
        col_sum = col.sum()
        if col_sum == 0:
            continue
        p_x_given_yj = col / col_sum
        hx_given_y += (col_sum / ct.sum().sum()) * entropy(p_x_given_yj, base=2)
    return (hx - hx_given_y) / hx


# ── Correlation Ratio (continuous-categorical association) ────────────────────
def _correlation_ratio(cat_series, cont_series):
    \"\"\"Correlation ratio eta — measures how much variance in the continuous
    variable is explained by the categorical grouping.
    \"\"\"
    groups = {}
    for cat_val, cont_val in zip(cat_series, cont_series):
        groups.setdefault(cat_val, []).append(cont_val)
    grand_mean = cont_series.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values())
    ss_total   = sum((v - grand_mean) ** 2 for v in cont_series)
    if ss_total == 0:
        return 0.0
    return np.sqrt(ss_between / ss_total)


# ── Full Correlation Difference ───────────────────────────────────────────────
def compute_correlation_diff(real, syn, cat_cols, cont_cols):
    \"\"\"Frobenius norm of (C_real - C_syn) using Pearson (cont x cont),
    Theil's U (cat x cat), and Correlation Ratio (cont x cat).  Lower is better.
    \"\"\"
    all_cols = [c for c in cont_cols + cat_cols if c in real.columns and c in syn.columns]
    n = len(all_cols)
    if n == 0:
        return np.nan

    def _corr_matrix(df):
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ci, cj = all_cols[i], all_cols[j]
                if i == j:
                    mat[i, j] = 1.0
                elif ci in cont_cols and cj in cont_cols:
                    mat[i, j] = df[[ci, cj]].corr().iloc[0, 1]
                elif ci in cat_cols and cj in cat_cols:
                    mat[i, j] = _theils_u(df[ci].astype(str), df[cj].astype(str))
                elif ci in cont_cols and cj in cat_cols:
                    mat[i, j] = _correlation_ratio(df[cj].astype(str), df[ci].astype(float))
                else:
                    mat[i, j] = _correlation_ratio(df[ci].astype(str), df[cj].astype(float))
        return np.nan_to_num(mat, nan=0.0)

    C_real = _corr_matrix(real)
    C_syn  = _corr_matrix(syn)
    return float(np.linalg.norm(C_real - C_syn, "fro"))


print("✓ Statistical similarity helpers defined")\
"""))

# ============================================================================
# Cell 8: Privacy Metric (DCR)
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 8 · Privacy Metric — Distance to Closest Record (DCR)
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors

def compute_dcr(real, syn, cont_cols, n_samples=2000):
    \"\"\"5th-percentile distance to closest real record.  Higher = more private.\"\"\"
    usable = [c for c in cont_cols if c in real.columns and c in syn.columns]
    if not usable:
        return np.nan

    n = min(n_samples, len(real), len(syn))
    r = real[usable].sample(n, random_state=42).fillna(0)
    s = syn[usable].sample(n, random_state=42).fillna(0)

    scaler = StandardScaler().fit(r)
    r_sc = scaler.transform(r)
    s_sc = scaler.transform(s)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(r_sc)
    dists, _ = nn.kneighbors(s_sc)
    return float(np.percentile(dists, 5))


print("✓ DCR privacy metric defined")\
"""))

# ============================================================================
# Cell 9: Real Baseline Evaluation
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 9 · Real Data Baseline (train-on-real, test-on-real)
# ──────────────────────────────────────────────────────────────────────────────
%%time

print("Computing real-data baseline (train on real, test on real)...")
real_baseline = evaluate_ml_utility(df_train, df_test, TARGET_COL, categorical_cols, continuous_cols)

print(f"\\nReal baseline (averaged across 5 classifiers):")
print(f"  Accuracy : {real_baseline['avg']['accuracy']:.2f}%")
print(f"  F1       : {real_baseline['avg']['f1']:.4f}")
print(f"  ROC-AUC  : {real_baseline['avg']['roc_auc']:.4f}")

print("\\nPer-model breakdown:")
for name, metrics in real_baseline["per_model"].items():
    print(f"  {name:20s}  Acc={metrics['accuracy']:.2f}%  F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")\
"""))

# ============================================================================
# Cell 10: Single Benchmark Run Function
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 10 · Single Benchmark Run
# ──────────────────────────────────────────────────────────────────────────────

def run_single_benchmark(
    name, generator, train_df, test_df, target,
    cat_cols, cont_cols, real_baseline_avg, seed,
):
    \"\"\"Fit a generator, produce synthetic data, evaluate on all 3 axes.

    Returns a flat dict with all metrics for one (synthesizer, seed) pair.
    \"\"\"
    np.random.seed(seed)

    t0 = time.time()

    # ── Fit ────────────────────────────────────────────────────────────────
    logger.info(f"[{name}] seed={seed} — Fitting...")
    generator.fit(train_df)
    fit_time = time.time() - t0

    # ── Generate (synthetic size == training size) ─────────────────────────
    t1 = time.time()
    n_synth = len(train_df)
    logger.info(f"[{name}] seed={seed} — Generating {n_synth} rows...")
    syn_df = generator.generate(n_synth)
    gen_time = time.time() - t1

    # ── ML Utility ─────────────────────────────────────────────────────────
    logger.info(f"[{name}] seed={seed} — Evaluating ML utility...")
    ml_result = evaluate_ml_utility(syn_df, test_df, target, cat_cols, cont_cols)
    ml_avg = ml_result["avg"]

    # Compute abs-difference from real baseline
    acc_diff = abs(real_baseline_avg["accuracy"] - ml_avg["accuracy"])
    f1_diff  = abs(real_baseline_avg["f1"]       - ml_avg["f1"])
    auc_diff = abs(real_baseline_avg["roc_auc"]  - ml_avg["roc_auc"])

    # ── Statistical Similarity ─────────────────────────────────────────────
    logger.info(f"[{name}] seed={seed} — Evaluating statistical similarity...")
    avg_jsd = compute_avg_jsd(train_df, syn_df, cat_cols)
    avg_wd  = compute_avg_wasserstein(train_df, syn_df, cont_cols)
    corr_diff = compute_correlation_diff(train_df, syn_df, cat_cols, cont_cols)

    # ── Privacy ────────────────────────────────────────────────────────────
    logger.info(f"[{name}] seed={seed} — Evaluating privacy (DCR)...")
    dcr = compute_dcr(train_df, syn_df, cont_cols)

    total_time = time.time() - t0

    result = {
        "synthesizer": name,
        "seed": seed,
        # ML Utility — absolute metric values (on synthetic-trained models)
        "accuracy_syn": ml_avg["accuracy"],
        "f1_syn": ml_avg["f1"],
        "roc_auc_syn": ml_avg["roc_auc"],
        # ML Utility — differences from real baseline
        "accuracy_diff": acc_diff,
        "f1_diff": f1_diff,
        "roc_auc_diff": auc_diff,
        # Statistical Similarity
        "avg_jsd": avg_jsd,
        "avg_wd": avg_wd,
        "corr_diff": corr_diff,
        # Privacy
        "dcr_5th": dcr,
        # Timing
        "fit_time_s": fit_time,
        "gen_time_s": gen_time,
        "total_time_s": total_time,
    }

    logger.info(
        f"[{name}] seed={seed} DONE — "
        f"Acc={ml_avg['accuracy']:.1f}%  F1={ml_avg['f1']:.3f}  AUC={ml_avg['roc_auc']:.3f}  "
        f"JSD={avg_jsd:.4f}  WD={avg_wd:.4f}  Corr={corr_diff:.4f}  DCR={dcr:.4f}  "
        f"Time={total_time:.1f}s"
    )

    # ── Save synthetic data for this run ───────────────────────────────────
    synth_dir = OUTPUT_DIR / name.replace(" ", "_").replace("+", "plus")
    synth_dir.mkdir(parents=True, exist_ok=True)
    syn_df.to_csv(synth_dir / f"synthetic_seed{seed}.csv", index=False)

    return result


print("✓ run_single_benchmark() defined")\
"""))

# ============================================================================
# Cell 11: Full Benchmark Loop
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 11 · Full Benchmark — Run All Synthesizers x All Seeds
# ──────────────────────────────────────────────────────────────────────────────
%%time

all_results = []
failed_synthesizers = []

for synth_name, class_name, kwargs in SYNTHESIZERS:
    print(f"\\n{'='*70}")
    print(f"  BENCHMARKING: {synth_name}")
    print(f"{'='*70}")

    # ── Lazy import of the generator class ────────────────────────────────
    try:
        from src.generation import (
            CTGANGenerator, TVAEGenerator, CopulaGANGenerator,
            GaussianCopulaGenerator, CTABGANPlusGenerator,
            WGANGPGenerator, TabDDPMGenerator,
        )
        gen_cls = {
            "CTGANGenerator": CTGANGenerator,
            "TVAEGenerator": TVAEGenerator,
            "CopulaGANGenerator": CopulaGANGenerator,
            "GaussianCopulaGenerator": GaussianCopulaGenerator,
            "CTABGANPlusGenerator": CTABGANPlusGenerator,
            "WGANGPGenerator": WGANGPGenerator,
            "TabDDPMGenerator": TabDDPMGenerator,
        }.get(class_name)

        if gen_cls is None:
            raise ImportError(f"Unknown generator class: {class_name}")

    except ImportError as e:
        logger.warning(f"SKIPPING {synth_name} — import failed: {e}")
        failed_synthesizers.append((synth_name, str(e)))
        continue

    # ── Run across all seeds ──────────────────────────────────────────────
    for seed in RANDOM_SEEDS:
        try:
            generator = gen_cls(**kwargs)
            result = run_single_benchmark(
                name=synth_name,
                generator=generator,
                train_df=df_train,
                test_df=df_test,
                target=TARGET_COL,
                cat_cols=categorical_cols,
                cont_cols=continuous_cols,
                real_baseline_avg=real_baseline["avg"],
                seed=seed,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"[{synth_name}] seed={seed} FAILED: {e}")
            failed_synthesizers.append((synth_name, f"seed={seed}: {e}"))
            continue

print(f"\\n{'='*70}")
print(f"Benchmark complete — {len(all_results)} successful runs")
if failed_synthesizers:
    print(f"\\nFailed synthesizers:")
    for name, err in failed_synthesizers:
        print(f"  ✗ {name}: {err}")\
"""))

# ============================================================================
# Cell 12: Aggregate Results (Mean +/- Std)
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 12 · Aggregate Results
# ──────────────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(all_results)
print(f"Raw results: {results_df.shape[0]} rows across {results_df['synthesizer'].nunique()} synthesizers")
results_df.head(10)\
"""))

# ============================================================================
# Cell 13: Build Final Results Table
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 13 · Final Results Table (CTAB-GAN+ Paper Format)
# ──────────────────────────────────────────────────────────────────────────────

# Metrics to aggregate (column_name, display_name, format_spec, lower_is_better)
METRICS = [
    ("accuracy_diff", "Accuracy Diff (%)", ".2f", True),
    ("f1_diff",       "F1-score Diff",     ".4f", True),
    ("roc_auc_diff",  "AUC Diff",          ".4f", True),
    ("avg_jsd",       "Avg JSD",           ".4f", True),
    ("avg_wd",        "Avg WD",            ".4f", True),
    ("corr_diff",     "Diff. Corr.",       ".4f", True),
    ("dcr_5th",       "DCR (5th %ile)",    ".4f", False),
    ("total_time_s",  "Time (s)",          ".1f", None),
]

# ── Compute mean +/- std per synthesizer ──────────────────────────────────────
def fmt_mean_std(series, spec):
    m = series.mean()
    s = series.std()
    return f"{m:{spec}} +/- {s:{spec}}"

table_rows = []
for synth_name in results_df["synthesizer"].unique():
    subset = results_df[results_df["synthesizer"] == synth_name]
    row = {"Method": synth_name}
    for col, display, spec, _ in METRICS:
        row[display] = fmt_mean_std(subset[col], spec)
    table_rows.append(row)

final_table = pd.DataFrame(table_rows)
final_table = final_table.set_index("Method")

print("\\n" + "="*100)
print("  FINAL BENCHMARK RESULTS — Classification (is_fraud)")
print("  Metrics show abs-difference from real baseline (lower = better, except DCR)")
print("="*100 + "\\n")
print(final_table.to_string())

# Also include the raw metric values (synthetic model performance) for reference
print("\\n\\n" + "="*100)
print("  SYNTHETIC MODEL ABSOLUTE PERFORMANCE")
print("="*100 + "\\n")

ABS_METRICS = [
    ("accuracy_syn", "Accuracy (%)", ".2f"),
    ("f1_syn",       "F1-score",     ".4f"),
    ("roc_auc_syn",  "ROC-AUC",     ".4f"),
]

abs_rows = []
# Add real baseline as first row
abs_rows.append({
    "Method": "Real (baseline)",
    **{d: f"{real_baseline['avg'][col.replace('_syn','')]:{s}}" for col, d, s in ABS_METRICS}
})
for synth_name in results_df["synthesizer"].unique():
    subset = results_df[results_df["synthesizer"] == synth_name]
    row = {"Method": synth_name}
    for col, display, spec in ABS_METRICS:
        row[display] = fmt_mean_std(subset[col], spec)
    abs_rows.append(row)

abs_table = pd.DataFrame(abs_rows).set_index("Method")
print(abs_table.to_string())\
"""))

# ============================================================================
# Cell 14: Visualisations (Bar Plots)
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 14 · Visualisations — Bar Plots
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Generator Benchmark — Metric Comparison (lower = better, except DCR)",
             fontsize=16, fontweight="bold", y=1.02)

plot_metrics = [
    ("accuracy_diff", "Accuracy Diff (%)", True),
    ("f1_diff",       "F1-score Diff",     True),
    ("roc_auc_diff",  "AUC Diff",          True),
    ("avg_jsd",       "Avg JSD",           True),
    ("avg_wd",        "Avg WD",            True),
    ("dcr_5th",       "DCR (5th %ile)",    False),
]

palette = sns.color_palette("husl", n_colors=results_df["synthesizer"].nunique())

for ax, (col, title, lower_better) in zip(axes.ravel(), plot_metrics):
    synth_order = results_df.groupby("synthesizer")[col].mean()
    if lower_better:
        synth_order = synth_order.sort_values()
    else:
        synth_order = synth_order.sort_values(ascending=False)

    sns.barplot(
        data=results_df,
        x="synthesizer",
        y=col,
        order=synth_order.index,
        palette=palette,
        ax=ax,
        errorbar="sd",
        capsize=0.15,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "benchmark_barplots.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\\n✓ Bar plots saved to {OUTPUT_DIR / 'benchmark_barplots.png'}")\
"""))

# ============================================================================
# Cell 15: Radar Chart
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 15 · Radar Chart — Multi-Axis Synthesizer Comparison
# ──────────────────────────────────────────────────────────────────────────────

radar_metrics = ["accuracy_diff", "f1_diff", "roc_auc_diff", "avg_jsd", "avg_wd", "corr_diff"]
radar_labels  = ["Acc Diff", "F1 Diff", "AUC Diff", "JSD", "WD", "Corr Diff"]

# Normalise each metric to [0, 1] (min-max across synthesizers)
agg = results_df.groupby("synthesizer")[radar_metrics].mean()

normed = agg.copy()
for col in radar_metrics:
    mn, mx = agg[col].min(), agg[col].max()
    normed[col] = (agg[col] - mn) / (mx - mn + 1e-10)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_metrics = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
colors = sns.color_palette("husl", n_colors=len(normed))

for idx, (synth_name, row) in enumerate(normed.iterrows()):
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, "o-", linewidth=2, label=synth_name, color=colors[idx])
    ax.fill(angles, values, alpha=0.08, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_title("Synthesizer Quality Radar\\n(closer to centre = better)",
             fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "benchmark_radar.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Radar chart saved to {OUTPUT_DIR / 'benchmark_radar.png'}")\
"""))

# ============================================================================
# Cell 16: Metric Heatmap
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 16 · Metric Heatmap
# ──────────────────────────────────────────────────────────────────────────────

heatmap_metrics = ["accuracy_diff", "f1_diff", "roc_auc_diff", "avg_jsd", "avg_wd", "corr_diff", "dcr_5th"]
heatmap_labels  = ["Acc Diff", "F1 Diff", "AUC Diff", "JSD", "WD", "Corr Diff", "DCR"]

hm_data = results_df.groupby("synthesizer")[heatmap_metrics].mean()
hm_data.columns = heatmap_labels

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    hm_data,
    annot=True,
    fmt=".4f",
    cmap="YlOrRd_r",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Metric Value"},
)
ax.set_title("Generator Benchmark Heatmap\\n(lower = better, except DCR)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Synthesizer")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "benchmark_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Heatmap saved to {OUTPUT_DIR / 'benchmark_heatmap.png'}")\
"""))

# ============================================================================
# Cell 17: Save Outputs
# ============================================================================
cells.append(code("""\
# ──────────────────────────────────────────────────────────────────────────────
# Cell 17 · Save Outputs
# ──────────────────────────────────────────────────────────────────────────────

# 1. Raw results CSV
results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
print(f"✓ Raw results    -> {OUTPUT_DIR / 'results.csv'}")

# 2. Final table CSV
final_table.to_csv(OUTPUT_DIR / "final_table.csv")
print(f"✓ Final table    -> {OUTPUT_DIR / 'final_table.csv'}")

# 3. Final table Markdown
md_str = final_table.to_markdown()
(OUTPUT_DIR / "final_table.md").write_text(md_str, encoding="utf-8")
print(f"✓ Final table MD -> {OUTPUT_DIR / 'final_table.md'}")

# 4. Metrics JSON
metrics_json = {
    "real_baseline": real_baseline["avg"],
    "synthesizers": {},
}
for synth_name in results_df["synthesizer"].unique():
    subset = results_df[results_df["synthesizer"] == synth_name]
    metrics_json["synthesizers"][synth_name] = {
        col: {"mean": float(subset[col].mean()), "std": float(subset[col].std())}
        for col in ["accuracy_diff", "f1_diff", "roc_auc_diff",
                     "avg_jsd", "avg_wd", "corr_diff", "dcr_5th",
                     "total_time_s"]
    }

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"✓ Metrics JSON   -> {OUTPUT_DIR / 'metrics.json'}")

print("\\n✓ All outputs saved successfully")\
"""))

# ============================================================================
# Cell 18: Conclusion
# ============================================================================
cells.append(md("""\
## Conclusion & Next Steps

### What This Notebook Produced

1. **Benchmark comparison** of 7 synthesizers across ML utility, statistical
   similarity, and privacy metrics
2. **Final results table** matching the CTAB-GAN+ paper format (Tables 3/4)
3. **Visualisations** — bar plots, radar chart, and heatmap for quick comparison
4. **Saved outputs** — CSV, JSON, and Markdown tables for downstream use

### Key Metrics Interpretation

| Metric | What It Measures | Ideal |
|---|---|---|
| Accuracy / F1 / AUC Diff | How close synthetic-trained models are to real-trained | → 0 |
| Avg JSD | Categorical distribution fidelity | → 0 |
| Avg WD | Continuous distribution fidelity | → 0 |
| Diff. Corr. | Feature correlation preservation | → 0 |
| DCR (5th %ile) | Privacy — distance from synthetic to nearest real | → high |

### Next Steps

- **Phase 3**: Use the winning synthesizer to generate the full 5M-row dataset
- Add **TabSyn** wrapper when library becomes available
- Consider per-column analysis for generators with high statistical divergence
- Run extended benchmark with 5 seeds for publication-quality results"""))


# ===========================================================================
# Assemble notebook
# ===========================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "phase2_benchmark_generators.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {output_path}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
