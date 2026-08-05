"""
src/evaluation/statistical_metrics.py
----------------------------------------
Statistical similarity metrics for synthetic data evaluation.

Implements the metrics from the CTAB-GAN+ paper (Benchmark_pipeline.md):
  - KS test per continuous column
  - Average Wasserstein Distance (continuous columns)
  - Average Jensen-Shannon Divergence (categorical columns)
  - Correlation matrix difference (Pearson, Theil's U, Correlation Ratio)
  - Frobenius norm of correlation difference
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ---------------------------------------------------------------------------
# KS Test (original function — preserved)
# ---------------------------------------------------------------------------

def compute_ks_test(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    continuous_cols: list[str],
) -> dict[str, dict]:
    """Kolmogorov-Smirnov test per continuous column.

    Returns
    -------
    dict mapping column → {statistic, p_value}
    """
    results = {}
    for col in continuous_cols:
        if col in real_data.columns and col in synthetic_data.columns:
            real_vals = real_data[col].dropna()
            syn_vals  = synthetic_data[col].dropna()
            stat, p_value = ks_2samp(real_vals, syn_vals)
            results[col] = {"statistic": float(stat), "p_value": float(p_value)}
    return results


# ---------------------------------------------------------------------------
# Average Wasserstein Distance
# ---------------------------------------------------------------------------

def compute_avg_wasserstein(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    continuous_cols: list[str],
) -> float:
    """Average Wasserstein distance across continuous columns.

    Scales each column to [0,1] using MinMaxScaler fitted on real data,
    then computes W1 distance per column and averages.

    Returns
    -------
    float — lower is better.
    """
    distances = []
    for col in continuous_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            continue
        real_vals = real_data[col].dropna().values.reshape(-1, 1)
        syn_vals  = synthetic_data[col].dropna().values.reshape(-1, 1)
        if len(real_vals) == 0 or len(syn_vals) == 0:
            continue
        scaler = MinMaxScaler()
        real_scaled = scaler.fit_transform(real_vals).flatten()
        syn_scaled  = scaler.transform(np.clip(syn_vals, real_vals.min(), real_vals.max())).flatten()
        distances.append(wasserstein_distance(real_scaled, syn_scaled))

    return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# Average Jensen-Shannon Divergence (categorical)
# ---------------------------------------------------------------------------

def compute_avg_jsd(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_cols: list[str],
) -> float:
    """Average Jensen-Shannon Divergence across categorical columns.

    Returns
    -------
    float — lower is better (0 = identical distributions).
    """
    jsds = []
    for col in categorical_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            continue
        real_vc = real_data[col].astype(str).value_counts(normalize=True)
        syn_vc  = synthetic_data[col].astype(str).value_counts(normalize=True)

        # Union of categories
        all_cats = set(real_vc.index) | set(syn_vc.index)
        p = np.array([real_vc.get(c, 0.0) for c in all_cats], dtype=float)
        q = np.array([syn_vc.get(c, 0.0)  for c in all_cats], dtype=float)

        # Smooth to avoid log(0)
        eps = 1e-10
        p = p + eps;  p /= p.sum()
        q = q + eps;  q /= q.sum()

        m = 0.5 * (p + q)
        jsd = 0.5 * np.sum(p * np.log(p / m + eps)) + 0.5 * np.sum(q * np.log(q / m + eps))
        jsds.append(float(np.clip(jsd, 0.0, math.log(2))))

    return float(np.mean(jsds)) if jsds else 0.0


# ---------------------------------------------------------------------------
# Theil's U (categorical → categorical association)
# ---------------------------------------------------------------------------

def theils_u(x: pd.Series, y: pd.Series) -> float:
    """Theil's U asymmetric uncertainty coefficient: U(x|y).

    Returns a value in [0, 1]. 1 = x fully determined by y.
    """
    from collections import Counter

    def entropy(series: pd.Series) -> float:
        counts = Counter(series.astype(str))
        total = sum(counts.values())
        return -sum((c / total) * math.log(c / total + 1e-10) for c in counts.values())

    h_x = entropy(x)
    if h_x == 0:
        return 1.0

    # Conditional entropy H(x|y)
    xy = pd.DataFrame({"x": x.astype(str), "y": y.astype(str)})
    h_xy = 0.0
    n = len(xy)
    for y_val, grp in xy.groupby("y"):
        p_y = len(grp) / n
        h_xy += p_y * entropy(grp["x"])

    return (h_x - h_xy) / h_x


# ---------------------------------------------------------------------------
# Correlation Ratio (continuous ~ categorical)
# ---------------------------------------------------------------------------

def correlation_ratio(categorical: pd.Series, continuous: pd.Series) -> float:
    """Correlation ratio η² measuring association between categorical and continuous."""
    cats  = categorical.astype(str)
    cont  = pd.to_numeric(continuous, errors="coerce").dropna()
    cats  = cats[cont.index]

    grand_mean = cont.mean()
    numerator  = sum(
        len(grp) * (grp.mean() - grand_mean) ** 2
        for _, grp in cont.groupby(cats)
    )
    denominator = ((cont - grand_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Full correlation matrix (Pearson + Theil's U + Correlation Ratio)
# ---------------------------------------------------------------------------

def compute_correlation_matrices(
    df: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Build a square association matrix over all columns.

    - continuous × continuous : Pearson
    - categorical × categorical : Theil's U (averaged both ways)
    - continuous × categorical : Correlation Ratio
    """
    all_cols = list(continuous_cols) + list(categorical_cols)
    all_cols = [c for c in all_cols if c in df.columns]
    n = len(all_cols)
    mat = np.zeros((n, n))

    for i, ci in enumerate(all_cols):
        for j, cj in enumerate(all_cols):
            if i == j:
                mat[i, j] = 1.0
                continue
            ci_is_cat = ci in categorical_cols
            cj_is_cat = cj in categorical_cols

            try:
                if not ci_is_cat and not cj_is_cat:
                    # Pearson
                    xi = pd.to_numeric(df[ci], errors="coerce").dropna()
                    xj = pd.to_numeric(df[cj], errors="coerce").dropna()
                    common = xi.index.intersection(xj.index)
                    if len(common) < 5:
                        mat[i, j] = 0.0
                    else:
                        mat[i, j] = float(np.corrcoef(xi[common], xj[common])[0, 1])

                elif ci_is_cat and cj_is_cat:
                    # Average Theil's U (both directions)
                    u1 = theils_u(df[ci].dropna(), df[cj].dropna())
                    u2 = theils_u(df[cj].dropna(), df[ci].dropna())
                    mat[i, j] = (u1 + u2) / 2.0

                else:
                    # Correlation Ratio
                    cat_col = ci if ci_is_cat else cj
                    con_col = cj if ci_is_cat else ci
                    mat[i, j] = correlation_ratio(df[cat_col], df[con_col])
            except Exception:
                mat[i, j] = 0.0

    return pd.DataFrame(mat, index=all_cols, columns=all_cols)


# ---------------------------------------------------------------------------
# Correlation Distance (original function — extended)
# ---------------------------------------------------------------------------

def compute_correlation_distance(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> float:
    """Frobenius norm of (real_corr - synthetic_corr) using full mixed-type matrix.

    Parameters
    ----------
    continuous_cols : list[str]
        Numeric columns.
    categorical_cols : list[str], optional
        Categorical columns. If None, only Pearson correlation is used
        (original behaviour).

    Returns
    -------
    float — lower is better.
    """
    if categorical_cols is None:
        # Original behaviour: Pearson only
        real_corr = real_data[continuous_cols].corr().fillna(0)
        syn_corr  = synthetic_data[continuous_cols].corr().fillna(0)
        diff = real_corr - syn_corr
        return float(np.linalg.norm(diff.values, "fro"))

    # Full mixed-type matrix
    real_mat = compute_correlation_matrices(real_data, continuous_cols, categorical_cols)
    syn_mat  = compute_correlation_matrices(synthetic_data, continuous_cols, categorical_cols)
    diff = real_mat.values - syn_mat.values
    return float(np.linalg.norm(diff, "fro"))
