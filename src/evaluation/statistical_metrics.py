import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def compute_ks_test(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, continuous_cols: list) -> dict:
    """Computes the Kolmogorov-Smirnov test for continuous columns."""
    results = {}
    for col in continuous_cols:
        if col in real_data.columns and col in synthetic_data.columns:
            real_vals = real_data[col].dropna()
            syn_vals = synthetic_data[col].dropna()
            stat, p_value = ks_2samp(real_vals, syn_vals)
            results[col] = {"statistic": stat, "p_value": p_value}
    return results

def compute_correlation_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, continuous_cols: list) -> float:
    """Computes the Frobenius norm of the difference between correlation matrices."""
    real_corr = real_data[continuous_cols].corr().fillna(0)
    syn_corr = synthetic_data[continuous_cols].corr().fillna(0)
    diff = real_corr - syn_corr
    return np.linalg.norm(diff.values, 'fro')
