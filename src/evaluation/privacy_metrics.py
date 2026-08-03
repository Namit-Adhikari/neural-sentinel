import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_dcr(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, continuous_cols: list, n_samples=1000) -> float:
    """Computes the Distance to Closest Record (DCR) for privacy evaluation."""
    # Subsample for speed
    n_samples = min(n_samples, len(real_data), len(synthetic_data))
    real_sample = real_data[continuous_cols].sample(n_samples, random_state=42)
    syn_sample = synthetic_data[continuous_cols].sample(n_samples, random_state=42)
    
    # Normalize
    real_mean = real_sample.mean()
    real_std = real_sample.std().replace(0, 1)
    
    real_norm = (real_sample - real_mean) / real_std
    syn_norm = (syn_sample - real_mean) / real_std
    
    # Fill NAs
    real_norm = real_norm.fillna(0)
    syn_norm = syn_norm.fillna(0)
    
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(real_norm)
    distances, _ = nn.kneighbors(syn_norm)
    
    # Return 5th percentile distance as a conservative privacy metric
    return float(np.percentile(distances, 5))
