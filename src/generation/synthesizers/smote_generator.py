"""
src/generation/synthesizers/smote_generator.py
-----------------------------------------------
SMOTE-based synthetic data generator.

Wraps imbalanced-learn's SMOTE / SMOTENC to generate an arbitrary number of
synthetic tabular rows by learning the local neighbourhood structure of the
seed data and interpolating new samples from it.

Strategy
--------
1. Label-encode all categorical columns.
2. Build a synthetic binary target so SMOTE has something to oversample:
   rows above the median amount (or any continuous column) → class 1,
   rest → class 0.  This lets SMOTE explore the full feature space.
3. Ask SMOTE to produce exactly ``num_rows`` new minority-class samples.
4. Decode categoricals back to their original string values.
5. Return a DataFrame with the same columns and dtypes as the seed data.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base_generator import BaseGenerator


class SMOTEGenerator(BaseGenerator):
    """Generate synthetic tabular rows using SMOTE interpolation.

    Parameters
    ----------
    config : dict, optional
        - k_neighbors : int, default 5
        - random_state : int, default None
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.k_neighbors: int = int(self.config.get("k_neighbors", 5))
        self.random_state = self.config.get("random_state", None)

        # State populated by fit()
        self._original_columns: list[str] = []
        self._cat_indices: list[int] = []           # positional indices of categorical cols
        self._cat_cols: list[str] = []              # names of categorical cols
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._X_encoded: pd.DataFrame | None = None  # numeric-only encoded seed data
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Learn the feature space from seed data.

        Parameters
        ----------
        data : pd.DataFrame
            Seed data (original accounts or transactions).
        """
        df = data.copy().reset_index(drop=True)

        # Drop columns that are pure identifiers or all-null — they add no signal
        df = self._drop_useless_columns(df)

        self._original_columns = df.columns.tolist()

        # Detect categorical columns
        self._cat_cols = []
        self._cat_indices = []
        for i, col in enumerate(df.columns):
            if df[col].dtype == "object" or df[col].dtype.name in ("category", "string"):
                self._cat_cols.append(col)
                self._cat_indices.append(i)
            elif df[col].dtype in (bool,) or str(df[col].dtype).startswith("bool"):
                self._cat_cols.append(col)
                self._cat_indices.append(i)

        # Label-encode categoricals; fill nulls first
        encoded = df.copy()
        for col in self._cat_cols:
            encoded[col] = encoded[col].fillna("__MISSING__").astype(str)
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col])
            self._label_encoders[col] = le

        # Fill remaining nulls in numeric columns with median
        for col in encoded.columns:
            if col not in self._cat_cols:
                encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
                encoded[col] = encoded[col].fillna(encoded[col].median())

        self._X_encoded = encoded.astype(float)
        self._fitted = True

    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate ``num_rows`` synthetic rows via SMOTE interpolation.

        Parameters
        ----------
        num_rows : int
            Exact number of synthetic rows to return.

        Returns
        -------
        pd.DataFrame with the same columns as the seed data.
        """
        if not self._fitted or self._X_encoded is None:
            raise RuntimeError("Call fit() before generate().")

        try:
            from imblearn.over_sampling import SMOTE, SMOTENC
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required: pip install imbalanced-learn==0.12.0"
            ) from exc

        rng = np.random.default_rng(self.random_state)
        X = self._X_encoded.values
        n_seed = len(X)

        # Build a simple binary target so SMOTE has a minority class to oversample.
        # Use median split on the first numeric column that has variance.
        y = self._make_binary_target(X, rng)

        minority_mask = y == 1
        n_minority = minority_mask.sum()
        n_majority = (~minority_mask).sum()

        # We want exactly num_rows NEW samples from the minority class.
        # Tell SMOTE to produce n_minority + num_rows minority samples.
        target_minority = int(n_minority) + num_rows

        k = min(self.k_neighbors, int(n_minority) - 1)
        if k < 1:
            k = 1

        sampling_strategy = {1: target_minority}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._cat_indices:
                smote = SMOTENC(
                    categorical_features=self._cat_indices,
                    k_neighbors=k,
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                )
            else:
                smote = SMOTE(
                    k_neighbors=k,
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                )
            X_res, y_res = smote.fit_resample(X, y)

        # The new rows are at the end of X_res (after the original n_seed rows).
        # Take exactly num_rows from the newly generated minority samples.
        new_rows = X_res[n_seed:]          # all newly created rows
        if len(new_rows) == 0:
            # Fallback: bootstrap from seed if SMOTE produced nothing new
            idx = rng.integers(0, n_seed, size=num_rows)
            new_rows = X[idx]
        elif len(new_rows) < num_rows:
            # Pad by bootstrapping from what SMOTE gave us
            pad_idx = rng.integers(0, len(new_rows), size=num_rows - len(new_rows))
            new_rows = np.vstack([new_rows, new_rows[pad_idx]])
        else:
            new_rows = new_rows[:num_rows]

        synthetic = pd.DataFrame(new_rows, columns=self._original_columns)

        # Decode categorical columns back to original labels
        for col in self._cat_cols:
            le = self._label_encoders[col]
            codes = synthetic[col].round().clip(0, len(le.classes_) - 1).astype(int)
            decoded = le.inverse_transform(codes)
            # Restore __MISSING__ → NaN
            synthetic[col] = [v if v != "__MISSING__" else np.nan for v in decoded]

        # Restore original dtypes as closely as possible
        synthetic = self._restore_dtypes(synthetic)

        return synthetic.reset_index(drop=True)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "SMOTEGenerator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls()
        instance.__dict__.update(state)
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _drop_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that carry no interpolation signal."""
        drop = []
        for col in df.columns:
            # All-null
            if df[col].isna().all():
                drop.append(col)
            # UUID-like strings (high cardinality object cols with unique values)
            elif df[col].dtype == "object" and df[col].nunique() == len(df):
                drop.append(col)
        if drop:
            df = df.drop(columns=drop)
        return df

    def _make_binary_target(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Create a balanced-ish binary target for SMOTE using median split."""
        # Try each column until we get a split with both classes present
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            if np.isnan(col).all():
                continue
            median = np.nanmedian(col)
            y = (col > median).astype(int)
            if 0 < y.sum() < len(y):
                return y
        # Fallback: random 50/50 split
        y = np.zeros(len(X), dtype=int)
        half = len(X) // 2
        y[rng.choice(len(X), size=half, replace=False)] = 1
        return y

    def _restore_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Best-effort dtype restoration after float round-trip."""
        for col in df.columns:
            if col in self._cat_cols:
                continue  # already decoded to strings
            try:
                # Try integer if values are whole numbers
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().all() and (numeric == numeric.round()).all():
                    df[col] = numeric.astype("Int64")
                else:
                    df[col] = numeric
            except Exception:
                pass
        return df
