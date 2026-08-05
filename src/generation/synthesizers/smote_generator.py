import os
import pandas as pd
import logging
from .base_generator import BaseGenerator

try:
    from synthyverse.generators.smote_generator import SMOTEGenerator as SynthyverseSMOTE
    SYNTHYVERSE_AVAILABLE = True
except ImportError:
    SYNTHYVERSE_AVAILABLE = False


class SMOTEGenerator(BaseGenerator):
    """
    SMOTE generator wrapper for tabular fraud data.

    Synthetic Minority Over-sampling Technique (SMOTE) creates synthetic samples
    via interpolation in feature space. 

    This wrapper uses the `synthyverse` library's SMOTE implementation.
    Install with: `pip install synthyverse`
    """

    def __init__(
        self,
        config: dict | None = None,
        k_neighbors: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.config = config or {}
        if not SYNTHYVERSE_AVAILABLE:
            raise ImportError(
                "synthyverse is required to run SMOTE. Install with: pip install synthyverse"
            )
        self.k_neighbors = self.config.get("k_neighbors", k_neighbors)
        self.n_jobs      = self.config.get("n_jobs",      n_jobs)
        self.random_state= self.config.get("random_state",random_state)
        self.target_column = self.config.get("target_column", "is_suspicious_tx")

        self.model = None
        self.is_fitted = False
        self._columns: list[str] | None = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit SMOTE on *data*."""
        self._columns = data.columns.tolist()

        logging.info("SMOTE fit — %d rows, %d cols", len(data), len(self._columns))

        # Detect discrete features automatically as Synthyverse requires them
        discrete_features = []
        for col in data.columns:
            if data[col].dtype == "object" or data[col].dtype.name == "category" or data[col].nunique() <= 20:
                discrete_features.append(col)

        # Handle case where the target column is not in the dataframe
        if self.target_column not in data.columns:
            fallback_target = discrete_features[0] if discrete_features else data.columns[-1]
            logging.warning("Target column '%s' not found. Using '%s' instead.", self.target_column, fallback_target)
            self.target_column = fallback_target

        self.model = SynthyverseSMOTE(
            target_column=self.target_column,
            k_neighbors=self.k_neighbors,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # SMOTEGenerator in synthyverse doesn't actually fit a model to generate randomly,
        # it just stores the data and will run SMOTE algorithm in `generate`.
        self.model.fit(data, discrete_features)
        self.is_fitted = True

    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logging.info("Generating %d rows using SMOTE...", num_rows)
        synthetic = self.model.generate(num_rows)
        
        # Ensure column order matches original
        if self._columns is not None:
            for col in self._columns:
                if col not in synthetic.columns:
                    synthetic[col] = 0
            synthetic = synthetic[self._columns]
            
        return synthetic

    def save(self, path: str) -> None:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        
        import pickle
        meta_path = str(path) + "_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "columns": self._columns,
                "target_column": self.target_column
            }, f)
        logging.info("SMOTE model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SMOTEGenerator":
        if not SYNTHYVERSE_AVAILABLE:
            raise ImportError("synthyverse is required. Install with: pip install synthyverse")

        instance = cls()
        instance.model = SynthyverseSMOTE.load(path)
        
        import pickle
        meta_path = str(path) + "_meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                state = pickle.load(f)
            instance._columns = state["columns"]
            instance.target_column = state["target_column"]
            
        instance.is_fitted = True
        return instance
