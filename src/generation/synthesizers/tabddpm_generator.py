import os
import pandas as pd
import logging
from .base_generator import BaseGenerator

try:
    from synthyverse.generators import TabDDPMGenerator as SynthyverseTabDDPM
    SYNTHYVERSE_AVAILABLE = True
except ImportError:
    SYNTHYVERSE_AVAILABLE = False


class TabDDPMGenerator(BaseGenerator):
    """
    TabDDPM generator wrapper for tabular fraud data.

    TabDDPM uses Denoising Diffusion Probabilistic Models adapted for
    mixed-type tabular data. It represents the state-of-the-art for tabular
    synthesis (2024+), avoiding the mode collapse issues typical of GANs
    while preserving complex non-normal distributions with high fidelity.

    This wrapper uses the `synthyverse` library's DDPM implementation, which
    provides a robust, pip-installable version of TabDDPM.

    Install with: `pip install synthyverse`
    """

    def __init__(
        self,
        config: dict | None = None,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 1e-3,
        num_timesteps: int = 1000,
        device: str = "cpu"
    ):
        self.config = config or {}
        if not SYNTHYVERSE_AVAILABLE:
            raise ImportError(
                "synthyverse is required to run TabDDPM. Install with: pip install synthyverse"
            )
        # Allow config to override defaults
        self.epochs = self.config.get("epochs", epochs)
        self.batch_size = self.config.get("batch_size", batch_size)
        self.lr = self.config.get("lr", lr)
        self.num_timesteps = self.config.get("num_timesteps", num_timesteps)
        
        # We need a target column for Synthyverse's TabDDPM.
        # Fallback to 'is_suspicious_tx' which is our benchmark default.
        self.target_column = self.config.get("target_column", "is_suspicious_tx")

        self.model = None
        self.is_fitted = False
        self._columns: list[str] | None = None

    # ------------------------------------------------------------------ #
    #  Fit
    # ------------------------------------------------------------------ #
    def fit(self, data: pd.DataFrame) -> None:
        """Train TabDDPM on *data*.
        
        The caller must pass only the *base* columns (Generator Input Schema).
        """
        self._columns = data.columns.tolist()

        logging.info(
            "TabDDPM fit — %d rows, %d cols",
            len(data),
            len(self._columns)
        )

        # Detect discrete features automatically as Synthyverse requires them
        discrete_features = []
        for col in data.columns:
            if data[col].dtype == "object" or data[col].dtype.name == "category" or data[col].nunique() <= 20:
                discrete_features.append(col)
                
        # Handle case where the target column is not in the dataframe
        if self.target_column not in data.columns:
            # Synthyverse TabDDPM *requires* a target column to initialize
            fallback_target = discrete_features[0] if discrete_features else data.columns[0]
            logging.warning(f"Target column '{self.target_column}' not found. Using '{fallback_target}' instead.")
            self.target_column = fallback_target

        self.model = SynthyverseTabDDPM(
            target_column=self.target_column,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            num_timesteps=self.num_timesteps
        )
        
        self.model.fit(data, discrete_features)
        self.is_fitted = True

    # ------------------------------------------------------------------ #
    #  Generate
    # ------------------------------------------------------------------ #
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logging.info("Generating %d rows using TabDDPM...", num_rows)
        synthetic = self.model.generate(num_rows)
        
        # Ensure column order matches original
        if self._columns is not None:
            for col in self._columns:
                if col not in synthetic.columns:
                    synthetic[col] = 0
            synthetic = synthetic[self._columns]
            
        return synthetic

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Save Synthyverse model
        self.model.save(path)
        
        # Save wrapper metadata
        import pickle
        meta_path = str(path) + "_meta.pkl"
        state = {
            "columns": self._columns,
            "target_column": self.target_column
        }
        with open(meta_path, "wb") as f:
            pickle.dump(state, f)
            
        logging.info("TabDDPM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TabDDPMGenerator":
        if not SYNTHYVERSE_AVAILABLE:
            raise ImportError("synthyverse is required. Install with: pip install synthyverse")

        instance = cls()
        
        # Load Synthyverse model
        instance.model = SynthyverseTabDDPM.load(path)
        
        # Load wrapper metadata
        import pickle
        import os
        meta_path = str(path) + "_meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                state = pickle.load(f)
            instance._columns = state["columns"]
            instance.target_column = state["target_column"]
            
        instance.is_fitted = True
        return instance
