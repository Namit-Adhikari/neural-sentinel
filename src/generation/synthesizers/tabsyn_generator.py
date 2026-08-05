import os
import logging
import pandas as pd
from .base_generator import BaseGenerator

try:
    from synthyverse.generators.tabsyn_generator import TabSynGenerator as SynthyverseTabSyn
    SYNTHYVERSE_AVAILABLE = True
except ImportError:
    SYNTHYVERSE_AVAILABLE = False


class TabSynGenerator(BaseGenerator):
    """
    TabSyn generator wrapper for tabular fraud data.

    TabSyn trains a VAE to learn a latent representation of mixed-type
    tabular data, then fits a score-based diffusion model in that latent
    space. It achieves state-of-the-art fidelity on mixed continuous/
    categorical tabular datasets.

    This wrapper uses the `synthyverse` library's TabSyn implementation.

    Install with: `pip install synthyverse`

    Paper: "Mixed-type tabular data synthesis with score-based diffusion
    in latent space" — Zhang et al. (2023)
    """

    def __init__(
        self,
        config: dict | None = None,
        vae_epochs: int = 200,
        diff_epochs: int = 500,
        batch_size: int = 4096,
        lr: float = 1e-3,
        vae_lr: float = 1e-3,
    ):
        self.config = config or {}
        if not SYNTHYVERSE_AVAILABLE:
            raise ImportError(
                "synthyverse is required to run TabSyn. Install with: pip install synthyverse"
            )
        # Allow config dict to override keyword defaults
        self.vae_epochs  = self.config.get("vae_epochs",  vae_epochs)
        self.diff_epochs = self.config.get("diff_epochs", diff_epochs)
        self.batch_size  = self.config.get("batch_size",  batch_size)
        self.lr          = self.config.get("lr",          lr)
        self.vae_lr      = self.config.get("vae_lr",      vae_lr)
        self.target_column = self.config.get("target_column", "is_suspicious_tx")

        self.model = None
        self.is_fitted = False
        self._columns: list[str] | None = None

    # ------------------------------------------------------------------ #
    #  Fit
    # ------------------------------------------------------------------ #
    def fit(self, data: pd.DataFrame) -> None:
        """Train TabSyn (VAE + diffusion) on *data*."""
        self._columns = data.columns.tolist()

        logging.info(
            "TabSyn fit — %d rows, %d cols",
            len(data), len(self._columns)
        )

        # Auto-detect discrete/categorical features (required by synthyverse)
        discrete_features = [
            col for col in data.columns
            if data[col].dtype == "object"
            or data[col].dtype.name == "category"
            or data[col].nunique() <= 20
        ]

        # Resolve target column
        target = self.target_column
        if target not in data.columns:
            fallback = discrete_features[0] if discrete_features else data.columns[0]
            logging.warning(
                "TabSyn: target column '%s' not found. Using '%s'.",
                target, fallback
            )
            target = fallback

        self.model = SynthyverseTabSyn(
            target_column=target,
            vae_num_epochs=self.vae_epochs,
            epochs=self.diff_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            vae_lr=self.vae_lr,
        )

        self.model.fit(data, discrete_features)
        self.is_fitted = True

    # ------------------------------------------------------------------ #
    #  Generate
    # ------------------------------------------------------------------ #
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logging.info("Generating %d rows using TabSyn...", num_rows)
        synthetic = self.model.generate(num_rows)

        # Align columns to original schema
        if self._columns is not None:
            for col in self._columns:
                if col not in synthetic.columns:
                    synthetic[col] = 0
            synthetic = synthetic[[c for c in self._columns if c in synthetic.columns]]

        return synthetic

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
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
                "target_column": self.target_column,
            }, f)
        logging.info("TabSyn model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TabSynGenerator":
        if not SYNTHYVERSE_AVAILABLE:
            raise ImportError("synthyverse is required. Install with: pip install synthyverse")

        instance = cls()
        instance.model = SynthyverseTabSyn.load(path)

        import pickle
        meta_path = str(path) + "_meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                state = pickle.load(f)
            instance._columns = state["columns"]
            instance.target_column = state["target_column"]

        instance.is_fitted = True
        return instance
