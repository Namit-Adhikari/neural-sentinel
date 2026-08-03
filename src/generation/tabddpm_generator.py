import os
import pandas as pd
import logging
from .base_generator import BaseGenerator

try:
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    SYNTHCITY_AVAILABLE = True
except ImportError:
    SYNTHCITY_AVAILABLE = False


class TabDDPMGenerator(BaseGenerator):
    """
    TabDDPM generator wrapper for tabular fraud data.

    TabDDPM uses Denoising Diffusion Probabilistic Models adapted for
    mixed-type tabular data. It represents the state-of-the-art for tabular
    synthesis (2024+), avoiding the mode collapse issues typical of GANs
    while preserving complex non-normal distributions with high fidelity.

    This wrapper uses the `synthcity` library's DDPM implementation, which
    provides a robust, pip-installable version of TabDDPM.

    Install with: `pip install synthcity`
    """

    def __init__(
        self,
        n_iter: int = 2000,
        batch_size: int = 1024,
        lr: float = 1e-3,
        num_timesteps: int = 1000,
        device: str = "cpu"
    ):
        if not SYNTHCITY_AVAILABLE:
            raise ImportError(
                "synthcity is required to run TabDDPM. Install with: pip install synthcity"
            )
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.lr = lr
        self.num_timesteps = num_timesteps
        self.device = device

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
            "TabDDPM fit — %d rows, %d cols (Device: %s)",
            len(data),
            len(self._columns),
            self.device
        )

        # Initialize the DDPM plugin from Synthcity
        self.model = Plugins().get(
            "ddpm",
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            lr=self.lr,
            num_timesteps=self.num_timesteps,
            device=self.device
        )

        # Synthcity requires data to be wrapped in a GenericDataLoader
        loader = GenericDataLoader(data)
        
        self.model.fit(loader)
        self.is_fitted = True

    # ------------------------------------------------------------------ #
    #  Generate
    # ------------------------------------------------------------------ #
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logging.info("Generating %d rows using TabDDPM...", num_rows)
        synthetic_loader = self.model.generate(num_rows)
        synthetic = synthetic_loader.dataframe()
        
        # Ensure column order matches original
        if self._columns is not None:
            synthetic = synthetic[self._columns]
            
        return synthetic

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        import pickle
        state = {
            "model": self.model,
            "columns": self._columns,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logging.info("TabDDPM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TabDDPMGenerator":
        if not SYNTHCITY_AVAILABLE:
            raise ImportError("synthcity is required. Install with: pip install synthcity")

        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls()
        instance.model = state["model"]
        instance._columns = state["columns"]
        instance.is_fitted = True
        return instance
