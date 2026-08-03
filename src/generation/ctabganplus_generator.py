import os
import pandas as pd
import logging
from .base_generator import BaseGenerator

try:
    from ctabganplus.synthesizer.ctabganplus import CTABGANPlus
    CTABGANPLUS_AVAILABLE = True
except ImportError:
    CTABGANPLUS_AVAILABLE = False


class CTABGANPlusGenerator(BaseGenerator):
    """
    CTAB-GAN+ generator wrapper for tabular fraud data.

    CTAB-GAN+ is a conditional tabular GAN with an auxiliary classifier that
    jointly optimises distributional fidelity *and* downstream ML utility.
    Key advantages over vanilla CTGAN for fraud data:

    * Built-in minority-class oversampling (handles extreme imbalance)
    * Gaussian-mixture + log-transform encoding for long-tailed amounts
    * Auxiliary classifier loss pushes the generator to produce samples that
      preserve discriminative feature relationships
    * Native mixed-type handling (continuous, categorical, integer)

    Wraps the ``ctabganplus`` PyPI package.
    Install with: ``pip install ctabganplus``
    """

    def __init__(
        self,
        epochs: int = 150,
        batch_size: int = 500,
        lr: float = 2e-4,
        class_dim: tuple = (256, 256, 256, 256),
        random_dim: int = 100,
        num_channels: int = 64,
        test_ratio: float = 0.20,
    ):
        if not CTABGANPLUS_AVAILABLE:
            raise ImportError(
                "ctabganplus is required.  Install with:  pip install ctabganplus"
            )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_dim = class_dim
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.test_ratio = test_ratio

        self.model: CTABGANPlus | None = None
        self.is_fitted = False
        self._columns: list[str] | None = None

    # ------------------------------------------------------------------ #
    #  Fit
    # ------------------------------------------------------------------ #
    def fit(self, data: pd.DataFrame) -> None:
        """Train CTAB-GAN+ on *data*.

        The caller is responsible for passing only the *base* columns that
        should be modelled (see EDA.md §1.15 – do not pass derived columns).
        """
        self._columns = data.columns.tolist()

        # Identify column types for CTAB-GAN+
        categorical_cols = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        integer_cols = [
            c
            for c in data.select_dtypes(include=["int"]).columns
            if c not in categorical_cols
        ]

        # Log-transformed columns (long-tailed amounts) benefit from
        # CTAB-GAN+'s mixed-mode encoding.
        log_columns: list[str] = []
        for col in data.select_dtypes(include=["float", "int"]).columns:
            if data[col].skew() > 2.0 and (data[col] >= 0).all():
                log_columns.append(col)

        logging.info(
            "CTAB-GAN+ fit — %d rows, %d cols (%d cat, %d int, %d log-skewed)",
            len(data),
            len(self._columns),
            len(categorical_cols),
            len(integer_cols),
            len(log_columns),
        )

        self.model = CTABGANPlus(
            raw_csv_path="",          # Unused – we pass a DataFrame directly
            test_ratio=self.test_ratio,
            categorical_columns=categorical_cols,
            log_columns=log_columns,
            integer_columns=integer_cols,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            class_dim=self.class_dim,
            random_dim=self.random_dim,
            num_channels=self.num_channels,
        )

        # CTABGANPlus.fit() can accept a DataFrame directly
        self.model.fit(df=data)
        self.is_fitted = True

    # ------------------------------------------------------------------ #
    #  Generate
    # ------------------------------------------------------------------ #
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted.  Call fit() first.")
        synthetic = self.model.generate_samples(num_rows)
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
        logging.info("CTAB-GAN+ model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "CTABGANPlusGenerator":
        if not CTABGANPLUS_AVAILABLE:
            raise ImportError("ctabganplus is required.")

        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls()
        instance.model = state["model"]
        instance._columns = state["columns"]
        instance.is_fitted = True
        return instance
