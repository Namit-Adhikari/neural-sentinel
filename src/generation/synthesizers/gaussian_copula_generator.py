import os
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from .base_generator import BaseGenerator

class GaussianCopulaGenerator(BaseGenerator):
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.metadata = None
        
    def fit(self, data: pd.DataFrame):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)
        self.model = GaussianCopulaSynthesizer(self.metadata)
        self.model.fit(data)
        
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.model:
            raise ValueError("Model not fitted.")
        return self.model.sample(num_rows)
        
    def save(self, path: str):
        if not self.model:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
    @classmethod
    def load(cls, path: str) -> "GaussianCopulaGenerator":
        instance = cls()
        instance.model = GaussianCopulaSynthesizer.load(path)
        instance.metadata = instance.model.metadata
        return instance
