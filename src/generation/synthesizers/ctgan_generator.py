import os
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from .base_generator import BaseGenerator

class CTGANGenerator(BaseGenerator):
    def __init__(self, config=None, epochs=30):
        self.config = config or {}
        self.model = None
        self.metadata = None
        self.epochs = self.config.get('epochs', epochs)
        
    def fit(self, data: pd.DataFrame):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)
        self.model = CTGANSynthesizer(self.metadata, epochs=self.epochs)
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
    def load(cls, path: str) -> "CTGANGenerator":
        instance = cls()
        instance.model = CTGANSynthesizer.load(path)
        # Hack to recover metadata if needed, though usually bound to model
        instance.metadata = instance.model.metadata
        return instance
