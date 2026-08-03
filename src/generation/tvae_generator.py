import os
import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from .base_generator import BaseGenerator

class TVAEGenerator(BaseGenerator):
    def __init__(self, epochs=30):
        self.model = None
        self.metadata = None
        self.epochs = epochs
        
    def fit(self, data: pd.DataFrame):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)
        self.model = TVAESynthesizer(self.metadata, epochs=self.epochs)
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
    def load(cls, path: str) -> "TVAEGenerator":
        instance = cls()
        instance.model = TVAESynthesizer.load(path)
        instance.metadata = instance.model.metadata
        return instance
