from abc import ABC, abstractmethod
import pandas as pd

class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit the generator to real data."""
        pass
        
    @abstractmethod
    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate num_rows of synthetic data."""
        pass
        
    @abstractmethod
    def save(self, path: str):
        """Save the fitted model."""
        pass
        
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseGenerator":
        """Load a fitted model from path."""
        pass
