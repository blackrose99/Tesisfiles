from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np

class PredictiveModel(ABC):
    """
    Abstract Strategy class for machine learning models.
    """
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PredictiveModel":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def set_params(self, **params) -> "PredictiveModel":
        pass
