from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class ModelRepository(ABC):
    """
    Abstract repository interface for trained models and preprocessors.
    """
    @abstractmethod
    def save_model(self, model: Any, preprocessor: Any, metrics: Dict[str, float], model_name: str) -> str:
        """
        Saves a trained model along with its preprocessor, metrics, and generates a version.
        Returns the version string.
        """
        pass

    @abstractmethod
    def load_model(self, version: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Loads a specific model version and preprocessor.
        Returns (model, preprocessor, metadata).
        """
        pass

    @abstractmethod
    def load_latest_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Loads the active/latest model and preprocessor.
        Returns (model, preprocessor, metadata).
        """
        pass

    @abstractmethod
    def get_model_history(self) -> List[Dict[str, Any]]:
        """
        Returns a list of metadata for all saved model versions.
        """
        pass

    @abstractmethod
    def log_monitoring_metrics(self, drift_metrics: Dict[str, Any]) -> None:
        """
        Logs data drift and concept drift metrics.
        """
        pass
