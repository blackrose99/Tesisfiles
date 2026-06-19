from abc import ABC, abstractmethod
from typing import List
from src.domain.entities.student import Student
from src.domain.entities.prediction import Prediction

class StudentRepository(ABC):
    """
    Abstract repository interface for Student entities.
    """
    @abstractmethod
    def load_from_excel(self, file_path: str) -> List[Student]:
        pass

    @abstractmethod
    def load_from_csv(self, file_path: str, delimiter: str = ",") -> List[Student]:
        pass

    @abstractmethod
    def save_predictions(self, predictions: List[Prediction], output_path: str) -> None:
        pass
