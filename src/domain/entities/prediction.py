from typing import Dict, Any, List

class Prediction:
    """
    Domain entity representing the model's prediction for a student.
    """
    def __init__(self,
                 code_student: str,
                 probability: float,
                 risk_level: str,  # "Bajo", "Moderado", "Alto"
                 is_dropout: bool,
                 shap_values: Dict[str, float]):
        self.code_student = code_student
        self.probability = probability
        self.risk_level = risk_level
        self.is_dropout = is_dropout
        self.shap_values = shap_values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_student": self.code_student,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "is_dropout": self.is_dropout,
            "shap_values": self.shap_values
        }
