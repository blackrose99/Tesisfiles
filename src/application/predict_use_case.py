import os
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple
from src.domain.entities.student import Student
from src.domain.entities.prediction import Prediction
from src.domain.repositories.student_repository import StudentRepository
from src.domain.repositories.model_repository import ModelRepository

class PredictUseCase:
    """
    Application Use Case to process predictions for individual or batch students.
    """
    def __init__(self,
                 student_repo: StudentRepository,
                 model_repo: ModelRepository):
        self.student_repo = student_repo
        self.model_repo = model_repo

    def execute_batch(self, students: List[Student], threshold: float = 0.5, calculate_shap: bool = False) -> List[Prediction]:
        """
        Runs predictions for a batch of student entities.
        """
        if not students:
            return []
            
        # Load active model and preprocessor
        model, prep_wrapper, metadata = self.model_repo.load_latest_model()
        preprocessor = prep_wrapper["preprocessor"]
        trained_features = prep_wrapper["trained_features"]
        
        # Convert students to dataframe
        df_raw = pd.DataFrame([s.to_dict() for s in students])
        
        # Transform data using fitted preprocessor
        df_transformed = preprocessor.transform(df_raw)
        
        # Filter for features the model was trained on
        X_scaled = df_transformed[trained_features].values
        
        # Predict probabilities
        probabilities = model.predict_proba(X_scaled)
        
        # Initialize SHAP explainer only if requested
        explainer = None
        if calculate_shap:
            background_path = "shap_background.npy"
            if os.path.exists(background_path):
                background = np.load(background_path)
                try:
                    # TreeExplainer is fast for tree-based models
                    # shap.Explainer selects the best explainer automatically
                    explainer = shap.Explainer(model.model, background)
                except Exception as e:
                    print(f"Error al inicializar explainer SHAP: {e}")
                
        # Construct prediction domain entities
        predictions = []
        for i, student in enumerate(students):
            prob = float(probabilities[i])
            is_dropout = prob >= threshold
            
            # Risk level definition
            if prob < 0.35:
                risk_level = "Bajo"
            elif prob < 0.70:
                risk_level = "Moderado"
            else:
                risk_level = "Alto"
                
            # Compute SHAP values for this student
            shap_dict = {}
            if calculate_shap and explainer is not None:
                try:
                    X_row = X_scaled[i].reshape(1, -1)
                    shap_exp = explainer(X_row)
                    
                    # Handle different SHAP output formats (e.g. multi-class, binary)
                    if hasattr(shap_exp, "values"):
                        s_vals = shap_exp.values[0]
                    else:
                        s_vals = shap_exp[0]
                        
                    # If model is binary, sometimes shape has an extra dimension
                    if s_vals.ndim == 2: # Shape (1, num_features) or (num_features, 2)
                        # For binary classifiers, take the contribution for class 1
                        if s_vals.shape[1] == 2:
                            s_vals = s_vals[:, 1]
                        else:
                            s_vals = s_vals.flatten()
                    elif s_vals.ndim == 1:
                        s_vals = s_vals.flatten()
                        
                    # Map back to feature names
                    for idx, feat in enumerate(trained_features):
                        if idx < len(s_vals):
                            shap_dict[feat] = float(s_vals[idx])
                except Exception as ex:
                    print(f"No se pudo calcular SHAP para estudiante {student.code_student}: {ex}")
                    
            pred_entity = Prediction(
                code_student=student.code_student,
                probability=prob,
                risk_level=risk_level,
                is_dropout=is_dropout,
                shap_values=shap_dict
            )
            predictions.append(pred_entity)
            
        return predictions

    def execute_individual(self, student: Student, threshold: float = 0.5) -> Prediction:
        """
        Runs prediction for a single student.
        """
        predictions = self.execute_batch([student], threshold, calculate_shap=True)
        return predictions[0]
