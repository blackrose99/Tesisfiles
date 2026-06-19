import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from src.domain.entities.student import Student
from src.domain.repositories.student_repository import StudentRepository
from src.domain.repositories.model_repository import ModelRepository
from src.infrastructure.ml.preprocessors import DataPreprocessor
from src.infrastructure.ml.evaluator import ModelEvaluator

class TrainUseCase:
    """
    Application Use Case to train, optimize, validate, and register student dropout models.
    """
    def __init__(self, 
                 student_repo: StudentRepository,
                 model_repo: ModelRepository,
                 random_state: int = 42):
        self.student_repo = student_repo
        self.model_repo = model_repo
        self.random_state = random_state

    def execute(self, raw_data_path: str, allowed_programs: List[str] = None, 
                exclude_features: List[str] = None, n_trials: int = 15) -> Dict[str, Any]:
        """
        Executes the training workflow.
        1. Loads data from Excel database.
        2. Normalizes Target column based on institutional definitions.
        3. Prepares training and testing sets.
        4. Trains and fits the preprocessor.
        5. Automatically compares models.
        6. Tunes hyperparameters for the best model.
        7. Saves final artifacts to registry.
        """
        print("Cargando base de datos...")
        # Load raw student records
        students = self.student_repo.load_from_excel(raw_data_path)
        
        # Convert to DataFrame for processing
        df_raw = pd.DataFrame([s.to_dict() for s in students])
        
        # INSTITUTIONAL DROP-OUT TARGET DEFINITION
        # 1 = Risk of dropout (excluidos, cancelaciones, PFI, inactivo)
        # 0 = Continuity (activo, graduado, sobresaliente, condicional, transferencias, etc.)
        dropout_statuses = [
            "EXCLUIDO NO RENOVACION DE MATRICULA",
            "PFI",
            "EXCLUIDO CANCELACION SEMESTRE",
            "INACTIVO"
        ]
        df_raw["TARGET"] = df_raw["SITUACION"].isin(dropout_statuses).astype(int)
        
        # Stratified partition (80% train / 20% hold-out test)
        df_train_raw, df_test_raw = train_test_split(
            df_raw, 
            test_size=0.20, 
            random_state=self.random_state, 
            stratify=df_raw["TARGET"]
        )
        
        # Initialize and fit Preprocessor
        preprocessor = DataPreprocessor(allowed_programs=allowed_programs)
        
        print("Ajustando pipeline de datos y feature engineering...")
        preprocessor.fit(df_train_raw)
        
        # Transform data
        X_train = preprocessor.transform(df_train_raw)
        X_test = preprocessor.transform(df_test_raw)
        
        # Target arrays
        y_train = df_train_raw["TARGET"].values
        y_test = df_test_raw["TARGET"].values
        
        # Exclude specific features from training if requested (e.g. to prevent temporal bias)
        features_to_use = list(preprocessor.feature_names_in_)
        if exclude_features:
            for f in exclude_features:
                if f in features_to_use:
                    features_to_use.remove(f)
                    
            X_train = X_train[features_to_use]
            X_test = X_test[features_to_use]
            
        X_train_np = X_train.values
        X_test_np = X_test.values
        
        # Evaluate model selection matrix
        evaluator = ModelEvaluator(random_state=self.random_state)
        print("Evaluando comparación de algoritmos...")
        comparison_df = evaluator.compare_models(X_train_np, y_train)
        print("\nResultados de comparación:")
        print(comparison_df.to_string(index=False))
        
        # Select best model based on F1-Score
        best_row = comparison_df.sort_values(by="f1", ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        print(f"\nModelo seleccionado automáticamente: {best_model_name} (F1 = {best_row['f1']:.4f})")
        
        # Optimize hyperparameters for the best model
        best_params = evaluator.optimize_hyperparameters(best_model_name, X_train_np, y_train, n_trials=n_trials)
        
        # Train final model on entire training set using tuned parameters
        print(f"Entrenando modelo final ({best_model_name}) con hiperparámetros óptimos...")
        model, test_metrics = evaluator.train_and_evaluate_final(
            best_model_name, X_train_np, y_train, X_test_np, y_test, params=best_params
        )
        print("\nMétricas finales en test:")
        for k, v in test_metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  confusion_matrix: {v}")
                
        # Register in repository
        # Preprocessor wrapper to remember features used
        preprocessor_wrapper = {
            "preprocessor": preprocessor,
            "exclude_features": exclude_features,
            "trained_features": features_to_use
        }
        
        print("Registrando modelo en repositorio...")
        version = self.model_repo.save_model(
            model=model, 
            preprocessor=preprocessor_wrapper, 
            metrics=test_metrics, 
            model_name=best_model_name
        )
        print(f"Modelo registrado exitosamente con versión: {version}")
        
        # Save background data for SHAP explanations (needed for Streamlit UI)
        # Select a representative sample from training features
        rng = np.random.default_rng(self.random_state)
        bg_size = min(100, len(X_train_np))
        bg_idx = rng.choice(len(X_train_np), size=bg_size, replace=False)
        np.save("shap_background.npy", X_train_np[bg_idx])
        np.save("shap_feature_names.npy", np.array(features_to_use))
        
        return {
            "version": version,
            "model_name": best_model_name,
            "metrics": test_metrics,
            "comparison_matrix": comparison_df.to_dict(orient="records"),
            "features_trained": features_to_use
        }
