import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, List, Tuple, Union
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
from src.infrastructure.ml.models import ModelFactory, PredictiveModel

# Silence optuna logs unless warning/error
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelEvaluator:
    """
    Evaluator class that performs cross-validation, model comparison,
    hyperparameter optimization with Optuna, and final training.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.k_folds = 5

    def cross_validate(self, model_type: str, X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Runs Stratified 5-Fold Cross Validation for a specific model type.
        """
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        
        metrics = {
            "accuracy": [], "precision": [], "recall": [], 
            "f1": [], "roc_auc": [], "balanced_accuracy": []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = ModelFactory.create_model(model_type, **(params or {}))
            
            # Compute class weights for trees / linear models if possible
            # Standard sklearn models support class_weight='balanced'
            # For XGBoost we can use scale_pos_weight
            # For LightGBM and CatBoost they support class_weight='balanced' or auto_class_weights
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            # Calculate metrics
            metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
            metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val, y_pred))
            
        # Return average metrics
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def compare_models(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Compares all base and advanced models and returns a DataFrame with their metrics.
        """
        models_to_compare = [
            "Logistic Regression", "Random Forest", "Extra Trees", 
            "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"
        ]
        
        results = []
        for model_name in models_to_compare:
            print(f"Evaluando modelo: {model_name}...")
            # Use balanced weights if possible to handle class imbalance
            params = {}
            name_lower = model_name.lower().replace(" ", "")
            if name_lower in ["logisticregression", "randomforest", "extratrees"]:
                params["class_weight"] = "balanced"
            elif name_lower == "lightgbm":
                params["class_weight"] = "balanced"
            elif name_lower == "catboost":
                params["auto_class_weights"] = "Balanced"
            elif name_lower == "xgboost":
                # scale_pos_weight = count(negative) / count(positive)
                ratio = (len(y) - np.sum(y)) / (np.sum(y) + 1e-9)
                params["scale_pos_weight"] = float(ratio)
                
            metrics = self.cross_validate(model_name, X, y, params)
            results.append({"Model": model_name, **metrics})
            
        return pd.DataFrame(results)

    def optimize_hyperparameters(self, model_type: str, X: np.ndarray, y: np.ndarray, n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimizes model hyperparameters using Optuna (Bayesian Optimization).
        """
        print(f"Optimizando hiperparámetros para {model_type} ({n_trials} ensayos)...")
        
        model_name = model_type.lower().replace(" ", "").replace("_", "")
        
        def objective(trial):
            params = {}
            
            # Define hyperparameter search spaces based on the model type
            if model_name in ["logisticregression", "lr"]:
                params["C"] = trial.suggest_float("C", 1e-4, 1e2, log=True)
                params["penalty"] = trial.suggest_categorical("penalty", ["l2"])
                params["class_weight"] = "balanced"
                
            elif model_name in ["randomforest", "rf"]:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 4, 15)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
                params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
                params["class_weight"] = "balanced"
                
            elif model_name in ["extratrees", "et"]:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 4, 15)
                params["class_weight"] = "balanced"
                
            elif model_name in ["gradientboosting", "gb"]:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
                params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
                
            elif model_name in ["xgboost", "xgb"]:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
                params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
                ratio = (len(y) - np.sum(y)) / (np.sum(y) + 1e-9)
                params["scale_pos_weight"] = float(ratio)
                
            elif model_name in ["lightgbm", "lgbm", "lgb"]:
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
                params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
                params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
                params["class_weight"] = "balanced"
                params["verbose"] = -1
                
            elif model_name in ["catboost", "cat"]:
                params["iterations"] = trial.suggest_int("iterations", 50, 300)
                params["depth"] = trial.suggest_int("depth", 4, 10)
                params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
                params["auto_class_weights"] = "Balanced"
                params["verbose"] = 0
                
            else:
                raise ValueError(f"Tuning not configured for {model_type}")

            # We optimize for F1-Score to balance precision and recall on imbalanced data
            metrics = self.cross_validate(model_type, X, y, params)
            return metrics["f1"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Mejor valor F1: {study.best_value:.4f}")
        print("Mejores parámetros:", study.best_params)
        
        # Add class weight configuration back to best params
        best_params = study.best_params.copy()
        if model_name in ["logisticregression", "randomforest", "extratrees", "lightgbm", "lgb"]:
            best_params["class_weight"] = "balanced"
        elif model_name in ["catboost", "cat"]:
            best_params["auto_class_weights"] = "Balanced"
        elif model_name in ["xgboost", "xgb"]:
            ratio = (len(y) - np.sum(y)) / (np.sum(y) + 1e-9)
            best_params["scale_pos_weight"] = float(ratio)
            
        return best_params

    def train_and_evaluate_final(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray, params: Dict[str, Any] = None) -> Tuple[PredictiveModel, Dict[str, Any]]:
        """
        Trains the final model on train set and evaluates on hold-out test set.
        """
        model = ModelFactory.create_model(model_type, **(params or {}))
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            }
        }
        
        return model, metrics
