import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Tuple
from src.domain.repositories.model_repository import ModelRepository

class MonitorUseCase:
    """
    Application Use Case to monitor model performance, data drift, and concept drift over time.
    """
    def __init__(self, model_repo: ModelRepository):
        self.model_repo = model_repo

    def detect_data_drift(self, incoming_features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects data drift in incoming predictions compared to training baseline (SHAP background).
        Uses Kolmogorov-Smirnov 2-sample test for numerical columns.
        """
        background_path = "shap_background.npy"
        feature_names_path = "shap_feature_names.npy"
        
        if not os.path.exists(background_path) or not os.path.exists(feature_names_path):
            return {"status": "error", "message": "Baseline training data not found. Train model first."}
            
        # Load training baseline (scaled features)
        baseline = np.load(background_path)
        feature_names = np.load(feature_names_path, allow_pickle=True).tolist()
        
        # Load latest model metadata to get preprocessor
        _, prep_wrapper, _ = self.model_repo.load_latest_model()
        preprocessor = prep_wrapper["preprocessor"]
        
        # Preprocess incoming features to scale them the same way
        incoming_transformed = preprocessor.transform(incoming_features_df)
        
        # Keep only the features the model was trained on
        incoming_scaled = incoming_transformed[feature_names].values
        
        drift_results = {}
        drift_detected_count = 0
        
        for idx, feat in enumerate(feature_names):
            if idx >= baseline.shape[1] or idx >= incoming_scaled.shape[1]:
                continue
                
            baseline_feat = baseline[:, idx]
            incoming_feat = incoming_scaled[:, idx]
            
            # Run KS test
            # H0: The two samples are from the same distribution
            stat, p_val = ks_2samp(baseline_feat, incoming_feat)
            
            # If p-value < 0.05, we reject H0 (distributions are significantly different)
            drifted = bool(p_val < 0.05)
            
            if drifted:
                drift_detected_count += 1
                
            drift_results[feat] = {
                "ks_statistic": float(stat),
                "p_value": float(p_val),
                "drift_detected": drifted
            }
            
        drift_fraction = drift_detected_count / len(feature_names) if feature_names else 0.0
        
        # Overall status
        # If > 30% of features show drift, trigger general alert
        overall_status = "Estable"
        if drift_fraction > 0.30:
            overall_status = "Drift Detectado"
        elif drift_fraction > 0.10:
            overall_status = "Advertencia"
            
        summary = {
            "status": "success",
            "overall_status": overall_status,
            "drift_fraction": float(drift_fraction),
            "total_features": len(feature_names),
            "drifted_features_count": drift_detected_count,
            "features_drift": drift_results
        }
        
        # Log drift metrics in model repository
        self.model_repo.log_monitoring_metrics(summary)
        
        return summary

    def check_concept_drift(self, y_true: np.ndarray, y_pred: np.ndarray, baseline_f1: float) -> Dict[str, Any]:
        """
        Detects concept drift (model performance degradation) by comparing F1 score of 
        new actuals against baseline training/validation F1.
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        current_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        current_acc = float(accuracy_score(y_true, y_pred))
        
        degradation = baseline_f1 - current_f1
        
        status = "Estable"
        if degradation > 0.15:
            status = "Drift Crítico"
        elif degradation > 0.05:
            status = "Advertencia"
            
        summary = {
            "metric": "F1-Score",
            "baseline_value": baseline_f1,
            "current_value": current_f1,
            "current_accuracy": current_acc,
            "degradation": degradation,
            "status": status
        }
        
        # Log metrics
        self.model_repo.log_monitoring_metrics({"concept_drift": summary})
        
        return summary
