import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, List, Tuple
from src.domain.repositories.model_repository import ModelRepository

class ModelRepositoryImpl(ModelRepository):
    """
    Concrete implementation of ModelRepository saving to local directory registry.
    """
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self.monitoring_file = os.path.join(registry_dir, "monitoring.json")
        
        os.makedirs(self.models_dir, exist_ok=True)
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w") as f:
                json.dump({"active_version": None, "history": []}, f, indent=4)
        if not os.path.exists(self.monitoring_file):
            with open(self.monitoring_file, "w") as f:
                json.dump([], f, indent=4)

    def save_model(self, model: Any, preprocessor: Any, metrics: Dict[str, float], model_name: str) -> str:
        # Load registry
        with open(self.registry_file, "r") as f:
            registry = json.load(f)
            
        version_num = len(registry["history"]) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = f"v{version_num}_{timestamp}"
        
        # Save model and preprocessor
        model_filename = f"model_{version_str}.joblib"
        prep_filename = f"preprocessor_{version_str}.joblib"
        
        # In case it's a keras model, handle it, but standard sklearn/xgb/lgb models use joblib
        model_path = os.path.join(self.models_dir, model_filename)
        prep_path = os.path.join(self.models_dir, prep_filename)
        
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, prep_path)
        
        # Metadata
        metadata = {
            "version": version_str,
            "model_name": model_name,
            "model_file": model_filename,
            "preprocessor_file": prep_filename,
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        registry["history"].append(metadata)
        registry["active_version"] = version_str
        
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=4)
            
        # Copy to root directory for easy access by legacy code / simpler loading
        # Let's save active paths in root
        joblib.dump(model, "modelo_desercion_nn.keras" if "keras" in str(type(model)).lower() else "model.joblib")
        joblib.dump(preprocessor, "scaler.joblib") # preprocessor contains scaling + encoding
        
        return version_str

    def load_model(self, version: str) -> Tuple[Any, Any, Dict[str, Any]]:
        with open(self.registry_file, "r") as f:
            registry = json.load(f)
            
        metadata = next((item for item in registry["history"] if item["version"] == version), None)
        if metadata is None:
            raise FileNotFoundError(f"Model version {version} not found in registry.")
            
        model_path = os.path.join(self.models_dir, metadata["model_file"])
        prep_path = os.path.join(self.models_dir, metadata["preprocessor_file"])
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        
        return model, preprocessor, metadata

    def load_latest_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        with open(self.registry_file, "r") as f:
            registry = json.load(f)
            
        active_version = registry["active_version"]
        if active_version is None:
            raise FileNotFoundError("No active model version in registry. Train a model first.")
            
        return self.load_model(active_version)

    def get_model_history(self) -> List[Dict[str, Any]]:
        with open(self.registry_file, "r") as f:
            registry = json.load(f)
        return registry["history"]

    def log_monitoring_metrics(self, drift_metrics: Dict[str, Any]) -> None:
        if os.path.exists(self.monitoring_file):
            with open(self.monitoring_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append({
            "timestamp": datetime.now().isoformat(),
            **drift_metrics
        })
        
        with open(self.monitoring_file, "w") as f:
            json.dump(logs, f, indent=4)
