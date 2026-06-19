import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.domain.services.predictive_model import PredictiveModel

class LogisticRegressionModel(PredictiveModel):
    """
    Logistic Regression Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"max_iter": 1000, "random_state": 42}
        default_params.update(params)
        self.model = LogisticRegression(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        coefs = self.model.coef_[0]
        abs_coefs = np.abs(coefs)
        total = np.sum(abs_coefs) + 1e-9
        normalized_coefs = abs_coefs / total
        return dict(zip(feature_names, [float(v) for v in normalized_coefs]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "LogisticRegressionModel":
        self.model.set_params(**params)
        return self


class RandomForestModel(PredictiveModel):
    """
    Random Forest Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"random_state": 42, "n_jobs": -1}
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "RandomForestModel":
        self.model.set_params(**params)
        return self


class ExtraTreesModel(PredictiveModel):
    """
    Extra Trees Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"random_state": 42, "n_jobs": -1}
        default_params.update(params)
        self.model = ExtraTreesClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "ExtraTreesModel":
        self.model.set_params(**params)
        return self


class GradientBoostingModel(PredictiveModel):
    """
    Gradient Boosting Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"random_state": 42}
        default_params.update(params)
        self.model = GradientBoostingClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "GradientBoostingModel":
        self.model.set_params(**params)
        return self


class XGBoostModel(PredictiveModel):
    """
    XGBoost Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"random_state": 42, "n_jobs": -1, "eval_metric": "logloss"}
        default_params.update(params)
        self.model = XGBClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "XGBoostModel":
        self.model.set_params(**params)
        return self


class LightGBMModel(PredictiveModel):
    """
    LightGBM Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"random_state": 42, "n_jobs": -1, "verbose": -1}
        default_params.update(params)
        self.model = LGBMClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        # Normalize LightGBM feature importances (which are raw split counts by default)
        total = np.sum(importances) + 1e-9
        normalized = importances / total
        return dict(zip(feature_names, [float(v) for v in normalized]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "LightGBMModel":
        self.model.set_params(**params)
        return self


class CatBoostModel(PredictiveModel):
    """
    CatBoost Model strategy wrapper.
    """
    def __init__(self, **params):
        default_params = {"random_state": 42, "verbose": 0}
        default_params.update(params)
        self.model = CatBoostClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        importances = self.model.feature_importances_
        # Normalize CatBoost feature importances (which sum to 100 by default)
        total = np.sum(importances) + 1e-9
        normalized = importances / total
        return dict(zip(feature_names, [float(v) for v in normalized]))

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **params) -> "CatBoostModel":
        # CatBoost doesn't fully support set_params in the same sklearn way for all variables, 
        # so we recreate the classifier if parameters change, but standard set_params works for most
        for k, v in params.items():
            if hasattr(self.model, k):
                setattr(self.model, k, v)
        return self


class ModelFactory:
    """
    Factory class to instantiate predictive models.
    """
    @staticmethod
    def create_model(model_type: str, **params) -> PredictiveModel:
        model_type = model_type.lower().replace(" ", "").replace("_", "")
        if model_type in ["logisticregression", "lr"]:
            return LogisticRegressionModel(**params)
        elif model_type in ["randomforest", "rf"]:
            return RandomForestModel(**params)
        elif model_type in ["extratrees", "et"]:
            return ExtraTreesModel(**params)
        elif model_type in ["gradientboosting", "gb"]:
            return GradientBoostingModel(**params)
        elif model_type in ["xgboost", "xgb"]:
            return XGBoostModel(**params)
        elif model_type in ["lightgbm", "lgbm", "lgb"]:
            return LightGBMModel(**params)
        elif model_type in ["catboost", "cat"]:
            return CatBoostModel(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
