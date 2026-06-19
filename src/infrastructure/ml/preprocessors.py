import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data Pipeline preprocessor implementing Cleaning, Feature Engineering, Imputation,
    Automatic Scaling Selection, and Categorical Encoding.
    """
    def __init__(self, allowed_programs: List[str] = None):
        self.allowed_programs = allowed_programs
        self.fitted = False
        
        # Scaling maps
        self.scalers = {}
        self.scaler_types = {}  # Store which scaler was selected for each feature
        
        # Categorical maps
        self.categorical_mappings = {}
        self.categorical_cols = [
            "PROGRAMA", "JORNADA", "GENERO", "CIUDADRESIDENCIA", 
            "INFE_SITUACIONPADRES", "CATEGORIA", "NIVEL_SISBEN"
        ]
        
        # Numeric columns to process
        self.numeric_cols = [
            "CREDITOSAPROBADOS", "UBICACION_SEMESTRAL", "PROMEDIO_GENERAL",
            "ESTRATO", "TIENE_SISBEN", "INFE_VIVECONFAMILIA",
            "INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS", 
            "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA",
            "EDAD_INGRESO", "ANIO_INGRESO", "MES_INGRESO",
            # Engineered features:
            "CREDITOS_POR_SEMESTRE", "RENDIMIENTO_GENERAL", 
            "TASA_VULNERABILIDAD", "APOYO_FAMILIAR"
        ]
        
        # Imputations values (computed during fit)
        self.imputation_values = {}

    def _clean_and_engineer(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        
        # Normalize column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Filter programs if specified (only during training, or if requested)
        if self.allowed_programs:
            df = df[df["PROGRAMA"].isin(self.allowed_programs)].copy()

        # Parse Datetimes and calculate Age
        def parse_date(col_name):
            if col_name not in df.columns:
                return pd.Series(pd.NaT, index=df.index)
            # Normalize text dates
            text = df[col_name].astype(str).str.strip()
            text = text.str.replace("\u202f", " ", regex=False)
            text = text.str.replace("\xa0", " ", regex=False)
            text = text.str.replace(r"(?i)\ba\.?\s*m\.?\b", "AM", regex=True)
            text = text.str.replace(r"(?i)\bp\.?\s*m\.?\b", "PM", regex=True)
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
            fallback = parsed.isna()
            if fallback.any():
                parsed.loc[fallback] = pd.to_datetime(text.loc[fallback], errors="coerce", dayfirst=False)
            return parsed

        ingreso = parse_date("ESTP_FECHAINGRESO")
        nacimiento = parse_date("FECHA_NACIMIENTO")
        
        df["EDAD_INGRESO"] = ((ingreso - nacimiento).dt.days / 365.25).round()
        df["ANIO_INGRESO"] = ingreso.dt.year
        df["MES_INGRESO"] = ingreso.dt.month
        df = df.drop(columns=["ESTP_FECHAINGRESO", "FECHA_NACIMIENTO"], errors="ignore")

        # Clean numeric columns of commas and spaces before parsing
        numeric_cols_to_clean = [
            "CREDITOSAPROBADOS", "UBICACION_SEMESTRAL", "PROMEDIO_GENERAL",
            "ESTRATO", "TIENE_SISBEN", "INFE_VIVECONFAMILIA",
            "INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS", 
            "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA"
        ]
        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)

        # Calculate Academic features
        df["CREDITOSAPROBADOS"] = pd.to_numeric(df["CREDITOSAPROBADOS"], errors="coerce")
        df["UBICACION_SEMESTRAL"] = pd.to_numeric(df["UBICACION_SEMESTRAL"], errors="coerce")
        df["PROMEDIO_GENERAL"] = pd.to_numeric(df["PROMEDIO_GENERAL"], errors="coerce")
        
        # Academic Progress Rate (credits approved per semester)
        # Avoid division by zero
        sem_loc = df["UBICACION_SEMESTRAL"].fillna(1).clip(lower=1)
        df["CREDITOS_POR_SEMESTRE"] = (df["CREDITOSAPROBADOS"].fillna(0) / sem_loc).astype(float)
        
        # Academic Efficiency
        df["RENDIMIENTO_GENERAL"] = (df["PROMEDIO_GENERAL"].fillna(0) * df["CREDITOSAPROBADOS"].fillna(0)).astype(float)

        # Handle Estrato anomalies (estrato must be 1 to 6)
        df["ESTRATO"] = pd.to_numeric(df["ESTRATO"], errors="coerce")
        df.loc[(df["ESTRATO"] < 1) | (df["ESTRATO"] > 6), "ESTRATO"] = np.nan

        # Clean binary columns
        df["TIENE_SISBEN"] = pd.to_numeric(df["TIENE_SISBEN"], errors="coerce")
        df.loc[~df["TIENE_SISBEN"].isin([0, 1]), "TIENE_SISBEN"] = np.nan

        df["INFE_VIVECONFAMILIA"] = pd.to_numeric(df["INFE_VIVECONFAMILIA"], errors="coerce")
        df.loc[~df["INFE_VIVECONFAMILIA"].isin([0, 1]), "INFE_VIVECONFAMILIA"] = np.nan

        # Clean other socio-demographic
        for col in ["INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS", "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean SISBEN Level
        if "NIVEL_SISBEN" in df.columns:
            # Extract main group letter (e.g. A4 -> A, B1 -> B)
            df["NIVEL_SISBEN"] = df["NIVEL_SISBEN"].astype(str).str.strip().str.upper()
            df["NIVEL_SISBEN"] = df["NIVEL_SISBEN"].apply(lambda x: x[0] if isinstance(x, str) and len(x) > 0 and x[0].isalpha() else x)
            df.loc[df["NIVEL_SISBEN"] == "N", "NIVEL_SISBEN"] = np.nan # Null represented as NAN or empty string
            df.loc[df["NIVEL_SISBEN"] == "NAN", "NIVEL_SISBEN"] = np.nan
        else:
            df["NIVEL_SISBEN"] = np.nan

        # Engineering Socioeconomic Index (Tasa de Vulnerabilidad)
        # Vulnerability increases if: lower stratum, has Sisben, lives alone
        estrato_v = df["ESTRATO"].fillna(3)
        tiene_sisben_v = df["TIENE_SISBEN"].fillna(0)
        vive_fam_v = df["INFE_VIVECONFAMILIA"].fillna(1)
        
        # High value = high vulnerability (normalize between 0 and 1)
        # Stratum: (6 - estrato) / 5 (since 1 is highest vulnerability, 6 is lowest)
        # Sisben: +0.5 vulnerability
        # Live alone: +0.3 vulnerability
        df["TASA_VULNERABILIDAD"] = ((6.0 - estrato_v) / 5.0) + (tiene_sisben_v * 0.5) + ((1.0 - vive_fam_v) * 0.3)

        # Support Family Index
        # Family support increases with: parents living together, more working members
        parents_v = df["INFE_SITUACIONPADRES"].astype(str).str.strip().str.upper().fillna("DESCONOCIDO")
        parents_score = parents_v.map({
            "VIVOS Y CONVIVEN": 1.0,
            "VIVOS Y SEPARADOS": 0.5,
            "MADRE VIVA - PADRE DIFUNTO": 0.4,
            "PADRE VIVO - MADRE DIFUNTA": 0.4,
            "DIFUNTOS": 0.1,
            "DESCONOCIDO": 0.5
        }).fillna(0.5)
        
        trabajadores = df["INFE_NUMMIEMBROSTRABAJA"].fillna(1).clip(upper=5)
        df["APOYO_FAMILIAR"] = parents_score * 0.7 + (trabajadores / 5.0) * 0.3

        # Drop ID / leakage / zero-variance columns
        drop_cols = ["CODESTUDIANTE", "CODIGOCIUDADR", "CODMATRICULA", "SEDE", "INFE_HERMANOSESTUDIANDOU"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        return df

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DataPreprocessor":
        df = self._clean_and_engineer(X)
        
        # Remove target-related columns from feature computation
        if "SITUACION" in df.columns:
            df = df.drop(columns=["SITUACION"])
        if "TARGET" in df.columns:
            df = df.drop(columns=["TARGET"])
            
        # 1. Compute Imputation Values (Medians for numericals, Modes for categoricals)
        for col in df.columns:
            if col in self.categorical_cols:
                self.imputation_values[col] = df[col].mode().iloc[0] if not df[col].mode().empty else "DESCONOCIDO"
            else:
                self.imputation_values[col] = df[col].median() if not pd.isna(df[col].median()) else 0.0

        # Fill nulls temporarily to train encoders
        df_imputed = df.copy()
        for col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(self.imputation_values.get(col, 0))

        # 2. Learn Categorical Mappings (with rare grouping)
        for col in self.categorical_cols:
            if col in df_imputed.columns:
                counts = df_imputed[col].value_counts()
                # Group categories that represent less than 1% of the dataset
                total_count = len(df_imputed)
                frequent_categories = counts[counts / total_count >= 0.01].index.tolist()
                
                # Ensure we have a mapping
                mapping = {cat: cat for cat in frequent_categories}
                self.categorical_mappings[col] = mapping

        # 3. Fit Scalers with Adaptive selection based on data statistics
        # We transform categorical columns first to compute the scaled columns
        df_encoded = self._encode_categoricals(df_imputed)
        
        # Keep list of expected columns after encoding
        self.feature_names_in_ = df_encoded.columns.tolist()
        if "SITUACION" in self.feature_names_in_:
            self.feature_names_in_.remove("SITUACION")
        if "TARGET" in self.feature_names_in_:
            self.feature_names_in_.remove("TARGET")
            
        # Determine scaling strategy for each column
        for col in self.feature_names_in_:
            col_data = df_encoded[col]
            
            # Simple outlier detection based on IQR
            q25, q75 = np.percentile(col_data, 25), np.percentile(col_data, 75)
            iqr = q75 - q25
            if iqr > 0:
                outliers = col_data[(col_data < q25 - 1.5 * iqr) | (col_data > q75 + 1.5 * iqr)]
                outlier_fraction = len(outliers) / len(col_data)
            else:
                outlier_fraction = 0.0

            # Selection rule:
            # - If outlier fraction > 5%, use RobustScaler (robust to outliers)
            # - If data is bounded and has no outliers, use MinMaxScaler
            # - Else use StandardScaler
            is_binary = col_data.nunique() <= 2
            
            if is_binary:
                # Binary features don't need heavy scaling, MinMaxScaler keeps them [0, 1]
                self.scaler_types[col] = "minmax"
                self.scalers[col] = MinMaxScaler().fit(col_data.values.reshape(-1, 1))
            elif outlier_fraction > 0.05:
                self.scaler_types[col] = "robust"
                self.scalers[col] = RobustScaler().fit(col_data.values.reshape(-1, 1))
            elif col_data.min() >= 0 and col_data.max() <= 5: # Likert scales or GPA
                self.scaler_types[col] = "minmax"
                self.scalers[col] = MinMaxScaler().fit(col_data.values.reshape(-1, 1))
            else:
                self.scaler_types[col] = "standard"
                self.scalers[col] = StandardScaler().fit(col_data.values.reshape(-1, 1))
                
        self.fitted = True
        return self

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        
        # Apply mapping for each categorical column, fall back to "OTROS"
        for col in self.categorical_cols:
            if col in df_out.columns:
                mapping = self.categorical_mappings.get(col, {})
                df_out[col] = df_out[col].astype(str).str.strip().str.upper()
                df_out[col] = df_out[col].map(mapping).fillna("OTROS")
                
        # Perform One-Hot Encoding via pandas get_dummies
        # To make it deterministic, we construct dummy variables for each known category
        for col in self.categorical_cols:
            if col in df_out.columns:
                known_cats = list(set(self.categorical_mappings.get(col, {}).values()))
                if "OTROS" not in known_cats:
                    known_cats.append("OTROS")
                
                # Add one column per known category
                for cat in sorted(known_cats):
                    dummy_col = f"{col}_{cat}"
                    df_out[dummy_col] = (df_out[col] == cat).astype(int)
                    
                df_out = df_out.drop(columns=[col])
                
        return df_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("DataPreprocessor has not been fitted yet.")
            
        df = self._clean_and_engineer(X)
        
        # Handle target column if it's there (drop it, we don't scale it)
        target_col = None
        if "SITUACION" in df.columns:
            target_col = df["SITUACION"]
            df = df.drop(columns=["SITUACION"])
        if "TARGET" in df.columns:
            df = df.drop(columns=["TARGET"])

        # Impute missing values
        for col in df.columns:
            val = self.imputation_values.get(col, 0.0)
            df[col] = df[col].fillna(val)

        # Apply Categorical Encoding
        df_encoded = self._encode_categoricals(df)

        # Align columns to match the fitted ones (fill missing with 0, drop extra)
        for col in self.feature_names_in_:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
                
        df_aligned = df_encoded[self.feature_names_in_].copy()

        # Apply Scalers
        for col in self.feature_names_in_:
            scaler = self.scalers[col]
            df_aligned[col] = scaler.transform(df_aligned[col].values.reshape(-1, 1)).flatten()

        return df_aligned
