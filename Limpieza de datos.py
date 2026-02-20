# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 22:36:01 2026

@author: dajos
"""
import pandas as pd
from datetime import datetime


# Cargar el archivo Excel
ruta_archivo = "Base de datos estudiantes.xlsx"
df = pd.read_excel(ruta_archivo)

# 1) Eliminar columnas que no se usarán en el estudio
df = df.drop(columns=[
    "CODESTUDIANTE",
    "CODIGOCIUDADR",
    "NIVEL_SISBEN",
    "CATEGORIA",
    "CODMATRICULA",
    "SEDE",
    "INFE_HERMANOSESTUDIANDOU"
])

# 2) Normalizar fechas (dejar fecha sin hora, en formato datetime)
df["ESTP_FECHAINGRESO"] = pd.to_datetime(df["ESTP_FECHAINGRESO"], errors="coerce").dt.normalize()
df["FECHA_NACIMIENTO"]  = pd.to_datetime(df["FECHA_NACIMIENTO"],  errors="coerce").dt.normalize()

# PUNTO 1) Fechas -> variables numéricas y luego eliminar fechas
df["EDAD_INGRESO"] = ((df["ESTP_FECHAINGRESO"] - df["FECHA_NACIMIENTO"]).dt.days / 365.25).round().astype("Int64")
df["ANIO_INGRESO"] = df["ESTP_FECHAINGRESO"].dt.year
df["MES_INGRESO"]  = df["ESTP_FECHAINGRESO"].dt.month
df = df.drop(columns=["ESTP_FECHAINGRESO", "FECHA_NACIMIENTO"])

# PUNTO 3) ESTRATO: reemplazar valores fuera de rango por vacío (NaN)
df["ESTRATO"] = pd.to_numeric(df["ESTRATO"], errors="coerce")
df.loc[(df["ESTRATO"] < 1) | (df["ESTRATO"] > 6), "ESTRATO"] = pd.NA

# 3) Conservar solo los programas definidos en el alcance
allowed_programs = [
    "INGENIERIA DE SISTEMAS",
    "TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS"
]
df = df.drop(df[~df["PROGRAMA"].isin(allowed_programs)].index)

# 4) Convertir SITUACION a binaria: 1 para estados definidos, 0 para el resto
situaciones_1 = [
    "EXCLUIDO NO RENOVACION DE MATRICULA",
    "PFI",
    "EXCLUIDO CANCELACION SEMESTRE",
    "INACTIVO"
]
df["SITUACION"] = df["SITUACION"].isin(situaciones_1).astype(int)

# 5) Recodificar CIUDADRESIDENCIA (1–4) y el resto como 5
df["CIUDADRESIDENCIA"] = df["CIUDADRESIDENCIA"].astype(str).str.strip()
map_ciudad = {
    "BUCARAMANGA": 1,
    "FLORIDABLANCA": 2,
    "GIRON": 3,
    "PIEDECUESTA": 4
}
df["CIUDADRESIDENCIA"] = df["CIUDADRESIDENCIA"].map(map_ciudad).fillna(5).astype(int)

# 6) Identificar automáticamente columnas para One-Hot Encoding
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols_onehot = [c for c in cat_cols if df[c].nunique(dropna=True) > 1]
print("Columnas a One-Hot:", cat_cols_onehot)

# 7) Aplicar One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=cat_cols_onehot)

# 8) Convertir booleanos a 0/1 (si existieran)
bool_cols = df_onehot.select_dtypes(include=["bool"]).columns.tolist()
if bool_cols:
    df_onehot[bool_cols] = df_onehot[bool_cols].astype(int)

# ---------------------------------------------------------
# Limpieza EXTRA (sin medias/medianas y sin columnas nuevas):
# reemplazar NaN por códigos "NO REPORTADO"
# ---------------------------------------------------------
# ESTRATO (válido 1–6): 0 = no reportado
if "ESTRATO" in df_onehot.columns:
    df_onehot["ESTRATO"] = df_onehot["ESTRATO"].fillna(0).astype(int)

# Conteos INFE_*: -1 = no reportado
cols_no_reportado_menos1 = [
    "INFE_NUMEROFAMILIARES",
    "INFE_NUMEROHERMANOS",
    "INFE_POSICIONENHERMANOS",
    "INFE_NUMMIEMBROSTRABAJA",
    "EDAD_INGRESO"
]
for c in cols_no_reportado_menos1:
    if c in df_onehot.columns:
        df_onehot[c] = pd.to_numeric(df_onehot[c], errors="coerce").fillna(-1).round().astype(int)

# TIENE_SISBEN: 0/1, y -1 = no reportado
if "TIENE_SISBEN" in df_onehot.columns:
    df_onehot["TIENE_SISBEN"] = pd.to_numeric(df_onehot["TIENE_SISBEN"], errors="coerce").round().fillna(-1).astype(int)

# QC: asegurar que ya no queden NaN
total_nan = int(df_onehot.isna().sum().sum())
print("Total NaN restantes:", total_nan)

print(df_onehot.head(5))
print("Filas:", len(df_onehot), "| Columnas:", df_onehot.shape[1])

# 9) Guardar Excel con nombre único
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
salida = f"Base_de_datos_estudiantes_ready_step4_{timestamp}.xlsx"

with pd.ExcelWriter(salida, engine="openpyxl") as writer:
    df_onehot.to_excel(writer, index=False, sheet_name="data")

print("Archivo nuevo generado:", salida)

