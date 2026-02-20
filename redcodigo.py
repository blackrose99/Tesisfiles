# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 14:31:50 2026

@author: dajos
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================================================
# 0) Configuración básica (reproducibilidad)
# =========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 1) Cargar dataset final (ya limpio, sin NaN, numérico)
# =========================================================
ruta = "Base_de_datos_estudiantes_ready_step4_20260215_014516.xlsx"
df = pd.read_excel(ruta)

# Variable objetivo (target)
y = df["SITUACION"].astype(int)

# Variables predictoras (features)
X = df.drop(columns=["SITUACION"])

# =========================================================
# 2) Partición de datos: Train / Validation / Test (estratificado)
#    - Test: 20% (evaluación final)
#    - Validation: 20% del Train (para Early Stopping)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)

# =========================================================
# 3) Escalamiento (StandardScaler)
#    IMPORTANTE:
#    - fit SOLO con train
#    - transform para val y test
# =========================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# =========================================================
# 4) Pesos de clase (opcional, recomendado si quieres balancear el aprendizaje)
#    - Ajusta el entrenamiento para no favorecer solo la clase mayoritaria.
# =========================================================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("Pesos de clase:", class_weight)

# =========================================================
# 5) Arquitectura del modelo (MLP feed-forward)
#    - 2 capas ocultas: 64 y 32 neuronas
#    - ReLU en ocultas
#    - Dropout para regularización
#    - Salida Sigmoid (probabilidad de SITUACION=1)
# =========================================================
input_dim = X_train_s.shape[1]

model = Sequential([
    Dense(64, activation="relu", input_shape=(input_dim,)),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(1, activation="sigmoid")
])

# =========================================================
# 6) Compilación
#    - Loss: binary_crossentropy (clasificación binaria)
#    - Optimizer: Adam
#    - Métricas: AUC, Precision, Recall (útiles para interpretar clase 1)
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

# =========================================================
# 7) Early Stopping + ReduceLROnPlateau
#    - EarlyStopping: detiene si val_auc no mejora
#    - restore_best_weights: conserva el mejor punto (mejor generalización)
# =========================================================
callbacks = [
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================================================
# 8) Entrenamiento
# =========================================================
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,   # si no lo quieres, elimina esta línea
    callbacks=callbacks,
    verbose=1
)

# =========================================================
# 9) Evaluación en TEST (resultado final del modelo)
# =========================================================
print("\n=== Evaluación en TEST ===")
test_metrics = model.evaluate(X_test_s, y_test, verbose=0)
for name, val in zip(model.metrics_names, test_metrics):
    print(f"{name}: {val:.4f}")

# Probabilidades (para AUC y para umbral)
y_proba = model.predict(X_test_s, verbose=0).ravel()

# AUC-ROC y AUC-PR (opcional, útil como complemento)
print("AUC-ROC (sklearn):", round(roc_auc_score(y_test, y_proba), 4))
print("AUC-PR  (sklearn):", round(average_precision_score(y_test, y_proba), 4))

# Predicción con umbral (por defecto 0.50)
threshold = 0.50
y_pred = (y_proba >= threshold).astype(int)

print("\nMatriz de confusión (threshold=0.50):")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación (threshold=0.50):")
print(classification_report(y_test, y_pred, digits=4))

# =========================================================
# 10) Guardar modelo (y opcionalmente el scaler)
# =========================================================
model.save("modelo_desercion_nn.keras")
print("\nModelo guardado en: modelo_desercion_nn.keras")