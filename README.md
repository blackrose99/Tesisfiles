# Plataforma de Alerta Temprana y Predicción de Deserción Estudiantil (Rama Beta)

Este proyecto implementa una plataforma inteligente de predicción de deserción académica y análisis de riesgo estudiantil. Cuenta con una arquitectura desacoplada basada en **Clean Architecture**, algoritmos avanzados de AutoML (CatBoost optimizado con Optuna), explicabilidad mediante SHAP, y un panel web interactivo en Streamlit.

---

## ⚡ Guía de Inicio Rápido (Copy-Paste)

Copia y pega **todo el bloque** correspondiente a tu sistema operativo en la terminal para descargar dependencias, entrenar el modelo campeón y ejecutar el servidor web local.

### 🔴 Windows PowerShell (Recomendado)
> [!IMPORTANT]
> Si PowerShell arroja un error de ejecución de scripts de administrador al intentar activar el entorno, **esta línea soluciona el problema de forma automática** ejecutando directamente los binarios del entorno virtual sin activar scripts restrictivos de Windows:
```powershell
python -m venv .venv; & .venv/Scripts/python.exe -m pip install --upgrade pip; & .venv/Scripts/python.exe -m pip install -r requirements.txt; & .venv/Scripts/python.exe train.py; & .venv/Scripts/streamlit.exe run app.py
```

*Si deseas forzar la activación clásica en PowerShell omitiendo las restricciones políticas de Windows de forma segura (sin privilegios de Administrador), puedes ejecutar:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; .venv\Scripts\Activate.ps1
```

### ⚪ Windows CMD (Consola Clásica)
```cmd
python -m venv .venv && .venv\Scripts\python.exe -m pip install --upgrade pip && .venv\Scripts\python.exe -m pip install -r requirements.txt && .venv\Scripts\python.exe train.py && .venv\Scripts\streamlit.exe run app.py
```

### 🔵 Linux / macOS
```bash
python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt && .venv/bin/python train.py && .venv/bin/streamlit run app.py
```

---

## 📁 Estructura General del Proyecto

```
Tesisfiles/
├── dataset/                     # Directorio unificado de insumos y bases de datos
│   ├── student_database.xlsx    # Archivo Excel original con datos históricos de estudiantes
│   ├── students_with_status.csv # Dataset depurado de estudiantes con estado resuelto
│   ├── students_without_status.csv # Dataset para predicción masiva sin etiquetas
│   ├── sample_50_students.csv   # Muestra rápida de 50 estudiantes para demostraciones
│   └── original_documents/      # Histórico original de documentos de la universidad
├── src/                         # Código fuente estructurado (Clean Architecture)
│   ├── domain/                  # Entidades nucleares e interfaces (SOLID)
│   │   ├── entities/            # Entidades Student y Prediction
│   │   ├── repositories/        # Contratos StudentRepository y ModelRepository
│   │   └── services/            # Interfaz de estrategia de modelo predictivo
│   ├── infrastructure/          # Implementaciones tecnológicas
│   │   ├── ml/                  # Algoritmos, preprocesadores y entrenadores
│   │   ├── reports/             # Generador de reportes en PDF
│   │   └── repositories/        # Repositorios concretos de persistencia (Excel/JSON)
│   └── application/             # Casos de uso de negocio (Use Cases)
│       ├── train_use_case.py    # Flujo de reentrenamiento, comparación y tuning
│       ├── predict_use_case.py  # Predicción en lotes / individuales y cálculo SHAP
│       └── monitor_use_case.py  # Detección de Data Drift (KS-Test)
├── model_registry/              # Historial estructurado y versionamiento de modelos
│   ├── registry.json            # Metadatos del modelo activo y métricas históricas
│   └── models/                  # Binarios de preprocesadores y modelos por versión
├── app.py                       # Interfaz gráfica web interactiva de Streamlit
├── train.py                     # Entrypoint de consola para el reentrenamiento del modelo
├── requirements.txt             # Dependencias del proyecto actualizadas
└── README.md                    # Este manual técnico
```

---

## ⚙️ Funcionalidades Clave

1. **AutoML y Selección**: Compara automáticamente en cada reentrenamiento 7 algoritmos diferentes (Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, y CatBoost) usando Validación Cruzada de 5 pliegues y selecciona el campeón.
2. **Optimización Bayesiana**: Sintoniza los parámetros del modelo campeón usando **Optuna** para exprimir el rendimiento de clasificación.
3. **Fuga Temporal Resuelta**: Se excluyó la variable de año de ingreso para asegurar que la inteligencia artificial prediga basándose en variables académicas/socioeconómicas en lugar del año calendario.
4. **Monitoreo de Data Drift**: Implementa la prueba estadística Kolmogorov-Smirnov (KS-test) para advertir al administrador cuando los perfiles de los nuevos estudiantes cargados difieran significativamente de la muestra de entrenamiento original.
5. **Reportes PDF**: Descarga fichas personalizadas para cada estudiante con planes de intervención sugeridos según su perfil de riesgo.

---

## 📦 Empaquetado para Windows (.exe)
Para distribuir la plataforma a usuarios finales que no tienen instalado Python, puedes compilar un ejecutable autocontenido de doble clic mediante **PyInstaller**.

1. Instala PyInstaller:
   ```bash
   .venv/bin/pip install pyinstaller   # Linux
   .venv\Scripts\pip.exe install pyinstaller # Windows
   ```
2. Ejecuta el comando de compilación:
   ```bash
   pyinstaller --name "AlertaTempranaAcademica" --onefile --add-data "model_registry:model_registry" --add-data "shap_background.npy:." --add-data "shap_feature_names.npy:." --add-data "src:src" app.py
   ```
3. Ubica el ejecutable en la carpeta `dist/AlertaTempranaAcademica.exe`. Distribúyelo junto a la carpeta `dataset/`. Los usuarios finales solo requerirán hacer **doble clic** para ejecutar la plataforma.
