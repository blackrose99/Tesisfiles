# Plataforma Inteligente de Alerta Temprana y Predicción de Deserción Estudiantil

Este proyecto implementa una plataforma avanzada para predecir el riesgo de deserción académica y generar recomendaciones automatizadas de intervención institucional. La plataforma cuenta con una arquitectura desacoplada basada en **Clean Architecture**, algoritmos avanzados de Machine Learning (CatBoost optimizado con Optuna), explicabilidad individual mediante SHAP (evaluada perezosamente bajo demanda) y un panel interactivo desarrollado en Streamlit.

---

## ⚡ Guía de Inicio Rápido (Comandos Uno a Uno)

Para garantizar una configuración exitosa libre de restricciones políticas en Windows PowerShell, ejecuta los siguientes comandos uno por uno en la consola:

### 🔴 Windows PowerShell

```powershell
# 1. Crear el entorno virtual de Python
python -m venv .venv

# 2. Omitir políticas de restricción de ejecución de scripts para la sesión actual
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 3. Activar el entorno virtual de Python (.venv)
.venv\Scripts\Activate.ps1

# 4. Actualizar el gestor de paquetes pip a su última versión
python -m pip install --upgrade pip

# 5. Instalar todas las dependencias requeridas del proyecto
pip install -r requirements.txt

# 6. Entrenar el modelo campeón (CatBoost optimizado con Optuna)
python train.py

# 7. Ejecutar el servidor web local interactivo de Streamlit
streamlit run app.py
```

### 🔵 Linux / macOS

```bash
# 1. Crear el entorno virtual de Python
python3 -m venv .venv

# 2. Activar el entorno virtual de Python
source .venv/bin/activate

# 3. Actualizar el gestor de paquetes pip
pip install --upgrade pip

# 4. Instalar las dependencias del proyecto
pip install -r requirements.txt

# 5. Ejecutar el reentrenamiento del modelo campeón
python train.py

# 6. Iniciar la aplicación en Streamlit
streamlit run app.py
```

---

## 🚀 Lanzador Rápido para Windows (`run_app.bat`)

Si estás en Windows y prefieres no interactuar con la línea de comandos, hemos diseñado un iniciador automatizado nativo de doble clic:

[![Descargar run_app.bat](https://img.shields.io/badge/Descargar-run__app.bat-blue?style=for-the-badge&logo=windows)](https://raw.githubusercontent.com/blackrose99/Tesisfiles/beta/run_app.bat)

Al hacer **doble clic** sobre `run_app.bat`:
1. Detecta automáticamente si tienes Python instalado (si no, abre el instalador oficial).
2. Crea el entorno virtual `.venv` e instala las dependencias de forma automatizada.
3. Entrena el modelo si no existe un binario activo en el sistema.
4. Lanza el servidor Streamlit y abre la aplicación en tu navegador de forma transparente.

Esta opción requiere tener Python instalable en el equipo. Si quieres una app que **no dependa de Python en absoluto**, usa el `.exe` de la siguiente sección.

---

## 📦 Descargar el ejecutable (`.exe`) — para usuarios sin conocimientos técnicos

No necesitas instalar Python ni usar la consola. Haz clic en el botón, espera a que descargue y abre el archivo:

[![Descargar AlertaTemprana.exe](https://img.shields.io/badge/Descargar-AlertaTemprana.exe-success?style=for-the-badge&logo=windows)](https://github.com/blackrose99/Tesisfiles/releases/download/latest-exe/AlertaTemprana.exe)

Este enlace **siempre apunta a la última versión compilada automáticamente**: un flujo de GitHub Actions ([.github/workflows/build-exe.yml](.github/workflows/build-exe.yml)) reentrena el modelo y reconstruye el `.exe` cada vez que se sube un cambio a las ramas `main` o `beta`, y lo publica sobrescribiendo el mismo release (`latest-exe`). No tienes que hacer nada manualmente para que el enlace quede actualizado tras un push.

### Persistencia: el `.exe` se comporta igual que Streamlit

La primera vez que ejecutas `AlertaTemprana.exe`, se crea **junto al propio archivo `.exe`** (no en una carpeta temporal) una estructura idéntica a la del proyecto: `dataset/`, `model_registry/`, `model.joblib`, `scaler.joblib`, etc. Todo lo que hagas después —reentrenar el modelo, subir un nuevo dataset, generar reportes— se guarda ahí y **persiste entre cierres y aperturas del programa**, exactamente igual que cuando corres `streamlit run app.py` desde el repositorio. Si quieres reiniciar a un estado limpio, borra esa carpeta de datos junto al `.exe`: se regenerará con el modelo de fábrica en el siguiente arranque.

> Recomendación: coloca `AlertaTemprana.exe` en su propia carpeta (por ejemplo `Documentos\AlertaTemprana\`) antes de abrirlo, ya que ahí es donde se guardarán los datos.

---

## 🛠️ Recompilar el `.exe` manualmente (para desarrolladores)

Normalmente no necesitas hacer esto: el `.exe` se reconstruye solo en GitHub Actions con cada push. Pero si quieres generar una build local (por ejemplo para probar un cambio antes de subirlo a la rama), hazlo desde una máquina Windows:

1. **Activa tu entorno virtual e instala PyInstaller**:
   ```powershell
   pip install pyinstaller
   ```
2. **Entrena el modelo si aún no existen `model.joblib` / `scaler.joblib` / los archivos `shap_*.npy`** (el workflow de CI siempre lo hace antes de compilar):
   ```powershell
   python train.py
   ```
3. **Ejecuta la compilación autocontenida de Streamlit**:
   ```powershell
   pyinstaller --name "AlertaTemprana" --onefile --clean --add-data "model_registry;model_registry" --add-data "model.joblib;." --add-data "scaler.joblib;." --add-data "shap_background.npy;." --add-data "shap_feature_names.npy;." --add-data "src;src" --add-data "dataset;dataset" --add-data "app.py;." run_streamlit.py
   ```
4. Ubica el ejecutable compilado en la carpeta `dist/AlertaTemprana.exe`. Este ejecutable encapsula Python, Streamlit y el modelo matemático, funcionando exactamente igual que en la consola, incluida la persistencia de datos descrita arriba.

---

## 🏛️ Arquitectura de Software

El proyecto se estructuró bajo la filosofía de **Clean Architecture** (Arquitectura Limpia) y principios **SOLID**, garantizando un bajo acoplamiento y alta cohesión entre sus capas:

```
src/
├── domain/            # Capa 1: Núcleo de Dominio (Entidades y Contratos abstractos)
│   ├── entities/      # Objetos puros de negocio (Student, Prediction) sin librerías externas
│   └── repositories/  # Interfaces/Contratos (StudentRepository, ModelRepository)
├── application/       # Capa 2: Casos de Uso (Lógica y Reglas de Negocio)
│   ├── train_use_case.py   # Orquestación de AutoML, tuning y evaluación
│   ├── predict_use_case.py # Inferencia en lote e individual con cálculo SHAP perezoso
│   └── monitor_use_case.py # Detección de Data Drift estadístico (KS-Test)
├── infrastructure/    # Capa 3: Implementaciones y Tecnologías Externas
│   ├── ml/            # Algoritmos específicos (CatBoost, XGBoost, Optuna) y preprocesadores
│   ├── reports/       # Lógica técnica de generación de PDF mediante FPDF2
│   └── repositories/  # Repositorios concretos de lectura Excel/CSV y guardado JSON
```

### Ventajas de esta arquitectura:
* **Independencia de Frameworks**: Cambiar la interfaz gráfica (de Streamlit a Django o FastAPI) no requiere modificar la lógica de predicción ni el modelo matemático.
* **Independencia del Modelo de ML**: Podemos reemplazar CatBoost por otra red neuronal o modelo en la capa de infraestructura sin alterar los casos de uso ni la entidad estudiante.
* **Testabilidad**: Es posible probar los flujos lógicos simulando repositorios en memoria (Mocking) sin necesidad de cargar la base de datos Excel física.
