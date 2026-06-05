# Predicción de Deserción Estudiantil

Aplicación Streamlit que predice la deserción estudiantil usando una red neuronal. Sube un archivo CSV o Excel con datos de estudiantes y obtén las predicciones al instante.

---

## Requisito: Python 3.9 — 3.12

Descarga e instala Python si no lo tienes:

| Sistema | Descarga |
|---|---|
| **Windows** | [python.org/downloads](https://www.python.org/downloads/) — al instalar, marca **"Add Python to PATH"** |
| **macOS** | `brew install python@3.12` o [python.org/downloads](https://www.python.org/downloads/) |
| **Linux (Ubuntu/Debian)** | `sudo apt install python3.12 python3.12-venv python3.12-dev` |
| **Linux (Arch)** | `sudo pacman -S python` |
| **Linux (Fedora)** | `sudo dnf install python3.12 python3.12-devel` |

Verifica que quedó bien instalado:

```bash
python3 --version
```

---

## Quick Start

Copia y pega **todo el bloque** correspondiente a tu sistema operativo en la terminal.

### Linux / macOS

```bash
git clone https://github.com/blackrose99/Tesisfiles.git && cd Tesisfiles && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && echo "✅ Todo listo. Pon tu archivo 'Base de datos estudiantes.xlsx' en la carpeta y luego ejecuta: python \"Limpieza de datos.py\" && python train.py && streamlit run app.py"
```

Después de ejecutar el bloque de arriba, **coloca tu archivo `Base de datos estudiantes.xlsx` en la carpeta `Tesisfiles/`** y ejecuta estos comandos uno por uno:

```bash
python "Limpieza de datos.py"
python train.py
streamlit run app.py
```

### Windows PowerShell

```powershell
git clone https://github.com/blackrose99/Tesisfiles.git; cd Tesisfiles; python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt; Write-Host "✅ Listo. Pon tu archivo 'Base de datos estudiantes.xlsx' en la carpeta y luego ejecuta: python `"Limpieza de datos.py`"; python train.py; streamlit run app.py"
```

Después, **coloca tu archivo `Base de datos estudiantes.xlsx` en la carpeta `Tesisfiles/`** y ejecuta:

```powershell
python "Limpieza de datos.py"
python train.py
streamlit run app.py
```

### Windows CMD

```cmd
git clone https://github.com/blackrose99/Tesisfiles.git && cd Tesisfiles && python -m venv .venv && .venv\Scripts\activate.bat && pip install -r requirements.txt && echo ✅ Listo. Pon tu archivo "Base de datos estudiantes.xlsx" en la carpeta y luego ejecuta: python "Limpieza de datos.py" ^&^& python train.py ^&^& streamlit run app.py
```

Después, **coloca tu archivo `Base de datos estudiantes.xlsx` en la carpeta `Tesisfiles/`** y ejecuta:

```cmd
python "Limpieza de datos.py"
python train.py
streamlit run app.py
```

---

## Explicación de cada paso

Solo para entender qué hace cada comando — si ya lo ejecutaste arriba, puedes saltarte esto.

| Comando | ¿Qué hace? |
|---|---|
| `git clone ...` | Descarga el proyecto a tu computadora |
| `cd Tesisfiles` | Entra a la carpeta del proyecto |
| `python -m venv .venv` | Crea un entorno virtual (aisla las librerías) |
| `source .venv/bin/activate` | Activa el entorno virtual |
| `pip install -r requirements.txt` | Instala todas las librerías necesarias |
| `python "Limpieza de datos.py"` | Limpia y prepara los datos para entrenar |
| `python train.py` | Entrena la red neuronal y guarda el modelo |
| `streamlit run app.py` | Inicia la interfaz web |

---

## Estructura del proyecto

```
Tesisfiles/
├── app.py                        # Interfaz Streamlit
├── train.py                      # Entrenamiento del modelo
├── Limpieza de datos.py          # Limpieza y preparación de datos
├── requirements.txt              # Dependencias del proyecto
├── modelo_desercion_nn.keras     # Modelo entrenado (se genera al ejecutar train.py)
├── scaler.joblib                 # Escalador (se genera al ejecutar train.py)
├── test50.csv                    # Ejemplo con 50 estudiantes para pruebas
├── README.md
├── Base de datos estudiantes.xlsx # TU archivo de datos (debes agregarlo)
├── archivos_procesados/          # Resultados de predicciones
│   └── resultados/
└── Documents/                    # Datos originales organizados por año
    ├── 2017-2019/
    ├── 2020-2021/
    └── 2022-2024/
```

---

## Descripción de archivos principales

| Archivo | Función |
|---|---|
| `app.py` | Interfaz web con Streamlit. Valida datos, predice y exporta resultados |
| `train.py` | Entrena la red neuronal y guarda `modelo_desercion_nn.keras` y `scaler.joblib` |
| `Limpieza de datos.py` | Procesa el Excel crudo y genera el archivo listo para entrenar |
| `requirements.txt` | Lista de librerías para instalar con `pip install -r requirements.txt` |
| `test50.csv` | Archivo de ejemplo con 50 estudiantes para probar la interfaz |
| `modelo_desercion_nn.keras` | Modelo de red neuronal entrenado |
| `scaler.joblib` | Escalador con las features que espera el modelo |

---

## Cómo usar la interfaz web

1. Ejecuta `streamlit run app.py`
2. Se abrirá el navegador con la aplicación
3. Descarga la **plantilla vacía** o el **archivo de ejemplo** desde la barra lateral
4. Completa el archivo con los datos de los estudiantes
5. Súbelo a la app (CSV o Excel)
6. Si hay errores de validación, la app los muestra fila por fila
7. Si todo está bien, descarga el CSV con las predicciones

### Columnas requeridas en el archivo de entrada

```
CODESTUDIANTE, ESTP_FECHAINGRESO, CREDITOSAPROBADOS, UBICACION_SEMESTRAL,
PROMEDIO_GENERAL, PROGRAMA, JORNADA, GENERO, FECHA_NACIMIENTO, CIUDADRESIDENCIA,
ESTRATO, TIENE_SISBEN, INFE_VIVECONFAMILIA, INFE_SITUACIONPADRES,
INFE_NUMEROFAMILIARES, INFE_NUMEROHERMANOS, INFE_POSICIONENHERMANOS,
INFE_NUMMIEMBROSTRABAJA
```

> El archivo **no debe incluir** la columna `SITUACION`. Esa es la variable que el modelo predice.

### Delimitador para CSV

La app detecta automáticamente si usas `,`, `;`, tabulación o `|`. Mantén un solo delimitador en todo el archivo.

---

## Solución de problemas

| Problema | Solución |
|---|---|
| `python: command not found` | Python no está instalado o no está en el PATH. Revisa la sección de requisitos |
| `pip: command not found` | En Linux: `sudo apt install python3-pip` o `sudo pacman -S python-pip` |
| `No module named ...` | No activaste el entorno virtual o no instalaste las dependencias. Ejecuta `source .venv/bin/activate` y luego `pip install -r requirements.txt` |
| El modelo no se carga en la app | Ejecuta `python train.py` primero para generar `modelo_desercion_nn.keras` y `scaler.joblib` |
| Error al leer el Excel | Asegúrate de que `Base de datos estudiantes.xlsx` está en la raíz del proyecto |
