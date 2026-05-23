# Prediccion de Desercion Estudiantil

Proyecto Streamlit para predecir desercion con un modelo de red neuronal.

## Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd Tesisfiles
```

## Crear y activar entorno virtual

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Instalar librerias

```bash
pip install streamlit pandas numpy joblib tensorflow scikit-learn plotly openpyxl
```

## Entrenar el modelo (opcional)

Entrena y genera los archivos del modelo y el scaler.

```bash
python train.py
```

Se crean los archivos:
- `modelo_desercion_nn.keras`
- `scaler.joblib`

## Ejecutar la interfaz

```bash
streamlit run app.py
```

La aplicacion abre en el navegador. Carga un archivo CSV o Excel usando la plantilla y descarga los resultados.

## Estructura del proyecto

```
Tesisfiles/
├─ app.py
├─ train.py
├─ modelo_desercion_nn.keras
├─ scaler.joblib
├─ test50.csv
├─ Base de datos estudiantes.xlsx
├─ Base_de_datos_estudiantes_ready_step4_20260215_014516.xlsx
└─ archivos_procesados/
   └─ resultados/
```

## Descripcion de archivos

- `app.py`: interfaz Streamlit, validaciones, prediccion y exportacion de resultados.
- `train.py`: entrenamiento del modelo, evaluacion y guardado de artefactos.
- `test50.csv`: ejemplo con 50 estudiantes para pruebas.
- `modelo_desercion_nn.keras`: modelo entrenado.
- `scaler.joblib`: scaler con las features esperadas por el modelo.
- `archivos_procesados/resultados/`: resultados generados al predecir.

## Plantilla y validaciones

La interfaz exige la estructura exacta de la plantilla (mismos encabezados). Si falta una columna, sobra una columna o hay datos invalidos, la app muestra una tabla de errores con fila y columna.

Columnas requeridas:

```
CODESTUDIANTE, ESTP_FECHAINGRESO, CREDITOSAPROBADOS, UBICACION_SEMESTRAL,
PROMEDIO_GENERAL, PROGRAMA, JORNADA, GENERO, FECHA_NACIMIENTO, CIUDADRESIDENCIA,
ESTRATO, TIENE_SISBEN, INFE_VIVECONFAMILIA, INFE_SITUACIONPADRES,
INFE_NUMEROFAMILIARES, INFE_NUMEROHERMANOS, INFE_POSICIONENHERMANOS,
INFE_NUMMIEMBROSTRABAJA
```

El archivo para prediccion no debe incluir la columna `SITUACION`.

## Delimitacion de CSV

Si subes CSV, la app detecta el delimitador (`,`, `;`, tabulacion o `|`). Mantener un delimitador consistente por archivo evita errores de lectura.

## Flujo recomendado

1) Descarga la plantilla desde la interfaz.
2) Llena el archivo con los datos requeridos.
3) Sube el archivo y revisa la tabla de errores si aparece.
4) Descarga el CSV de resultados.
