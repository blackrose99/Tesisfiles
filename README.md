# Predicción de Deserción Estudiantil

Proyecto Streamlit para predecir la deserción estudiantil usando un modelo de
red neuronal entrenado. La aplicación valida la plantilla de entrada, realiza la
predicción y permite descargar los resultados.

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
.venv\\Scripts\\Activate.ps1
```

## Instalar dependencias

```bash
pip install streamlit pandas numpy joblib tensorflow scikit-learn plotly openpyxl
```

## Entrenar el modelo (opcional)

Si quieres volver a entrenar el modelo y regenerar los artefactos, ejecuta:

```bash
python train.py
```

Al finalizar se generan (si corresponde):

- `modelo_desercion_nn.keras` — modelo entrenado
- `scaler.joblib` — scaler con las features esperadas por el modelo

## Ejecutar la interfaz

```bash
streamlit run app.py
```

La aplicación se abrirá en el navegador. Carga un archivo CSV o Excel usando la
plantilla y, si todo es válido, podrás descargar los resultados con las
predicciones.

## Estructura del proyecto

```
Tesisfiles/
├─ app.py
├─ train.py
├─ Limpieza de datos.py
├─ modelo_desercion_nn.keras
├─ scaler.joblib
├─ test50.csv
├─ README.md
├─ archivos_procesados/
│  └─ resultados/
└─ Documents/
   ├─ 2017-2019/
   │  ├─ "Con situacion 2017-2019.xlsx - Situacion.csv"
   │  └─ "Sin situacion 2017-2019.xlsx - Sheet1.csv"
   ├─ 2020-2021/
   │  ├─ "Con situacion 2020-2021.xlsx - Situacion.csv"
   │  └─ "Sin situacion 2020-2021.xlsx - Sheet1.csv"
   └─ 2022-2024/
      ├─ "Con situacion 2022-2024.xlsx - Situacion.csv"
      └─ "Sin situacion 2022-2025.xlsx - Sheet1.csv"
```

> Nota: la estructura refleja los archivos y carpetas presentes en el
> repositorio. Ajusta las rutas si mueves o renombras archivos.

## Descripción de archivos principales

- `app.py`: interfaz Streamlit, validaciones, predicción y exportación de resultados.
- `train.py`: script de entrenamiento del modelo y guardado de artefactos.
- `Limpieza de datos.py`: script de limpieza / preparación de datos.
- `test50.csv`: ejemplo con 50 estudiantes para pruebas.
- `modelo_desercion_nn.keras`: modelo entrenado (si existe).
- `scaler.joblib`: scaler con las features esperadas por el modelo (si existe).
- `archivos_procesados/resultados/`: carpeta con CSVs generados tras la predicción.

## Plantilla y validaciones

La aplicación exige la estructura exacta de la plantilla (mismos encabezados).
Si falta una columna, sobra una columna o hay datos inválidos, la app mostrará
una tabla de errores indicando fila y columna.

Columnas requeridas:

```
CODESTUDIANTE, ESTP_FECHAINGRESO, CREDITOSAPROBADOS, UBICACION_SEMESTRAL,
PROMEDIO_GENERAL, PROGRAMA, JORNADA, GENERO, FECHA_NACIMIENTO, CIUDADRESIDENCIA,
ESTRATO, TIENE_SISBEN, INFE_VIVECONFAMILIA, INFE_SITUACIONPADRES,
INFE_NUMEROFAMILIARES, INFE_NUMEROHERMANOS, INFE_POSICIONENHERMANOS,
INFE_NUMMIEMBROSTRABAJA
```

El archivo para predicción no debe incluir la columna `SITUACION`.

## Delimitador para CSV

Si subes un CSV, la app detecta el delimitador (`,`, `;`, tabulación o `|`).
Mantén un delimitador consistente por archivo para evitar errores de lectura.

## Flujo recomendado

1. Descarga la plantilla desde la interfaz.
2. Completa el archivo con los datos requeridos.
3. Sube el archivo y revisa la tabla de errores si aparece.
4. Si todo es válido, descarga el CSV con los resultados.

---

Si quieres, puedo además generar una plantilla CSV con los encabezados requeridos
o crear un `requirements.txt` para gestionar dependencias. ¿Lo hago?
