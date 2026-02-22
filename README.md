# 🎓 Sistema de Predicción de Deserción Estudiantil

Sistema inteligente de predicción de deserción estudiantil basado en Machine Learning que procesa automáticamente datos de estudiantes y genera predicciones con visualizaciones interactivas.

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Ejecución](#-ejecución)
- [Uso del Sistema](#-uso-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Archivos de Entrada](#-archivos-de-entrada)
- [Archivos de Salida](#-archivos-de-salida)
- [Solución de Problemas](#-solución-de-problemas)
- [Documentación Adicional](#-documentación-adicional)

---

## ✨ Características

- ✅ **Limpieza automática de datos** - Replica el proceso de `Limpieza de datos.py`
- ✅ **Predicción con Red Neuronal** - Modelo entrenado con TensorFlow/Keras
- ✅ **Interfaz web interactiva** - Dashboard construido con Streamlit
- ✅ **Visualizaciones avanzadas** - Gráficos interactivos con Plotly
- ✅ **Evaluación del modelo** - Matriz de confusión, ROC curve, métricas detalladas
- ✅ **Logs en tiempo real** - Seguimiento completo del proceso
- ✅ **Gestión automática de archivos** - Organización en carpetas (base/limpio/resultados)
- ✅ **Soporte CSV y Excel** - Acepta ambos formatos de entrada

---

## 💻 Requisitos del Sistema

### Sistema Operativo
- Linux (probado en Ubuntu/Debian)
- Windows 10/11
- macOS 10.14+

### Software Requerido
- **Python**: 3.8 o superior
- **pip**: Gestor de paquetes de Python
- **Git** (opcional): Para clonar el repositorio

### Recursos Recomendados
- **RAM**: 4 GB mínimo (8 GB recomendado)
- **Espacio en disco**: 500 MB para el entorno virtual y dependencias
- **Procesador**: CPU moderna (GPU no requerida, el modelo funciona en CPU)

---

## 🚀 Instalación

### Paso 1: Clonar o Descargar el Proyecto

```bash
# Opción 1: Si tienes Git
git clone <URL_DEL_REPOSITORIO>
cd Tesisfiles

# Opción 2: Descarga manual
# Descomprime el archivo ZIP en una carpeta de tu elección
cd ruta/a/Tesisfiles
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# En Linux/Mac
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar todas las dependencias
pip install -r requirements.txt
```

**Dependencias principales:**
- `streamlit` - Framework web para la interfaz
- `pandas` - Procesamiento de datos
- `numpy` - Cálculos numéricos
- `tensorflow` - Motor de predicción (Red Neuronal)
- `scikit-learn` - Métricas y evaluación
- `plotly` - Visualizaciones interactivas
- `openpyxl` - Soporte para archivos Excel

### Paso 4: Verificar Instalación

```bash
# Verificar que Python puede importar las librerías
python3 -c "import streamlit; import tensorflow; import pandas; print('✅ Todas las dependencias instaladas correctamente')"
```

---

## ▶️ Ejecución

### Iniciar la Aplicación

```bash
# Desde la carpeta Tesisfiles/
streamlit run app.py
```

### Resultado esperado

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Acceder a la Aplicación

1. Abre tu navegador web
2. Ve a: **http://localhost:8501**
3. La interfaz del dashboard se cargará automáticamente

### Detener la Aplicación

- **Desde terminal**: Presiona `Ctrl + C`
- **Cerrar pestaña**: La aplicación seguirá corriendo en segundo plano
- **Forzar detención**: `pkill -f "streamlit run"`

---

## 📖 Uso del Sistema

### Flujo de Trabajo Básico

1. **Preparar datos**
   - Descarga `plantilla_carga_estudiantes.csv` desde el dashboard
   - Completa con los datos de tus estudiantes (18 columnas obligatorias)
   - O usa `ejemplo_datos_estudiantes.csv` para probar

2. **Subir archivo**
   - Click en "📤 Subir archivo CSV/Excel"
   - Selecciona tu archivo (CSV o XLSX)

3. **Configurar umbral** (opcional)
   - Ajusta el umbral de clasificación (por defecto 0.5)
   - Mayor umbral = menos estudiantes clasificados como "No aprueba"

4. **Ver resultados**
   - El sistema procesará automáticamente los datos
   - Observa los logs en tiempo real
   - Explora las visualizaciones y métricas

5. **Descargar resultados**
   - Click en "💾 Descargar Resultados (CSV)"
   - Archivo generado: `resultados_prediccion.csv`

### Columnas Obligatorias del Archivo de Entrada

El archivo **DEBE** contener estas 18 columnas:

```
1.  ESTP_FECHAINGRESO          (Fecha: YYYY-MM-DD)
2.  CREDITOSAPROBADOS           (Número: entero)
3.  UBICACION_SEMESTRAL         (Número: semestre actual)
4.  PROMEDIO_GENERAL            (Número: 0.0 - 5.0)
5.  PROGRAMA                    (Texto: nombre del programa)
6.  JORNADA                     (Texto: DIURNA/NOCTURNA)
7.  GENERO                      (Texto: M/F)
8.  FECHA_NACIMIENTO            (Fecha: YYYY-MM-DD)
9.  CIUDADRESIDENCIA            (Texto: ciudad)
10. ESTRATO                     (Número: 1-6)
11. TIENE_SISBEN                (Número: 0/1)
12. INFE_VIVECONFAMILIA         (Texto: SI/NO)
13. INFE_SITUACIONPADRES        (Texto: categoría)
14. INFE_NUMEROFAMILIARES       (Número: entero)
15. INFE_NUMEROHERMANOS         (Número: entero)
16. INFE_POSICIONENHERMANOS     (Número: entero)
17. INFE_NUMMIEMBROSTRABAJA     (Número: entero)
18. SITUACION (OPCIONAL)        (Texto: estado académico - solo para evaluación)
```

**📖 Ver:** `INSTRUCCIONES_CARGA_DATOS.md` para detalles completos

---

## 📁 Estructura del Proyecto

```
Tesisfiles/
│
├── app.py                                 # 🎯 Aplicación principal Streamlit
├── modelo_desercion_nn.keras              # 🧠 Modelo de Red Neuronal entrenado
├── scaler.joblib                          # 📏 Normalizador StandardScaler
├── Limpieza de datos.py                   # 🧹 Script de limpieza (referencia)
│
├── requirements.txt                       # 📦 Dependencias Python
├── README.md                              # 📖 Este archivo
│
├── plantilla_carga_estudiantes.csv        # 📋 Plantilla vacía (18 columnas)
├── ejemplo_datos_estudiantes.csv          # 📊 Datos de ejemplo (20 estudiantes)
│
├── INSTRUCCIONES_CARGA_DATOS.md           # 📘 Guía de columnas y formatos
├── README_ESTRUCTURA_CARPETAS.md          # 📁 Documentación de carpetas
│
├── Base de datos estudiantes.xlsx         # 💾 Datos originales de entrenamiento
│
└── archivos_procesados/                   # 🗂️ Carpeta automática (se crea al usar)
    ├── base/                              # Archivos originales (backup)
    ├── limpio/                            # Archivos post-limpieza (26 columnas)
    └── resultados/                        # Predicciones finales (3 columnas)
```

---

## 📥 Archivos de Entrada

### Formato Aceptado
- **CSV**: Delimitado por comas, UTF-8
- **Excel**: XLSX (formato moderno)

### Plantillas Disponibles

1. **`plantilla_carga_estudiantes.csv`**
   - Solo encabezados (18 columnas)
   - Para completar con tus datos

2. **`ejemplo_datos_estudiantes.csv`**
   - 20 estudiantes de ejemplo
   - Incluye columna SITUACION para evaluar el modelo
   - Datos realistas de prueba

### Validaciones Automáticas

El sistema valida y corrige automáticamente:
- ✅ Fechas en diferentes formatos
- ✅ Valores faltantes (relleno con códigos especiales)
- ✅ ESTRATO fuera de rango (1-6)
- ✅ Programas no válidos (filtrado automático)
- ✅ Valores categóricos (conversión a numérico)

---

## 📤 Archivos de Salida

### Archivo de Resultados

**Nombre:** `resultados_prediccion_YYYYMMDD_HHMMSS.csv`

**Ubicación:** `archivos_procesados/resultados/`

**Estructura:**
```csv
identificador,resultado_modelo,probabilidad_estadistica
ESTUDIANTE_0001,Aprueba,0.2345
ESTUDIANTE_0002,No aprueba,0.8712
ESTUDIANTE_0003,Aprueba,0.4521
```

**Columnas:**
- `identificador`: ID generado automáticamente
- `resultado_modelo`: Clasificación (Aprueba / No aprueba)
- `probabilidad_estadistica`: Probabilidad de deserción (0.0 - 1.0)

### Archivos Guardados Automáticamente

Por cada procesamiento se generan 3 archivos:

1. **`base/archivo_base_*.csv`** - Copia del archivo original
2. **`limpio/archivo_limpio_*.xlsx`** - Datos después de limpieza (26 columnas)
3. **`resultados/resultados_prediccion_*.csv`** - Predicciones finales

---

## 🔧 Solución de Problemas

### Error: "No se puede importar streamlit"

```bash
# Solución: Reinstalar streamlit
pip install --upgrade streamlit
```

### Error: "TensorFlow no encontrado"

```bash
# Solución: Instalar TensorFlow
pip install tensorflow>=2.13.0

# En sistemas con recursos limitados, usa la versión CPU:
pip install tensorflow-cpu
```

### Error: "CUDA drivers not found" (Advertencia)

**No es un error crítico**. El modelo funciona perfectamente en CPU.
- Advertencia esperada si no tienes GPU NVIDIA
- El modelo se ejecutará en CPU (suficiente para este uso)

### Error: "Archivo vacío después de limpieza"

**Causa:** Ningún estudiante pertenece a los programas válidos

**Solución:** Verificar que el archivo contenga estudiantes de:
- `INGENIERIA DE SISTEMAS`
- `TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS`

### Error: "Puerto 8501 ya en uso"

```bash
# Solución 1: Detener la aplicación existente
pkill -f "streamlit run"

# Solución 2: Usar otro puerto
streamlit run app.py --server.port 8502
```

### Error: "Permisos denegados en archivos_procesados/"

```bash
# Linux/Mac: Dar permisos de escritura
chmod -R 755 archivos_procesados/

# Windows: Desmarcar "Solo lectura" en propiedades de la carpeta
```

### El modelo predice siempre la misma clase

**Causa:** Datos muy similares o umbral mal configurado

**Solución:** 
1. Ajustar el umbral de clasificación (0.3 - 0.7)
2. Verificar que los datos tengan diversidad
3. Revisar el archivo de ejemplo para comparar

### Logs no se muestran en el dashboard

**Causa:** Error en la inicialización de session_state

**Solución:**
1. Recargar la página (F5)
2. Borrar caché de Streamlit: `streamlit cache clear`

---

## 📚 Documentación Adicional

### Archivos de Documentación

- **`INSTRUCCIONES_CARGA_DATOS.md`**
  - Guía completa de las 18 columnas obligatorias
  - Formatos aceptados y reglas de validación
  - Ejemplos de archivos CSV correctos
  - Errores comunes y soluciones

- **`README_ESTRUCTURA_CARPETAS.md`**
  - Descripción de la carpeta `archivos_procesados/`
  - Contenido de base/, limpio/, resultados/
  - Gestión y limpieza de archivos
  - Estimación de espacio en disco

### Recursos Externos

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Python](https://plotly.com/python/)

---

## 🤝 Contribuciones

### Reportar Problemas

Si encuentras errores o tienes sugerencias:
1. Revisa la sección de [Solución de Problemas](#-solución-de-problemas)
2. Consulta los archivos de documentación
3. Verifica los logs del dashboard

### Mejoras Futuras

Posibles extensiones del sistema:
- [ ] Soporte para más programas académicos
- [ ] Exportación de reportes en PDF
- [ ] API REST para integración con otros sistemas
- [ ] Dashboard de administración con histórico
- [ ] Reentrenamiento automático del modelo

---

## 📄 Licencia

Este proyecto es parte de una tesis académica sobre predicción de deserción estudiantil.

---

## 📞 Soporte

Para dudas o problemas:
1. Revisa este README
2. Consulta `INSTRUCCIONES_CARGA_DATOS.md`
3. Revisa los logs del dashboard
4. Verifica los errores en la terminal

---

## 🎓 Créditos

**Sistema de Predicción de Deserción Estudiantil**
- Desarrollado como parte de proyecto de tesis
- Framework: Streamlit + TensorFlow
- Visualizaciones: Plotly

---

## 🚀 Inicio Rápido (TL;DR)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar aplicación
streamlit run app.py

# 3. Abrir navegador
# http://localhost:8501

# 4. Subir archivo (usar ejemplo_datos_estudiantes.csv para probar)

# 5. Descargar resultados
```

**¡Listo!** El sistema está funcionando. 🎉
