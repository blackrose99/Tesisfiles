import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from joblib import load as joblib_load
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="🎓 Predicción de Deserción Estudiantil", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Sistema de predicción de deserción estudiantil usando Machine Learning"
    }
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Header con estilo mejorado
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 3rem;">🎓 Predicción de Deserción Estudiantil</h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 1.2rem; margin-top: 0.5rem;">Sistema Inteligente de Análisis Predictivo</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 1.5rem;">
    <h3 style="color: #667eea; margin-top: 0;">📊 ¿Cómo funciona?</h3>
    <ul style="color: #495057; line-height: 1.8;">
        <li><strong>Paso 1:</strong> Descarga la plantilla CSV o usa el ejemplo de 20 estudiantes</li>
        <li><strong>Paso 2:</strong> Completa los datos de tus estudiantes</li>
        <li><strong>Paso 3:</strong> Sube el archivo y obtén predicciones instantáneas</li>
        <li><strong>Paso 4:</strong> Descarga los resultados con probabilidades y clasificaciones</li>
    </ul>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "modelo_desercion_nn.keras"
SCALER_PATH = "scaler.joblib"

# Inicializar logs en session state
if 'process_logs' not in st.session_state:
    st.session_state.process_logs = []

def add_log(message, log_type="info"):
    """Añade un mensaje al log con timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.process_logs.append({
        'timestamp': timestamp,
        'message': message,
        'type': log_type
    })

def clean_data(df_raw):
    """Limpieza automática REAL - replica exactamente Limpieza de datos.py"""
    add_log("⚙️ Iniciando proceso de limpieza automática de datos...", "info")
    df = df_raw.copy()
    
    # Guardar SITUACION si existe (para evaluación posterior)
    has_situacion = 'SITUACION' in df.columns
    situacion_original = df['SITUACION'].copy() if has_situacion else None
    
    # 1) Eliminar columnas que no se usarán (las que están en el archivo original pero no en el limpio)
    columnas_eliminar = [
        "CODESTUDIANTE", "CODIGOCIUDADR", "NIVEL_SISBEN", 
        "CATEGORIA", "CODMATRICULA", "SEDE", "INFE_HERMANOSESTUDIANDOU"
    ]
    cols_eliminadas = [c for c in columnas_eliminar if c in df.columns]
    if cols_eliminadas:
        df = df.drop(columns=cols_eliminadas)
        add_log(f"🗑️ Eliminadas {len(cols_eliminadas)} columnas innecesarias: {', '.join(cols_eliminadas)}", "info")
    
    # 2) Normalizar y procesar fechas
    add_log("📅 Procesando fechas (ESTP_FECHAINGRESO, FECHA_NACIMIENTO)...", "info")
    df["ESTP_FECHAINGRESO"] = pd.to_datetime(df["ESTP_FECHAINGRESO"], errors="coerce")
    df["FECHA_NACIMIENTO"] = pd.to_datetime(df["FECHA_NACIMIENTO"], errors="coerce")
    
    # Crear variables numéricas desde fechas
    df["EDAD_INGRESO"] = ((df["ESTP_FECHAINGRESO"] - df["FECHA_NACIMIENTO"]).dt.days / 365.25).round().astype("Int64")
    df["ANIO_INGRESO"] = df["ESTP_FECHAINGRESO"].dt.year
    df["MES_INGRESO"] = df["ESTP_FECHAINGRESO"].dt.month
    
    # Eliminar columnas de fecha originales
    df = df.drop(columns=["ESTP_FECHAINGRESO", "FECHA_NACIMIENTO"])
    add_log("  ✓ Fechas convertidas a EDAD_INGRESO, ANIO_INGRESO, MES_INGRESO", "success")
    
    # 3) Validar y limpiar ESTRATO (debe estar entre 1-6)
    add_log("🏘️ Validando ESTRATO (rango 1-6)...", "info")
    df["ESTRATO"] = pd.to_numeric(df["ESTRATO"], errors="coerce")
    fuera_rango = ((df["ESTRATO"] < 1) | (df["ESTRATO"] > 6)).sum()
    if fuera_rango > 0:
        add_log(f"  ⚠️ {fuera_rango} valores de ESTRATO fuera de rango - marcados como NaN", "warning")
        df.loc[(df["ESTRATO"] < 1) | (df["ESTRATO"] > 6), "ESTRATO"] = pd.NA
    
    # 4) Filtrar solo programas permitidos
    add_log("🎓 Filtrando programas académicos...", "info")
    allowed_programs = [
        "INGENIERIA DE SISTEMAS",
        "TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS"
    ]
    rows_before = len(df)
    df = df[df["PROGRAMA"].isin(allowed_programs)]
    if len(df) < rows_before:
        add_log(f"  ⚠️ Eliminados {rows_before - len(df)} estudiantes de programas no válidos", "warning")
    add_log(f"  ✓ Mantenidos solo {', '.join(allowed_programs)}", "success")
    
    # 5) Convertir SITUACION a binaria (1 = desertor, 0 = activo)
    if has_situacion:
        add_log("🎯 Codificando SITUACION como binaria...", "info")
        situaciones_desertor = [
            "EXCLUIDO NO RENOVACION DE MATRICULA",
            "PFI",
            "EXCLUIDO CANCELACION SEMESTRE",
            "INACTIVO"
        ]
        df["SITUACION"] = df["SITUACION"].isin(situaciones_desertor).astype(int)
        add_log(f"  ✓ SITUACION: 1={', '.join(situaciones_desertor[:2])}..., 0=otros", "success")
    
    # 6) Recodificar CIUDADRESIDENCIA
    add_log("🏙️ Recodificando CIUDADRESIDENCIA...", "info")
    df["CIUDADRESIDENCIA"] = df["CIUDADRESIDENCIA"].astype(str).str.strip().str.upper()
    map_ciudad = {
        "BUCARAMANGA": 1,
        "FLORIDABLANCA": 2,
        "GIRON": 3,
        "PIEDECUESTA": 4
    }
    df["CIUDADRESIDENCIA"] = df["CIUDADRESIDENCIA"].map(map_ciudad).fillna(5).astype(int)
    add_log("  ✓ Ciudades: BUCARAMANGA=1, FLORIDABLANCA=2, GIRON=3, PIEDECUESTA=4, otras=5", "success")
    
    # 7) One-Hot Encoding automático de columnas categóricas
    add_log("🔢 Aplicando One-Hot Encoding a variables categóricas...", "info")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols_onehot = [c for c in cat_cols if df[c].nunique(dropna=True) > 1]
    
    if cat_cols_onehot:
        add_log(f"  📊 Columnas a codificar: {', '.join(cat_cols_onehot)}", "info")
        df = pd.get_dummies(df, columns=cat_cols_onehot, dtype=int)
        add_log(f"  ✓ One-Hot Encoding completado - {len(df.columns)} columnas totales", "success")
    
    # 8) Convertir booleanos a 0/1
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # 9) Limpieza final: rellenar NaN con códigos especiales
    add_log("🧹 Limpieza final: rellenando valores faltantes...", "info")
    
    # ESTRATO: 0 = no reportado
    if "ESTRATO" in df.columns:
        na_count = df["ESTRATO"].isna().sum()
        if na_count > 0:
            df["ESTRATO"] = df["ESTRATO"].fillna(0).astype(int)
            add_log(f"  ✓ ESTRATO: {na_count} valores faltantes → 0 (no reportado)", "success")
    
    # Variables de familia: -1 = no reportado
    cols_familia = [
        "INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS", 
        "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA", "EDAD_INGRESO"
    ]
    for c in cols_familia:
        if c in df.columns:
            na_count = df[c].isna().sum()
            if na_count > 0:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).round().astype(int)
                add_log(f"  ✓ {c}: {na_count} valores faltantes → -1", "success")
    
    # TIENE_SISBEN: -1 = no reportado
    if "TIENE_SISBEN" in df.columns:
        na_count = df["TIENE_SISBEN"].isna().sum()
        if na_count > 0:
            df["TIENE_SISBEN"] = pd.to_numeric(df["TIENE_SISBEN"], errors="coerce").fillna(-1).round().astype(int)
            add_log(f"  ✓ TIENE_SISBEN: {na_count} valores faltantes → -1", "success")
    
    # Verificar que no queden NaN
    total_nan = df.isna().sum().sum()
    if total_nan > 0:
        add_log(f"⚠️ Quedan {total_nan} NaN - rellenando con 0", "warning")
        df = df.fillna(0)
    
    add_log(f"✅ Limpieza completada: {len(df)} registros × {len(df.columns)} columnas", "success")
    add_log(f"📊 Archivo listo para el modelo (columnas: {', '.join(df.columns[:5])}...)", "info")
    
    return df


def make_template_df():
    """Plantilla con las 18 columnas REALES obligatorias (sin SITUACION para predicción)"""
    cols = [
        "ESTP_FECHAINGRESO",
        "CREDITOSAPROBADOS",
        "UBICACION_SEMESTRAL",
        "PROMEDIO_GENERAL",
        "PROGRAMA",
        "JORNADA",
        "GENERO",
        "FECHA_NACIMIENTO",
        "CIUDADRESIDENCIA",
        "ESTRATO",
        "TIENE_SISBEN",
        "INFE_VIVECONFAMILIA",
        "INFE_SITUACIONPADRES",
        "INFE_NUMEROFAMILIARES",
        "INFE_NUMEROHERMANOS",
        "INFE_POSICIONENHERMANOS",
        "INFE_NUMMIEMBROSTRABAJA"
    ]
    return pd.DataFrame(columns=cols)


def make_example_df():
    """Carga el archivo de ejemplo con 20 estudiantes diversos (incluye SITUACION para evaluación)"""
    try:
        ejemplo_path = "ejemplo_datos_estudiantes.csv"
        if os.path.exists(ejemplo_path):
            df = pd.read_csv(ejemplo_path)
            return df
        else:
            # Si no existe, retornar plantilla vacía
            st.warning(f"Archivo de ejemplo no encontrado: {ejemplo_path}")
            return make_template_df()
    except Exception as e:
        st.error(f"Error al cargar ejemplo: {e}")
        return make_template_df()




def safe_load_model(path):
    if os.path.exists(path):
        try:
            return load_model(path)
        except Exception as e:
            st.warning(f"No se pudo cargar el modelo: {e}")
            return None
    else:
        st.warning(f"Modelo no encontrado en: {path}. La predicción no estará disponible.")
        return None

model = safe_load_model(MODEL_PATH)

def safe_load_scaler(path):
    if os.path.exists(path):
        try:
            return joblib_load(path)
        except Exception as e:
            st.warning(f"No se pudo cargar el scaler: {e}")
            return None
    else:
        st.info(f"Scaler no encontrado en: {path}. Si el modelo fue entrenado con scaler, guarda el scaler como 'scaler.joblib' en la raíz.")
        return None

scaler = safe_load_scaler(SCALER_PATH)

def prepare_features(df_cleaned, label_col='SITUACION'):
    """Extrae las 25 features del DataFrame limpio (26 columnas - SITUACION)"""
    add_log("🔧 Preparando features para el modelo...", "info")
    
    # Separar features de la etiqueta
    if label_col in df_cleaned.columns:
        X = df_cleaned.drop(columns=[label_col])
        add_log(f"  ℹ️ Columna '{label_col}' excluida (etiqueta real)", "info")
    else:
        X = df_cleaned.copy()
        add_log(f"  ℹ️ No se encontró columna '{label_col}' - procesando todas las columnas", "info")
    
    # Verificar que tengamos exactamente 25 features
    if len(X.columns) != 25:
        add_log(f"⚠️ ADVERTENCIA: Se esperaban 25 features pero se encontraron {len(X.columns)}", "warning")
        add_log(f"  Columnas encontradas: {', '.join(X.columns[:10])}...", "info")
        
        # Ajustar número de features si es necesario
        if len(X.columns) < 25:
            # Agregar columnas faltantes con ceros
            for i in range(len(X.columns), 25):
                X[f'FEATURE_EXTRA_{i}'] = 0
            add_log(f"  ✓ Agregadas {25 - len(X.columns)} columnas con ceros", "warning")
        elif len(X.columns) > 25:
            # Tomar solo las primeras 25
            X = X.iloc[:, :25]
            add_log(f"  ✓ Tomadas solo las primeras 25 columnas", "warning")
    
    add_log(f"✅ Features preparadas: {len(X)} estudiantes × {len(X.columns)} variables", "success")
    return X

def predict_probs(model, X):
    """Predice probabilidades con logs detallados por estudiante"""
    if model is None:
        add_log("⚠️ Modelo no disponible - retornando predicciones vacías", "error")
        return np.full(len(X), np.nan)
    
    try:
        add_log(f"🧠 Cargando {len(X)} estudiantes al modelo...", "info")
        add_log(f"📊 Características generadas: {X.shape[1]} features por estudiante", "info")
        
        # Predicción batch completa
        preds = model.predict(X, verbose=0)
        preds = np.array(preds).reshape(-1)
        preds = np.clip(preds, 0, 1)
        
        add_log(f"✅ Predicciones completadas para todos los estudiantes", "success")
        
        return preds
    except Exception as e:
        add_log(f"❌ Error al predecir: {str(e)}", "error")
        return np.full(len(X), np.nan)
    

# Sidebar mejorado con diseño atractivo
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">📁 Archivos</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📥 Descargar Plantillas", expanded=True):
        st.markdown("##### 📋 Plantilla vacía")
        st.caption("Descarga esta plantilla para completar con tus datos")
        tpl_df = make_template_df()
        st.download_button(
            "⬇️ Plantilla (solo encabezados)", 
            tpl_df.to_csv(index=False).encode('utf-8'), 
            file_name='plantilla_estudiantes.csv', 
            mime='text/csv', 
            key='btn_plantilla',
            use_container_width=True
        )
        
        st.markdown("---")
        
        st.markdown("##### 📊 Archivo de ejemplo")
        st.caption("20 estudiantes con datos diversos para probar el modelo")
        example_df = make_example_df()
        st.download_button(
            "⬇️ Ejemplo (20 estudiantes)", 
            example_df.to_csv(index=False).encode('utf-8'), 
            file_name='ejemplo_20_estudiantes.csv', 
            mime='text/csv', 
            key='btn_ejemplo20',
            use_container_width=True
        )
        
        st.markdown("---")
        
        st.markdown("##### ℹ️ Instrucciones")
        st.info("""
        **Columnas Obligatorias (18):**
        - ESTP_FECHAINGRESO, CREDITOSAPROBADOS
        - UBICACION_SEMESTRAL, PROMEDIO_GENERAL
        - PROGRAMA, JORNADA, GENERO
        - FECHA_NACIMIENTO, CIUDADRESIDENCIA
        - ESTRATO, TIENE_SISBEN
        - INFE_VIVECONFAMILIA, INFE_SITUACIONPADRES
        - INFE_NUMEROFAMILIARES, INFE_NUMEROHERMANOS
        - INFE_POSICIONENHERMANOS, INFE_NUMMIEMBROSTRABAJA
        
        📖 Ver INSTRUCCIONES_CARGA_DATOS.md para detalles completos
        """)

uploaded_file = st.file_uploader(
    "📤 Subir archivo CSV/Excel con datos de estudiantes", 
    type=["csv", "xlsx"], 
    key='file_uploader_main',
    help="Sube un archivo con las 18 columnas obligatorias (ver instrucciones)"
)

# Create stable placeholders
main_content = st.container()

if uploaded_file is not None:
    from datetime import datetime
    
    # Leer archivo según extensión
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # Generar timestamp para nombres de archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h3 style="color: white; text-align: center; margin: 0;">⚙️ Configuración</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### 🎯 Umbral de clasificación")
        threshold = st.slider(
            "Probabilidad mínima para clasificar como 'No aprueba'", 
            0.0, 1.0, 0.5, 0.01, 
            key='slider_threshold',
            help="Valores mayores a este umbral se clasifican como 'No aprueba'"
        )
        
        # Mostrar información del archivo
        st.markdown("---")
        st.markdown("##### 📊 Información del archivo original")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📁 Registros", len(df_raw))
        with col2:
            st.metric("📋 Columnas", len(df_raw.columns))
    
    with main_content:
        # Limpiar logs anteriores
        st.session_state.process_logs = []
        
        # Guardar archivo original
        base_filename = f"archivo_base_{timestamp}.{'csv' if uploaded_file.name.endswith('.csv') else 'xlsx'}"
        base_path = os.path.join("archivos_procesados", "base", base_filename)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        if uploaded_file.name.endswith('.csv'):
            df_raw.to_csv(base_path, index=False)
        else:
            df_raw.to_excel(base_path, index=False)
        add_log(f"💾 Archivo original guardado: {base_path}", "success")
        
        # Panel de logs en tiempo real
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h3 style="color: white; text-align: center; margin: 0;">📋 Registro de Proceso en Tiempo Real</h3>
        </div>
        """, unsafe_allow_html=True)
        
        log_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # PASO 1: Limpieza automática de datos
        status_text.text("🔄 Paso 1/4: Limpiando datos...")
        progress_bar.progress(0.25)
        
        df_cleaned = clean_data(df_raw)
        
        # Guardar archivo limpio
        clean_filename = f"archivo_limpio_{timestamp}.xlsx"
        clean_path = os.path.join("archivos_procesados", "limpio", clean_filename)
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        df_cleaned.to_excel(clean_path, index=False)
        add_log(f"💾 Archivo limpio guardado: {clean_path}", "success")
        
        # Mostrar logs de limpieza
        with log_container:
            with st.expander("📜 Ver logs de limpieza de datos", expanded=True):
                for log in st.session_state.process_logs:
                    if log['type'] == 'success':
                        st.success(f"[{log['timestamp']}] {log['message']}")
                    elif log['type'] == 'warning':
                        st.warning(f"[{log['timestamp']}] {log['message']}")
                    elif log['type'] == 'error':
                        st.error(f"[{log['timestamp']}] {log['message']}")
                    else:
                        st.info(f"[{log['timestamp']}] {log['message']}")
        
        # Verificar si hay datos después de limpieza
        if len(df_cleaned) == 0:
            st.error("❌ No quedan datos después de la limpieza. Verifica que el archivo contenga estudiantes de programas válidos.")
            st.stop()
        
        # PASO 2: Detectar si existe SITUACION (para evaluación)
        status_text.text("🔄 Paso 2/4: Validando estructura de datos...")
        progress_bar.progress(0.40)
        
        has_situacion = 'SITUACION' in df_cleaned.columns
        if has_situacion:
            add_log("✓ Columna SITUACION detectada - se habilitará evaluación del modelo", "success")
        else:
            add_log("ℹ️ No se encontró columna SITUACION - solo se harán predicciones", "info")
        
        add_log(f"✅ Datos listos: {len(df_cleaned)} estudiantes × {len(df_cleaned.columns)} variables", "success")
        
        # PASO 3: Preparar características y cargar al modelo
        status_text.text("🔄 Paso 3/4: Preparando características y cargando al modelo...")
        progress_bar.progress(0.60)
        
        X = prepare_features(df_cleaned, label_col='SITUACION' if has_situacion else None)
        
        # Aplicar scaler si existe
        if scaler is not None and not X.empty:
            try:
                add_log("📏 Aplicando normalización (scaler)...", "info")
                X_scaled = scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                add_log("✓ Normalización aplicada correctamente", "success")
            except Exception as e:
                add_log(f"⚠️ No se aplicó el scaler: {e}", "warning")
        
        # PASO 4: Predicciones
        status_text.text("🔄 Paso 4/4: Generando predicciones...")
        progress_bar.progress(0.80)
        
        probs = predict_probs(model, X)
        
        # Generar IDs para resultados (usar índice + 1)
        estudiante_ids = [f"ESTUDIANTE_{i+1:04d}" for i in range(len(df_cleaned))]
        
        # Procesar resultados estudiante por estudiante con logs
        add_log("📝 Procesando resultados individuales...", "info")
        
        df_results = pd.DataFrame()
        df_results['IDENTIFICADOR'] = estudiante_ids
        df_results['prob_desercion'] = probs
        
        # Clasificar y loggear algunos ejemplos
        valid_mask = np.isfinite(probs)
        if valid_mask.any():
            df_results['estadistica_%'] = df_results['prob_desercion'].apply(
                lambda p: (np.sum(probs <= p) / len(probs) * 100) if np.isfinite(p) else np.nan
            )
        else:
            df_results['estadistica_%'] = np.nan
        
        df_results['resultado_modelo'] = np.where(
            df_results['prob_desercion'] >= threshold, 'No aprueba', 'Aprueba'
        )
        
        # Loggear muestra de resultados
        add_log(f"📊 Resumen de análisis:", "info")
        aprueban = (df_results['resultado_modelo'] == 'Aprueba').sum()
        no_aprueban = (df_results['resultado_modelo'] == 'No aprueba').sum()
        add_log(f"  ✅ Estudiantes que aprueban: {aprueban} ({aprueban/len(df_results)*100:.1f}%)", "success")
        add_log(f"  ❌ Estudiantes en riesgo: {no_aprueban} ({no_aprueban/len(df_results)*100:.1f}%)", "warning")
        
        # Ejemplos de clasificación
        add_log("📋 Ejemplos de clasificaciones:", "info")
        for idx in df_results.head(5).index:
            estudiante_id = df_results.loc[idx, 'IDENTIFICADOR']
            prob = df_results.loc[idx, 'prob_desercion']
            resultado = df_results.loc[idx, 'resultado_modelo']
            emoji = "✅" if resultado == "Aprueba" else "❌"
            add_log(f"  {emoji} {estudiante_id}: {prob:.3f} → {resultado}", "info")
        
        # Guardar resultados
        result_filename = f"resultados_prediccion_{timestamp}.csv"
        result_path = os.path.join("archivos_procesados", "resultados", result_filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        out_csv = df_results[['IDENTIFICADOR', 'resultado_modelo', 'prob_desercion']].copy()
        out_csv.columns = ['identificador', 'resultado_modelo', 'probabilidad_estadistica']
        out_csv.to_csv(result_path, index=False)
        add_log(f"💾 Resultados guardados: {result_path}", "success")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Proceso completado")
        
        # Actualizar logs finales
        with log_container:
            with st.expander("📜 Ver todos los logs del proceso", expanded=False):
                for log in st.session_state.process_logs:
                    if log['type'] == 'success':
                        st.success(f"[{log['timestamp']}] {log['message']}")
                    elif log['type'] == 'warning':
                        st.warning(f"[{log['timestamp']}] {log['message']}")
                    elif log['type'] == 'error':
                        st.error(f"[{log['timestamp']}] {log['message']}")
                    else:
                        st.info(f"[{log['timestamp']}] {log['message']}")

        # Mostrar información de archivos guardados
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: white; text-align: center; margin: 0;">📁 Archivos Guardados</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.info(f"**📥 Archivo Base**\n\n`{base_filename}`\n\nUbicación: `archivos_procesados/base/`")
        with col_info2:
            st.success(f"**🧹 Archivo Limpio**\n\n`{clean_filename}`\n\nUbicación: `archivos_procesados/limpio/`")
        with col_info3:
            st.success(f"**📊 Resultados**\n\n`{result_filename}`\n\nUbicación: `archivos_procesados/resultados/`")
        
        # Mostrar success message y botón de descarga con diseño mejorado
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: white; text-align: center; margin: 0;">✅ Predicciones completadas exitosamente</h3>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">Los resultados están listos para descargar</p>
        </div>
        """, unsafe_allow_html=True)
        
        # CSV de salida con botón destacado
        col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
        with col_btn2:
            st.download_button(
                "💾 Descargar Resultados (CSV)", 
                out_csv.to_csv(index=False).encode('utf-8'), 
                file_name='resultados_prediccion.csv', 
                mime='text/csv', 
                key='btn_download_results',
                use_container_width=True
            )

        # Layout dashboard con diseño mejorado
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1rem; border-radius: 10px; margin: 2rem 0 1rem 0;">
            <h2 style="color: white; text-align: center; margin: 0;">📊 Análisis de Resultados</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("#### 📈 Resumen General")
            
            # Info tooltip para el resumen
            st.info("""
            **📊 ¿Qué muestra este resumen?**
            - Este panel muestra la clasificación final de los estudiantes según el modelo predictivo
            - **Aprueban**: Estudiantes con probabilidad de deserción menor al umbral configurado
            - **No aprueban**: Estudiantes con alta probabilidad de deserción que requieren intervención
            """)
            
            counts = df_results['resultado_modelo'].value_counts().reindex(['Aprueba','No aprueba']).fillna(0).astype(int)
            
            # Métricas destacadas
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                total = len(df_results)
                st.metric("👥 Total Estudiantes", total)
            with mcol2:
                aprueba_pct = (int(counts.get('Aprueba', 0)) / total * 100) if total > 0 else 0
                st.metric("✅ Aprueban", f"{int(counts.get('Aprueba', 0))}", f"{aprueba_pct:.1f}%")
            with mcol3:
                noaprueba_pct = (int(counts.get('No aprueba', 0)) / total * 100) if total > 0 else 0
                st.metric("❌ No aprueban", f"{int(counts.get('No aprueba', 0))}", f"{noaprueba_pct:.1f}%")
            
            # Gráfico de barras mejorado
            fig_bar = px.bar(
                x=counts.index, 
                y=counts.values, 
                labels={'x':'Estado','y':'Cantidad de Estudiantes'}, 
                color=counts.index, 
                color_discrete_map={'Aprueba':'#2ecc71','No aprueba':'#e74c3c'},
                title='Clasificación de Estudiantes'
            )
            fig_bar.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig_bar, use_container_width=True, key='plot_bar_resumen')
            
            with st.expander("ℹ️ ¿Cómo interpretar este gráfico?"):
                st.markdown("""
                **Gráfico de Barras - Clasificación de Estudiantes**
                
                - **Barra Verde (Aprueba)**: Estudiantes con probabilidad de deserción menor al umbral
                - **Barra Roja (No aprueba)**: Estudiantes en riesgo que necesitan atención
                
                **💡 Recomendación**: 
                - Si > 70% aprueba: El grupo tiene buen rendimiento general
                - Si < 50% aprueba: Se requiere intervención urgente del programa
                """)

            # Histograma mejorado
            st.markdown("#### 📊 Distribución de Probabilidades de Deserción")
            fig_hist = px.histogram(
                df_results, 
                x='prob_desercion', 
                nbins=40,
                color_discrete_sequence=['#667eea'],
                labels={'prob_desercion': 'Probabilidad de Deserción'}
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True, key='plot_hist_prob')
            
            with st.expander("ℹ️ ¿Cómo interpretar este histograma?"):
                st.markdown("""
                **Histograma de Probabilidades**
                
                - **Eje X**: Probabilidad de deserción (0 = muy bajo riesgo, 1 = muy alto riesgo)
                - **Eje Y**: Número de estudiantes en cada rango de probabilidad
                - **Barras altas cerca de 0**: Muchos estudiantes con bajo riesgo ✅
                - **Barras altas cerca de 1**: Muchos estudiantes en riesgo alto ⚠️
                
                **💡 Interpretación**:
                - **Distribución hacia la izquierda**: Grupo saludable académicamente
                - **Distribución hacia la derecha**: Grupo en riesgo generalizado
                - **Distribución bimodal (dos picos)**: Grupo polarizado - atender a los de alto riesgo
                """)

        with col2:
            st.markdown("#### 📉 Estadísticas Clave")
            
            st.info("""
            **📊 Percentiles**
            
            Los percentiles dividen el grupo en cuartos según el riesgo:
            - **P25**: El 25% de estudiantes están por debajo de este valor
            - **P50 (Mediana)**: Valor central - divide al grupo en dos mitades
            - **P75**: El 75% de estudiantes están por debajo de este valor
            
            **Ejemplo**: Si P50 = 0.3, significa que la mitad del grupo tiene probabilidad < 0.3
            """)
            
            if valid_mask.any():
                p25 = np.nanpercentile(probs,25)
                p50 = np.nanpercentile(probs,50)
                p75 = np.nanpercentile(probs,75)
                p_mean = np.nanmean(probs)
                
                st.metric("📊 Promedio", f"{p_mean:.3f}", 
                         help="Promedio de probabilidad de deserción del grupo")
                st.metric("📉 Percentil 25", f"{p25:.3f}",
                         help="25% de estudiantes tienen probabilidad menor a este valor")
                st.metric("📊 Mediana (P50)", f"{p50:.3f}",
                         help="Valor central - divide al grupo en dos mitades iguales")
                st.metric("📈 Percentil 75", f"{p75:.3f}",
                         help="75% de estudiantes tienen probabilidad menor a este valor")
                
                # Interpretación mejorada
                st.markdown("---")
                st.markdown("##### 💡 Análisis del Grupo")
                if p50 < 0.3:
                    st.success("🟢 **Riesgo Bajo**: La mayoría del grupo está en buen estado académico")
                elif p50 < 0.7:
                    st.warning("🟡 **Riesgo Moderado**: Se recomienda monitoreo y apoyo preventivo")
                else:
                    st.error("🔴 **Riesgo Alto**: Se requiere intervención inmediata del programa")
                
                # Estadísticas adicionales
                st.markdown("---")
                st.markdown("##### 📈 Estadísticas Adicionales")
                p_std = np.nanstd(probs)
                p_min = np.nanmin(probs)
                p_max = np.nanmax(probs)
                
                st.metric("📏 Desviación Estándar", f"{p_std:.3f}",
                         help="Mide la dispersión de las probabilidades - mayor valor = grupo más heterogéneo")
                st.metric("⬇️ Mínimo", f"{p_min:.3f}",
                         help="Estudiante con menor riesgo de deserción")
                st.metric("⬆️ Máximo", f"{p_max:.3f}",
                         help="Estudiante con mayor riesgo de deserción")
            else:
                st.info("No hay probabilidades válidas")

        # If label column present, show confusion and false positives
        st.markdown("---")
        if has_situacion:
            label_col = 'SITUACION'
            if label_col in df_cleaned.columns:
                with st.container():
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; text-align: center; margin: 0;">🔄 Evaluación del Modelo con Datos Reales</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("""
                    **📊 ¿Qué es esta sección?**
                    
                    Ya que tienes datos reales de deserción (columna SITUACION), podemos evaluar qué tan bien predice el modelo.
                    Aquí comparamos las predicciones del modelo vs. la realidad para medir su precisión.
                    """)
                    
                    y_true = df_cleaned[label_col].values
                    # try convert to binary 0/1
                    try:
                        y_true = np.array(y_true, dtype=float)
                        y_true = np.where(y_true>=0.5,1,0)
                    except Exception:
                        st.warning("⚠️ La columna etiqueta no es numérica; intenta convertir a 0/1 antes de subir.")
                        y_true = None

                    if y_true is not None and valid_mask.any():
                        y_pred = np.where(df_results['prob_desercion'] >= threshold, 1, 0)
                        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                        
                        # Calcular métricas
                        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        # Primera fila: Matriz de confusión y métricas principales
                        st.markdown("#### 📊 Matriz de Confusión y Métricas")
                        
                        ccol1, ccol2 = st.columns(2)
                        with ccol1:
                            cm_fig = go.Figure(data=go.Heatmap(
                                z=[[tn, fp],[fn, tp]], 
                                x=['Pred: Aprueba','Pred: No aprueba'], 
                                y=['Real: Aprueba','Real: No aprueba'], 
                                colorscale='Blues',
                                text=[[f'VN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'VP: {tp}']],
                                texttemplate='%{text}',
                                textfont={"size": 16}
                            ))
                            cm_fig.update_layout(title='Matriz de confusión', height=400)
                            st.plotly_chart(cm_fig, use_container_width=True, key='plot_confusion_matrix')
                            
                            with st.expander("ℹ️ ¿Cómo leer la matriz de confusión?"):
                                st.markdown("""
                                **Matriz de Confusión - Guía de Lectura**
                                
                                - **VN (Verdaderos Negativos)**: ✅ Predijo "Aprueba" y SÍ aprobó
                                - **FP (Falsos Positivos)**: ⚠️ Predijo "No aprueba" pero SÍ aprobó (Error Tipo I)
                                - **FN (Falsos Negativos)**: ⚠️ Predijo "Aprueba" pero NO aprobó (Error Tipo II)  
                                - **VP (Verdaderos Positivos)**: ✅ Predijo "No aprueba" y NO aprobó
                                
                                **💡 Lo ideal**: VN y VP altos, FP y FN bajos
                                
                                **⚠️ Errores más peligrosos**:
                                - **FN (Falsos Negativos)**: El modelo dice que aprueba pero deserta → No recibe ayuda necesaria
                                - **FP (Falsos Positivos)**: El modelo dice que deserta pero aprueba → Se le da ayuda innecesaria (menos grave)
                                """)
                        
                        with ccol2:
                            st.markdown("##### 📊 Métricas de Rendimiento")
                            
                            st.info("""
                            **Guía de Métricas:**
                            - **Exactitud**: % total de predicciones correctas
                            - **Precisión**: De los que predijo "deserta", % que realmente desertaron
                            - **Recall**: De los que realmente desertaron, % que detectó el modelo
                            - **Especificidad**: De los que realmente aprobaron, % que identificó correctamente
                            - **F1-Score**: Balance entre Precisión y Recall (0-1, mayor es mejor)
                            - **AUC ROC**: Capacidad de discriminación del modelo (0.5-1, >0.7 es bueno)
                            """)
                            
                            m1, m2 = st.columns(2)
                            with m1:
                                st.metric("🎯 Exactitud", f"{accuracy:.2%}",
                                         help="Porcentaje de predicciones correctas del total")
                                st.metric("✅ Precisión", f"{precision:.2%}",
                                         help="De los que predijo 'deserta', cuántos realmente desertaron")
                                st.metric("📈 Recall", f"{recall:.2%}",
                                         help="De los que realmente desertaron, cuántos detectó")
                            with m2:
                                st.metric("🔍 Especificidad", f"{specificity:.2%}",
                                         help="De los que realmente aprobaron, cuántos identificó correctamente")
                                st.metric("⚖️ F1-Score", f"{f1_score:.3f}",
                                         help="Balance entre Precisión y Recall (0-1)")
                                try:
                                    auc = roc_auc_score(y_true, df_results['prob_desercion'])
                                    st.metric("📐 AUC ROC", f"{auc:.3f}",
                                             help="Capacidad general del modelo (>0.7 es bueno)")
                                except Exception:
                                    pass
                        
                        # Segunda fila: Gráfico de barras de métricas de confusión
                        st.markdown("#### 📊 Distribución de predicciones")
                        metrics_df = pd.DataFrame({
                            'Métrica': ['Verdaderos Negativos (VN)', 'Falsos Positivos (FP)', 
                                       'Falsos Negativos (FN)', 'Verdaderos Positivos (VP)'],
                            'Cantidad': [tn, fp, fn, tp],
                            'Color': ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4']
                        })
                        
                        fig_metrics = go.Figure(data=[
                            go.Bar(
                                x=metrics_df['Métrica'],
                                y=metrics_df['Cantidad'],
                                marker_color=metrics_df['Color'],
                                text=metrics_df['Cantidad'],
                                textposition='auto'
                            )
                        ])
                        fig_metrics.update_layout(
                            title='Distribución de clasificaciones',
                            xaxis_title='Tipo de predicción',
                            yaxis_title='Cantidad',
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_metrics, use_container_width=True, key='plot_metrics_bar')

                        # Tercera fila: ROC Curve y distribución de probabilidades por clase
                        st.markdown("#### 📈 Curvas de Evaluación")
                        rcol1, rcol2 = st.columns(2)
                        
                        with rcol1:
                            try:
                                fpr, tpr, _ = roc_curve(y_true, df_results['prob_desercion'])
                                roc_fig = go.Figure()
                                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#1f77b4', width=3)))
                                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
                                roc_fig.update_layout(
                                    xaxis_title='Tasa de Falsos Positivos (FPR)', 
                                    yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                                    title='Curva ROC',
                                    height=400
                                )
                                st.plotly_chart(roc_fig, use_container_width=True, key='plot_roc_curve')
                                
                                with st.expander("ℹ️ ¿Qué es la Curva ROC?"):
                                    st.markdown("""
                                    **Curva ROC (Receiver Operating Characteristic)**
                                    
                                    Evalúa la capacidad del modelo para distinguir entre clases en todos los umbrales posibles.
                                    
                                    - **Línea Azul (ROC)**: Rendimiento del modelo
                                    - **Línea Gris Punteada**: Modelo aleatorio (50/50)
                                    
                                    **💡 Interpretación**:
                                    - **Curva cercana a la esquina superior izquierda**: Excelente modelo
                                    - **Curva sobre la diagonal**: Modelo sin capacidad predictiva
                                    - **AUC = 1.0**: Modelo perfecto
                                    - **AUC = 0.7-0.8**: Modelo aceptable
                                    - **AUC = 0.8-0.9**: Modelo bueno
                                    - **AUC > 0.9**: Modelo excelente
                                    """)
                            except Exception:
                                pass
                        
                        with rcol2:
                            # Distribución de probabilidades por clase real
                            dist_df = pd.DataFrame({
                                'probabilidad': df_results['prob_desercion'],
                                'clase_real': ['Deserta' if y == 1 else 'Aprueba' for y in y_true]
                            })
                            fig_dist = px.histogram(
                                dist_df, 
                                x='probabilidad', 
                                color='clase_real',
                                nbins=30,
                                title='Distribución por Clase Real',
                                labels={'probabilidad': 'Probabilidad de deserción', 'clase_real': 'Clase real'},
                                color_discrete_map={'Aprueba': '#2ca02c', 'Deserta': '#d62728'},
                                barmode='overlay',
                                opacity=0.7
                            )
                            fig_dist.update_layout(height=400)
                            st.plotly_chart(fig_dist, use_container_width=True, key='plot_prob_distribution')
                            
                            with st.expander("ℹ️ ¿Cómo interpretar este gráfico?"):
                                st.markdown("""
                                **Distribución de Probabilidades por Clase Real**
                                
                                Compara las probabilidades asignadas según si el estudiante realmente aprobó o desertó.
                                
                                - **Barras Verdes**: Estudiantes que realmente aprobaron
                                - **Barras Rojas**: Estudiantes que realmente desertaron
                                
                                **💡 Modelo ideal**:
                                - Verdes concentradas a la IZQUIERDA (probabilidades bajas)
                                - Rojas concentradas a la DERECHA (probabilidades altas)
                                - **Poca superposición** = Modelo discrimina bien
                                
                                **⚠️ Señales de alerta**:
                                - Mucha superposición = Modelo confunde las clases
                                - Distribuciones similares = Modelo no aprende patrones
                                """)

                        # Mostrar ejemplos de cada categoría
                        st.markdown("#### 📋 Ejemplos por categoría de predicción")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            f"✅ Verdaderos Negativos ({tn})",
                            f"⚠️ Falsos Positivos ({fp})",
                            f"⚠️ Falsos Negativos ({fn})",
                            f"❌ Verdaderos Positivos ({tp})"
                        ])
                        
                        with tab1:
                            if tn > 0:
                                vn_mask = (y_true == 0) & (y_pred == 0)
                                st.caption("✅ Predicción correcta: El modelo predice que aprueba y efectivamente aprueba")
                                st.dataframe(df_results.loc[vn_mask, :].head(10), use_container_width=True, key='df_true_negatives')
                            else:
                                st.info("No hay verdaderos negativos en esta predicción")
                        
                        with tab2:
                            if fp > 0:
                                fp_mask = (y_true == 0) & (y_pred == 1)
                                st.caption("⚠️ Error Tipo I: El modelo predice deserción pero el estudiante aprueba")
                                st.dataframe(df_results.loc[fp_mask, :].head(10), use_container_width=True, key='df_false_positives')
                            else:
                                st.info("No hay falsos positivos en esta predicción")
                        
                        with tab3:
                            if fn > 0:
                                fn_mask = (y_true == 1) & (y_pred == 0)
                                st.caption("⚠️ Error Tipo II: El modelo predice que aprueba pero el estudiante deserta")
                                st.dataframe(df_results.loc[fn_mask, :].head(10), use_container_width=True, key='df_false_negatives')
                            else:
                                st.info("No hay falsos negativos en esta predicción")
                        
                        with tab4:
                            if tp > 0:
                                vp_mask = (y_true == 1) & (y_pred == 1)
                                st.caption("❌ Predicción correcta: El modelo predice deserción y efectivamente deserta")
                                st.dataframe(df_results.loc[vp_mask, :].head(10), use_container_width=True, key='df_true_positives')
                            else:
                                st.info("No hay verdaderos positivos en esta predicción")

else:
    # Mensaje de bienvenida cuando no hay archivo
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; font-size: 2rem; margin-bottom: 1rem;">👋 ¡Bienvenido al Sistema de Predicción!</h2>
        <p style="color: #f0f0f0; font-size: 1.2rem; margin: 0;">Para comenzar, sube un archivo CSV con los datos de los estudiantes</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instrucciones visuales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center; height: 250px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📥</div>
            <h4 style="color: #667eea;">Paso 1</h4>
            <p style="color: #6c757d;">Descarga la plantilla o el archivo de ejemplo desde el menú lateral</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center; height: 250px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">✏️</div>
            <h4 style="color: #667eea;">Paso 2</h4>
            <p style="color: #6c757d;">Completa el CSV con la información de tus estudiantes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center; height: 250px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🚀</div>
            <h4 style="color: #667eea;">Paso 3</h4>
            <p style="color: #6c757d;">Sube el archivo y obtén predicciones instantáneas con gráficos detallados</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información adicional
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2196F3;">
        <h4 style="color: #2196F3; margin-top: 0;">📋 Campos requeridos en el CSV:</h4>
        <ul style="color: #495057; line-height: 1.8;">
            <li><strong>IDENTIFICADOR:</strong> Código único del estudiante</li>
            <li><strong>EDAD:</strong> Edad del estudiante (número)</li>
            <li><strong>PROMEDIO:</strong> Promedio académico (0.0 - 5.0)</li>
            <li><strong>ASISTENCIA_%:</strong> Porcentaje de asistencia (0 - 100)</li>
            <li><strong>CREDITOS_APROBADOS:</strong> Número de créditos aprobados</li>
            <li><strong>NUM_REPROBADOS:</strong> Número de materias reprobadas</li>
            <li><strong>HORAS_ESTUDIO_SEMANA:</strong> Horas de estudio por semana</li>
            <li><strong>TRABAJA:</strong> SI o NO</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffc107;">
        <p style="color: #856404; margin: 0;"><strong>💡 Tip:</strong> El modelo utiliza 25 características derivadas automáticamente de tus datos para hacer predicciones más precisas.</p>
    </div>
    """, unsafe_allow_html=True)

