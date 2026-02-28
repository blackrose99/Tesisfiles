import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load as joblib_load
from tensorflow.keras.models import load_model
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="🎓 Predicción de Deserción Estudiantil",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH  = "modelo_desercion_nn.keras"
SCALER_PATH = "scaler.joblib"

# ─────────────────────────────────────────────────────────────────
#  CARGAR MODELO Y SCALER (cacheado)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    mdl, scl = None, None
    if os.path.exists(MODEL_PATH):
        try:
            mdl = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
    else:
        st.error(f"Modelo no encontrado: `{MODEL_PATH}`")

    if os.path.exists(SCALER_PATH):
        try:
            scl = joblib_load(SCALER_PATH)
        except Exception as e:
            st.error(f"No se pudo cargar el scaler: {e}")

    return mdl, scl

model, scaler = load_artifacts()

# ─────────────────────────────────────────────────────────────────
#  PLANTILLAS
# ─────────────────────────────────────────────────────────────────
def make_template_df():
    return pd.DataFrame(columns=[
        "CODESTUDIANTE", "ESTP_FECHAINGRESO", "CREDITOSAPROBADOS",
        "UBICACION_SEMESTRAL", "PROMEDIO_GENERAL", "PROGRAMA", "JORNADA",
        "GENERO", "FECHA_NACIMIENTO", "CIUDADRESIDENCIA", "ESTRATO",
        "TIENE_SISBEN", "INFE_VIVECONFAMILIA", "INFE_SITUACIONPADRES",
        "INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS",
        "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA",
    ])

def make_example_df():
    if os.path.exists("ejemplo_datos_estudiantes.csv"):
        return pd.read_csv("ejemplo_datos_estudiantes.csv")
    return make_template_df()

# ─────────────────────────────────────────────────────────────────
#  LIMPIEZA — replica el pipeline que generó el Excel "ready_step4"
#  que usó el script de entrenamiento.
#
#  Reglas clave (copiadas del script de entrenamiento original):
#  - CIUDADRESIDENCIA → numérico (1-4, resto=5)
#  - Fechas → EDAD_INGRESO, ANIO_INGRESO, MES_INGRESO
#  - ESTRATO fuera de rango → 0
#  - Columnas categóricas → pd.get_dummies (ONE-HOT, dtype=int)
#  - NaN en familia → -1,  ESTRATO → 0,  resto → 0
# ─────────────────────────────────────────────────────────────────
def clean_data(df_raw):
    df = df_raw.copy()

    # Guardar identificadores
    codigos = df["CODESTUDIANTE"].astype(str).tolist() if "CODESTUDIANTE" in df.columns else None

    # 1) Columnas que no entran al modelo
    drop_cols = [
        "CODESTUDIANTE", "CODIGOCIUDADR", "NIVEL_SISBEN", "CATEGORIA",
        "CODMATRICULA", "SEDE", "INFE_HERMANOSESTUDIANDOU", "SITUACION",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2) Fechas → variables numéricas
    if "ESTP_FECHAINGRESO" in df.columns:
        # Limpiar texto basura como "12/12/2018 10:45:51 AM CANCELACION SEMESTRE"
        df["ESTP_FECHAINGRESO"] = (
            df["ESTP_FECHAINGRESO"].astype(str)
            .str.extract(r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)")[0]
        )
    df["ESTP_FECHAINGRESO"] = pd.to_datetime(df["ESTP_FECHAINGRESO"], errors="coerce", dayfirst=True)
    df["FECHA_NACIMIENTO"]  = pd.to_datetime(df["FECHA_NACIMIENTO"],  errors="coerce", dayfirst=True)

    df["EDAD_INGRESO"] = (
        (df["ESTP_FECHAINGRESO"] - df["FECHA_NACIMIENTO"]).dt.days / 365.25
    ).round().astype("Int64")
    df["ANIO_INGRESO"] = df["ESTP_FECHAINGRESO"].dt.year
    df["MES_INGRESO"]  = df["ESTP_FECHAINGRESO"].dt.month
    df = df.drop(columns=["ESTP_FECHAINGRESO", "FECHA_NACIMIENTO"])

    # 3) ESTRATO: fuera de rango → NaN (luego se rellena con 0)
    df["ESTRATO"] = pd.to_numeric(df["ESTRATO"], errors="coerce")
    df.loc[(df["ESTRATO"] < 1) | (df["ESTRATO"] > 6), "ESTRATO"] = pd.NA

    # 4) Solo programas válidos
    programas_validos = [
        "INGENIERIA DE SISTEMAS",
        "TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS",
    ]
    if "PROGRAMA" in df.columns:
        df = df[df["PROGRAMA"].isin(programas_validos)]

    # 5) CIUDADRESIDENCIA → numérico
    if "CIUDADRESIDENCIA" in df.columns:
        mapa_ciudad = {"BUCARAMANGA": 1, "FLORIDABLANCA": 2, "GIRON": 3, "PIEDECUESTA": 4}
        df["CIUDADRESIDENCIA"] = (
            df["CIUDADRESIDENCIA"].astype(str).str.strip().str.upper()
            .map(mapa_ciudad).fillna(5).astype(int)
        )

    # 6) ONE-HOT ENCODING de columnas categóricas
    #    Igual que pd.get_dummies en el pipeline de entrenamiento
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if df[c].nunique(dropna=True) > 1]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)

    # 7) Booleanos → int
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)

    # 8) Valores faltantes
    if "ESTRATO" in df.columns:
        df["ESTRATO"] = df["ESTRATO"].fillna(0).astype(int)

    for c in ["INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS",
              "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA", "EDAD_INGRESO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).round().astype(int)

    if "TIENE_SISBEN" in df.columns:
        df["TIENE_SISBEN"] = pd.to_numeric(df["TIENE_SISBEN"], errors="coerce").fillna(-1).round().astype(int)

    # Todo a numérico, NaN residuales → 0
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    return df, codigos


def prepare_for_model(df_cleaned, scaler):
    """
    Alinea el DataFrame a las columnas exactas con que se entrenó el scaler,
    respetando orden y cantidad. Columnas faltantes (categorías no vistas
    en este lote) se rellenan con 0. Columnas sobrantes se descartan.
    Luego aplica la transformación del scaler.
    """
    expected_cols = list(scaler.feature_names_in_)

    # Agregar columnas faltantes con 0
    for col in expected_cols:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0

    # Seleccionar y ordenar exactamente las columnas del scaler
    X = df_cleaned[expected_cols]

    # Aplicar normalización (mismo espacio que el entrenamiento)
    X_scaled = scaler.transform(X)
    return X_scaled


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            padding:2rem;border-radius:15px;margin-bottom:2rem;
            box-shadow:0 4px 6px rgba(0,0,0,0.1);">
    <h1 style="color:white;text-align:center;margin:0;font-size:3rem;">
        🎓 Predicción de Deserción Estudiantil
    </h1>
    <p style="color:#f0f0f0;text-align:center;font-size:1.2rem;margin-top:0.5rem;">
        Sistema Inteligente de Análisis Predictivo
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#f8f9fa;padding:1.5rem;border-radius:10px;
            border-left:5px solid #667eea;margin-bottom:1.5rem;">
    <h3 style="color:#667eea;margin-top:0;">🔍 ¿Qué significa la probabilidad?</h3>
    <p style="color:#495057;margin:0;line-height:1.8;">
        La red neuronal devuelve <strong>P(aprueba)</strong>: un valor entre 0 y 1
        que indica qué tan probable es que el estudiante deserte.<br><br>
        <strong>Ejemplo:</strong> <code>p_aprueba = 0.80</code> → 80% de riesgo de deserción.<br>
        El <strong>umbral</strong> (slider lateral) define el corte:
        <code>P ≥ umbral</code> →
        <span style="color:#e74c3c;font-weight:bold;">No aprueba</span> &nbsp;|&nbsp;
        <code>P &lt; umbral</code> →
        <span style="color:#27ae60;font-weight:bold;">Aprueba</span>.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  ALERTA SCALER FALTANTE
# ─────────────────────────────────────────────────────────────────
if scaler is None:
    st.error("""
❌ **`scaler.joblib` no encontrado.**

El scaler es indispensable — sin él los datos llegan al modelo en escala diferente
a la del entrenamiento y las probabilidades son inválidas.

**Solución:** agrega estas dos líneas al final del script de entrenamiento y ejecútalo:

```python
from joblib import dump
dump(scaler, 'scaler.joblib')
```

Luego copia `scaler.joblib` a la misma carpeta que `app.py`.
""")

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding:1rem;border-radius:10px;margin-bottom:1rem;">
        <h2 style="color:white;text-align:center;margin:0;">📁 Archivos</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📥 Descargar Plantillas", expanded=True):
        st.markdown("##### 📋 Plantilla vacía")
        st.download_button(
            "⬇️ Plantilla (solo encabezados)",
            make_template_df().to_csv(index=False).encode("utf-8"),
            file_name="plantilla_estudiantes.csv", mime="text/csv",
            key="btn_plantilla", use_container_width=True,
        )
        st.markdown("---")
        st.markdown("##### 📊 Archivo de ejemplo")
        st.download_button(
            "⬇️ Ejemplo (20 estudiantes)",
            make_example_df().to_csv(index=False).encode("utf-8"),
            file_name="ejemplo_20_estudiantes.csv", mime="text/csv",
            key="btn_ejemplo20", use_container_width=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);
                padding:1rem;border-radius:10px;margin-bottom:1rem;">
        <h3 style="color:white;text-align:center;margin:0;">⚙️ Configuración</h3>
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Umbral P(aprueba) → 'No aprueba'",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        key="slider_threshold",
        help="Si P(aprueba) ≥ umbral → 'No aprueba'. Bájalo para detectar más casos en riesgo.",
    )

# ─────────────────────────────────────────────────────────────────
#  CARGA DEL ARCHIVO
# ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Subir archivo CSV/Excel con datos de estudiantes",
    type=["csv", "xlsx"], key="file_uploader_main",
)

# ─────────────────────────────────────────────────────────────────
#  PROCESAMIENTO
# ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    if model is None:
        st.error("❌ Modelo no encontrado.")
        st.stop()

    if scaler is None:
        st.error("❌ `scaler.joblib` no encontrado. Lee las instrucciones arriba.")
        st.stop()

    df_raw = (
        pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith(".csv")
        else pd.read_excel(uploaded_file)
    )

    with st.sidebar:
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("📁 Registros", len(df_raw))
        c2.metric("📋 Columnas",  len(df_raw.columns))

    # ── Paso 1: Limpieza ─────────────────────────────────────────
    with st.spinner("🧹 Limpiando datos..."):
        df_cleaned, codigos = clean_data(df_raw)

    if len(df_cleaned) == 0:
        st.error("❌ No quedaron registros. Verifica que el archivo contenga programas válidos.")
        st.stop()

    st.success(f"✅ Limpieza completada — {len(df_cleaned)} estudiantes.")

    # ── Paso 2: Alinear al scaler y escalar ──────────────────────
    with st.spinner("📏 Escalando datos..."):
        try:
            X_scaled = prepare_for_model(df_cleaned.copy(), scaler)
        except Exception as e:
            st.error(f"❌ Error al escalar: {e}")
            st.stop()

    st.success(f"✅ Datos escalados — {X_scaled.shape[1]} features.")

    # ── Diagnóstico de columnas ───────────────────────────────────
    with st.expander("🔍 Diagnóstico: columnas del scaler vs columnas generadas", expanded=False):
        scaler_cols  = set(scaler.feature_names_in_)
        cleaned_cols = set(df_cleaned.columns)
        missing = sorted(scaler_cols - cleaned_cols)
        extra   = sorted(cleaned_cols - scaler_cols)
        col_a, col_b = st.columns(2)
        col_a.metric("Scaler espera", len(scaler_cols))
        col_b.metric("Limpieza generó", len(cleaned_cols))
        if missing:
            st.warning(f"⚠️ Faltan {len(missing)} columnas (se rellenan con 0): {missing}")
        if extra:
            st.info(f"ℹ️ Sobran {len(extra)} columnas (se descartan): {extra}")
        if not missing and not extra:
            st.success("✅ Columnas perfectamente alineadas.")
        with st.expander("Ver columnas del scaler (entrenamiento)"):
            st.code(", ".join(scaler.feature_names_in_))
        with st.expander("Ver columnas generadas por limpieza"):
            st.code(", ".join(df_cleaned.columns))

    # ── Paso 3: Predicción ───────────────────────────────────────
    with st.spinner("🧠 Generando predicciones..."):
        try:
            probs = model.predict(X_scaled, verbose=0).reshape(-1)
            probs = np.clip(probs, 0, 1)
        except Exception as e:
            st.error(f"❌ Error al predecir: {e}")
            st.stop()

    st.success("✅ Predicciones completadas.")

    # ── Construir resultados ─────────────────────────────────────
    ids = (
        codigos if (codigos and len(codigos) == len(df_cleaned))
        else [f"EST_{i+1:04d}" for i in range(len(df_cleaned))]
    )
    df_results = pd.DataFrame({
        "identificador":    ids,
        "p_aprueba":      np.round(probs, 6),
        "resultado_modelo": np.where(probs >= threshold, "Aprueba", "No aprueba"),
    })

    os.makedirs(os.path.join("archivos_procesados", "resultados"), exist_ok=True)
    df_results.to_csv(
        os.path.join("archivos_procesados", "resultados",
                     f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
        index=False,
    )

    # ─────────────────────────────────────────────────────────────
    #  DASHBOARD
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#fa709a 0%,#fee140 100%);
                padding:1rem;border-radius:10px;margin:2rem 0 1rem 0;">
        <h2 style="color:white;text-align:center;margin:0;">📊 Análisis de Resultados</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        counts = (
            df_results["resultado_modelo"]
            .value_counts()
            .reindex(["Aprueba", "No aprueba"])
            .fillna(0).astype(int)
        )
        total = len(df_results)

        m1, m2, m3 = st.columns(3)
        m1.metric("👥 Total", total)
        m2.metric("✅ Aprueban",
                  counts.get("Aprueba", 0),
                  f"{counts.get('Aprueba', 0) / total * 100:.1f}%")
        m3.metric("❌ No aprueban",
                  counts.get("No aprueba", 0),
                  f"{counts.get('No aprueba', 0) / total * 100:.1f}%")

        fig_bar = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Estado", "y": "Estudiantes"},
            color=counts.index,
            color_discrete_map={"Aprueba": "#2ecc71", "No aprueba": "#e74c3c"},
            title="Clasificación de Estudiantes",
        )
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="plot_bar")

        st.markdown("#### 📊 Distribución de P(aprueba)")
        st.caption("Estudiantes a la derecha del umbral → Aprueba. A la izquierda → No aprueba (riesgo de deserción).")
        fig_hist = px.histogram(
            df_results, x="p_aprueba", nbins=40,
            color_discrete_sequence=["#667eea"],
            labels={"p_aprueba": "P(aprueba)"},
        )
        fig_hist.add_vline(
            x=threshold, line_dash="dash", line_color="red",
            annotation_text=f"Umbral: {threshold:.2f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="plot_hist")

        st.markdown("#### 📋 Detalle por estudiante")
        st.caption(
            "**p_aprueba** = probabilidad de aprobar/graduarse (salida sigmoid). "
            "🟥 ≥ 0.7 riesgo alto · 🟨 0.5–0.7 riesgo medio · 🟩 < 0.5 riesgo bajo."
        )

        def color_resultado(val):
            return "color:#e74c3c;font-weight:bold" if val == "No aprueba" else "color:#27ae60;font-weight:bold"

        def color_prob(val):
            if   val >= 0.7: return "background-color:#e8f8e8"
            elif val >= 0.5: return "background-color:#fff3cd"
            return "background-color:#fde8e8"

        st.dataframe(
            df_results.style
                .applymap(color_resultado, subset=["resultado_modelo"])
                .applymap(color_prob,      subset=["p_aprueba"]),
            use_container_width=True,
            height=420,
        )

    with col2:
        st.markdown("#### 📉 Estadísticas de P(aprueba)")
        pmean = float(np.nanmean(probs))
        p25   = float(np.nanpercentile(probs, 25))
        p50   = float(np.nanpercentile(probs, 50))
        p75   = float(np.nanpercentile(probs, 75))
        pstd  = float(np.nanstd(probs))
        pmin  = float(np.nanmin(probs))
        pmax  = float(np.nanmax(probs))

        st.metric("📊 Promedio",       f"{pmean:.3f}", help="Riesgo promedio del grupo")
        st.metric("📉 Percentil 25",   f"{p25:.3f}",  help="25% más seguro está por debajo")
        st.metric("📊 Mediana",        f"{p50:.3f}",  help="Valor central del grupo")
        st.metric("📈 Percentil 75",   f"{p75:.3f}",  help="75% del grupo está por debajo")
        st.metric("📏 Desv. Estándar", f"{pstd:.3f}", help="Heterogeneidad del grupo")
        st.metric("⬇️ Mínimo",         f"{pmin:.3f}", help="Estudiante con menor riesgo")
        st.metric("⬆️ Máximo",         f"{pmax:.3f}", help="Estudiante con mayor riesgo")

        st.markdown("---")
        st.markdown("##### 💡 Diagnóstico del grupo")
        if p50 < 0.3:
            st.success("🟢 **Riesgo Bajo**  \nLa mayoría tiene bajo riesgo de deserción.")
        elif p50 < 0.7:
            st.warning("🟡 **Riesgo Moderado**  \nSe recomienda monitoreo preventivo.")
        else:
            st.error("🔴 **Riesgo Alto**  \nSe requiere intervención inmediata.")

    st.markdown("---")
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        st.download_button(
            "💾 Descargar Resultados (CSV)",
            df_results.to_csv(index=False).encode("utf-8"),
            file_name="resultados_prediccion.csv", mime="text/csv",
            key="btn_download", use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────
#  PANTALLA DE BIENVENIDA
# ─────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;
                background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                border-radius:15px;margin:2rem 0;">
        <h2 style="color:white;font-size:2rem;margin-bottom:1rem;">
            👋 ¡Bienvenido al Sistema de Predicción!
        </h2>
        <p style="color:#f0f0f0;font-size:1.2rem;margin:0;">
            Sube un archivo CSV con los datos de los estudiantes para comenzar
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    for c, icon, paso, desc in [
        (col1, "📥", "Paso 1", "Descarga la plantilla desde el menú lateral"),
        (col2, "✏️", "Paso 2", "Completa el CSV con los datos de tus estudiantes"),
        (col3, "🚀", "Paso 3", "Sube el archivo y obtén predicciones al instante"),
    ]:
        with c:
            st.markdown(f"""
            <div style="background-color:#f8f9fa;padding:2rem;border-radius:10px;
                        text-align:center;height:220px;">
                <div style="font-size:3rem;margin-bottom:1rem;">{icon}</div>
                <h4 style="color:#667eea;">{paso}</h4>
                <p style="color:#6c757d;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background-color:#e7f3ff;padding:1.5rem;border-radius:10px;
                border-left:5px solid #2196F3;">
        <h4 style="color:#2196F3;margin-top:0;">📋 Columnas requeridas en el CSV:</h4>
        <p style="color:#495057;line-height:2.2;">
            CODESTUDIANTE &nbsp;·&nbsp; ESTP_FECHAINGRESO &nbsp;·&nbsp; CREDITOSAPROBADOS &nbsp;·&nbsp;
            UBICACION_SEMESTRAL &nbsp;·&nbsp; PROMEDIO_GENERAL &nbsp;·&nbsp; PROGRAMA &nbsp;·&nbsp;
            JORNADA &nbsp;·&nbsp; GENERO &nbsp;·&nbsp; FECHA_NACIMIENTO &nbsp;·&nbsp;
            CIUDADRESIDENCIA &nbsp;·&nbsp; ESTRATO &nbsp;·&nbsp; TIENE_SISBEN &nbsp;·&nbsp;
            INFE_VIVECONFAMILIA &nbsp;·&nbsp; INFE_SITUACIONPADRES &nbsp;·&nbsp;
            INFE_NUMEROFAMILIARES &nbsp;·&nbsp; INFE_NUMEROHERMANOS &nbsp;·&nbsp;
            INFE_POSICIONENHERMANOS &nbsp;·&nbsp; INFE_NUMMIEMBROSTRABAJA
        </p>
        <div style="background-color:#fff3cd;padding:0.75rem;border-radius:5px;margin-top:1rem;">
            <strong>⚠️ No incluyas la columna SITUACION</strong> — es la variable objetivo.
        </div>
    </div>
    """, unsafe_allow_html=True)