import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
import io
from joblib import load as joblib_load
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, roc_curve,
)
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="🎓 Predicción de Deserción Estudiantil",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH  = "modelo_desercion_nn.keras"
SCALER_PATH = "scaler.joblib"

TEMPLATE_COLUMNS = [
    "CODESTUDIANTE", "ESTP_FECHAINGRESO", "CREDITOSAPROBADOS",
    "UBICACION_SEMESTRAL", "PROMEDIO_GENERAL", "PROGRAMA", "JORNADA",
    "GENERO", "FECHA_NACIMIENTO", "CIUDADRESIDENCIA", "ESTRATO",
    "TIENE_SISBEN", "INFE_VIVECONFAMILIA", "INFE_SITUACIONPADRES",
    "INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS",
    "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA",
    "CODIGOCIUDADR",
]

PROGRAMAS_VALIDOS = [
    "INGENIERIA DE SISTEMAS",
    "TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS",
]

NUMERIC_RANGES = {
    "CREDITOSAPROBADOS": (0, None),
    "UBICACION_SEMESTRAL": (1, None),
    "PROMEDIO_GENERAL": (0, 5),
    "ESTRATO": (1, 6),
    "INFE_NUMEROFAMILIARES": (0, None),
    "INFE_NUMEROHERMANOS": (0, None),
    "INFE_POSICIONENHERMANOS": (0, None),
    "INFE_NUMMIEMBROSTRABAJA": (0, None),
}

ALLOWED_SETS = {
    "TIENE_SISBEN": {0, 1},
}


def parse_datetime_series(series):
    text = series.astype(str).str.strip()
    text = text.str.replace("\u202f", " ", regex=False)
    text = text.str.replace("\xa0", " ", regex=False)
    text = text.str.replace(r"(?i)\ba\.?\s*m\.?\b", "AM", regex=True)
    text = text.str.replace(r"(?i)\bp\.?\s*m\.?\b", "PM", regex=True)
    text = text.str.replace(r"(?i)\ba\.\s*m\.?", "AM", regex=True)
    text = text.str.replace(r"(?i)\bp\.\s*m\.?", "PM", regex=True)
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(text.loc[fallback_mask], errors="coerce", dayfirst=False)
    return parsed


def parse_decimal_series(series):
    text = series.astype(str).str.strip()
    text = text.str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")

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

def make_template_df():
    return pd.DataFrame(columns=TEMPLATE_COLUMNS)

def make_example_df():
    if os.path.exists("test50.csv"):
        return pd.read_csv("test50.csv")
    return make_template_df()

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        content = uploaded_file.getvalue()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        sample = text[:2048]
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
        df.columns = df.columns.astype(str).str.strip()
        return df, delimiter
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str).str.strip()
    return df, None


def validate_schema(df_raw):
    errors = []
    cols = list(df_raw.columns)
    missing = [c for c in TEMPLATE_COLUMNS if c not in cols]
    for c in missing:
        errors.append({
            "Fila": 1,
            "Columna": c,
            "Valor": "",
            "Detalle": "Columna requerida faltante.",
        })
    return errors


def collect_value_errors(df_raw):
    errors = []

    def add_error(idx, col, val, detalle):
        errors.append({
            "Fila": int(idx) + 2,
            "Columna": col,
            "Valor": "" if pd.isna(val) else str(val),
            "Detalle": detalle,
        })

    if "CODESTUDIANTE" in df_raw.columns:
        mask = df_raw["CODESTUDIANTE"].isna() | (df_raw["CODESTUDIANTE"].astype(str).str.strip() == "")
        for idx in df_raw[mask].index:
            add_error(idx, "CODESTUDIANTE", df_raw.at[idx, "CODESTUDIANTE"], "Valor requerido.")

    for col in ["ESTP_FECHAINGRESO", "FECHA_NACIMIENTO"]:
        if col in df_raw.columns:
            parsed = parse_datetime_series(df_raw[col])
            mask = parsed.isna()
            for idx in df_raw[mask].index:
                add_error(idx, col, df_raw.at[idx, col], "Fecha invalida o vacia.")

    if "PROGRAMA" in df_raw.columns:
        valid_set = {p.upper() for p in PROGRAMAS_VALIDOS}
        normalized = df_raw["PROGRAMA"].astype(str).str.strip().str.upper()
        mask = ~normalized.isin(valid_set)
        for idx in df_raw[mask].index:
            add_error(idx, "PROGRAMA", df_raw.at[idx, "PROGRAMA"], "Programa no valido.")

    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df_raw.columns:
            numeric = parse_decimal_series(df_raw[col]) if col == "PROMEDIO_GENERAL" else pd.to_numeric(df_raw[col], errors="coerce")
            for idx in df_raw[numeric.isna()].index:
                add_error(idx, col, df_raw.at[idx, col], "Debe ser numerico.")
            if min_val is not None:
                mask = numeric < min_val
                for idx in df_raw[mask.fillna(False)].index:
                    add_error(idx, col, df_raw.at[idx, col], f"Debe ser >= {min_val}.")
            if max_val is not None:
                mask = numeric > max_val
                for idx in df_raw[mask.fillna(False)].index:
                    add_error(idx, col, df_raw.at[idx, col], f"Debe ser <= {max_val}.")

    for col, allowed in ALLOWED_SETS.items():
        if col in df_raw.columns:
            numeric = pd.to_numeric(df_raw[col], errors="coerce")
            for idx in df_raw[numeric.isna()].index:
                add_error(idx, col, df_raw.at[idx, col], f"Debe ser numerico ({sorted(allowed)}).")
            mask = ~numeric.isin(allowed)
            for idx in df_raw[mask.fillna(False)].index:
                add_error(idx, col, df_raw.at[idx, col], f"Valor permitido: {sorted(allowed)}.")

    return errors


def invalid_row_indices_from_errors(errors):
    return sorted({int(error["Fila"]) - 2 for error in errors if int(error.get("Fila", 0)) >= 2})


def clean_data(df_raw, keep_situacion=False):
    df = df_raw.copy()
    codigos = df["CODESTUDIANTE"].astype(str).tolist() if "CODESTUDIANTE" in df.columns else None

    situacion = None
    if keep_situacion and "SITUACION" in df.columns:
        situacion = df["SITUACION"].astype(int).values

    drop_cols = [
        "CODESTUDIANTE", "CODIGOCIUDADR", "NIVEL_SISBEN", "CATEGORIA",
        "CODMATRICULA", "SEDE", "INFE_HERMANOSESTUDIANDOU", "SITUACION",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "ESTP_FECHAINGRESO" in df.columns:
        df["ESTP_FECHAINGRESO"] = parse_datetime_series(df["ESTP_FECHAINGRESO"])
    if "FECHA_NACIMIENTO" in df.columns:
        df["FECHA_NACIMIENTO"] = parse_datetime_series(df["FECHA_NACIMIENTO"])
    if "ESTP_FECHAINGRESO" in df.columns and "FECHA_NACIMIENTO" in df.columns:
        df["EDAD_INGRESO"] = ((df["ESTP_FECHAINGRESO"] - df["FECHA_NACIMIENTO"]).dt.days / 365.25).round().astype("Int64")
        df["ANIO_INGRESO"] = df["ESTP_FECHAINGRESO"].dt.year
        df["MES_INGRESO"]  = df["ESTP_FECHAINGRESO"].dt.month
        df = df.drop(columns=["ESTP_FECHAINGRESO", "FECHA_NACIMIENTO"])

    if "PROMEDIO_GENERAL" in df.columns:
        df["PROMEDIO_GENERAL"] = parse_decimal_series(df["PROMEDIO_GENERAL"])

    df["ESTRATO"] = pd.to_numeric(df["ESTRATO"], errors="coerce")
    df.loc[(df["ESTRATO"] < 1) | (df["ESTRATO"] > 6), "ESTRATO"] = pd.NA

    if "PROGRAMA" in df.columns:
        mask = df["PROGRAMA"].isin(PROGRAMAS_VALIDOS)
        if situacion is not None:
            situacion = situacion[mask.values]
        if codigos is not None:
            codigos = [c for c, m in zip(codigos, mask) if m]
        df = df[mask]

    if "CIUDADRESIDENCIA" in df.columns:
        mapa_ciudad = {"BUCARAMANGA": 1, "FLORIDABLANCA": 2, "GIRON": 3, "PIEDECUESTA": 4}
        df["CIUDADRESIDENCIA"] = (
            df["CIUDADRESIDENCIA"].astype(str).str.strip().str.upper()
            .map(mapa_ciudad).fillna(5).astype(int)
        )

    cat_cols = [c for c in df.select_dtypes(include="object").columns if df[c].nunique(dropna=True) > 1]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)

    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)

    if "ESTRATO" in df.columns:
        df["ESTRATO"] = df["ESTRATO"].fillna(0).astype(int)
    for c in ["INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS",
              "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA", "EDAD_INGRESO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).round().astype(int)
    if "TIENE_SISBEN" in df.columns:
        df["TIENE_SISBEN"] = pd.to_numeric(df["TIENE_SISBEN"], errors="coerce").fillna(-1).round().astype(int)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)

    return df, codigos, situacion


def prepare_for_model(df_cleaned, scaler):
    expected_cols = list(scaler.feature_names_in_)
    for col in expected_cols:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0
    X = df_cleaned[expected_cols]
    return scaler.transform(X)


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

if scaler is None:
    st.error("""❌ **`scaler.joblib` no encontrado.** Agrega al final del script de entrenamiento:
```python
from joblib import dump
dump(scaler, 'scaler.joblib')
```
Luego copia `scaler.joblib` junto a `app.py`.""")

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
            "⬇️ Ejemplo (50 estudiantes)",
            make_example_df().to_csv(index=False).encode("utf-8"),
            file_name="ejemplo_50_estudiantes.csv", mime="text/csv",
            key="btn_ejemplo50", use_container_width=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);
                padding:1rem;border-radius:10px;margin-bottom:1rem;">
        <h3 style="color:white;text-align:center;margin:0;">⚙️ Configuración</h3>
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Umbral P(deserta)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        key="slider_threshold",
        help="P(deserta) ≥ umbral → Deserta (riesgo) | P(deserta) < umbral → No deserta",
    )

    st.markdown(f"""
    <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:8px;
                padding:0.6rem;text-align:center;margin-top:0.5rem;">
        <strong>Umbral activo: {threshold:.2f}</strong><br>
        <small>P ≥ {threshold:.2f} → 🚨 Deserta<br>P &lt; {threshold:.2f} → ✅ No deserta</small>
    </div>
    """, unsafe_allow_html=True)

tab_pred, tab_stats = st.tabs(["🎯 Predicción", "📈 Estadísticas del Modelo"])

with tab_pred:

    st.markdown("""
    <div style="background-color:#f8f9fa;padding:1.2rem;border-radius:10px;
                border-left:5px solid #e74c3c;margin-bottom:1.5rem;">
        <p style="color:#495057;margin:0;line-height:1.8;">
            El modelo devuelve <strong>P(deserta)</strong>: probabilidad de que el estudiante
            <em>deserte / no complete</em> sus estudios (<code>SITUACION=1</code> en entrenamiento).<br>
            <code>P ≥ umbral</code> →
            <span style="color:#e74c3c;font-weight:bold;">Deserta (en riesgo)</span> &nbsp;|&nbsp;
            <code>P &lt; umbral</code> →
            <span style="color:#27ae60;font-weight:bold;">No deserta</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Subir CSV/Excel con datos de estudiantes (sin columna SITUACION)",
        type=["csv", "xlsx"], key="file_uploader_pred",
    )

    if uploaded_file is not None:
        if model is None or scaler is None:
            st.error("❌ Falta el modelo o el scaler.")
            st.stop()

        try:
            df_raw, delimiter = read_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"❌ No se pudo leer el archivo: {e}")
            st.stop()

        schema_errors = validate_schema(df_raw)
        if schema_errors:
            st.error("❌ El archivo no cumple con la estructura requerida.")
            st.dataframe(pd.DataFrame(schema_errors), use_container_width=True, hide_index=True)
            st.stop()

        value_errors = collect_value_errors(df_raw)
        invalid_indices = invalid_row_indices_from_errors(value_errors)
        valid_mask = ~df_raw.index.isin(invalid_indices)
        df_valid_raw = df_raw.loc[valid_mask].copy()

        if value_errors:
            st.warning(f"⚠️ Se detectaron {len(invalid_indices)} registros con errores. Solo se procesarán los válidos.")
            st.dataframe(pd.DataFrame(value_errors), use_container_width=True, hide_index=True)

        with st.sidebar:
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Registros", len(df_raw))
            c2.metric("Válidos", len(df_valid_raw))
            if delimiter:
                st.caption(f"Delimitador CSV detectado: `{delimiter}`")

        if len(df_valid_raw) == 0:
            st.error("❌ No quedaron registros válidos para procesar.")
            st.stop()

        with st.spinner("🧹 Limpiando datos..."):
            df_cleaned, codigos, _ = clean_data(df_valid_raw, keep_situacion=False)

        if len(df_cleaned) == 0:
            st.error("❌ No quedaron registros. Verifica los programas en el archivo.")
            st.stop()
        st.success(f"✅ Limpieza completada — {len(df_cleaned)} estudiantes.")

        with st.spinner("📏 Escalando..."):
            try:
                X_scaled = prepare_for_model(df_cleaned.copy(), scaler)
            except Exception as e:
                st.error(f"❌ Error al escalar: {e}")
                st.stop()
        st.success(f"✅ Datos escalados — {X_scaled.shape[1]} features.")

        with st.spinner("🧠 Prediciendo..."):
            try:
                probs = model.predict(X_scaled, verbose=0).reshape(-1)
                probs = np.clip(probs, 0, 1)
            except Exception as e:
                st.error(f"❌ Error al predecir: {e}")
                st.stop()
        st.success("✅ Predicciones completadas.")

        ids = codigos if (codigos and len(codigos) == len(df_cleaned)) else [f"EST_{i+1:04d}" for i in range(len(df_cleaned))]

        st.session_state["probs"] = probs
        st.session_state["ids"]   = ids
        st.session_state["value_errors"] = value_errors
        st.session_state["invalid_count"] = len(invalid_indices)
        st.session_state["valid_count"] = len(df_cleaned)
        st.session_state["total_count"] = len(df_raw)

    if "probs" in st.session_state:

        probs     = st.session_state["probs"]
        ids       = st.session_state["ids"]
        invalid_count = st.session_state.get("invalid_count", 0)
        total_count   = st.session_state.get("total_count", len(ids))
        valid_count   = st.session_state.get("valid_count", len(ids))
        value_errors  = st.session_state.get("value_errors", [])

        if invalid_count > 0:
            st.info(f"Se procesaron {valid_count} registros válidos de {total_count}. {invalid_count} quedaron fuera por errores de formato o contenido.")
            st.dataframe(pd.DataFrame(value_errors), use_container_width=True, hide_index=True)

        df_results = pd.DataFrame({
            "identificador":    ids,
            "p_desercion":      np.round(probs, 4),
            "resultado_modelo": np.where(probs >= threshold, "Deserta", "No deserta"),
        })

        os.makedirs(os.path.join("archivos_procesados", "resultados"), exist_ok=True)
        df_results.to_csv(
            os.path.join("archivos_procesados", "resultados",
                         f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            index=False,
        )

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#fa709a 0%,#fee140 100%);
                    padding:1rem;border-radius:10px;margin:1.5rem 0 1rem 0;">
            <h2 style="color:white;text-align:center;margin:0;">
                📊 Resultados &nbsp;·&nbsp; Umbral activo: {threshold:.2f}
            </h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            counts = df_results["resultado_modelo"].value_counts().reindex(["No deserta", "Deserta"]).fillna(0).astype(int)
            total  = len(df_results)
            m1, m2, m3 = st.columns(3)
            m1.metric("👥 Total", total)
            m2.metric("✅ No desertan", counts.get("No deserta", 0), f"{counts.get('No deserta', 0)/total*100:.1f}%")
            m3.metric("🚨 Desertan",    counts.get("Deserta", 0),    f"{counts.get('Deserta', 0)/total*100:.1f}%")

            fig_bar = px.bar(
                x=counts.index, y=counts.values,
                color=counts.index,
                color_discrete_map={"No deserta": "#2ecc71", "Deserta": "#e74c3c"},
                labels={"x": "Estado", "y": "Estudiantes"},
                title=f"Clasificación de Estudiantes (umbral={threshold:.2f})",
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True, key="pred_bar")

            st.markdown("#### 📊 Distribución de P(deserta)")
            st.caption("Derecha del umbral → Deserta (en riesgo). Izquierda → No deserta.")
            fig_hist = px.histogram(df_results, x="p_desercion", nbins=40,
                                    color_discrete_sequence=["#667eea"], labels={"p_desercion": "P(deserta)"})
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                               annotation_text=f"Umbral {threshold:.2f}", annotation_position="top right")
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True, key="pred_hist")

            st.markdown("#### 📋 Detalle por estudiante")
            st.caption(f"p_desercion con umbral={threshold:.2f}: 🚨 ≥ {threshold:.2f} Deserta · ✅ < {threshold:.2f} No deserta")

            def color_res(val):
                return "color:#27ae60;font-weight:bold" if val == "No deserta" else "color:#e74c3c;font-weight:bold"

            def color_p(val):
                if val >= threshold:
                    diff = val - threshold
                    if diff >= 0.15:
                        return "background-color:#fde8e8"
                    else:
                        return "background-color:#fff3cd"
                return "background-color:#e8f8e8"

            st.dataframe(
                df_results.style
                    .map(color_res, subset=["resultado_modelo"])
                    .map(color_p,   subset=["p_desercion"]),
                use_container_width=True, height=420,
            )

        with col2:
            st.markdown("#### 📉 Estadísticas")
            pmean = float(np.nanmean(probs));  p25 = float(np.nanpercentile(probs, 25))
            p50   = float(np.nanpercentile(probs, 50)); p75 = float(np.nanpercentile(probs, 75))
            pstd  = float(np.nanstd(probs));   pmin = float(np.nanmin(probs)); pmax = float(np.nanmax(probs))
            st.metric("Promedio P(deserta)", f"{pmean:.3f}")
            st.metric("Percentil 25",  f"{p25:.3f}")
            st.metric("Mediana",       f"{p50:.3f}")
            st.metric("Percentil 75",  f"{p75:.3f}")
            st.metric("Desv. Estándar",f"{pstd:.3f}")
            st.metric("Mínimo",        f"{pmin:.3f}")
            st.metric("Máximo",        f"{pmax:.3f}")
            st.markdown("---")

            st.markdown(f"**Umbral activo:** `{threshold:.2f}`")
            pct_deserta = counts.get("Deserta", 0) / total * 100
            if pct_deserta >= 60:
                st.error(f"🔴 **Alto riesgo grupal** ({pct_deserta:.1f}% desertan)")
            elif pct_deserta >= 30:
                st.warning(f"🟡 **Riesgo moderado** ({pct_deserta:.1f}% desertan)")
            else:
                st.success(f"🟢 **Grupo de bajo riesgo** ({pct_deserta:.1f}% desertan)")

        st.markdown("---")
        _, cb, _ = st.columns([1, 2, 1])
        with cb:
            st.download_button("💾 Descargar Resultados (CSV)",
                               df_results.to_csv(index=False).encode("utf-8"),
                               file_name=f"resultados_umbral{threshold:.2f}.csv", mime="text/csv",
                               key="btn_dl_pred", use_container_width=True)

    elif uploaded_file is None:
        st.markdown("""
        <div style="text-align:center;padding:3rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    border-radius:15px;margin:2rem 0;">
            <h2 style="color:white;margin:0;">👋 Sube un archivo CSV para comenzar</h2>
        </div>
        """, unsafe_allow_html=True)

with tab_stats:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding:1rem;border-radius:10px;margin-bottom:1.5rem;">
        <h3 style="color:white;text-align:center;margin:0;">🏗️ Arquitectura del Modelo</h3>
    </div>
    """, unsafe_allow_html=True)

    if model is not None:
        c1, c2, c3, c4 = st.columns(4)
        total_params    = model.count_params()
        trainable       = sum(np.prod(w.shape) for w in model.trainable_weights)
        non_trainable   = total_params - trainable
        n_layers        = len(model.layers)
        c1.metric("🧱 Capas",             n_layers)
        c2.metric("🔢 Parámetros totales", f"{total_params:,}")
        c3.metric("✏️ Entrenables",        f"{trainable:,}")
        c4.metric("🔒 No entrenables",     f"{non_trainable:,}")

        st.markdown("##### Detalle de capas")
        layers_data = []
        for layer in model.layers:
            cfg   = layer.get_config()
            ltype = layer.__class__.__name__
            params = layer.count_params()
            shape  = str(layer.output_shape) if hasattr(layer, "output_shape") else "—"
            extra  = ""
            if ltype == "Dense":
                extra = f"activación: {cfg.get('activation','?')} | neuronas: {cfg.get('units','?')}"
            elif ltype == "Dropout":
                extra = f"rate: {cfg.get('rate','?')}"
            layers_data.append({"Capa": layer.name, "Tipo": ltype, "Parámetros": params,
                                 "Output shape": shape, "Detalle": extra})
        st.dataframe(pd.DataFrame(layers_data), use_container_width=True, hide_index=True)

        if scaler is not None:
            st.markdown("##### Scaler (StandardScaler)")
            sc1, sc2 = st.columns(2)
            sc1.metric("Features de entrada", len(scaler.feature_names_in_))
            sc2.metric("Archivo", SCALER_PATH)
            with st.expander("Ver nombres de features"):
                st.code(", ".join(scaler.feature_names_in_))
    else:
        st.error("❌ Modelo no cargado.")

    st.markdown("---")
