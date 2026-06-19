import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from src.infrastructure.repositories.student_repository_impl import StudentRepositoryImpl
from src.infrastructure.repositories.model_repository_impl import ModelRepositoryImpl
from src.application.predict_use_case import PredictUseCase
from src.application.train_use_case import TrainUseCase
from src.application.monitor_use_case import MonitorUseCase
from src.domain.entities.student import Student
from src.infrastructure.reports.pdf_generator import PDFGenerator

# Set up page config
st.set_page_config(
    page_title="Alerta Temprana - Predicción de Deserción",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------
# THEME SYSTEM (light/dark palettes with guaranteed text/background contrast)
# -----------------
THEMES = {
    "Oscuro": {
        "app_bg": "#10121c",
        "sidebar_bg": "#161827",
        "text_primary": "#f2f3f7",
        "text_secondary": "#a7acc2",
        "border": "rgba(255,255,255,0.12)",
        "card_bg": "rgba(255,255,255,0.05)",
        "input_bg": "#1c1f30",
        "button_bg": "#1f2235",
        "button_border": "rgba(255,255,255,0.16)",
        "header_grad": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
        "accent": "#7c93ff",
        "success_text": "#34d399",
        "success_bg": "rgba(52, 211, 153, 0.14)",
        "success_border": "#2ecc71",
        "danger_text": "#f87171",
        "danger_bg": "rgba(248, 113, 113, 0.14)",
        "danger_border": "#e74c3c",
        "warning_text": "#fbbf24",
        "warning_bg": "rgba(251, 191, 36, 0.14)",
        "warning_border": "#f1c40f",
    },
    "Claro": {
        "app_bg": "#f7f8fc",
        "sidebar_bg": "#ffffff",
        "text_primary": "#1a1a2e",
        "text_secondary": "#555b6e",
        "border": "rgba(20,20,40,0.12)",
        "card_bg": "#ffffff",
        "input_bg": "#ffffff",
        "button_bg": "#eef0f6",
        "button_border": "rgba(20,20,40,0.16)",
        "header_grad": "linear-gradient(135deg, #2b2f6b 0%, #1a1a2e 100%)",
        "accent": "#3b5bdb",
        "success_text": "#157347",
        "success_bg": "rgba(46, 204, 113, 0.14)",
        "success_border": "#2ecc71",
        "danger_text": "#b3261e",
        "danger_bg": "rgba(231, 76, 60, 0.12)",
        "danger_border": "#e74c3c",
        "warning_text": "#8a5d00",
        "warning_bg": "rgba(241, 196, 15, 0.18)",
        "warning_border": "#f1c40f",
    },
}

if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "Oscuro"
THEME_MODE = st.session_state["ui_theme"]
PAL = THEMES[THEME_MODE]

# -----------------
# SVG ICON SYSTEM (replaces emoji glyphs in custom UI elements)
# -----------------
ICONS = {
    "graduation_cap": '<path d="M12 3 1 9l11 6 9-4.9V17h2V9z"/><path d="M5 11.5V16c0 1.66 3.13 3 7 3s7-1.34 7-3v-4.5"/>',
    "settings": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "check_circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert_triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "folder": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "bar_chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "sun": '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>',
    "trending_up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "clipboard_check": '<polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
    "dollar_sign": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
}


def svg_icon(name, size=18, color="currentColor", stroke_width=1.8):
    body = ICONS[name]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-3px;flex-shrink:0;">{body}</svg>'
    )


def section_header(icon_name, text, level=3, color=None):
    tag = f"h{level}"
    size = {2: 26, 3: 20, 4: 17}.get(level, 17)
    col = color or PAL["text_primary"]
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:0.55rem;color:{col};margin:0.2rem 0 0.8rem 0;">'
        f'{svg_icon(icon_name, size=size, color=col)}<span>{text}</span></{tag}>',
        unsafe_allow_html=True
    )


def rec_item(icon_name, color, label, detail):
    st.markdown(
        f'<div style="display:flex;gap:0.5rem;align-items:flex-start;margin:0.35rem 0;">'
        f'<span style="margin-top:0.15rem;flex-shrink:0;">{svg_icon(icon_name, size=16, color=color)}</span>'
        f'<span><strong style="color:{color};">{label}</strong>: {detail}</span></div>',
        unsafe_allow_html=True
    )


def status_box(icon_name, title, lines, kind="success"):
    text_c = PAL[f"{kind}_text"]
    bg_c = PAL[f"{kind}_bg"]
    border_c = PAL[f"{kind}_border"]
    body_lines = "".join(
        f'<span style="font-size:0.85rem;color:{PAL["text_secondary"]};">{line}</span><br>' for line in lines
    )
    st.markdown(
        f"""
        <div style="background-color:{bg_c};padding:10px 12px;border-radius:8px;border:1px solid {border_c};margin-bottom:1rem;">
            <strong style="color:{text_c};display:flex;align-items:center;gap:0.4rem;">
                {svg_icon(icon_name, size=16, color=text_c)}<span>{title}</span>
            </strong>
            {body_lines}
        </div>
        """,
        unsafe_allow_html=True
    )


# Custom Styling (theme-aware: backgrounds, text and accents derive from PAL so contrast
# stays correct in both Claro and Oscuro modes)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}

    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {{
        background-color: {PAL["app_bg"]} !important;
        color: {PAL["text_primary"]} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {PAL["sidebar_bg"]} !important;
        color: {PAL["text_primary"]} !important;
        border-right: 1px solid {PAL["border"]};
    }}

    /* Scoped to markdown-rendered text only, so it never bleeds into
       widgets (buttons, inputs) that manage their own bg/text pairing. */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
        color: {PAL["text_primary"]};
    }}

    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: {PAL["text_primary"]} !important;
    }}

    [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stFileUploader"] {{
        border: 1px solid {PAL["border"]} !important;
        border-radius: 10px;
        background-color: {PAL["card_bg"]};
    }}

    [data-baseweb="tab-list"] {{
        border-bottom: 1px solid {PAL["border"]} !important;
    }}

    [data-baseweb="tab"] {{
        color: {PAL["text_secondary"]} !important;
    }}

    [data-baseweb="tab"][aria-selected="true"] {{
        color: {PAL["accent"]} !important;
    }}

    /* Buttons (download/form-submit/regular) keep their own readable bg+text pair
       regardless of theme, instead of inheriting Streamlit's unconfigured default. */
    [data-testid^="stBaseButton-"] {{
        background-color: {PAL["button_bg"]} !important;
        border: 1px solid {PAL["button_border"]} !important;
    }}
    [data-testid^="stBaseButton-"], [data-testid^="stBaseButton-"] * {{
        color: {PAL["text_primary"]} !important;
    }}
    [data-testid^="stBaseButton-"]:hover {{
        border-color: {PAL["accent"]} !important;
    }}
    [data-testid^="stBaseButton-"]:hover, [data-testid^="stBaseButton-"]:hover * {{
        color: {PAL["accent"]} !important;
    }}

    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div,
    [data-baseweb="input"] {{
        background-color: {PAL["input_bg"]} !important;
        color: {PAL["text_primary"]} !important;
        border-color: {PAL["border"]} !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background-color: {PAL["card_bg"]} !important;
    }}

    .main-header {{
        background: {PAL["header_grad"]};
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        color: white;
    }}

    .main-header * {{
        color: white !important;
    }}

    .card {{
        background: {PAL["card_bg"]};
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid {PAL["border"]};
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .card:hover {{
        transform: translateY(-2px);
        border-color: {PAL["accent"]};
    }}

    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {PAL["text_primary"]};
    }}

    .metric-label {{
        font-size: 0.9rem;
        color: {PAL["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .alert-banner {{
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid;
        color: {PAL["text_primary"]};
    }}

    .alert-banner strong {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    .alert-danger {{
        background-color: {PAL["danger_bg"]};
        border-left-color: {PAL["danger_border"]};
    }}
    .alert-danger strong {{ color: {PAL["danger_text"]}; }}

    .alert-warning {{
        background-color: {PAL["warning_bg"]};
        border-left-color: {PAL["warning_border"]};
    }}
    .alert-warning strong {{ color: {PAL["warning_text"]}; }}

    .alert-success {{
        background-color: {PAL["success_bg"]};
        border-left-color: {PAL["success_border"]};
    }}
    .alert-success strong {{ color: {PAL["success_text"]}; }}
</style>
""", unsafe_allow_html=True)

# Instantiate application repositories and services
@st.cache_resource
def get_services():
    student_repo = StudentRepositoryImpl()
    model_repo = ModelRepositoryImpl()
    predict_use_case = PredictUseCase(student_repo, model_repo)
    train_use_case = TrainUseCase(student_repo, model_repo)
    monitor_use_case = MonitorUseCase(model_repo)
    return student_repo, model_repo, predict_use_case, train_use_case, monitor_use_case

student_repo, model_repo, predict_use_case, train_use_case, monitor_use_case = get_services()

# Constants
TEMPLATE_COLUMNS = [
    "CODESTUDIANTE", "ESTP_FECHAINGRESO", "CREDITOSAPROBADOS",
    "UBICACION_SEMESTRAL", "PROMEDIO_GENERAL", "PROGRAMA", "JORNADA",
    "GENERO", "FECHA_NACIMIENTO", "CIUDADRESIDENCIA", "ESTRATO",
    "TIENE_SISBEN", "INFE_VIVECONFAMILIA", "INFE_SITUACIONPADRES",
    "INFE_NUMEROFAMILIARES", "INFE_NUMEROHERMANOS",
    "INFE_POSICIONENHERMANOS", "INFE_NUMMIEMBROSTRABAJA",
]

PROGRAMAS_VALIDOS = [
    "INGENIERIA DE SISTEMAS",
    "TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS",
]

# Validation rules
NUMERIC_RANGES = {
    "CREDITOSAPROBADOS": (0, 160),
    "UBICACION_SEMESTRAL": (1, 14),
    "PROMEDIO_GENERAL": (0.0, 5.0),
    "ESTRATO": (1, 6),
    "TIENE_SISBEN": (0, 1),
    "INFE_VIVECONFAMILIA": (0, 1),
}

# Header Section
st.markdown(f"""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700; letter-spacing: -1px; display:flex; align-items:center; justify-content:center; gap:0.6rem;">
        {svg_icon("graduation_cap", size=38, color="white", stroke_width=1.6)}<span>Sistema de Alerta Temprana Estudiantil</span>
    </h1>
    <p style="margin: 0.5rem 0 0 0; color: #a0a5b5; font-size: 1.2rem; font-weight: 300;">
        Plataforma Predictiva de Permanencia y Deserción Académica (Rama Beta)
    </p>
</div>
""", unsafe_allow_html=True)

# Active Model Status check
active_model_exists = True
active_version = None
model_name = None
model_metrics = {}

try:
    history = model_repo.get_model_history()
    with open(model_repo.registry_file, "r") as f:
        registry_data = json.load(f)
    active_version = registry_data.get("active_version")
        
    if not active_version and history:
        active_version = history[-1]["version"]
        
    if active_version:
        # Load active metadata
        meta = next((item for item in history if item["version"] == active_version), None)
        if meta:
            model_name = meta["model_name"]
            model_metrics = meta["metrics"]
            active_model_exists = True
        else:
            active_model_exists = False
    else:
        active_model_exists = False
except Exception:
    active_model_exists = False

# Sidebar Config
with st.sidebar:
    section_header("settings", "Panel de Configuración", level=3)

    selected_theme = st.segmented_control(
        "Tema de la interfaz",
        options=["Oscuro", "Claro"],
        key="ui_theme",
        label_visibility="collapsed"
    )
    if selected_theme is None:
        # segmented_control can be deselected to None; keep the last valid theme
        st.session_state["ui_theme"] = THEME_MODE

    if active_model_exists:
        status_box(
            "check_circle", "Modelo Activo Cargado",
            [f"Algoritmo: {model_name}", f"Versión: {active_version}"],
            kind="success"
        )
    else:
        status_box(
            "alert_triangle", "Sin Modelo Activo",
            ["Entrena un modelo en la pestaña de Administración."],
            kind="danger"
        )

    threshold = st.slider(
        "Umbral P(deserta)",
        min_value=0.0, max_value=1.0, value=0.70, step=0.01,
        help="Probabilidades por encima de este valor marcan al estudiante como En Riesgo."
    )

    st.markdown("---")
    section_header("download", "Descarga de Plantillas", level=3)

    # Download buttons
    template_df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    st.download_button(
        "Plantilla Vacía (CSV)",
        template_df.to_csv(index=False).encode("utf-8"),
        "plantilla_estudiantes.csv",
        "text/csv",
        icon=":material/download:",
        use_container_width=True
    )

    if os.path.exists("dataset/sample_50_students.csv"):
        with open("dataset/sample_50_students.csv", "rb") as f:
            st.download_button(
                "Ejemplo de Entrada (50 Estudiantes)",
                f.read(),
                "ejemplo_50_estudiantes.csv",
                "text/csv",
                icon=":material/download:",
                use_container_width=True
            )

# UI Tabs Configuration
tab_dash, tab_batch, tab_indiv, tab_admin = st.tabs([
    ":material/monitoring: Métricas & Dashboard",
    ":material/folder_open: Predicción Masiva",
    ":material/person: Predicción Individual",
    ":material/settings: Administración & Control"
])

# -----------------
# TAB: DASHBOARD
# -----------------
with tab_dash:
    if not active_model_exists:
        st.warning("No se ha encontrado un modelo entrenado activo en el registro. Dirígete a la pestaña 'Administración & Control' para reentrenar el modelo champion.", icon=":material/warning:")
    else:
        section_header("trending_up", "Rendimiento Histórico del Modelo Activo")
        
        # Grid metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="card"><span class="metric-label">Accuracy</span><div class="metric-value">{model_metrics.get("accuracy", 0.0):.4f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card"><span class="metric-label">F1-Score</span><div class="metric-value">{model_metrics.get("f1", 0.0):.4f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card"><span class="metric-label">Precision</span><div class="metric-value">{model_metrics.get("precision", 0.0):.4f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="card"><span class="metric-label">Recall</span><div class="metric-value">{model_metrics.get("recall", 0.0):.4f}</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="card"><span class="metric-label">ROC-AUC</span><div class="metric-value">{model_metrics.get("roc_auc", 0.0):.4f}</div></div>', unsafe_allow_html=True)
            
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            # Render confusion matrix
            matrix = model_metrics.get("confusion_matrix", {})
            if matrix:
                tn, fp, fn, tp = matrix.get("tn", 0), matrix.get("fp", 0), matrix.get("fn", 0), matrix.get("tp", 0)
                cm_data = [[tn, fp], [fn, tp]]
                fig_cm = px.imshow(
                    cm_data,
                    text_auto=True,
                    aspect="auto",
                    labels=dict(x="Predicción", y="Realidad"),
                    x=["No Deserta", "Deserta"],
                    y=["No Deserta", "Deserta"],
                    color_continuous_scale="Blues",
                    title="Matriz de Confusión (Hold-out Test Set)"
                )
                fig_cm.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)
                
        with col_chart2:
            # Feature Importance (loads if exists in registry/npy)
            if os.path.exists("shap_feature_names.npy") and os.path.exists("shap_background.npy"):
                feature_names = np.load("shap_feature_names.npy", allow_pickle=True)
                # Compute absolute mean importance of baseline
                bg = np.load("shap_background.npy")
                # Approximate feature importances from baseline scaling variations or load model coefficients
                # Here we fetch from the model's metadata or background
                try:
                    # Let's show feature importance of training champion
                    # We can load model coefficients or importances
                    model, prep_wrapper, _ = model_repo.load_latest_model()
                    importances = model.get_feature_importances(feature_names.tolist())
                    
                    df_imp = pd.DataFrame(list(importances.items()), columns=["Variable", "Importancia"]).sort_values("Importancia", ascending=True).tail(10)
                    fig_imp = px.bar(
                        df_imp,
                        x="Importancia",
                        y="Variable",
                        orientation="h",
                        color="Importancia",
                        color_continuous_scale="Viridis",
                        title="Importancia de Variables (Top 10)"
                    )
                    fig_imp.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as ex:
                    st.info(f"No se pudo mostrar la importancia de variables: {ex}")

# -----------------
# TAB: BATCH PREDICTION
# -----------------
with tab_batch:
    section_header("folder", "Carga Masiva de Registros de Estudiantes")
    uploaded_file = st.file_uploader("Sube tu archivo (Excel o CSV)", type=["xlsx", "csv"])

    if uploaded_file is not None:
        if not active_model_exists:
            st.error("No hay un modelo activo entrenado. Ve a Control para entrenarlo primero.", icon=":material/error:")
        else:
            filename = uploaded_file.name
            
            with st.spinner("Leyendo archivo..."):
                try:
                    if filename.endswith(".xlsx"):
                        df_input = pd.read_excel(uploaded_file)
                    else:
                        content = uploaded_file.getvalue()
                        try:
                            text = content.decode("utf-8-sig")
                        except UnicodeDecodeError:
                            text = content.decode("latin-1")
                        
                        import csv
                        sample = text[:2048]
                        delimiter = ","
                        try:
                            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                            delimiter = dialect.delimiter
                        except csv.Error:
                            delimiter = ","
                        df_input = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")
                    df_input.columns = df_input.columns.astype(str).str.strip()
                except Exception as e:
                    st.error(f"Error al leer el archivo: {e}")
                    st.stop()
            
            # Validation step
            section_header("search", "Validaciones del Dataset", level=4)

            # 1. Schema check
            missing_cols = [c for c in TEMPLATE_COLUMNS if c not in df_input.columns]

            if missing_cols:
                st.error(f"Estructura de columnas inválida. Columnas faltantes: {', '.join(missing_cols)}", icon=":material/error:")
            else:
                st.success("Estructura de columnas válida.", icon=":material/check_circle:")
                
                # 2. Value checks
                value_errors = []
                
                def clean_val(v):
                    if pd.isna(v):
                        return np.nan
                    try:
                        return float(str(v).strip().replace(" ", "").replace(",", "."))
                    except Exception:
                        return np.nan
                        
                for idx, row in df_input.iterrows():
                    # Check PROMEDIO_GENERAL
                    pg = clean_val(row.get("PROMEDIO_GENERAL"))
                    if pd.isna(pg) or pg < 0.0 or pg > 5.0:
                        value_errors.append({"Fila": idx + 2, "Columna": "PROMEDIO_GENERAL", "Valor": "" if pd.isna(row.get("PROMEDIO_GENERAL")) else str(row.get("PROMEDIO_GENERAL")), "Error": "El promedio debe estar entre 0.0 y 5.0"})
                    
                    # Check CREDITOSAPROBADOS
                    ca = clean_val(row.get("CREDITOSAPROBADOS"))
                    if pd.isna(ca) or ca < 0:
                        value_errors.append({"Fila": idx + 2, "Columna": "CREDITOSAPROBADOS", "Valor": "" if pd.isna(row.get("CREDITOSAPROBADOS")) else str(row.get("CREDITOSAPROBADOS")), "Error": "Créditos aprobados deben ser mayores o iguales a 0"})
                        
                    # Check UBICACION_SEMESTRAL
                    us = clean_val(row.get("UBICACION_SEMESTRAL"))
                    if pd.isna(us) or us < 1 or us > 14:
                        value_errors.append({"Fila": idx + 2, "Columna": "UBICACION_SEMESTRAL", "Valor": "" if pd.isna(row.get("UBICACION_SEMESTRAL")) else str(row.get("UBICACION_SEMESTRAL")), "Error": "Ubicación semestral debe ser entre 1 y 14"})
                        
                    # Check ESTRATO
                    est = clean_val(row.get("ESTRATO"))
                    if not pd.isna(est) and (est < 1 or est > 6):
                        value_errors.append({"Fila": idx + 2, "Columna": "ESTRATO", "Valor": "" if pd.isna(row.get("ESTRATO")) else str(row.get("ESTRATO")), "Error": "El estrato debe ser un número entero entre 1 y 6"})
                
                if value_errors:
                    st.warning(f"Se encontraron {len(value_errors)} registros con anomalías de datos. Serán ignorados durante la predicción.", icon=":material/warning:")
                    st.dataframe(pd.DataFrame(value_errors), use_container_width=True, hide_index=True)

                    # Filter only clean rows
                    error_rows = {err["Fila"] - 2 for err in value_errors}
                    df_clean_input = df_input.drop(index=list(error_rows)).copy()
                else:
                    st.success("Todos los valores están validados correctamente.", icon=":material/check_circle:")
                    df_clean_input = df_input.copy()

                if df_clean_input.empty:
                    st.error("No quedan registros válidos para procesar.", icon=":material/error:")
                else:
                    # Execute prediction use case
                    students_list = []
                    for _, row in df_clean_input.iterrows():
                        students_list.append(Student.from_dict(row.to_dict()))
                        
                    with st.spinner("Procesando predicciones con modelos optimizados..."):
                        try:
                            predictions = predict_use_case.execute_batch(students_list, threshold=threshold)
                        except Exception as e:
                            st.error(f"Error al predecir: {e}")
                            st.stop()
                            
                    # Construct display table
                    results_data = []
                    for idx, pred in enumerate(predictions):
                        orig_student = students_list[idx]
                        results_data.append({
                            "CODESTUDIANTE": orig_student.code_student,
                            "PROGRAMA": orig_student.programa,
                            "PROMEDIO": orig_student.promedio_general,
                            "CREDITOS": orig_student.creditos_aprobados,
                            "SEMESTRE": orig_student.ubicacion_semestral,
                            "P(DESERTA)": pred.probability,
                            "RIESGO": pred.risk_level,
                            "RESULTADO": "Deserta (Riesgo)" if pred.is_dropout else "No Deserta (Estable)",
                            "prediction_object": pred,
                            "student_object": orig_student
                        })
                        
                    df_res = pd.DataFrame(results_data)
                    
                    # Quick stats
                    section_header("bar_chart", "Distribución de Riesgo del Lote", level=4)
                    c1, c2, c3 = st.columns(3)
                    total_p = len(df_res)
                    risk_counts = df_res["RIESGO"].value_counts()

                    with c1:
                        st.metric(":material/groups: Total Procesados", total_p)
                    with c2:
                        high_risk_n = int(risk_counts.get("Alto", 0))
                        st.metric(":material/emergency: Alto Riesgo", high_risk_n, f"{high_risk_n/total_p*100:.1f}%")
                    with c3:
                        mod_risk_n = int(risk_counts.get("Moderado", 0))
                        st.metric(":material/warning: Moderado Riesgo", mod_risk_n, f"{mod_risk_n/total_p*100:.1f}%")
                        
                    col_p1, col_p2 = st.columns([1, 1])
                    with col_p1:
                        fig_pie = px.pie(
                            df_res, 
                            names="RIESGO",
                            color="RIESGO",
                            color_discrete_map={"Bajo": "#2ecc71", "Moderado": "#f1c40f", "Alto": "#e74c3c"},
                            title="Proporción por Nivel de Riesgo"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_p2:
                        fig_hist = px.histogram(
                            df_res, 
                            x="P(DESERTA)",
                            nbins=20,
                            color_discrete_sequence=["#16213e"],
                            title="Distribución de Probabilidades de Deserción"
                        )
                        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Umbral {threshold:.2f}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                    # Detailed results table
                    section_header("clipboard_check", "Detalle de Predicciones", level=4)
                    st.caption("Selecciona una fila para ver el análisis explicativo individual (SHAP) y generar su reporte PDF.")
                    
                    # Remove object columns for display
                    df_display = df_res.drop(columns=["prediction_object", "student_object"])
                    
                    ev = st.dataframe(
                        df_display,
                        column_config={
                            "P(DESERTA)": st.column_config.NumberColumn("Probabilidad", format="%.4f"),
                        },
                        hide_index=True,
                        use_container_width=True,
                        selection_mode="single-row",
                        on_select="rerun"
                    )
                    
                    # Download CSV/Excel
                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        st.download_button(
                            "Descargar Resultados (CSV)",
                            df_display.to_csv(index=False).encode("utf-8"),
                            "resultados_predicciones.csv",
                            "text/csv",
                            icon=":material/download:",
                            use_container_width=True
                        )
                    with c_dl2:
                        # Write to Excel in memory
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_display.to_excel(writer, index=False, sheet_name="Resultados")
                        st.download_button(
                            "Descargar Resultados (Excel)",
                            output.getvalue(),
                            "resultados_predicciones.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            icon=":material/download:",
                            use_container_width=True
                        )
                        
                    # Handle single row selection
                    if ev and ev.selection and ev.selection.rows:
                        selected_idx = ev.selection.rows[0]
                        selected_row = df_res.iloc[selected_idx]
                        selected_student = selected_row["student_object"]
                        
                        # Calculate SHAP values dynamically in real-time
                        with st.spinner("Calculando explicabilidad SHAP en tiempo real..."):
                            try:
                                selected_pred = predict_use_case.execute_individual(selected_student, threshold=threshold)
                            except Exception as e:
                                selected_pred = selected_row["prediction_object"]
                                st.error(f"Error al calcular SHAP: {e}")
                        
                        st.markdown("---")
                        section_header(
                            "search",
                            f"Explicabilidad Individual para el Estudiante: <code>{selected_student.code_student}</code>"
                        )
                        
                        col_sh1, col_sh2 = st.columns([1, 1])
                        
                        with col_sh1:
                            st.markdown("#### Factores de Impacto (SHAP)")
                            # Plot horizontal bar chart for individual SHAP values
                            shap_vals = selected_pred.shap_values
                            if shap_vals:
                                df_shap_ind = pd.DataFrame(list(shap_vals.items()), columns=["Variable", "Impacto"]).sort_values("Impacto", ascending=True)
                                df_shap_ind["Efecto"] = np.where(df_shap_ind["Impacto"] >= 0, "Aumenta Riesgo (Rojo)", "Disminuye Riesgo (Verde)")
                                fig_ind_shap = px.bar(
                                    df_shap_ind.tail(10), 
                                    x="Impacto", 
                                    y="Variable",
                                    orientation="h",
                                    color="Efecto",
                                    color_discrete_map={"Aumenta Riesgo (Rojo)": "#e74c3c", "Disminuye Riesgo (Verde)": "#2ecc71"},
                                    title="Principales Factores Influyentes"
                                )
                                st.plotly_chart(fig_ind_shap, use_container_width=True)
                            else:
                                st.info("No hay valores de explicabilidad SHAP disponibles.")
                                
                        with col_sh2:
                            st.markdown("#### Recomendaciones de Intervención")
                            recs = []
                            if selected_student.promedio_general < 3.3:
                                recs.append(("alert_triangle", PAL["danger_text"], "Rendimiento Académico Crítico", "Inscribir prioritariamente en tutorías dirigidas y mentorías de pares."))
                            if selected_student.creditos_aprobados / max(1, selected_student.ubicacion_semestral) < 10:
                                recs.append(("bar_chart", PAL["warning_text"], "Bajo Ritmo de Aprobación", "Agendar cita de asesoría académica para reorganizar carga y asignaturas matriculadas."))
                            if selected_student.estrato is not None and selected_student.estrato <= 2:
                                recs.append(("dollar_sign", PAL["warning_text"], "Estrato Socioeconómico Vulnerable", "Enviar reporte a Bienestar para tramitación de becas alimentarias o de transporte."))
                            if selected_student.jornada == "NOCTURNA":
                                recs.append(("moon", PAL["accent"], "Estudiante de Jornada Nocturna", "Brindar flexibilidad en tutorías mediante grabaciones y tutorías asincrónicas."))

                            if not recs:
                                recs.append(("check_circle", PAL["success_text"], "Estudiante de Bajo Riesgo", "Mantener seguimiento semestral ordinario."))

                            for icon_name, color, label, detail in recs:
                                rec_item(icon_name, color, label, detail)

                            # Generate and download PDF report
                            pdf_path = PDFGenerator.generate_student_report(selected_student, selected_pred)
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    "Descargar Reporte en PDF",
                                    pdf_file.read(),
                                    file_name=f"Reporte_Riesgo_{selected_student.code_student}.pdf",
                                    mime="application/pdf",
                                    icon=":material/picture_as_pdf:",
                                    use_container_width=True
                                )

                            # Check for Data Drift of this batch!
                            with st.expander(":material/query_stats: Análisis de Drift en este Lote (Monitoreo)"):
                                drift_res = monitor_use_case.detect_data_drift(df_clean_input)
                                if drift_res["status"] == "success":
                                    st.write(f"Estado de Drift: **{drift_res['overall_status'].upper()}**")
                                    st.write(f"Proporción de variables con drift detectado: `{drift_res['drift_fraction']*100:.1f}%`")
                                    if drift_res["drift_fraction"] > 0.30:
                                        st.warning("El dataset presenta diferencias de distribución significativas frente a los datos de entrenamiento iniciales. Se recomienda reentrenamiento.", icon=":material/warning:")
                                else:
                                    st.write("No se pudo correr el detector de drift.")

# -----------------
# TAB: INDIVIDUAL PREDICTION
# -----------------
with tab_indiv:
    section_header("user", "Formulario de Predicción para Estudiante Individual")
    
    with st.form("individual_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            code = st.text_input("Código de Estudiante", value="EST_999")
            programa = st.selectbox("Programa Académico", PROGRAMAS_VALIDOS)
            promedio = st.number_input("Promedio General (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
            creditos = st.number_input("Créditos Aprobados", min_value=0, max_value=160, value=30)
            semestre = st.number_input("Semestre Actual", min_value=1, max_value=14, value=3)
            
        with col_f2:
            jornada = st.selectbox("Jornada", ["DIURNA", "NOCTURNA"])
            genero = st.selectbox("Género", ["M", "F"])
            estrato = st.selectbox("Estrato Socioeconómico", [1, 2, 3, 4, 5, 6], index=1)
            tiene_sisben = st.selectbox("Tiene Sisbén", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            vive_familia = st.selectbox("Vive con la Familia", [0, 1], index=1, format_func=lambda x: "Sí" if x == 1 else "No")
            
        with col_f3:
            situacion_padres = st.selectbox("Situación de los Padres", [
                "VIVOS Y CONVIVEN", "VIVOS Y SEPARADOS", 
                "MADRE VIVA - PADRE DIFUNTO", "PADRE VIVO - MADRE DIFUNTA", "DIFUNTOS"
            ])
            familiares = st.number_input("Número de Familiares en Hogar", min_value=1, max_value=20, value=4)
            hermanos = st.number_input("Número de Hermanos", min_value=0, max_value=20, value=2)
            posicion = st.number_input("Posición entre Hermanos", min_value=1, max_value=20, value=2)
            trabajan = st.number_input("Miembros Familiares que Trabajan", min_value=0, max_value=10, value=2)
            edad = st.number_input("Edad al Ingreso", min_value=15, max_value=80, value=18)
            
        submit = st.form_submit_button("Calcular Riesgo de Deserción", use_container_width=True)
        
    if submit:
        if not active_model_exists:
            st.error("No hay un modelo activo entrenado.", icon=":material/error:")
        else:
            # Map input to dictionary format expected by Student entity
            # Simulate date parameters for pipeline extraction
            # Assuming registration date is today, birthdate is today minus (edad * 365.25)
            birth_year = datetime.now().year - edad
            birth_date = datetime(birth_year, 6, 15)
            reg_date = datetime.now()
            
            student_data = {
                "CODESTUDIANTE": code,
                "ESTP_FECHAINGRESO": reg_date,
                "CREDITOSAPROBADOS": creditos,
                "UBICACION_SEMESTRAL": semestre,
                "PROMEDIO_GENERAL": promedio,
                "PROGRAMA": programa,
                "JORNADA": jornada,
                "GENERO": genero,
                "FECHA_NACIMIENTO": birth_date,
                "CIUDADRESIDENCIA": "BUCARAMANGA", # Default for demo
                "ESTRATO": estrato,
                "TIENE_SISBEN": tiene_sisben,
                "INFE_VIVECONFAMILIA": vive_familia,
                "INFE_SITUACIONPADRES": situacion_padres,
                "INFE_NUMEROFAMILIARES": familiares,
                "INFE_NUMEROHERMANOS": hermanos,
                "INFE_POSICIONENHERMANOS": posicion,
                "INFE_NUMMIEMBROSTRABAJA": trabajan,
            }
            
            student = Student.from_dict(student_data)
            
            with st.spinner("Procesando predicción individual..."):
                try:
                    prediction = predict_use_case.execute_individual(student, threshold=threshold)
                except Exception as e:
                    st.error(f"Error al realizar predicción: {e}")
                    st.stop()
                    
            st.markdown("---")
            section_header("bar_chart", "Resultado del Diagnóstico")
            
            col_diag1, col_diag2 = st.columns([1, 2])
            
            with col_diag1:
                # Gauge plot of probability
                prob_pct = prediction.probability * 100
                risk = prediction.risk_level
                
                # Colors
                color_risk = "#2ecc71" if risk == "Bajo" else "#f1c40f" if risk == "Moderado" else "#e74c3c"
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_pct,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Riesgo: {risk.upper()}", 'font': {'size': 20, 'color': color_risk}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': PAL["text_secondary"]},
                        'bar': {'color': color_risk},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': PAL["border"],
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(46, 204, 113, 0.1)'},
                            {'range': [35, 70], 'color': 'rgba(241, 196, 15, 0.1)'},
                            {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.1)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_diag2:
                st.markdown("#### Variables más Influyentes (Análisis SHAP)")
                shap_vals = prediction.shap_values
                if shap_vals:
                    df_shap_ind = pd.DataFrame(list(shap_vals.items()), columns=["Variable", "Impacto"]).sort_values("Impacto", ascending=True)
                    df_shap_ind["Efecto"] = np.where(df_shap_ind["Impacto"] >= 0, "Incrementa Riesgo", "Disminuye Riesgo")
                    fig_ind_shap = px.bar(
                        df_shap_ind.tail(10), 
                        x="Impacto", 
                        y="Variable",
                        orientation="h",
                        color="Efecto",
                        color_discrete_map={"Incrementa Riesgo": "#e74c3c", "Disminuye Riesgo": "#2ecc71"},
                        height=260
                    )
                    fig_ind_shap.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_ind_shap, use_container_width=True)
                    
            # PDF download and Actionable tips
            col_act1, col_act2 = st.columns([1, 1])
            with col_act1:
                st.markdown("#### Recomendaciones de Intervención:")
                recs = []
                if promedio < 3.3:
                    recs.append(("user", PAL["danger_text"], "Vincular a Mentorías", "Derivar a tutorías de refuerzo en la facultad."))
                if creditos / semestre < 10:
                    recs.append(("calendar", PAL["warning_text"], "Consejería de Matrícula", "Agendar reunión académica para planificar créditos semestrales."))
                if estrato <= 2:
                    recs.append(("dollar_sign", PAL["warning_text"], "Bienestar Social", "Inscribir en programas de subsidio alimentario o subsidio de transporte."))
                if jornada == "NOCTURNA":
                    recs.append(("clock", PAL["accent"], "Flexibilidad Horaria", "Facilitar horarios extendidos para consultas de tutoría."))
                if not recs:
                    recs.append(("check_circle", PAL["success_text"], "Sin Acciones Pendientes", "No requiere acciones de contingencia inmediatas."))

                for icon_name, color, label, detail in recs:
                    rec_item(icon_name, color, label, detail)

            with col_act2:
                # PDF report download
                pdf_path = PDFGenerator.generate_student_report(student, prediction)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        "Descargar Reporte en PDF",
                        pdf_file.read(),
                        file_name=f"Reporte_Riesgo_{code}.pdf",
                        mime="application/pdf",
                        icon=":material/picture_as_pdf:",
                        use_container_width=True
                    )

# -----------------
# TAB: ADMINISTRATION
# -----------------
with tab_admin:
    section_header("settings", "Administración del Modelo de Clasificación")
    
    # 1. Active version switching
    st.markdown("#### 1. Historial de Versiones en el Registro")
    try:
        history_list = model_repo.get_model_history()
        if history_list:
            df_history = pd.DataFrame(history_list)
            # Display history
            st.dataframe(
                df_history[["version", "model_name", "trained_at", "metrics"]],
                use_container_width=True
            )
            
            # Form to change active version
            with st.form("active_switch_form"):
                versions = [item["version"] for item in history_list]
                active_switch = st.selectbox("Seleccionar Versión del Modelo Activo", versions, index=versions.index(active_version) if active_version in versions else 0)
                switch_submit = st.form_submit_button("Actualizar Modelo Activo", use_container_width=True)
                
            if switch_submit:
                with open(model_repo.registry_file, "r") as f:
                    import json
                    reg_dict = json.load(f)
                reg_dict["active_version"] = active_switch
                with open(model_repo.registry_file, "w") as f:
                    json.dump(reg_dict, f, indent=4)
                st.success(f"Modelo activo actualizado a la versión: {active_switch}. Recarga la página para aplicar.", icon=":material/check_circle:")
                st.rerun()
        else:
            st.info("No hay modelos guardados en el historial del registro.")
    except Exception as e:
        st.error(f"Error al cargar el historial del registro: {e}")
        
    # 2. Retraining triggers
    st.markdown("#### 2. Reentrenamiento y Mejora Continua (AutoML)")
    st.markdown("Sube una versión actualizada del archivo **'dataset/student_database.xlsx'** para reajustar los modelos preprocesadores, optimizar hiperparámetros con búsqueda bayesiana (Optuna) y seleccionar el mejor clasificador de forma automática.")
    
    retrain_file = st.file_uploader("Sube el archivo 'student_database.xlsx' para reentrenamiento", type=["xlsx"])
    optuna_trials = st.slider("Ensayos de Optimización (Optuna Trials)", min_value=5, max_value=50, value=15)
    
    if retrain_file is not None:
        retrain_btn = st.button("Iniciar Pipeline de Reentrenamiento", icon=":material/rocket_launch:", use_container_width=True)

        if retrain_btn:
            # Overwrite file locally
            temp_path = "dataset/student_database.xlsx"
            with open(temp_path, "wb") as f:
                f.write(retrain_file.getbuffer())

            st.success("Archivo de datos cargado en dataset/.", icon=":material/check_circle:")
            
            # Execute training use case
            with st.spinner("Ejecutando pipeline de reentrenamiento. Esto puede tomar unos minutos..."):
                try:
                    res = train_use_case.execute(
                        raw_data_path=temp_path,
                        allowed_programs=None,
                        exclude_features=["ANIO_INGRESO", "MES_INGRESO"],
                        n_trials=optuna_trials
                    )
                    
                    st.success(f"¡Modelo reentrenado con éxito! Champion seleccionado: {res['model_name']} ({res['version']})", icon=":material/celebration:")
                    st.markdown(f"""
                    **Métricas del nuevo campeón:**
                    - Accuracy: `{res['metrics']['accuracy']:.4f}`
                    - F1-Score: `{res['metrics']['f1']:.4f}`
                    - ROC-AUC: `{res['metrics']['roc_auc']:.4f}`
                    """)
                    st.info("Recarga la aplicación para comenzar a utilizar este nuevo modelo.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error durante el reentrenamiento: {e}")
