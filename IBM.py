import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NOVA | IBM HR Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "attrition_model.joblib"
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# ============================================================
# MODEL / DATA CONFIG
# These match the Logistic Regression notebook supplied.
# ============================================================
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
]

NUMERICAL_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

FEATURES = NUMERICAL_COLS + ["Gender", "OverTime"] + CATEGORICAL_COLS + DROP_COLS

# ============================================================
# CUSTOM THEME
# ============================================================
COLORS = {
    "bg": "#070A0F",
    "panel": "#10161F",
    "panel_2": "#141C26",
    "panel_3": "#18212D",
    "border": "#263241",
    "border_light": "#334155",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "muted_2": "#64748B",
    "primary": "#38D9C5",
    "blue": "#60A5FA",
    "purple": "#A78BFA",
    "orange": "#FBBF74",
    "red": "#FB7185",
    "green": "#4ADE80",
}

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 82% -5%, rgba(56,217,197,0.075), transparent 24%),
            radial-gradient(circle at 5% 35%, rgba(96,165,250,0.045), transparent 25%),
            {COLORS["bg"]};
        color: {COLORS["text"]};
    }}

    .main .block-container {{
        max-width: 1480px;
        padding: 1.6rem 2rem 4rem 2rem;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS["text"]} !important;
        letter-spacing: -0.5px;
    }}

    p {{
        color: {COLORS["muted"]};
    }}

    .section-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 0.15rem;
    }}

    .section-subtitle {{
        color: {COLORS["muted"]};
        font-size: 0.83rem;
        margin-bottom: 1.1rem;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #090D13 0%, #0B1017 100%);
        border-right: 1px solid {COLORS["border"]};
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding: 1.35rem 1rem 1rem 1rem;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 8px 4px 8px;
    }}

    .brand-icon {{
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(56,217,197,0.18), rgba(96,165,250,0.08));
        border: 1px solid rgba(56,217,197,0.25);
        font-size: 20px;
    }}

    .brand-name {{
        color: {COLORS["text"]};
        font-size: 1.25rem;
        font-weight: 850;
        letter-spacing: -0.7px;
        line-height: 1;
    }}

    .brand-sub {{
        color: {COLORS["muted_2"]};
        font-size: 0.61rem;
        font-weight: 700;
        letter-spacing: 1.3px;
        margin-top: 5px;
    }}

    .sidebar-label {{
        color: {COLORS["muted_2"]};
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin: 18px 8px 7px 8px;
    }}

    div[data-testid="stRadio"] > div {{
        gap: 4px;
    }}

    div[data-testid="stRadio"] label {{
        border-radius: 10px;
        padding: 9px 10px !important;
        margin: 1px 0;
        border: 1px solid transparent;
        background: transparent;
    }}

    div[data-testid="stRadio"] label:hover {{
        background: rgba(255,255,255,0.035);
        border-color: {COLORS["border"]};
    }}

    div[data-testid="stRadio"] label[data-checked="true"] {{
        background: rgba(56,217,197,0.09);
        border-color: rgba(56,217,197,0.18);
    }}

    div[data-testid="stRadio"] label p {{
        color: {COLORS["muted"]};
        font-size: 0.82rem;
        font-weight: 650;
    }}

    div[data-testid="stRadio"] label[data-checked="true"] p {{
        color: {COLORS["primary"]};
        font-weight: 750;
    }}

    .sidebar-card {{
        background: rgba(255,255,255,0.025);
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 13px;
        margin-top: 8px;
    }}

    .sidebar-card-title {{
        color: {COLORS["muted_2"]};
        font-size: 0.6rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .sidebar-card-value {{
        color: {COLORS["text"]};
        font-size: 0.86rem;
        font-weight: 750;
    }}

    .sidebar-card-small {{
        color: {COLORS["muted"]};
        font-size: 0.69rem;
        margin-top: 3px;
    }}

    .eyebrow {{
        color: {COLORS["primary"]};
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .page-title {{
        color: {COLORS["text"]};
        font-size: 1.8rem;
        font-weight: 850;
        letter-spacing: -1px;
        margin: 0;
    }}

    .page-description {{
        color: {COLORS["muted"]};
        font-size: 0.78rem;
        margin-top: 6px;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(74,222,128,0.07);
        border: 1px solid rgba(74,222,128,0.18);
        color: #86EFAC;
        font-size: 0.69rem;
        font-weight: 750;
    }}

    .status-dot {{
        width: 7px;
        height: 7px;
        background: #4ADE80;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(74,222,128,0.7);
    }}

    .kpi-card {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, #121923, #0E141C);
        border: 1px solid {COLORS["border"]};
        border-radius: 15px;
        padding: 17px 18px;
        min-height: 116px;
    }}

    .kpi-card::after {{
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -35px;
        top: -35px;
        background: rgba(56,217,197,0.05);
        border-radius: 50%;
    }}

    .kpi-label {{
        color: {COLORS["muted_2"]};
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .kpi-value {{
        color: {COLORS["text"]};
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.8px;
        margin-top: 9px;
    }}

    .kpi-icon {{
        position: absolute;
        right: 16px;
        top: 15px;
        width: 31px;
        height: 31px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(56,217,197,0.08);
        border: 1px solid rgba(56,217,197,0.12);
        font-size: 14px;
    }}

    .card-heading {{
        display: flex;
        align-items: center;
        gap: 9px;
        color: {COLORS["text"]};
        font-size: 0.92rem;
        font-weight: 800;
        margin-bottom: 4px;
    }}

    .card-heading-icon {{
        width: 29px;
        height: 29px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: rgba(56,217,197,0.10);
        border: 1px solid rgba(56,217,197,0.12);
    }}

    .card-caption {{
        color: {COLORS["muted_2"]};
        font-size: 0.7rem;
        margin-bottom: 15px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(145deg, rgba(20,28,38,0.98), rgba(15,21,29,0.98));
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }}

    label {{
        color: #B8C2CF !important;
        font-size: 0.69rem !important;
        font-weight: 700 !important;
    }}

    div[data-baseweb="input"] {{
        background: #0A1017 !important;
        border: 1px solid #293544 !important;
        border-radius: 9px !important;
    }}

    div[data-baseweb="input"]:focus-within {{
        border-color: {COLORS["primary"]} !important;
        box-shadow: 0 0 0 1px rgba(56,217,197,0.3) !important;
    }}

    div[data-baseweb="select"] > div {{
        background: #0A1017 !important;
        border: 1px solid #293544 !important;
        border-radius: 9px !important;
    }}

    input {{
        color: {COLORS["text"]} !important;
        font-weight: 650 !important;
    }}

    .stButton > button {{
        width: 100%;
        min-height: 46px;
        border-radius: 10px;
        border: 1px solid {COLORS["primary"]};
        background: linear-gradient(135deg, #38D9C5, #20BFAF);
        color: #04110F;
        font-size: 0.76rem;
        font-weight: 850;
        box-shadow: 0 7px 24px rgba(56,217,197,0.09);
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 11px 30px rgba(56,217,197,0.18);
    }}

    .prediction-result {{
        background:
            radial-gradient(circle at 85% 15%, rgba(56,217,197,0.12), transparent 35%),
            linear-gradient(145deg, #121C24, #0D141C);
        border: 1px solid rgba(56,217,197,0.22);
        border-radius: 17px;
        padding: 22px;
        margin-bottom: 12px;
    }}

    .prediction-label {{
        color: {COLORS["muted_2"]};
        font-size: 0.62rem;
        font-weight: 850;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }}

    .prediction-value {{
        color: {COLORS["text"]};
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-top: 8px;
    }}

    .prediction-sub {{
        color: {COLORS["muted"]};
        font-size: 0.76rem;
        margin-top: 7px;
    }}

    .risk-high {{
        display: inline-block;
        margin-top: 14px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(251,113,133,0.08);
        border: 1px solid rgba(251,113,133,0.20);
        color: #FDA4AF;
        font-size: 0.63rem;
        font-weight: 800;
    }}

    .risk-low {{
        display: inline-block;
        margin-top: 14px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(74,222,128,0.08);
        border: 1px solid rgba(74,222,128,0.18);
        color: #86EFAC;
        font-size: 0.63rem;
        font-weight: 800;
    }}

    .feature-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 7px;
        border-radius: 6px;
        background: rgba(96,165,250,0.07);
        border: 1px solid rgba(96,165,250,0.12);
        color: #93C5FD;
        font-size: 0.59rem;
        font-weight: 750;
    }}

    .footer {{
        text-align: center;
        color: {COLORS["muted_2"]};
        font-size: 0.64rem;
        padding: 15px 0 0 0;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL LOAD
# The notebook's FunctionTransformer references binary_cleanup,
# so define the same function before loading the joblib pipeline.
# ============================================================
def binary_cleanup(data):
    data = data.copy()
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})
    return data.drop(columns=DROP_COLS)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data["Attrition"] = data["Attrition"].astype(str).str.strip().str.capitalize()
    return data

try:
    model = load_model()
    df = load_dataset()
except Exception as error:
    st.error("Unable to load the model or dataset.")
    st.code(str(error))
    st.info(
        "Keep these files in the same folder as app.py: "
        "attrition_model.joblib and WA_Fn-UseC_-HR-Employee-Attrition(2).csv"
    )
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def plot_layout(fig, height=350, showlegend=False):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            color=COLORS["muted"],
            size=11,
        ),
        height=height,
        showlegend=showlegend,
        margin=dict(l=8, r=8, t=18, b=8),
        hoverlabel=dict(
            bgcolor=COLORS["panel_3"],
            bordercolor=COLORS["border_light"],
            font=dict(color=COLORS["text"]),
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.08)",
            linecolor="rgba(148,163,184,0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.08)",
            linecolor="rgba(148,163,184,0.12)",
        ),
    )
    return fig

def card_header(icon, title, subtitle=None):
    subtitle_html = f'<div class="card-caption">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="card-heading">
            <div class="card-heading-icon">{icon}</div>
            <span>{title}</span>
        </div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )

def kpi_card(icon, label, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODEL METADATA
# ============================================================
try:
    model_classes = list(model.named_steps["model"].classes_)
except Exception:
    model_classes = [0, 1]

# Evaluate on the same 80/20 split used in the notebook.
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

X_eval = df.drop(columns=["Attrition"])
y_eval = df["Attrition"].map({"Yes": 1, "No": 0})
_, X_test, _, y_test = train_test_split(
    X_eval, y_eval, test_size=0.2, random_state=42
)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">👥</div>
            <div>
                <div class="brand-name">NOVA</div>
                <div class="brand-sub">HR ATTRITION INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "◈  Overview",
            "◎  Attrition Prediction",
            "◉  HR Analytics",
            "◆  Model Intelligence",
        ],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[-1]

    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Algorithm</div>
            <div class="sidebar-card-value">Logistic Regression</div>
            <div class="sidebar-card-small">Binary classification • Pipeline preprocessing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Dataset</div>
            <div class="sidebar-card-value">{len(df):,} employees</div>
            <div class="sidebar-card-small">Target: Attrition • {len(df.columns)} columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">System</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">Status</div>
            <div class="sidebar-card-value">● Model Online</div>
            <div class="sidebar-card-small">Prediction engine ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top:22px;padding:0 8px;color:#475569;font-size:0.62rem;">
            NOVA / v1.0<br>
            IBM HR Attrition Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# GLOBAL HEADER
# ============================================================
header_left, header_right = st.columns([5, 1], vertical_alignment="center")

with header_left:
    st.markdown(
        """
        <div class="eyebrow">MACHINE LEARNING • HUMAN RESOURCES ANALYTICS</div>
        <div class="page-title">IBM HR Attrition Intelligence</div>
        <div class="page-description">
            Employee attrition analysis and prediction powered by Logistic Regression.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right;">
            <span class="status-pill">
                <span class="status-dot"></span>
                MODEL READY
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A high-level view of employee attrition and the prediction environment.</div>',
        unsafe_allow_html=True,
    )

    total = len(df)
    attrition_yes = int((df["Attrition"] == "Yes").sum())
    attrition_no = int((df["Attrition"] == "No").sum())
    attrition_rate = attrition_yes / total * 100
    avg_income = df["MonthlyIncome"].mean()
    avg_age = df["Age"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("▦", "EMPLOYEES", f"{total:,}")
    with k2:
        kpi_card("⚠", "ATTRITION RATE", f"{attrition_rate:.1f}%")
    with k3:
        kpi_card("⌁", "AVG MONTHLY INCOME", f"${avg_income:,.0f}")
    with k4:
        kpi_card("◌", "MODEL ACCURACY", f"{accuracy*100:.1f}%")

    st.write("")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.container(border=True):
            card_header("◒", "Attrition Distribution", "Employees who stayed versus employees who left.")
            counts = df["Attrition"].value_counts().rename_axis("Attrition").reset_index(name="Employees")
            fig = px.bar(
                counts,
                x="Attrition",
                y="Employees",
                text="Employees",
                category_orders={"Attrition": ["No", "Yes"]},
            )
            fig.update_traces(marker_color=[COLORS["green"], COLORS["red"]], textposition="outside")
            fig = plot_layout(fig, height=350)
            fig.update_yaxes(title="Employees")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            card_header("◉", "Attrition by Overtime", "Overtime is one of the strongest HR risk signals.")
            ot = pd.crosstab(df["OverTime"], df["Attrition"]).reset_index()
            for col in ["No", "Yes"]:
                if col not in ot.columns:
                    ot[col] = 0
            fig = px.bar(
                ot,
                x="OverTime",
                y=["No", "Yes"],
                barmode="group",
                labels={"value": "Employees", "variable": "Attrition"},
            )
            fig.update_traces(marker_line_width=0)
            fig = plot_layout(fig, height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            card_header("▤", "Attrition by Job Role", "Distribution of attrition across employee roles.")
            role = pd.crosstab(df["JobRole"], df["Attrition"]).reset_index()
            for col in ["No", "Yes"]:
                if col not in role.columns:
                    role[col] = 0
            role = role.sort_values("Yes", ascending=True)
            fig = px.bar(
                role,
                x="Yes",
                y="JobRole",
                orientation="h",
                text="Yes",
            )
            fig.update_traces(marker_color=COLORS["red"], textposition="outside")
            fig = plot_layout(fig, height=430)
            fig.update_xaxes(title="Employees with Attrition")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            card_header("⌁", "Age vs Monthly Income", "Employee age and compensation, colored by attrition status.")
            fig = px.scatter(
                df,
                x="Age",
                y="MonthlyIncome",
                color="Attrition",
                color_discrete_map={"No": COLORS["green"], "Yes": COLORS["red"]},
                opacity=0.65,
                hover_data=["JobRole", "Department", "OverTime"],
            )
            fig = plot_layout(fig, height=430, showlegend=True)
            fig.update_xaxes(title="Age")
            fig.update_yaxes(title="Monthly Income")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        card_header("✓", "Dataset Health", "Quality indicators for the loaded HR dataset.")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("MISSING VALUES", f"{int(df.isna().sum().sum()):,}")
        with h2:
            st.metric("DUPLICATE ROWS", f"{int(df.duplicated().sum()):,}")
        with h3:
            st.metric("NUMERIC COLUMNS", len(df.select_dtypes(include=np.number).columns))
        with h4:
            st.metric("TOTAL COLUMNS", len(df.columns))

# ============================================================
# ATTRITION PREDICTION
# ============================================================
elif page == "Attrition Prediction":
    st.markdown('<div class="section-title">Attrition Prediction Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter an employee profile and estimate the probability of attrition.</div>',
        unsafe_allow_html=True,
    )

    input_col, output_col = st.columns([1.55, 0.85], gap="large")

    with input_col:
        with st.container(border=True):
            card_header("⚙", "Employee Profile", "Inputs match the trained Logistic Regression pipeline.")

            st.markdown('<div class="feature-badge">01 • PERSONAL & WORK PROFILE</div>', unsafe_allow_html=True)
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 18, 65, 35)
            with c2:
                gender = st.selectbox("Gender", ["Male", "Female"])
            with c3:
                marital = st.selectbox("Marital Status", sorted(df["MaritalStatus"].dropna().unique()))

            c1, c2, c3 = st.columns(3)
            with c1:
                travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].dropna().unique()))
            with c2:
                department = st.selectbox("Department", sorted(df["Department"].dropna().unique()))
            with c3:
                job_role = st.selectbox("Job Role", sorted(df["JobRole"].dropna().unique()))

            st.write("")
            st.markdown('<div class="feature-badge">02 • JOB & SATISFACTION</div>', unsafe_allow_html=True)
            st.write("")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                job_level = st.slider("Job Level", 1, 5, 2)
            with c2:
                job_involvement = st.slider("Job Involvement", 1, 4, 3)
            with c3:
                job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            with c4:
                environment_sat = st.slider("Environment Satisfaction", 1, 4, 3)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                relationship_sat = st.slider("Relationship Satisfaction", 1, 4, 3)
            with c2:
                work_life = st.slider("Work-Life Balance", 1, 4, 3)
            with c3:
                stock_option = st.slider("Stock Option Level", 0, 3, 1)
            with c4:
                performance = st.slider("Performance Rating", 1, 4, 3)

            st.write("")
            st.markdown('<div class="feature-badge">03 • COMPENSATION & CAREER</div>', unsafe_allow_html=True)
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                monthly_income = st.number_input("Monthly Income", min_value=100.0, value=5000.0, step=100.0)
            with c2:
                monthly_rate = st.number_input("Monthly Rate", min_value=1000.0, value=15000.0, step=500.0)
            with c3:
                daily_rate = st.number_input("Daily Rate", min_value=100.0, value=800.0, step=10.0)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                hourly_rate = st.number_input("Hourly Rate", min_value=1.0, value=60.0, step=1.0)
            with c2:
                salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
            with c3:
                total_years = st.number_input("Total Working Years", 0, 45, 8)
            with c4:
                companies = st.number_input("Num Companies Worked", 0, 15, 2)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                years_company = st.number_input("Years at Company", 0, 40, 4)
            with c2:
                years_role = st.number_input("Years in Current Role", 0, 20, 2)
            with c3:
                years_promo = st.number_input("Years Since Last Promotion", 0, 15, 1)
            with c4:
                years_manager = st.number_input("Years With Current Manager", 0, 20, 2)

            st.write("")
            st.markdown('<div class="feature-badge">04 • OTHER FACTORS</div>', unsafe_allow_html=True)
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                education = st.slider("Education", 1, 5, 3)
            with c2:
                education_field = st.selectbox("Education Field", sorted(df["EducationField"].dropna().unique()))
            with c3:
                overtime = st.selectbox("Overtime", ["No", "Yes"])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                distance = st.number_input("Distance From Home", 1, 30, 10)
            with c2:
                training = st.slider("Training Times Last Year", 0, 6, 3)
            with c3:
                # The notebook uses median imputation for this numeric feature.
                education_display = education
                num_dummy = st.number_input("Employee Number", min_value=1, value=10001)
            with c4:
                st.markdown(
                    '<div style="color:#64748B;font-size:0.67rem;padding-top:31px;">Employee Number is dropped before model prediction.</div>',
                    unsafe_allow_html=True,
                )

            predict_button = st.button("✦  Predict Employee Attrition Risk", use_container_width=True)

    with output_col:
        with st.container(border=True):
            card_header("◈", "Risk Assessment", "Model probability and classification output.")

            if predict_button:
                sample = pd.DataFrame([{
                    "Age": age,
                    "DailyRate": daily_rate,
                    "DistanceFromHome": distance,
                    "Education": education,
                    "EnvironmentSatisfaction": environment_sat,
                    "Gender": gender,
                    "HourlyRate": hourly_rate,
                    "JobInvolvement": job_involvement,
                    "JobLevel": job_level,
                    "JobSatisfaction": job_satisfaction,
                    "MonthlyIncome": monthly_income,
                    "MonthlyRate": monthly_rate,
                    "NumCompaniesWorked": companies,
                    "OverTime": overtime,
                    "PercentSalaryHike": salary_hike,
                    "PerformanceRating": performance,
                    "RelationshipSatisfaction": relationship_sat,
                    "StockOptionLevel": stock_option,
                    "TotalWorkingYears": total_years,
                    "TrainingTimesLastYear": training,
                    "WorkLifeBalance": work_life,
                    "YearsAtCompany": years_company,
                    "YearsInCurrentRole": years_role,
                    "YearsSinceLastPromotion": years_promo,
                    "YearsWithCurrManager": years_manager,
                    "BusinessTravel": travel,
                    "Department": department,
                    "EducationField": education_field,
                    "JobRole": job_role,
                    "MaritalStatus": marital,
                    "EmployeeCount": 1,
                    "Over18": "Y",
                    "StandardHours": 80,
                    "EmployeeNumber": num_dummy,
                }])

                try:
                    pred = int(model.predict(sample)[0])
                    prob = float(model.predict_proba(sample)[0, 1])

                    if pred == 1:
                        title = "High Attrition Risk"
                        sub = "The model predicts that this employee is likely to leave."
                        badge = "⚠ ACTION RECOMMENDED"
                        badge_class = "risk-high"
                    else:
                        title = "Low Attrition Risk"
                        sub = "The model predicts that this employee is likely to stay."
                        badge = "✓ LOWER RISK"
                        badge_class = "risk-low"

                    st.markdown(
                        f"""
                        <div class="prediction-result">
                            <div class="prediction-label">PREDICTION</div>
                            <div class="prediction-value">{title}</div>
                            <div class="prediction-sub">{sub}</div>
                            <div class="{badge_class}">{badge}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            number={"suffix": "%", "font": {"size": 25, "color": COLORS["text"]}},
                            title={"text": "Probability of Attrition", "font": {"color": COLORS["muted"], "size": 12}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": COLORS["muted_2"]},
                                "bar": {"color": COLORS["red"] if prob >= 0.5 else COLORS["primary"], "thickness": 0.28},
                                "bgcolor": COLORS["panel_3"],
                                "borderwidth": 0,
                            },
                        )
                    )
                    gauge.update_layout(
                        height=230,
                        margin=dict(l=10, r=10, t=25, b=5),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

                    r1, r2 = st.columns(2)
                    with r1:
                        st.metric("ATTRITION PROBABILITY", f"{prob*100:.1f}%")
                    with r2:
                        st.metric("MODEL OUTPUT", "YES" if pred == 1 else "NO")

                    with st.expander("View employee input summary"):
                        summary = pd.DataFrame({
                            "Feature": ["Age", "Job Role", "Department", "Overtime", "Monthly Income", "Job Satisfaction", "Years at Company"],
                            "Value": [age, job_role, department, overtime, f"${monthly_income:,.0f}", job_satisfaction, years_company],
                        })
                        st.dataframe(summary, use_container_width=True, hide_index=True)

                except Exception as error:
                    st.error("Prediction failed.")
                    st.code(str(error))
            else:
                st.markdown(
                    """
                    <div style="text-align:center;padding:55px 18px;color:#94A3B8;">
                        <div style="font-size:34px;">◈</div>
                        <div style="color:#F8FAFC;font-weight:800;font-size:0.95rem;margin-top:10px;">
                            Ready for prediction
                        </div>
                        <div style="color:#64748B;font-size:0.72rem;margin-top:6px;">
                            Configure the employee profile and run the model.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.container(border=True):
        card_header("⇢", "Prediction Workflow", "The same preprocessing flow used in the supplied notebook.")
        w1, w2, w3, w4 = st.columns(4)
        steps = [
            ("01", "Employee Input", "HR profile values"),
            ("02", "Cleanup", "Binary encoding + dropped fields"),
            ("03", "Preprocessing", "Imputation + scaling + one-hot encoding"),
            ("04", "Logistic Regression", "Attrition class + probability"),
        ]
        for col, (num, title, desc) in zip([w1, w2, w3, w4], steps):
            with col:
                st.markdown(
                    f"""
                    <div style="min-height:120px;padding:15px;background:rgba(255,255,255,0.02);
                    border:1px solid #263241;border-radius:12px;">
                        <div style="color:#38D9C5;font-size:0.60rem;font-weight:850;letter-spacing:1px;">STEP {num}</div>
                        <div style="color:#F8FAFC;font-size:0.88rem;font-weight:800;margin-top:10px;">{title}</div>
                        <div style="color:#94A3B8;font-size:0.68rem;line-height:1.5;margin-top:7px;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ============================================================
# HR ANALYTICS
# ============================================================
elif page == "HR Analytics":
    st.markdown('<div class="section-title">HR Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Explore the employee dataset, attrition drivers and workforce patterns.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["▦ Attrition Analysis", "◉ Feature Relationships", "▧ Correlation Matrix"])

    with tab1:
        st.write("")
        left, right = st.columns(2, gap="large")

        with left:
            with st.container(border=True):
                card_header("◒", "Attrition by Department", "Compare attrition across business units.")
                dep = pd.crosstab(df["Department"], df["Attrition"]).reset_index()
                for c in ["No", "Yes"]:
                    if c not in dep.columns:
                        dep[c] = 0
                fig = px.bar(dep, x="Department", y=["No", "Yes"], barmode="group")
                fig.update_traces(marker_line_width=0)
                fig = plot_layout(fig, height=380, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with right:
            with st.container(border=True):
                card_header("◉", "Attrition by Business Travel", "Travel frequency versus employee exits.")
                travel_df = pd.crosstab(df["BusinessTravel"], df["Attrition"]).reset_index()
                for c in ["No", "Yes"]:
                    if c not in travel_df.columns:
                        travel_df[c] = 0
                fig = px.bar(travel_df, x="BusinessTravel", y=["No", "Yes"], barmode="group")
                fig.update_traces(marker_line_width=0)
                fig = plot_layout(fig, height=380, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.container(border=True):
            card_header("▤", "Monthly Income by Attrition", "Compensation distribution for employees who stayed and left.")
            fig = px.box(
                df,
                x="Attrition",
                y="MonthlyIncome",
                color="Attrition",
                color_discrete_map={"No": COLORS["green"], "Yes": COLORS["red"]},
                points=False,
            )
            fig = plot_layout(fig, height=360, showlegend=False)
            fig.update_yaxes(title="Monthly Income")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        st.write("")
        options = [
            "Age", "MonthlyIncome", "JobSatisfaction", "EnvironmentSatisfaction",
            "DistanceFromHome", "YearsAtCompany", "TotalWorkingYears",
            "PercentSalaryHike", "JobInvolvement", "WorkLifeBalance",
        ]
        selected = st.selectbox("Select feature", options)

        with st.container(border=True):
            card_header("◉", f"{selected} vs Attrition", "Distribution of the selected employee feature by attrition class.")
            fig = px.histogram(
                df,
                x=selected,
                color="Attrition",
                marginal="box",
                barmode="overlay",
                opacity=0.65,
                color_discrete_map={"No": COLORS["green"], "Yes": COLORS["red"]},
            )
            fig = plot_layout(fig, height=470, showlegend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        st.write("")
        with st.container(border=True):
            card_header("▧", "Feature Correlation Matrix", "Correlation between numerical HR variables.")
            numeric = df.select_dtypes(include=np.number).copy()
            corr = numeric.corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr.to_numpy(),
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    zmin=-1,
                    zmax=1,
                    colorscale=[
                        [0.0, "#0B1017"],
                        [0.5, "#183B4A"],
                        [1.0, "#38D9C5"],
                    ],
                    colorbar=dict(title="Correlation"),
                    hovertemplate="%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=650,
                margin=dict(l=10, r=10, t=15, b=10),
                font=dict(color=COLORS["muted"], size=10),
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# MODEL INTELLIGENCE
# ============================================================
elif page == "Model Intelligence":
    st.markdown('<div class="section-title">Model Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Understand the Logistic Regression model, preprocessing and evaluation results.</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("ML", "ALGORITHM", "LOGISTIC REG.")
    with k2:
        kpi_card("44", "MODEL FEATURES", "44")
    with k3:
        kpi_card("↗", "ACCURACY", f"{accuracy*100:.1f}%")
    with k4:
        kpi_card("◎", "ROC-AUC", f"{auc:.3f}")

    st.write("")

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container(border=True):
            card_header("▤", "Confusion Matrix", "Results from the same 80/20 split used in the notebook.")
            cm_fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted Stay", "Predicted Leave"],
                    y=["Actual Stay", "Actual Leave"],
                    colorscale=[
                        [0.0, "#0B1017"],
                        [1.0, "#38D9C5"],
                    ],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 18, "color": COLORS["text"]},
                    hovertemplate="Count: %{z}<extra></extra>",
                    showscale=False,
                )
            )
            cm_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            card_header("◎", "Classification Metrics", "Precision, recall and F1-score for each class.")
            metrics_df = pd.DataFrame([
                {
                    "Class": "Stay (0)",
                    "Precision": report["0"]["precision"],
                    "Recall": report["0"]["recall"],
                    "F1": report["0"]["f1-score"],
                },
                {
                    "Class": "Leave (1)",
                    "Precision": report["1"]["precision"],
                    "Recall": report["1"]["recall"],
                    "F1": report["1"]["f1-score"],
                },
            ])
            metrics_display = metrics_df.copy()
            for c in ["Precision", "Recall", "F1"]:
                metrics_display[c] = metrics_display[c].map(lambda x: f"{x:.3f}")
            st.dataframe(metrics_display, use_container_width=True, hide_index=True, height=145)
            st.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}")
            st.metric("Weighted F1", f"{report['weighted avg']['f1-score']:.3f}")

    # Coefficients
    with st.container(border=True):
        card_header("↕", "Logistic Regression Coefficients", "Magnitude and direction of the transformed model features.")

        try:
            preprocessor = model.named_steps["preprocessing"]
            lr = model.named_steps["model"]
            feature_names = preprocessor.get_feature_names_out()
            coefficients = lr.coef_.flatten()

            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": coefficients,
            })
            coef_df["Absolute Impact"] = coef_df["Coefficient"].abs()
            top_coef = coef_df.sort_values("Absolute Impact", ascending=False).head(20).sort_values("Coefficient")

            fig = px.bar(
                top_coef,
                x="Coefficient",
                y="Feature",
                orientation="h",
            )
            fig.update_traces(
                marker_color=[
                    COLORS["red"] if x < 0 else COLORS["primary"]
                    for x in top_coef["Coefficient"]
                ]
            )
            fig = plot_layout(fig, height=600)
            fig.update_xaxes(title="Coefficient")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            table = coef_df.sort_values("Absolute Impact", ascending=False).head(20).copy()
            table["Coefficient"] = table["Coefficient"].round(4)
            table["Absolute Impact"] = table["Absolute Impact"].round(4)
            st.dataframe(table, use_container_width=True, hide_index=True, height=360)
        except Exception as error:
            st.warning("Coefficient analysis is unavailable.")
            st.code(str(error))

    with st.container(border=True):
        card_header("⇢", "Model Pipeline", "Architecture used in the supplied Logistic Regression notebook.")
        p1, p2, p3, p4 = st.columns(4)
        pipeline_cards = [
            ("01", "Binary Cleanup", "Gender and OverTime are converted to binary values; four unused fields are dropped."),
            ("02", "Numeric Pipeline", "Median imputation followed by StandardScaler."),
            ("03", "Categorical Pipeline", "Most-frequent imputation followed by one-hot encoding."),
            ("04", "Logistic Regression", "Binary Attrition prediction with probability output."),
        ]
        for col, (num, title, desc) in zip([p1, p2, p3, p4], pipeline_cards):
            with col:
                st.markdown(
                    f"""
                    <div style="min-height:150px;padding:16px;background:rgba(255,255,255,0.02);
                    border:1px solid #263241;border-radius:12px;">
                        <div style="color:#38D9C5;font-size:0.60rem;font-weight:850;letter-spacing:1px;">STEP {num}</div>
                        <div style="color:#F8FAFC;font-size:0.90rem;font-weight:800;margin-top:12px;">{title}</div>
                        <div style="color:#94A3B8;font-size:0.69rem;line-height:1.55;margin-top:8px;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <div class="footer">
        NOVA &nbsp;•&nbsp; IBM HR Attrition Intelligence &nbsp;•&nbsp;
        Logistic Regression &nbsp;•&nbsp; Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)