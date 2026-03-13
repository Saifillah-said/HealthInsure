"""
Health-InsurTech — Application Streamlit
Prédiction éthique des frais de santé
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
import hashlib
import os
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health-InsurTech",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LOG_FILE = "app_logs.log"

def setup_logger():
    logger = logging.getLogger("HealthInsurTech")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)
    return logger

logger = setup_logger()

def log_event(level: str, message: str, user: str = "anonymous"):
    full_msg = f"[user={user}] {message}"
    getattr(logger, level, logger.info)(full_msg)
    # keep last 200 lines of logs in session for admin view
    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = []
    st.session_state.log_buffer.append(
        f"{datetime.now().strftime('%H:%M:%S')} | {level.upper()} | {full_msg}"
    )
    if len(st.session_state.log_buffer) > 200:
        st.session_state.log_buffer.pop(0)

# ─────────────────────────────────────────────────────────────
# USERS DB (hashed passwords — demo only)
# ─────────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

USERS_DB = {
    "admin": {
        "password_hash": _hash("admin123"),
        "role": "admin",
        "full_name": "Administrateur",
    },
    "actuary": {
        "password_hash": _hash("actuary123"),
        "role": "actuary",
        "full_name": "Marie Dupont (Actuaire)",
    },
    "client": {
        "password_hash": _hash("client123"),
        "role": "client",
        "full_name": "Jean Martin (Client)",
    },
}

ROLE_PERMISSIONS = {
    "admin":    ["dashboard", "simulator", "bias_audit", "logs", "admin_panel"],
    "actuary":  ["dashboard", "simulator", "bias_audit"],
    "client":   ["simulator"],
}

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Root ── */
    :root {
        --navy:   #0d1b2a;
        --teal:   #00b4d8;
        --teal2:  #0077b6;
        --mint:   #06d6a0;
        --gold:   #ffd166;
        --coral:  #ef476f;
        --card:   #132338;
        --border: rgba(0,180,216,0.18);
        --text:   #e8f4fd;
        --muted:  #8aa8c0;
        --radius: 14px;
    }

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif !important;
        background-color: var(--navy) !important;
        color: var(--text) !important;
    }

    /* ── Header band ── */
    .app-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a2744 60%, #00374f 100%);
        border-bottom: 2px solid var(--teal);
        padding: 1.4rem 2rem 1.2rem;
        margin: -1rem -1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .app-header h1 {
        font-size: 1.7rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header .badge {
        background: var(--teal);
        color: var(--navy);
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Cards ── */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--teal);
        margin-bottom: 0.4rem;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
    }
    .card-sub {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.15rem;
    }

    /* ── Result box ── */
    .result-box {
        background: linear-gradient(135deg, #003f5c 0%, #0a2744 100%);
        border: 2px solid var(--teal);
        border-radius: var(--radius);
        padding: 1.8rem 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(0,180,216,0.15);
    }
    .result-label { font-size: 0.75rem; letter-spacing: 2px; color: var(--teal); text-transform: uppercase; }
    .result-amount {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: var(--mint);
        line-height: 1.1;
    }
    .result-range { font-size: 0.85rem; color: var(--muted); margin-top: 0.3rem; }

    /* ── RGPD banner ── */
    .rgpd-overlay {
        position: fixed; inset: 0;
        background: rgba(5,15,25,0.92);
        backdrop-filter: blur(6px);
        z-index: 9999;
        display: flex; align-items: center; justify-content: center;
    }
    .rgpd-box {
        background: var(--card);
        border: 2px solid var(--teal);
        border-radius: 18px;
        padding: 2.5rem 3rem;
        max-width: 580px;
        box-shadow: 0 8px 60px rgba(0,0,0,0.6);
    }

    /* ── Login ── */
    .login-wrap {
        max-width: 420px;
        margin: 6vh auto 0;
    }

    /* ── Role badge ── */
    .role-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .role-admin   { background: var(--coral); color: #fff; }
    .role-actuary { background: var(--gold);  color: var(--navy); }
    .role-client  { background: var(--mint);  color: var(--navy); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0a1929 !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: var(--muted) !important; font-size: 0.82rem; }

    /* ── Inputs ── */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: #0d2035 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }
    .stSlider [data-baseweb="slider"] { padding: 0.3rem 0; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--teal2), var(--teal)) !important;
        color: var(--navy) !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 0.55rem 1.4rem !important;
        font-family: 'Sora', sans-serif !important;
        letter-spacing: 0.5px;
        transition: all .2s;
    }
    .stButton > button:hover { filter: brightness(1.12) !important; transform: translateY(-1px); }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 600;
        border-radius: 8px 8px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--card) !important;
        color: var(--teal) !important;
        border-bottom: 2px solid var(--teal) !important;
    }

    /* ── Metric ── */
    [data-testid="metric-container"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.8rem 1rem;
    }
    [data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--teal) !important; font-family: 'JetBrains Mono', monospace !important; }

    /* ── Log viewer ── */
    .log-viewer {
        background: #060f18;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #7ec8e3;
        max-height: 320px;
        overflow-y: auto;
        line-height: 1.7;
    }

    /* ── Alerts ── */
    .alert-success { background: rgba(6,214,160,0.12); border-left: 3px solid var(--mint); padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; color: var(--mint); font-size: 0.85rem; }
    .alert-warn    { background: rgba(255,209,102,0.12); border-left: 3px solid var(--gold); padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; color: var(--gold); font-size: 0.85rem; }
    .alert-info    { background: rgba(0,180,216,0.10); border-left: 3px solid var(--teal); padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; color: var(--teal); font-size: 0.85rem; }

    /* ── Plotly backgrounds ── */
    .js-plotly-plot .plotly { border-radius: var(--radius); overflow: hidden; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .result-amount { font-size: 2.2rem; }
        .app-header h1 { font-size: 1.2rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "rgpd_accepted": False,
        "authenticated": False,
        "username": None,
        "role": None,
        "full_name": None,
        "active_page": "simulator",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try to find the CSV
    for path in ["data/raw/insurance_data.csv", "/mnt/user-data/uploads/insurance_data.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Keep only needed columns
            keep = ["age", "bmi", "children", "smoker", "region", "sex", "charges"]
            df = df[keep].dropna()
            return df
    # Fallback: generate synthetic data
    np.random.seed(42)
    n = 1338
    ages = np.random.randint(18, 65, n)
    bmis = np.random.normal(30, 6, n).clip(15, 55)
    children = np.random.randint(0, 6, n)
    smoker = np.random.choice(["yes", "no"], n, p=[0.2, 0.8])
    region = np.random.choice(["southwest", "southeast", "northwest", "northeast"], n)
    sex = np.random.choice(["male", "female"], n)
    charges = (ages * 250 + bmis * 300 + children * 500
               + (smoker == "yes") * 20000
               + np.random.normal(0, 2000, n)).clip(1000)
    return pd.DataFrame(dict(age=ages, bmi=bmis, children=children,
                              smoker=smoker, region=region, sex=sex, charges=charges))

@st.cache_resource
def train_model(df: pd.DataFrame):
    df_enc = pd.get_dummies(df, columns=["smoker", "region", "sex"], drop_first=False)
    for col in ["smoker_no", "sex_male", "region_southwest"]:
        if col in df_enc.columns:
            df_enc.drop(columns=[col], inplace=True)
    feature_cols = [c for c in df_enc.columns if c != "charges"]
    X = df_enc[feature_cols]
    y = df_enc["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_sc, y_train)
    # Calibration offsets
    train_pred = ridge.predict(X_tr_sc)
    X_train_df = X_train.copy()
    X_train_df["err"] = train_pred - y_train.values
    X_train_df["smoker_int"] = X_train["smoker_yes"].astype(int) if "smoker_yes" in X_train.columns else 0
    offsets = X_train_df.groupby("smoker_int")["err"].mean().to_dict()
    mae = mean_absolute_error(y_test, ridge.predict(X_te_sc))
    r2  = r2_score(y_test, ridge.predict(X_te_sc))
    return ridge, scaler, feature_cols, offsets, {"mae": mae, "r2": r2, "n_train": len(X_train)}

def predict_charges(ridge, scaler, feature_cols, offsets,
                    age, bmi, children, smoker, region, sex):
    row = {f: 0 for f in feature_cols}
    row["age"] = age
    row["bmi"] = bmi
    row["children"] = children
    if "smoker_yes" in row and smoker == "yes":
        row["smoker_yes"] = 1
    for r in ["northeast", "northwest", "southeast"]:
        key = f"region_{r}"
        if key in row:
            row[key] = 1 if region == r else 0
    if "sex_female" in row and sex == "female":
        row["sex_female"] = 1
    X_in = pd.DataFrame([row])[feature_cols]
    X_sc = scaler.transform(X_in)
    pred = ridge.predict(X_sc)[0]
    smoker_int = 1 if smoker == "yes" else 0
    pred_cal = pred - offsets.get(smoker_int, 0)
    return max(500, pred_cal)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(13,27,42,0)",
    plot_bgcolor="rgba(13,27,42,0)",
    font_family="Sora",
    font_color="#e8f4fd",
    colorway=["#00b4d8", "#06d6a0", "#ffd166", "#ef476f", "#8172B2"],
    margin=dict(t=50, b=30, l=20, r=20),
)

# ─────────────────────────────────────────────────────────────
# RGPD CONSENT
# ─────────────────────────────────────────────────────────────
def rgpd_screen():
    st.markdown("""
    <div style='
        max-width:560px; margin: 8vh auto 0;
        background: #132338;
        border: 2px solid #00b4d8;
        border-radius: 18px;
        padding: 2.5rem 3rem;
        box-shadow: 0 8px 60px rgba(0,0,0,0.6);
    '>
        <div style='font-size:2.2rem; margin-bottom:0.6rem;'>🔒</div>
        <h2 style='color:#00b4d8; font-size:1.35rem; margin-bottom:0.6rem;'>Confidentialité & Données de Santé</h2>
        <p style='color:#8aa8c0; font-size:0.88rem; line-height:1.7; margin-bottom:1.2rem;'>
            Cette application collecte des <strong style='color:#e8f4fd;'>données de santé sensibles</strong>
            (IMC, statut fumeur) conformément au <strong style='color:#e8f4fd;'>RGPD (Art. 9)</strong>
            et à la directive ePrivacy.
        </p>
        <div style='background:#0d2035; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1.2rem; font-size:0.82rem; line-height:1.9; color:#8aa8c0;'>
            ✅ &nbsp;Données <strong style='color:#e8f4fd'>non stockées</strong> sur nos serveurs<br>
            ✅ &nbsp;Calcul <strong style='color:#e8f4fd'>local et éphémère</strong> — effacé à la fermeture<br>
            ✅ &nbsp;Aucune transmission à des tiers publicitaires<br>
            ✅ &nbsp;Droit d'accès, de rectification et d'effacement (Art. 17)
        </div>
        <p style='color:#8aa8c0; font-size:0.78rem; margin-bottom:1.4rem;'>
            En cliquant sur <em>Accepter</em>, vous consentez au traitement de vos données
            de santé à des fins de simulation tarifaire uniquement.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("✅ Accepter", use_container_width=True):
            st.session_state.rgpd_accepted = True
            log_event("info", "RGPD consent accepted")
            st.rerun()
    with col3:
        if st.button("❌ Refuser", use_container_width=True):
            st.markdown("<p style='text-align:center;color:#ef476f;margin-top:1rem;'>Accès refusé. Fermez cet onglet.</p>", unsafe_allow_html=True)
            st.stop()

# ─────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────
def login_page():
    st.markdown("""
    <div style='text-align:center; margin: 6vh auto 2rem;'>
        <div style='font-size:3rem;'>🏥</div>
        <h1 style='font-size:1.8rem; color:#fff; margin:0.3rem 0 0.2rem;'>Health-InsurTech</h1>
        <p style='color:#8aa8c0; font-size:0.88rem;'>Plateforme de simulation tarifaire éthique</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1.2, 1])
    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>Connexion sécurisée</p>", unsafe_allow_html=True)
        username = st.text_input("Identifiant", placeholder="ex: admin")
        password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Se connecter →", use_container_width=True):
            user = USERS_DB.get(username)
            if user and user["password_hash"] == _hash(password):
                st.session_state.authenticated = True
                st.session_state.username  = username
                st.session_state.role      = user["role"]
                st.session_state.full_name = user["full_name"]
                # Default page per role
                perms = ROLE_PERMISSIONS[user["role"]]
                st.session_state.active_page = perms[0]
                log_event("info", f"Login successful, role={user['role']}", username)
                st.rerun()
            else:
                log_event("warning", f"Failed login attempt for username='{username}'")
                st.error("Identifiant ou mot de passe incorrect.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; margin-top:1rem;'>
        <p style='color:#8aa8c0; font-size:0.75rem;'>
        Comptes de démonstration :<br>
        <code>admin / admin123</code> &nbsp;·&nbsp;
        <code>actuary / actuary123</code> &nbsp;·&nbsp;
        <code>client / client123</code>
        </p></div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    role  = st.session_state.role
    perms = ROLE_PERMISSIONS[role]
    role_class = f"role-{role}"

    with st.sidebar:
        st.markdown(f"""
        <div style='padding: 0.8rem 0 1.2rem;'>
            <div style='font-size:1.05rem; font-weight:700; color:#fff; margin-bottom:0.3rem;'>
                {st.session_state.full_name}
            </div>
            <span class='role-pill {role_class}'>{role}</span>
        </div>
        <hr style='margin-bottom:1rem;'>
        """, unsafe_allow_html=True)

        nav_items = {
            "simulator":   ("🧮", "Simulateur Tarifaire"),
            "dashboard":   ("📊", "Dashboard Analytique"),
            "bias_audit":  ("⚖️", "Audit des Biais"),
            "logs":        ("📋", "Journaux d'Accès"),
            "admin_panel": ("🛡️", "Administration"),
        }

        for page_key, (icon, label) in nav_items.items():
            if page_key in perms:
                is_active = st.session_state.active_page == page_key
                btn_style = "background: rgba(0,180,216,0.15); border: 1px solid rgba(0,180,216,0.3);" if is_active else ""
                if st.button(
                    f"{icon} {label}",
                    key=f"nav_{page_key}",
                    use_container_width=True,
                ):
                    st.session_state.active_page = page_key
                    log_event("info", f"Navigated to {page_key}", st.session_state.username)
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem; color:#8aa8c0;'>🔒 Session sécurisée<br>Données non conservées</p>",
                    unsafe_allow_html=True)
        if st.button("⬅️ Déconnexion", use_container_width=True):
            log_event("info", "User logged out", st.session_state.username)
            for k in ["authenticated", "username", "role", "full_name", "active_page"]:
                st.session_state[k] = None if k != "authenticated" else False
            st.session_state.active_page = "simulator"
            st.rerun()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
def render_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class='app-header'>
        <span style='font-size:1.6rem;'>🏥</span>
        <div>
            <h1>{title}</h1>
            {f"<p style='color:#8aa8c0;font-size:0.82rem;margin:0;'>{subtitle}</p>" if subtitle else ""}
        </div>
        <div style='margin-left:auto;'>
            <span class='badge'>Beta</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: SIMULATOR
# ─────────────────────────────────────────────────────────────
def page_simulator(df, ridge, scaler, feature_cols, offsets, metrics):
    render_header("Simulateur Tarifaire", "Estimez vos frais médicaux annuels en temps réel")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>🧬 Vos informations de santé</p>", unsafe_allow_html=True)

        age      = st.slider("Âge", 18, 80, 35, help="Votre âge en années")
        bmi      = st.slider("IMC (Indice de Masse Corporelle)", 15.0, 55.0, 27.0, step=0.1,
                             help="Calculé depuis votre taille et poids")

        # BMI category indicator
        if bmi < 18.5:
            bmi_label, bmi_color = "Sous-poids", "#ffd166"
        elif bmi < 25:
            bmi_label, bmi_color = "Normal", "#06d6a0"
        elif bmi < 30:
            bmi_label, bmi_color = "Surpoids", "#ffd166"
        else:
            bmi_label, bmi_color = "Obésité", "#ef476f"

        st.markdown(f"<span style='font-size:0.78rem; color:{bmi_color};'>▶ {bmi_label}</span>",
                    unsafe_allow_html=True)

        children = st.selectbox("Nombre d'enfants à charge", [0, 1, 2, 3, 4, 5])
        smoker   = st.radio("Statut tabagique", ["Non-fumeur", "Fumeur"],
                            horizontal=True, help="Donnée de santé sensible — utilisée uniquement pour l'estimation")
        sex      = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        region   = st.selectbox("Région d'assurance", ["southwest", "southeast", "northwest", "northeast"])

        st.markdown("<br>", unsafe_allow_html=True)
        simulate = st.button("🔮 Calculer mon estimation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='alert-info' style='margin-top:0.5rem;'>
        ℹ️ Cette simulation est indicative et non contractuelle. Les données saisies ne sont pas conservées.
        </div>""", unsafe_allow_html=True)

    with col_result:
        smoker_val  = "yes" if smoker == "Fumeur" else "no"
        sex_val     = "female" if sex == "Femme" else "male"
        estimation  = predict_charges(ridge, scaler, feature_cols, offsets,
                                       age, bmi, children, smoker_val, region, sex_val)
        monthly     = estimation / 12
        low, high   = estimation * 0.88, estimation * 1.12

        if simulate:
            log_event("info",
                      f"Simulation: age={age}, bmi={bmi:.1f}, children={children}, "
                      f"smoker={smoker_val}, region={region}, result={estimation:.0f}€",
                      st.session_state.username)

        st.markdown(f"""
        <div class='result-box'>
            <div class='result-label'>Estimation annuelle</div>
            <div class='result-amount'>{estimation:,.0f} €</div>
            <div class='result-range'>Fourchette probable : {low:,.0f} € — {high:,.0f} €</div>
            <hr style='border-color:rgba(0,180,216,0.2); margin: 1rem 0;'>
            <div style='display:flex; justify-content:center; gap:2rem;'>
                <div>
                    <div style='font-size:0.7rem; color:#8aa8c0; letter-spacing:1px; text-transform:uppercase;'>/ mois</div>
                    <div style='font-family:JetBrains Mono, monospace; font-size:1.4rem; color:#ffd166; font-weight:700;'>{monthly:,.0f} €</div>
                </div>
                <div>
                    <div style='font-size:0.7rem; color:#8aa8c0; letter-spacing:1px; text-transform:uppercase;'>R² modèle</div>
                    <div style='font-family:JetBrains Mono, monospace; font-size:1.4rem; color:#00b4d8; font-weight:700;'>{metrics["r2"]:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge
        st.markdown("<br>", unsafe_allow_html=True)
        df_q = df["charges"].quantile([0.25, 0.5, 0.75, 0.9])
        risk_pct = min(100, int((estimation / df["charges"].max()) * 100))

        if estimation < df_q[0.25]:
            risk_label, risk_color = "Risque faible", "#06d6a0"
        elif estimation < df_q[0.5]:
            risk_label, risk_color = "Risque modéré", "#ffd166"
        elif estimation < df_q[0.75]:
            risk_label, risk_color = "Risque élevé", "#DD8452"
        else:
            risk_label, risk_color = "Risque très élevé", "#ef476f"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": f"<span style='font-size:0.85rem'>{risk_label}</span>", "font": {"color": "#e8f4fd"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8aa8c0"},
                "bar": {"color": risk_color},
                "bgcolor": "#0d1b2a",
                "steps": [
                    {"range": [0,  25], "color": "rgba(6,214,160,0.15)"},
                    {"range": [25, 50], "color": "rgba(255,209,102,0.12)"},
                    {"range": [50, 75], "color": "rgba(221,132,82,0.12)"},
                    {"range": [75, 100],"color": "rgba(239,71,111,0.15)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "value": risk_pct},
            },
            number={"suffix": "%", "font": {"color": risk_color, "family": "JetBrains Mono"}},
        ))
        fig_gauge.update_layout(**PLOT_LAYOUT, height=220, margin=dict(t=40, b=0, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key drivers bar
        st.markdown("<div class='card' style='margin-top:0.5rem;'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>📌 Facteurs principaux</p>", unsafe_allow_html=True)
        drivers = {
            "Âge": age / 80,
            "IMC": (bmi - 15) / 40,
            "Tabac": 1.0 if smoker_val == "yes" else 0.05,
            "Enfants": children / 5,
        }
        for label, val in drivers.items():
            pct = int(val * 100)
            color = "#ef476f" if label == "Tabac" and smoker_val == "yes" else "#00b4d8"
            st.markdown(f"""
            <div style='margin-bottom:0.5rem;'>
                <div style='display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:2px;'>
                    <span>{label}</span><span style='color:{color};'>{pct}%</span>
                </div>
                <div style='background:#0d2035; border-radius:4px; height:6px;'>
                    <div style='width:{pct}%; background:{color}; height:6px; border-radius:4px; transition:width .4s;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────
def page_dashboard(df):
    render_header("Dashboard Analytique", "Corrélations IMC, âge et frais médicaux")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='card'><div class='card-title'>Clients</div>
        <div class='card-value'>{len(df):,}</div><div class='card-sub'>dans le dataset</div></div>""",
        unsafe_allow_html=True)
    with k2:
        avg = df["charges"].mean()
        st.markdown(f"""<div class='card'><div class='card-title'>Frais moyens</div>
        <div class='card-value'>{avg:,.0f}€</div><div class='card-sub'>par an</div></div>""",
        unsafe_allow_html=True)
    with k3:
        pct_smokers = (df["smoker"] == "yes").mean() * 100
        st.markdown(f"""<div class='card'><div class='card-title'>Fumeurs</div>
        <div class='card-value'>{pct_smokers:.1f}%</div><div class='card-sub'>de la clientèle</div></div>""",
        unsafe_allow_html=True)
    with k4:
        avg_bmi = df["bmi"].mean()
        st.markdown(f"""<div class='card'><div class='card-title'>IMC moyen</div>
        <div class='card-value'>{avg_bmi:.1f}</div><div class='card-sub'>surpoids</div></div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 IMC vs Frais", "🎂 Âge vs Frais", "🌍 Par région", "📦 Distributions"])

    with tab1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig = px.scatter(
                df, x="bmi", y="charges",
                color="smoker",
                color_discrete_map={"yes": "#ef476f", "no": "#00b4d8"},
                size="charges", size_max=14,
                opacity=0.65,
                labels={"bmi": "IMC", "charges": "Frais annuels (€)", "smoker": "Fumeur"},
                title="Corrélation IMC ↔ Frais médicaux",
            )
            # Trendlines manuelles numpy — sans statsmodels
            for _sv, _col in [("yes", "#ef476f"), ("no", "#00b4d8")]:
                _sub = df[df["smoker"] == _sv].sort_values("bmi")
                if len(_sub) > 2:
                    _z = np.polyfit(_sub["bmi"], _sub["charges"], 1)
                    _xl = np.linspace(_sub["bmi"].min(), _sub["bmi"].max(), 100)
                    fig.add_trace(go.Scatter(x=_xl, y=np.poly1d(_z)(_xl), mode="lines",
                        line=dict(color=_col, width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip"))
            fig.update_layout(**PLOT_LAYOUT, height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            # Correlation heatmap
            corr = df[["age", "bmi", "children", "charges"]].corr()
            fig_h = px.imshow(
                corr,
                color_continuous_scale=[[0, "#0d1b2a"], [0.5, "#0077b6"], [1, "#06d6a0"]],
                text_auto=".2f",
                title="Matrice de corrélation",
            )
            fig_h.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(fig_h, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig = px.scatter(
                df, x="age", y="charges",
                color="smoker",
                color_discrete_map={"yes": "#ef476f", "no": "#00b4d8"},
                facet_col="sex",
                opacity=0.55,
                labels={"age": "Âge", "charges": "Frais (€)", "smoker": "Fumeur", "sex": "Sexe"},
                title="Âge ↔ Frais par sexe",
            )
            # Trendlines manuelles par sexe × fumeur
            for _sex in ["male", "female"]:
                for _sv, _col in [("yes", "#ef476f"), ("no", "#00b4d8")]:
                    _sub = df[(df["smoker"] == _sv) & (df["sex"] == _sex)].sort_values("age")
                    if len(_sub) > 2:
                        _z = np.polyfit(_sub["age"], _sub["charges"], 1)
                        _xl = np.linspace(_sub["age"].min(), _sub["age"].max(), 80)
                        fig.add_trace(go.Scatter(x=_xl, y=np.poly1d(_z)(_xl), mode="lines",
                            line=dict(color=_col, width=1.8, dash="dot"),
                            showlegend=False, hoverinfo="skip"))
            fig.update_layout(**PLOT_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            # Age buckets
            df2 = df.copy()
            df2["tranche"] = pd.cut(df2["age"], bins=[17,25,35,45,55,65,80],
                                     labels=["18-25","26-35","36-45","46-55","56-65","65+"])
            age_avg = df2.groupby(["tranche", "smoker"])["charges"].mean().reset_index()
            fig2 = px.bar(age_avg, x="tranche", y="charges", color="smoker",
                          barmode="group",
                          color_discrete_map={"yes": "#ef476f", "no": "#00b4d8"},
                          labels={"tranche": "Tranche d'âge", "charges": "Frais moyens (€)"},
                          title="Frais moyens par tranche d'âge")
            fig2.update_layout(**PLOT_LAYOUT, height=380)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            reg_avg = df.groupby("region")["charges"].mean().reset_index()
            fig = px.bar(reg_avg, x="region", y="charges",
                         color="charges",
                         color_continuous_scale=["#0077b6", "#00b4d8", "#06d6a0"],
                         labels={"region": "Région", "charges": "Frais moyens (€)"},
                         title="Frais moyens par région")
            fig.update_layout(**PLOT_LAYOUT, height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.box(df, x="region", y="charges", color="smoker",
                          color_discrete_map={"yes": "#ef476f", "no": "#00b4d8"},
                          labels={"region": "Région", "charges": "Frais (€)"},
                          title="Distribution par région & tabac")
            fig2.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df, x="charges", nbins=50,
                               color="smoker",
                               color_discrete_map={"yes": "#ef476f", "no": "#00b4d8"},
                               barmode="overlay", opacity=0.7,
                               labels={"charges": "Frais annuels (€)"},
                               title="Distribution des frais médicaux")
            fig.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.violin(df, y="charges", x="sex", color="smoker",
                             box=True, points="outliers",
                             color_discrete_map={"yes": "#ef476f", "no": "#00b4d8"},
                             labels={"sex": "Sexe", "charges": "Frais (€)"},
                             title="Violin — Frais par sexe & tabac")
            fig2.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: BIAS AUDIT
# ─────────────────────────────────────────────────────────────
def page_bias_audit(df, ridge, scaler, feature_cols, offsets):
    render_header("Audit des Biais", "Analyse d'équité du modèle prédictif")

    df_enc = pd.get_dummies(df, columns=["smoker", "region", "sex"], drop_first=False)
    for col in ["smoker_no", "sex_male", "region_southwest"]:
        if col in df_enc.columns:
            df_enc.drop(columns=[col], inplace=True)
    fc2 = [c for c in df_enc.columns if c != "charges"]
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(df_enc[fc2], df_enc["charges"], test_size=0.2, random_state=42)
    from sklearn.preprocessing import StandardScaler
    sc2 = StandardScaler()
    _, X_te_sc = (lambda Xtr, Xte: (sc2.fit(Xtr), sc2.transform(Xte)))(
        df_enc[fc2].iloc[:int(len(df_enc)*0.8)], X_test
    )
    y_pred = ridge.predict(X_te_sc)
    df_t = X_test.copy()
    df_t["charges_reel"] = y_test.values
    df_t["prediction"]   = y_pred
    df_t["erreur"]       = y_pred - y_test.values
    df_t["Fumeur"]       = df_t["smoker_yes"].astype(int).map({1: "Fumeur", 0: "Non-fumeur"}) if "smoker_yes" in df_t.columns else "Non-fumeur"
    df_t["Sexe"]         = df_t["sex_female"].astype(int).map({1: "Femme", 0: "Homme"}) if "sex_female" in df_t.columns else "Homme"
    def get_region(row):
        for r in ["northeast", "northwest", "southeast"]:
            if f"region_{r}" in row.index and row[f"region_{r}"] == 1:
                return r
        return "southwest"
    df_t["Region"] = df_t.apply(get_region, axis=1)

    col_a, col_b = st.columns(2)
    with col_a:
        bias_s = df_t.groupby("Fumeur").apply(
            lambda g: (g["erreur"] / g["charges_reel"]).mean() * 100
        ).reset_index().rename(columns={0: "biais_pct"})
        fig = px.bar(bias_s, x="Fumeur", y="biais_pct",
                     color="biais_pct",
                     color_continuous_scale=["#06d6a0", "#ffd166", "#ef476f"],
                     labels={"biais_pct": "Biais (%)"},
                     title="Biais moyen par statut fumeur")
        fig.add_hline(y=0, line_color="#fff", line_dash="dash", opacity=0.4)
        fig.add_hrect(y0=-15, y1=15, fillcolor="rgba(6,214,160,0.05)",
                      line_width=0, annotation_text="Zone acceptable (±15%)")
        fig.update_layout(**PLOT_LAYOUT, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        bias_r = df_t.groupby("Region").apply(
            lambda g: (g["erreur"] / g["charges_reel"]).mean() * 100
        ).reset_index().rename(columns={0: "biais_pct"})
        fig2 = px.bar(bias_r, x="Region", y="biais_pct",
                      color="biais_pct",
                      color_continuous_scale=["#06d6a0", "#ffd166", "#ef476f"],
                      labels={"biais_pct": "Biais (%)"},
                      title="Biais moyen par région")
        fig2.add_hline(y=0, line_color="#fff", line_dash="dash", opacity=0.4)
        fig2.update_layout(**PLOT_LAYOUT, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = px.box(df_t, x="Fumeur", y="erreur", color="Fumeur",
                      color_discrete_map={"Fumeur": "#ef476f", "Non-fumeur": "#00b4d8"},
                      title="Distribution erreurs par statut fumeur",
                      labels={"erreur": "Erreur prédiction (€)"})
        fig3.add_hline(y=0, line_color="#fff", line_dash="dash", opacity=0.4)
        fig3.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig3, use_container_width=True)
    with col_d:
        fig4 = px.scatter(df_t, x="charges_reel", y="prediction",
                          color="Fumeur",
                          color_discrete_map={"Fumeur": "#ef476f", "Non-fumeur": "#00b4d8"},
                          opacity=0.5, size_max=8,
                          labels={"charges_reel": "Réel (€)", "prediction": "Prédit (€)"},
                          title="Réel vs Prédit — par groupe fumeur")
        rng = [df_t["charges_reel"].min(), df_t["charges_reel"].max()]
        fig4.add_shape(type="line", x0=rng[0], x1=rng[1], y0=rng[0], y1=rng[1],
                       line=dict(color="#fff", dash="dash", width=1.5))
        fig4.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class='alert-warn'>
    ⚖️ &nbsp;<strong>Interprétation</strong> : Un biais positif signifie que le modèle sur-estime les frais (pénalisation). 
    La zone acceptable est ±15%. Un biais structurel doit être corrigé par calibration ou repondération.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: LOGS
# ─────────────────────────────────────────────────────────────
def page_logs():
    render_header("Journaux d'Accès", "Traçabilité des actions utilisateurs")

    logs = st.session_state.get("log_buffer", [])

    col1, col2 = st.columns([3, 1])
    with col1:
        filter_level = st.selectbox("Filtrer par niveau", ["Tous", "INFO", "WARNING", "ERROR"])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Rafraîchir"):
            st.rerun()

    filtered = logs if filter_level == "Tous" else [l for l in logs if filter_level in l]
    log_html = "<br>".join(
        f"<span style='color:{'#06d6a0' if 'INFO' in l else '#ffd166' if 'WARNING' in l else '#ef476f'}'>{l}</span>"
        for l in reversed(filtered[-80:])
    ) or "<span style='color:#8aa8c0'>Aucun log disponible.</span>"

    st.markdown(f"<div class='log-viewer'>{log_html}</div>", unsafe_allow_html=True)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("⬇️ Télécharger les logs complets", f,
                               file_name="healthinsurtech_logs.log", mime="text/plain")

# ─────────────────────────────────────────────────────────────
# PAGE: ADMIN
# ─────────────────────────────────────────────────────────────
def page_admin():
    render_header("Administration", "Gestion des accès et des utilisateurs")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='card-title'>👥 Utilisateurs enregistrés</p>", unsafe_allow_html=True)
    users_df = pd.DataFrame([
        {"Identifiant": u, "Nom": d["full_name"], "Rôle": d["role"],
         "Permissions": ", ".join(ROLE_PERMISSIONS[d["role"]])}
        for u, d in USERS_DB.items()
    ])
    st.dataframe(users_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>🔐 Matrice des permissions</p>", unsafe_allow_html=True)
        all_perms = ["simulator", "dashboard", "bias_audit", "logs", "admin_panel"]
        matrix = {}
        for role, perms in ROLE_PERMISSIONS.items():
            matrix[role] = {p: ("✅" if p in perms else "❌") for p in all_perms}
        perm_df = pd.DataFrame(matrix).T
        st.dataframe(perm_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>⚙️ Paramètres système</p>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.82rem; line-height:2; color:#8aa8c0;'>
        🔒 Authentification &nbsp;&nbsp;<span style='color:#06d6a0;'>Active</span><br>
        📋 Journalisation &nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#06d6a0;'>Active</span><br>
        🍪 Consentement RGPD &nbsp;<span style='color:#06d6a0;'>Activé</span><br>
        🔐 Chiffrement mots de passe &nbsp;<span style='color:#06d6a0;'>SHA-256</span><br>
        📦 Version modèle &nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#00b4d8;'>Ridge v1.2</span><br>
        🗓️ Dernier retrainage &nbsp;<span style='color:#ffd166;'>Simulation (live)</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    init_session()
    inject_css()

    df = load_data()
    ridge, scaler, feature_cols, offsets, metrics = train_model(df)

    # ── 1. RGPD ──
    if not st.session_state.rgpd_accepted:
        rgpd_screen()
        return

    # ── 2. AUTH ──
    if not st.session_state.authenticated:
        login_page()
        return

    # ── 3. APP ──
    render_sidebar()
    page = st.session_state.active_page
    perms = ROLE_PERMISSIONS[st.session_state.role]

    if page not in perms:
        st.error("🚫 Accès refusé. Vous n'avez pas les droits nécessaires.")
        return

    if page == "simulator":
        page_simulator(df, ridge, scaler, feature_cols, offsets, metrics)
    elif page == "dashboard":
        page_dashboard(df)
    elif page == "bias_audit":
        page_bias_audit(df, ridge, scaler, feature_cols, offsets)
    elif page == "logs":
        page_logs()
    elif page == "admin_panel":
        page_admin()

if __name__ == "__main__":
    main()
