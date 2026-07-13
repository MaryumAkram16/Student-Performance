import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(
    page_title="Performance Index Predictor",
    page_icon="🎯",
    layout="centered"
)

# ── Theme tokens + CSS ──────────────────────────────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
    --ink: #1B1F2B;
    --muted: #6B7280;
    --primary: #1E2A47;
    --accent: #C98A1C;
    --good: #1E8E63;
    --warn: #C98A1C;
    --bad: #C0392B;
    --border: #E5E7EF;
    --surface: #FFFFFF;
    --bg: #F6F7FB;
}

.stApp {
    background: var(--bg);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2px;
}
.app-header .icon-badge {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    background: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}
.app-header h1 {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2rem;
    color: var(--primary);
    margin: 0;
    line-height: 1.1;
}
.app-subcaption {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 6px 0 0 62px;
}

/* Section labels */
.section-label {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

/* Card containers (targets Streamlit's bordered container) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(20, 24, 40, 0.04);
}

/* Sliders */
[data-testid="stSlider"] label p {
    font-weight: 500;
    color: var(--ink);
    font-size: 0.92rem;
}
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}
div[data-baseweb="slider"] > div > div {
    background: var(--primary) !important;
}

/* Radio */
[data-testid="stRadio"] label p {
    font-weight: 500;
    font-size: 0.92rem;
}

/* Result number */
.score-row {
    display: flex;
    align-items: baseline;
    gap: 18px;
    margin: 4px 0 2px 0;
}
.score-number {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 3.2rem;
    line-height: 1;
}
.grade-badge {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.4rem;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    flex-shrink: 0;
}
.score-caption {
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Progress bar recolor */
[data-testid="stProgress"] div[role="progressbar"] > div {
    background-color: var(--band-color, var(--primary)) !important;
}

/* Status message pill */
.status-pill {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 500;
    font-size: 0.9rem;
    margin-top: 10px;
}

/* Feature importance bars */
.feat-row {
    margin-bottom: 12px;
}
.feat-name {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--ink);
    margin-bottom: 4px;
}
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
}

footer, #MainMenu {visibility: hidden;}
</style>
""")


# ── Load the trained model (cached so it only loads once) ──────
@st.cache_resource
def load_model():
    return joblib.load("performance_index_rf_model.pkl")

model = load_model()

# ── Header ───────────────────────────────────────────────────
st.html("""
<div class="app-header">
    <div class="icon-badge">🎯</div>
    <h1>Performance Index Predictor</h1>
</div>
<div class="app-subcaption">Random Forest model trained on Student_Performance.csv &nbsp;·&nbsp; R² 0.986</div>
""")

st.write("")

# ── Inputs ───────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="section-label">Your Inputs</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider("Hours Studied", 0, 10, 5)
        previous_scores = st.slider("Previous Scores", 0, 100, 70)
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    with col2:
        sample_papers = st.slider("Sample Question Papers Practiced", 0, 10, 3)
        extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

extracurricular_val = 1 if extracurricular == "Yes" else 0

st.write("")

# ── Predict ──────────────────────────────────────────────────
features = pd.DataFrame([{
    "Hours Studied": hours_studied,
    "Previous Scores": previous_scores,
    "Sleep Hours": sleep_hours,
    "Sample Question Papers Practiced": sample_papers,
    "Extracurricular Activities_Yes": extracurricular_val
}])
prediction = model.predict(features)[0]
clamped = float(np.clip(prediction, 0, 100))


def grade_letter(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


if clamped >= 70:
    band_color = "#1E8E63"
    status_bg = "#E9F6EF"
    status_text = "Strong predicted performance."
elif clamped >= 40:
    band_color = "#C98A1C"
    status_bg = "#FBF1DE"
    status_text = "On track — moderate predicted performance."
else:
    band_color = "#C0392B"
    status_bg = "#FBE9E7"
    status_text = "Predicted performance is low — may need support."

# ── Result display ───────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="section-label">Predicted Performance Index</div>', unsafe_allow_html=True)
    st.html(f"""
    <div class="score-row">
        <div class="grade-badge" style="background:{band_color};">{grade_letter(clamped)}</div>
        <div>
            <div class="score-number" style="color:{band_color};">{clamped:.1f}</div>
            <div class="score-caption">out of 100</div>
        </div>
    </div>
    """)

    st.html(f'<style>:root {{ --band-color: {band_color}; }}</style>')
    st.progress(min(clamped / 100, 1.0))

    st.markdown(
        f'<div class="status-pill" style="background:{status_bg}; color:{band_color};">{status_text}</div>',
        unsafe_allow_html=True
    )

st.write("")

# ── Feature importance (optional extra insight) ────────────────
with st.expander("How much does each feature matter?"):
    importances = model.feature_importances_
    feature_names = [
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Papers Practiced",
        "Extracurricular Activities"
    ]
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        st.markdown(f'<div class="feat-name">{name}</div>', unsafe_allow_html=True)
        st.progress(float(imp))

st.write("")
st.caption("Model: RandomForestRegressor (scikit-learn) · Runs live on this server, not hardcoded.")