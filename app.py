import math
import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(
    page_title="Performance Index Predictor",
    page_icon="🎯",
    layout="centered"
)

# ── Theme tokens + CSS (Nocturne dark palette) ──────────────────
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #161826;
    --surface: #232532;
    --text: #E9E9ED;
    --muted: #9397AB;
    --divider: rgba(233,233,237,0.16);
    --accent: #9184D9;
    --accent-300: #D2CEFD;
    --accent-600: #796CBF;
    --accent-800: #423A6A;
    --accent-100: #F5F4FF;
    --accent2-800: #423E5D;
    --neutral-500: #9397AB;
    --neutral-800: #3F424D;
    --neutral-100: #F3F5FE;
}

.stApp {
    background: var(--bg);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* Header */
.app-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.6rem;
    color: var(--text);
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}
.app-subcaption {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0;
}

/* Section kicker labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
}

/* Card containers via Streamlit's key-scoped wrapper */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--divider);
    border-radius: 14px;
}
.st-key-hero div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        radial-gradient(120% 100% at 15% 0%, #2B2741 0%, #1C1E2E 45%, #161826 78%),
        linear-gradient(160deg, #262A60 0%, transparent 55%);
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}

/* Sliders */
[data-testid="stSlider"] label p {
    font-weight: 500;
    color: var(--text);
    font-size: 0.9rem;
}
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--bg) !important;
}
div[data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
}

/* Segmented radio (Yes/No) */
[data-testid="stRadio"] label p {
    font-weight: 500;
    font-size: 0.9rem;
}
[data-testid="stRadio"] > div {
    gap: 0 !important;
}

/* Feature bars */
.feat-row { margin-bottom: 12px; }
.feat-name {
    font-size: 0.8rem;
    color: var(--text);
    width: 170px;
    display: inline-block;
}
[data-testid="stExpander"] {
    border: 1px solid var(--divider);
    border-radius: 12px;
    background: var(--surface);
}
[data-testid="stExpander"] summary p {
    font-size: 0.85rem;
    font-weight: 500;
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
    <h1>🎯 Performance Index Predictor</h1>
    <p class="app-subcaption">Random Forest model trained on Student_Performance.csv &nbsp;·&nbsp; R² 0.986</p>
</div>
""")

st.write("")

# ── Hero placeholder (appears first visually; filled in after inputs are read) ──
hero_slot = st.container(border=True, key="hero")

st.write("")

# ── Inputs ───────────────────────────────────────────────────
with st.container(border=True, key="inputs"):
    st.markdown('<div class="section-label">Your Inputs</div>', unsafe_allow_html=True)
    hours_studied = st.slider("Hours Studied", 0, 10, 5)
    previous_scores = st.slider("Previous Scores", 0, 100, 70)
    sample_papers = st.slider("Sample Papers Practiced", 0, 10, 3)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

extracurricular_val = 1 if extracurricular == "Yes" else 0

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

if clamped >= 70:
    band_color = "#9184D9"
    tag_bg, tag_text = "#423A6A", "#F5F4FF"
    status_text = "Strong predicted performance."
elif clamped >= 40:
    band_color = "#796CBF"
    tag_bg, tag_text = "#423E5D", "#F5F4FF"
    status_text = "On track — moderate predicted performance."
else:
    band_color = "#9397AB"
    tag_bg, tag_text = "#3F424D", "#F3F5FE"
    status_text = "Predicted performance is low — may need support."

radius = 94
circumference = 2 * math.pi * radius
dash = (clamped / 100) * circumference
dasharray = f"{dash:.1f} {circumference:.1f}"

# ── Hero dial markup (rendered via iframe so SVG survives — st.html() strips it) ──
hero_html = f"""
<html>
<head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600&display=swap" rel="stylesheet">
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }}
    html, body {{ background: transparent; }}
</style>
</head>
<body>
<div style="text-align:center; padding:8px 0 4px 0;">
    <div style="font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
                color:#9397AB; text-align:left; margin-bottom:14px;">Predicted Performance Index</div>
    <div style="position:relative; width:220px; height:220px; margin:6px auto 0;">
        <svg width="220" height="220" viewBox="0 0 220 220" style="transform:rotate(-90deg);">
            <circle cx="110" cy="110" r="{radius}" fill="none" stroke="#292B31" stroke-width="16"></circle>
            <circle cx="110" cy="110" r="{radius}" fill="none" stroke="{band_color}"
                    stroke-width="16" stroke-linecap="round"
                    stroke-dasharray="{dasharray}"></circle>
        </svg>
        <div style="position:absolute; inset:0; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;">
            <div style="font-size:52px; font-weight:600; color:{band_color}; line-height:1;">{clamped:.1f}</div>
            <div style="font-size:11px; letter-spacing:0.06em; text-transform:uppercase;
                        color:#9397AB; margin-top:6px;">out of 100</div>
        </div>
    </div>
    <span style="display:inline-block; margin-top:14px; padding:6px 14px; border-radius:999px;
                 font-size:12.5px; background:{tag_bg}; color:{tag_text};">{status_text}</span>
</div>
</body>
</html>
"""

with hero_slot:
    components.html(hero_html, height=340, scrolling=False)

st.write("")

# ── Feature importance ──────────────────────────────────────────
with st.expander("How much does each feature matter?"):
    importances = model.feature_importances_
    feature_names = [
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Papers Practiced",
        "Extracurricular Activities"
    ]
    rows = ""
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        pct = imp * 100
        rows += f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
            <div class="feat-name">{name}</div>
            <div style="flex:1; height:6px; border-radius:999px; background:#292B31; overflow:hidden;">
                <div style="height:100%; border-radius:999px; background:#9184D9; width:{pct:.0f}%;"></div>
            </div>
            <div style="font-size:11.5px; color:#75798C; width:34px; text-align:right;">{pct:.0f}%</div>
        </div>
        """
    st.html(rows)

st.write("")
st.markdown(
    '<p style="font-size:12px; color:#75798C;">Model: RandomForestRegressor (scikit-learn) '
    '· Runs live on this server, not hardcoded.</p>',
    unsafe_allow_html=True
)
