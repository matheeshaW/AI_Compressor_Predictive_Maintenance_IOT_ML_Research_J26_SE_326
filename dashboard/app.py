"""
dashboard/app.py
Unified Industrial AI Compressor Platform Dashboard (J26-SE-326)
Master Application Coordinator
"""

import os
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Industrial AI Compressor Intelligence Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load theme CSS
THEME_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles", "theme.css")
if os.path.exists(THEME_CSS_PATH):
    with open(THEME_CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <h2 style="margin: 0; color: #00f2fe; font-size: 1.4rem;">J26-SE-326</h2>
        <p style="margin: 2px 0 0 0; color: #8a99ad; font-size: 0.85rem;">Industrial Compressor Intelligence</p>
    </div>
""", unsafe_allow_html=True)

# Telemetry mode toggle
st.sidebar.markdown("### 📡 Telemetry Mode")
telemetry_mode = st.sidebar.radio(
    "Source",
    ["Deterministic Replay (MetroPT-3)", "Live Serial Stream (ESP32-S3)"],
    index=0
)

# Component View Selector
st.sidebar.markdown("### 🧩 Intelligence Modules")
view_selection = st.sidebar.selectbox(
    "Select Component View",
    [
        "🧠 C02: AI Predictive Maintenance & RUL (Herath)",
        "⚡ C01: Edge Sensing & DSP Waveforms (Matheesha)",
        "🌐 C03: Industrial Digital Twin (Kahakotuwa)",
        "🎯 C04: RL Prescriptive Maintenance (Liyanage)"
    ],
    index=0
)

# Route to selected view
if "C02" in view_selection:
    from views.c02_predictive_view import render_c02_predictive_view
    render_c02_predictive_view()
else:
    st.info(f"Selected: {view_selection}. View under active development for group integration.")
