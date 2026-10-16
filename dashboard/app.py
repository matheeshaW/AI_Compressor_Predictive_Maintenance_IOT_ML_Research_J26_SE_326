"""
dashboard/app.py
Unified Industrial AI Compressor Platform Dashboard (J26-SE-326)
Master Application Coordinator
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure dashboard directory and repo root are on sys.path
DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
        "⚡ C01: Edge Sensing & DSP Waveforms (Matheesha)",
        "🧠 C02: AI Predictive Maintenance & RUL (Herath)",
        "🌐 C03: Industrial Digital Twin (Kahakotuwa)",
        "🎯 C04: RL Prescriptive Maintenance (Liyanage)"
    ],
    index=0
)

# Route to selected view
if "C01" in view_selection:
    import pandas as pd
    from views.c01_edge_telemetry_view import render_c01_view
    from utils.telemetry_simulator import CompressorTelemetrySimulator
    if "simulator" not in st.session_state:
        st.session_state.simulator = CompressorTelemetrySimulator()
    pkt = st.session_state.simulator.step()
    render_c01_view(pkt, pd.DataFrame([pkt]))
elif "C02" in view_selection:
    from views.c02_predictive_view import render_c02_predictive_view
    render_c02_predictive_view()
elif "C03" in view_selection:
    from views.c03_digital_twin_view import render_c03_view
    render_c03_view()
elif "C04" in view_selection:
    import pandas as pd
    from views.c04_prescriptive_view import render_c04_view
    from utils.telemetry_simulator import CompressorTelemetrySimulator
    if "simulator" not in st.session_state:
        st.session_state.simulator = CompressorTelemetrySimulator()
    pkt = st.session_state.simulator.step()
    render_c04_view(pkt, pd.DataFrame([pkt]))
else:
    st.info(f"Selected: {view_selection}. View under active development for group integration.")