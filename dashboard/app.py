#!/usr/bin/env python3
"""
app.py
======
Industrial AI Compressor Intelligence Platform — Next-Gen Unified Dashboard
SLIIT 4th Year Research Project: J26-SE-326
"""

import os
import sys
import time
import pandas as pd
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="AIR-GUARD AI | Industrial Compressor Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure dashboard root is in sys.path
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from utils.telemetry_simulator import CompressorTelemetrySimulator
from utils.serial_stream_client import SerialStreamClient
from components.header import render_header
from components.alert_banner import render_alert_banner
from views.c01_edge_telemetry_view import render_c01_view
from views.c02_predictive_view import render_c02_view
from views.c03_digital_twin_view import render_c03_view
from views.c04_prescriptive_view import render_c04_view

# Load and inject custom dark glassmorphism CSS
THEME_CSS_PATH = os.path.join(DASHBOARD_DIR, "styles", "theme.css")
if os.path.exists(THEME_CSS_PATH):
    with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "simulator" not in st.session_state:
    st.session_state.simulator = CompressorTelemetrySimulator()
if "serial_client" not in st.session_state:
    st.session_state.serial_client = SerialStreamClient()
if "history" not in st.session_state:
    st.session_state.history = []
if "is_running" not in st.session_state:
    st.session_state.is_running = True
if "telemetry_mode" not in st.session_state:
    st.session_state.telemetry_mode = "Simulation Replay"


# ==============================================================================
# SIDEBAR CONTROLS & FAULT INJECTION RIG
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Mission Control")
    st.markdown("Configure operational inputs and trigger physical fault injections for live viva demonstrations.")

    # 1. Telemetry Source Toggle
    st.markdown("##### 🔌 Telemetry Source")
    mode = st.radio(
        "Select Active Data Pipeline:",
        ["Simulation Replay", "Live Serial (ESP32-S3)"],
        index=0 if st.session_state.telemetry_mode == "Simulation Replay" else 1
    )
    st.session_state.telemetry_mode = mode

    if mode == "Live Serial (ESP32-S3)":
        ports = st.session_state.serial_client.list_available_ports()
        selected_port = st.selectbox("Detected USB Serial COM Ports:", ports if ports else ["No Ports Detected"])
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🔗 Connect", use_container_width=True):
                if ports and selected_port != "No Ports Detected":
                    success = st.session_state.serial_client.connect(selected_port)
                    if success:
                        st.success("Connected to ESP32-S3!")
                    else:
                        st.error("Failed to connect.")
        with col_c2:
            if st.button("🔌 Disconnect", use_container_width=True):
                st.session_state.serial_client.close()
                st.info("Port closed.")

    st.markdown("---")

    # 2. Live Fault Injection Trigger Rig
    st.markdown("##### 💥 Live Fault Injection Rig")
    st.caption("Perturb the machine in real-time to demonstrate multi-modal anomaly detection and automated relay tripping:")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        if st.button("🟢 Nominal Run", use_container_width=True):
            st.session_state.simulator.set_fault_regime("NOMINAL")
            st.toast("Operating regime set to NOMINAL")

        if st.button("💨 Air Leak Fault", use_container_width=True):
            st.session_state.simulator.set_fault_regime("AIR_LEAK")
            st.toast("Injected Pneumatic Air Leak!")

    with f_col2:
        if st.button("🔨 Bearing Knock", use_container_width=True):
            st.session_state.simulator.set_fault_regime("BEARING_VALVE_KNOCK")
            st.toast("Injected Mechanical Bearing Knock!")

        if st.button("🚨 Overpressure", use_container_width=True):
            st.session_state.simulator.set_fault_regime("CRITICAL_OVERPRESSURE")
            st.toast("Injected Pressure Ramp -> Overpressure Cutoff!")

    if st.button("🔥 Thermal Overheat", use_container_width=True):
        st.session_state.simulator.set_fault_regime("THERMAL_OVERHEAT")
        st.toast("Injected Thermal Overheating!")

    st.markdown("---")

    # 3. Stream Controls & CSV Export
    st.markdown("##### ⏯️ Stream Playback")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("▶️ Step / Tick", use_container_width=True):
            pass
    with p_col2:
        if st.button("🧹 Clear Logs", use_container_width=True):
            st.session_state.history = []
            st.session_state.simulator = CompressorTelemetrySimulator()
            st.rerun()

    # CSV Download
    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Session CSV",
            data=csv_data,
            file_name=f"compressor_telemetry_session_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================================================================
# DATA INGESTION & DISPATCH
# ==============================================================================
current_packet = None
if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
    raw_packet = st.session_state.serial_client.read_packet()
    if raw_packet:
        current_packet = raw_packet
        current_packet["step"] = len(st.session_state.history) + 1
else:
    # Use deterministic physical simulator
    current_packet = st.session_state.simulator.step()

if current_packet:
    st.session_state.history.append(current_packet)
    # Keep last 50 samples for rolling buffer
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

history_df = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()


# ==============================================================================
# MAIN VIEW COORDINATOR
# ==============================================================================
# 1. Top Executive Control Bar
is_tripped = current_packet.get("trip_relay", False) if current_packet else False
render_header(
    is_live_serial=(st.session_state.telemetry_mode == "Live Serial (ESP32-S3)"),
    is_tripped=is_tripped
)

# 2. Safety Interlock Alert Banner
fault_cause = current_packet.get("fault_cause", "NORMAL") if current_packet else "NORMAL"
render_alert_banner(is_tripped=is_tripped, fault_cause=fault_cause, latency_us=7.12)

# 3. Component Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 C01: Edge Sensing & DSP",
    "🧠 C02: AI Predictive & RUL",
    "🧬 C03: Digital Twin Behavioral",
    "🤖 C04: RL Prescriptive Policies"
])

with tab1:
    render_c01_view(current_packet, history_df)

with tab2:
    render_c02_view(current_packet, history_df)

with tab3:
    render_c03_view(current_packet, history_df)

with tab4:
    render_c04_view(current_packet, history_df)
