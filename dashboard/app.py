#!/usr/bin/env python3
"""
app.py
======
Industrial AI Compressor Intelligence Platform — Next-Gen Unified Dashboard
SLIIT 4th Year Research Project: J26-SE-326
Integrates All 4 Research Components:
- C01: Edge Sensing & DSP Waveforms (Matheesha)
- C02: AI Predictive Maintenance & RUL (Herath)
- C03: Industrial Digital Twin & Behavioral Intelligence (Kahakotuwa)
- C04: RL Prescriptive Maintenance Policies (Liyanage)
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="AIR-GUARD AI | Industrial Compressor Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure dashboard root and repo root are in sys.path
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DASHBOARD_DIR, ".."))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
    st.markdown("Configure operational inputs and trigger physical fault injections for live demonstrations.")

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
            if st.button("🔗 Connect", use_container_width=True, disabled=st.session_state.serial_client.is_connected):
                if ports and selected_port != "No Ports Detected":
                    st.session_state.serial_client.close()
                    success = st.session_state.serial_client.connect(selected_port)
                    if success:
                        st.toast(f"Connected to {selected_port}!")
                        st.rerun()
                    else:
                        st.error(f"Cannot open {selected_port}. Close Arduino IDE Serial Monitor first!")
        with col_c2:
            if st.button("🔌 Disconnect", use_container_width=True, disabled=not st.session_state.serial_client.is_connected):
                st.session_state.serial_client.close()
                st.toast("Serial port closed.")
                st.rerun()

        if st.session_state.serial_client.is_connected:
            st.markdown(
                f"""<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 8px; padding: 0.45rem 0.75rem; margin-top: 0.5rem; text-align: center;">
                    <span style="color: #10B981; font-weight: 700; font-size: 0.85rem;">🟢 ONLINE: {st.session_state.serial_client.port}</span><br>
                    <span style="color: #94A3B8; font-size: 0.72rem; font-family: monospace;">115200 Baud | Zero-Lag Telemetry</span>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 0.45rem 0.75rem; margin-top: 0.5rem; text-align: center;">
                    <span style="color: #94A3B8; font-weight: 600; font-size: 0.8rem;">⚪ PORT DISCONNECTED</span>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 2. Hardware Actuator & Safety Interlock Control
    st.markdown("##### ⚡ Actuator & Interlock Control")
    st.caption("Direct hardware relay control & emergency unlatch:")

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        if st.button("🟢 Start", use_container_width=True, help="Arm relay and start compressor"):
            if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
                st.session_state.serial_client.send_command("CMD:START")
                st.toast("Sent CMD:START to ESP32-S3")
            else:
                st.session_state.simulator.manual_start()
                st.toast("Simulator: Compressor armed and started")

    with act_col2:
        if st.button("⏹️ E-Stop", use_container_width=True, help="Emergency power isolation (de-energize relay)"):
            if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
                st.session_state.serial_client.send_command("CMD:STOP")
                st.toast("Sent CMD:STOP to ESP32-S3 (Relay Cut)")
            else:
                st.session_state.simulator.manual_stop()
                st.toast("Simulator: Manual E-Stop triggered")

    if st.button("🔄 Reset Safety Interlock", use_container_width=True, type="primary", help="Unlatch safety trip, silence buzzer, and re-arm relay"):
        if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
            ok = st.session_state.serial_client.send_command("CMD:RESET")
            if ok:
                st.toast("Sent CMD:RESET to ESP32-S3 (Interlock Cleared)")
            else:
                st.error("Failed to send CMD:RESET over serial")
        else:
            st.session_state.simulator.reset_interlock()
            st.toast("Simulator: Interlock cleared to NOMINAL")
        st.rerun()

    st.markdown("---")

    # 3. Live Fault Injection Trigger Rig
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

    # 4. Stream Controls & CSV Export
    st.markdown("##### ⏯️ Stream Playback")
    auto_stream = st.checkbox("⚡ Auto-Stream Telemetry", value=True)
    refresh_rate = st.slider("Stream Interval (sec)", min_value=0.15, max_value=2.0, value=1.30, step=0.05)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("▶️ Step / Tick", use_container_width=True):
            pass
    with p_col2:
        if st.button("🧹 Clear Logs", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_steps = 0
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

    st.markdown("---")

    # 4. View Presentation Mode & Component Selector
    st.markdown("##### 🧩 Intelligence Modules View")
    presentation_mode = st.radio(
        "Presentation Mode:",
        ["Unified Multi-Tab (All Components)", "Individual Component Focus"],
        index=0
    )

    selected_component = None
    if presentation_mode == "Individual Component Focus":
        selected_component = st.selectbox(
            "Select Component View",
            [
                "⚡ C01: Edge Sensing & DSP Waveforms (Matheesha)",
                "🧠 C02: AI Predictive Maintenance & RUL (Herath)",
                "🌐 C03: Industrial Digital Twin (Kahakotuwa)",
                "🎯 C04: RL Prescriptive Maintenance (Liyanage)"
            ],
            index=2
        )


# ==============================================================================
# DATA INGESTION & DISPATCH
# ==============================================================================
if "total_steps" not in st.session_state:
    st.session_state.total_steps = 0

current_packet = None
if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
    raw_packet = st.session_state.serial_client.read_packet()
    if raw_packet:
        current_packet = raw_packet
        st.session_state.total_steps += 1
        current_packet["step"] = st.session_state.total_steps
    elif st.session_state.history:
        current_packet = st.session_state.history[-1]
else:
    # Use deterministic physical simulator
    current_packet = st.session_state.simulator.step()
    st.session_state.total_steps += 1
    current_packet["step"] = st.session_state.total_steps

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
is_anomalous = current_packet.get("is_anomalous", False) if current_packet else False
render_header(
    is_live_serial=(st.session_state.telemetry_mode == "Live Serial (ESP32-S3)"),
    is_tripped=is_tripped,
    has_anomaly=is_anomalous
)

# 2. Safety Interlock Alert Banner
fault_cause = current_packet.get("fault_cause", "NORMAL") if current_packet else "NORMAL"
render_alert_banner(is_tripped=is_tripped, fault_cause=fault_cause, is_anomalous=is_anomalous, latency_us=7.12)

# 3. View Routing (Unified Tabs OR Individual Component Focus)
if presentation_mode == "Individual Component Focus" and selected_component is not None:
    if "C01" in selected_component:
        render_c01_view(current_packet, history_df)
    elif "C02" in selected_component:
        render_c02_view(current_packet, history_df)
    elif "C03" in selected_component:
        render_c03_view(current_packet, history_df)
    elif "C04" in selected_component:
        render_c04_view(current_packet, history_df)
else:
    # Component Navigation Tabs (Unified master view combining all research components)
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


# Continuous Live Auto-Stream Loop
if auto_stream and st.session_state.is_running:
    time.sleep(refresh_rate)
    st.rerun()
