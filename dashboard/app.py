#!/usr/bin/env python3
"""
app.py
======
AIR-GUARD Industrial Asset Performance Management (APM) Platform
Unified Turbomachinery Health Monitoring, Edge DSP & Digital Twin Intelligence

Integrated Modules:
- Module 01: High-Frequency Multi-Modal Edge Sensing & DSP Telemetry
- Module 02: Predictive AI Diagnostics & RUL Prognostics Engine
- Module 03: Industrial Digital Twin & Behavioral Intelligence
- Module 04: Reinforcement Learning Prescriptive Maintenance Engine
"""

import os
import sys
import time
import textwrap
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="AIR-GUARD APM | Industrial Turbomachinery Intelligence",
    page_icon="⚙️",
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
from views.unified_system_view import render_unified_system_view

# Initialize Theme State
if "theme_choice" not in st.session_state:
    st.session_state["theme_choice"] = "Google Dark (Console)"

# Load Base CSS
THEME_CSS_PATH = os.path.join(DASHBOARD_DIR, "styles", "theme.css")
if os.path.exists(THEME_CSS_PATH):
    with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject dynamic theme palette overrides based on user choice
is_light = (st.session_state.get("theme_choice") == "Google Light (Material)")
if is_light:
    st.markdown("""
    <style>
    :root {
        --bg-primary: #F8F9FA !important;
        --bg-secondary: #FFFFFF !important;
        --bg-card: #FFFFFF !important;
        --bg-card-hover: #F8F9FA !important;
        --border-subtle: #DADCE0 !important;
        --border-accent: #1A73E8 !important;
        --header-bg: rgba(255, 255, 255, 0.96) !important;
        --sidebar-bg: #F1F3F4 !important;
        --text-primary: #202124 !important;
        --text-secondary: #5F6368 !important;
        --text-muted: #80868B !important;
        --google-blue: #1A73E8 !important;
        --google-green: #1E8E3E !important;
        --google-amber: #F9AB00 !important;
        --google-red: #D93025 !important;
        --google-teal: #007B83 !important;
        --google-purple: #9334E6 !important;
        --neon-cyan: #007B83 !important;
        --neon-blue: #1A73E8 !important;
        --neon-emerald: #1E8E3E !important;
        --neon-amber: #F9AB00 !important;
        --neon-rose: #D93025 !important;
        --neon-violet: #9334E6 !important;
        --card-shadow: 0 1px 3px rgba(60,64,67,0.12), 0 1px 2px rgba(60,64,67,0.24) !important;
        --card-shadow-hover: 0 4px 12px rgba(60,64,67,0.18) !important;
    }
    html, body, [class*="stApp"] {
        background: #F8F9FA !important;
        color: #202124 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #F1F3F4 !important;
        border-right: 1px solid #DADCE0 !important;
    }
    .stMarkdown, p, span, label, h1, h2, h3, h4, h5 {
        color: #202124;
    }
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF !important;
        border: 1px solid #DADCE0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5F6368 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(26, 115, 232, 0.1) !important;
        color: #1A73E8 !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    :root {
        --bg-primary: #1A1D20 !important;
        --bg-secondary: #202124 !important;
        --bg-card: rgba(36, 40, 45, 0.90) !important;
        --bg-card-hover: rgba(44, 49, 54, 0.95) !important;
        --border-subtle: rgba(255, 255, 255, 0.08) !important;
        --border-accent: rgba(138, 180, 248, 0.5) !important;
        --header-bg: rgba(32, 37, 43, 0.94) !important;
        --sidebar-bg: #1E2226 !important;
        --text-primary: #E8EAED !important;
        --text-secondary: #9AA0A6 !important;
        --text-muted: #80868B !important;
        --google-blue: #8AB4F8 !important;
        --google-green: #81C995 !important;
        --google-amber: #FDD663 !important;
        --google-red: #F28B82 !important;
        --google-teal: #78D9EC !important;
        --google-purple: #C58AF9 !important;
        --neon-cyan: #78D9EC !important;
        --neon-blue: #8AB4F8 !important;
        --neon-emerald: #81C995 !important;
        --neon-amber: #FDD663 !important;
        --neon-rose: #F28B82 !important;
        --neon-violet: #C58AF9 !important;
        --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        --card-shadow-hover: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
    }
    html, body, [class*="stApp"] {
        background: radial-gradient(circle at 15% 15%, #23272C 0%, #15181B 100%) !important;
        color: #E8EAED !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1D20 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    </style>
    """, unsafe_allow_html=True)

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


# SIDEBAR CONTROLS & OPERATIONAL CONSOLE
with st.sidebar:
    st.markdown("### Operational Control Console")
    st.caption("Operational parameters, acquisition interface, and controlled perturbation injection.")

    # 1. Telemetry Source Toggle
    st.markdown("##### Telemetry Acquisition Pipeline")
    mode = st.radio(
        "Select Active Pipeline:",
        ["Simulation Replay", "Live Serial (ESP32-S3)"],
        index=0 if st.session_state.telemetry_mode == "Simulation Replay" else 1
    )
    st.session_state.telemetry_mode = mode

    if mode == "Live Serial (ESP32-S3)":
        ports = st.session_state.serial_client.list_available_ports()
        selected_port = st.selectbox("Detected USB Serial COM Ports:", ports if ports else ["No Ports Detected"])
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Connect Serial", key="btn_connect_serial", use_container_width=True, disabled=st.session_state.serial_client.is_connected):
                if ports and selected_port != "No Ports Detected":
                    st.session_state.serial_client.close()
                    success = st.session_state.serial_client.connect(selected_port)
                    if success:
                        st.toast(f"Connected to {selected_port}!")
                        st.rerun()
                    else:
                        st.error(f"Cannot open {selected_port}. Close Arduino IDE Serial Monitor first!")
        with col_c2:
            if st.button("Disconnect", key="btn_disconnect_serial", use_container_width=True, disabled=not st.session_state.serial_client.is_connected):
                st.session_state.serial_client.close()
                st.toast("Serial port closed.")
                st.rerun()

        if st.session_state.serial_client.is_connected:
            st.markdown(
                f"""<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 8px; padding: 0.45rem 0.75rem; margin-top: 0.5rem; text-align: center;">
                    <span style="color: #10B981; font-weight: 700; font-size: 0.85rem;">ONLINE: {st.session_state.serial_client.port}</span><br>
                    <span style="color: #94A3B8; font-size: 0.72rem; font-family: monospace;">115200 Baud | Zero-Lag DMA Stream</span>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 0.45rem 0.75rem; margin-top: 0.5rem; text-align: center;">
                    <span style="color: #94A3B8; font-weight: 600; font-size: 0.8rem;">PORT DISCONNECTED</span>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 2. Hardware Actuator & Safety Interlock Control
    st.markdown("##### Hardware Actuators & Interlock")
    st.caption("Direct relay interlock control & emergency trip reset:")

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        if st.button("Start Motor", key="btn_start_machine", use_container_width=True, help="Arm relay and start compressor"):
            if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
                st.session_state.serial_client.send_command("CMD:START")
                st.toast("Sent CMD:START to ESP32-S3")
            else:
                st.session_state.simulator.manual_start()
                st.toast("Simulator: Compressor armed and started")

    with act_col2:
        if st.button("Emergency Stop", key="btn_estop_machine", use_container_width=True, help="Emergency power isolation (de-energize relay)"):
            if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
                st.session_state.serial_client.send_command("CMD:STOP")
                st.toast("Sent CMD:STOP to ESP32-S3 (Relay Cut)")
            else:
                st.session_state.simulator.manual_stop()
                st.toast("Simulator: Manual E-Stop triggered")

    if st.button("Reset Safety Interlock", key="btn_reset_interlock_main", use_container_width=True, type="primary", help="Unlatch safety trip, silence buzzer, and re-arm relay"):
        if st.session_state.telemetry_mode == "Live Serial (ESP32-S3)" and st.session_state.serial_client.is_connected:
            ok = st.session_state.serial_client.send_command("CMD:RESET")
            if ok:
                st.toast("Sent CMD:RESET to ESP32-S3 (Interlock Cleared)")
            else:
                st.error("Failed to send CMD:RESET over serial")
        else:
            st.session_state.simulator.reset_interlock()
            st.session_state["active_fault_regime"] = "NOMINAL"
            st.session_state["c03_active_payload"] = None
            st.session_state["c03_active_scenario"] = None
            if "c03_controller" in st.session_state:
                ctrl = st.session_state.c03_controller
                ctrl.engine.buffer.clear()
                ctrl.engine.tracker.persistence_counter = 0
                ctrl.engine.tracker.drift_detected = False
                ctrl.engine.tracker._recent_deviations.clear()
                ctrl.engine.tracker._composite_score = 0.0
            st.toast("Simulator: Interlock cleared to NOMINAL")
        st.rerun()

    cal_col1, cal_col2 = st.columns([1.6, 1])
    is_live = (st.session_state.telemetry_mode == "Live Serial (ESP32-S3)")
    with cal_col1:
        tare_active = st.session_state.get("pressure_tared", False)
        tare_offset = st.session_state.get("pressure_tare_psi", 0.0)
        if is_live:
            btn_txt = "Tare Pressure (0 PSI)" if not tare_active else "Tare Active (Calibrated)"
        else:
            btn_txt = "Tare Pressure (0 PSI)" if tare_offset == 0.0 else f"Tared (-{tare_offset:.1f} PSI)"

        if st.button(btn_txt, key="btn_tare_pressure_sidebar", use_container_width=True, help="Tare pressure sensor offset to 0.0 PSI in open air"):
            st.session_state["pressure_tared"] = True
            if is_live and st.session_state.serial_client.is_connected:
                st.session_state.serial_client.send_command("CMD:TARE_P")
                st.toast("ESP32-S3: Transducer zeroed to 0.0 PSI baseline on-chip")
            else:
                curr_raw = st.session_state.get("last_raw_pressure", 4.70)
                st.session_state["pressure_tare_psi"] = curr_raw
                st.toast(f"Simulator: Pressure zeroed (offset: -{curr_raw:.2f} PSI)")
            st.rerun()
    with cal_col2:
        if st.button("Reset Tare", key="btn_reset_tare_sidebar", use_container_width=True, help="Revert pressure calibration to factory offset"):
            st.session_state["pressure_tared"] = False
            st.session_state["pressure_tare_psi"] = 0.0
            if is_live and st.session_state.serial_client.is_connected:
                st.session_state.serial_client.send_command("CMD:RESET_CAL_P")
                st.toast("ESP32-S3: Pressure calibration reset to 5V factory default (0.500V)")
            else:
                st.toast("Simulator: Pressure calibration reset to factory default")
            st.rerun()

    st.markdown("---")

    # 3. Live Fault Injection Trigger Rig
    st.markdown("##### Controlled Fault Perturbation Rig")
    st.caption("Inject calibrated physical perturbations to verify real-time multi-modal anomaly classification and sub-50ms relay tripping:")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        if st.button("Nominal Baseline", key="btn_fault_nominal_sidebar", use_container_width=True):
            st.session_state.simulator.set_fault_regime("NOMINAL")
            st.session_state["active_fault_regime"] = "NOMINAL"
            st.session_state["c03_active_payload"] = None
            st.session_state["c03_active_scenario"] = None
            if "c03_controller" in st.session_state:
                ctrl = st.session_state.c03_controller
                ctrl.engine.buffer.clear()
                ctrl.engine.tracker.persistence_counter = 0
                ctrl.engine.tracker.drift_detected = False
                ctrl.engine.tracker._recent_deviations.clear()
                ctrl.engine.tracker._composite_score = 0.0
            st.toast("Operating regime set to NOMINAL")
            st.rerun()

        if st.button("Pneumatic Line Leak", key="btn_fault_air_leak_sidebar", use_container_width=True):
            st.session_state.simulator.set_fault_regime("AIR_LEAK")
            st.session_state["active_fault_regime"] = "AIR_LEAK"
            st.session_state["c03_active_payload"] = None
            st.session_state["c03_active_scenario"] = None
            st.toast("Injected Pneumatic Air Leak!")
            st.rerun()

    with f_col2:
        if st.button("Bearing Impact Knock", key="btn_fault_bearing_knock_sidebar", use_container_width=True):
            st.session_state.simulator.set_fault_regime("BEARING_VALVE_KNOCK")
            st.session_state["active_fault_regime"] = "BEARING_VALVE_KNOCK"
            st.session_state["c03_active_payload"] = None
            st.session_state["c03_active_scenario"] = None
            st.toast("Injected Mechanical Bearing Knock!")
            st.rerun()

        if st.button("Discharge Overpressure", key="btn_fault_overpressure_sidebar", use_container_width=True):
            st.session_state.simulator.set_fault_regime("CRITICAL_OVERPRESSURE")
            st.session_state["active_fault_regime"] = "CRITICAL_OVERPRESSURE"
            st.session_state["c03_active_payload"] = None
            st.session_state["c03_active_scenario"] = None
            st.toast("Injected Pressure Ramp -> Overpressure Cutoff!")
            st.rerun()

    if st.button("Cylinder Thermal Overheat", key="btn_fault_overheat_sidebar", use_container_width=True):
        st.session_state.simulator.set_fault_regime("THERMAL_OVERHEAT")
        st.session_state["active_fault_regime"] = "THERMAL_OVERHEAT"
        st.session_state["c03_active_payload"] = None
        st.session_state["c03_active_scenario"] = None
        st.toast("Injected Thermal Overheating!")
        st.rerun()


    st.markdown("---")

    # 4. Stream Controls & CSV Export
    st.markdown("##### Telemetry Stream Controls")
    auto_stream = st.checkbox("Auto-Stream Active", value=True)
    refresh_rate = st.slider("Stream Interval (sec)", min_value=0.15, max_value=2.0, value=1.30, step=0.05)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("Step Frame", key="btn_sim_step", use_container_width=True):
            pass
    with p_col2:
        if st.button("Clear History Buffer", key="btn_clear_logs", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_steps = 0
            st.session_state.simulator = CompressorTelemetrySimulator()
            st.session_state["active_fault_regime"] = "NOMINAL"
            st.session_state["c03_active_payload"] = None
            st.session_state["c03_active_scenario"] = None
            st.rerun()

    # CSV Download
    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Session Log (CSV)",
            data=csv_data,
            file_name=f"compressor_telemetry_session_{int(time.time())}.csv",
            mime="text/csv",
            width='stretch'
        )

    st.markdown("---")

    # 5. View Presentation Mode & Module Selector
    st.markdown("##### System Intelligence Modules")
    presentation_mode = st.radio(
        "Presentation Mode:",
        ["Unified System Cockpit (All-in-One)", "Modular Deep-Dive Views"],
        index=0,
        help="Switch between the unified real-time industrial cockpit and individual module deep-dive analytics"
    )

    selected_component = None
    if presentation_mode == "Modular Deep-Dive Views":
        selected_component = st.selectbox(
            "Select Deep-Dive View:",
            [
                "All Modules (Tabbed Deep-Dive)",
                "Module 01: High-Frequency Edge Sensing & DSP",
                "Module 02: Predictive AI Diagnostics & RUL",
                "Module 03: Industrial Digital Twin & State Space",
                "Module 04: Prescriptive Policy Optimization"
            ],
            index=0
        )

    st.markdown("---")

    # 6. Interface Styling & Theme
    st.markdown("##### Interface Styling")
    theme_choice = st.radio(
        "Color Palette:",
        ["Google Dark (Console)", "Google Light (Material)"],
        index=0 if st.session_state.get("theme_choice", "Google Dark (Console)") == "Google Dark (Console)" else 1,
        horizontal=True,
        key="theme_radio"
    )
    if theme_choice != st.session_state.get("theme_choice"):
        st.session_state["theme_choice"] = theme_choice
        st.rerun()


# DATA INGESTION & DISPATCH
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
    # Retain raw uncalibrated pressure for taring calculations
    st.session_state["last_raw_pressure"] = current_packet.get("pressure_psi", 0.0)
    # In Live Serial mode, the ESP32-S3 firmware zeroes pressureZeroOffsetV directly on-chip.
    # Software tare subtraction is strictly for synthetic simulation mode to prevent double-subtraction.
    if st.session_state.telemetry_mode != "Live Serial (ESP32-S3)":
        tare_offset = st.session_state.get("pressure_tare_psi", 0.0)
        if tare_offset > 0.0 and "pressure_psi" in current_packet:
            current_packet["raw_pressure_psi"] = current_packet["pressure_psi"]
            current_packet["pressure_psi"] = max(0.0, round(current_packet["pressure_psi"] - tare_offset, 2))

    st.session_state.history.append(current_packet)
    # Keep last 50 samples for rolling buffer
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

history_df = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()


# MAIN VIEW COORDINATOR
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

# 3. View Routing (Unified System Cockpit OR Modular Deep-Dive Views)
is_dark = (st.session_state.get("theme_choice", "Google Dark (Console)") == "Google Dark (Console)")

if presentation_mode == "Unified System Cockpit (All-in-One)":
    render_unified_system_view(current_packet, history_df, is_dark=is_dark)
elif presentation_mode == "Modular Deep-Dive Views":
    if selected_component == "Module 01: High-Frequency Edge Sensing & DSP":
        render_c01_view(current_packet, history_df)
    elif selected_component == "Module 02: Predictive AI Diagnostics & RUL":
        render_c02_view(current_packet, history_df)
    elif selected_component == "Module 03: Industrial Digital Twin & State Space":
        render_c03_view(current_packet, history_df)
    elif selected_component == "Module 04: Prescriptive Policy Optimization":
        render_c04_view(current_packet, history_df)
    else:
        # All Modules (Tabbed Deep-Dive)
        tab1, tab2, tab3, tab4 = st.tabs([
            "Module 01: High-Frequency Edge & DSP",
            "Module 02: Predictive AI & RUL Prognostics",
            "Module 03: Industrial Digital Twin",
            "Module 04: Prescriptive Policy Engine"
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

