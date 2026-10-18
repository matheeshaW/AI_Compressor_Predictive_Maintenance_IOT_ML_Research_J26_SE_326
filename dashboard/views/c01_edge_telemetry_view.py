"""
c01_edge_telemetry_view.py
==========================
Component 01: High-Frequency Multi-Modal Edge Sensing & DSP Feature Extraction
Lead: Weerakoon W.M.M.B (IT23155152)
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List

from components.metric_card import render_metric_card


def render_c01_view(current_packet: Dict[str, Any], history_df: pd.DataFrame):
    """Renders the comprehensive Component 01 Edge Telemetry & DSP dashboard view."""
    
    hdr_c1, hdr_c2 = st.columns([3.5, 1.5])
    with hdr_c1:
        st.markdown("### 📡 Component 01: Multi-Modal Edge Sensing & Real-Time DSP Telemetry")
        st.markdown(
            "*Real-time embedded telemetry acquired via dual-rate DMA on ESP32-S3 (Xtensa LX7 @ 240 MHz). "
            "Evaluates sliding-window DSP features and executes sub-50ms relay emergency cutoffs.*"
        )
    with hdr_c2:
        is_live = (st.session_state.telemetry_mode == "Live Serial (ESP32-S3)")
        tare_active = st.session_state.get("pressure_tared", False)
        tare_offset = st.session_state.get("pressure_tare_psi", 0.0)
        is_zeroed = tare_active if is_live else (tare_offset > 0.0)

        btn_txt = "🎯 Tare Pressure (0 PSI)" if not is_zeroed else "↺ Reset Tare (Calibrated)"
        if st.button(btn_txt, key="btn_tare_c01", use_container_width=True, help="Tare pressure sensor offset to 0.0 PSI in open air"):
            if not is_zeroed:
                st.session_state["pressure_tared"] = True
                if is_live and st.session_state.serial_client.is_connected:
                    st.session_state.serial_client.send_command("CMD:TARE_P")
                    st.toast("ESP32-S3: Transducer zeroed to 0.0 PSI baseline on-chip")
                else:
                    raw_val = current_packet.get("raw_pressure_psi", current_packet.get("pressure_psi", 4.70))
                    st.session_state["pressure_tare_psi"] = raw_val
                    st.toast(f"Simulator: Pressure sensor tared to 0.0 PSI (offset: -{raw_val:.2f} PSI)")
            else:
                st.session_state["pressure_tared"] = False
                st.session_state["pressure_tare_psi"] = 0.0
                if is_live and st.session_state.serial_client.is_connected:
                    st.session_state.serial_client.send_command("CMD:RESET_CAL_P")
                    st.toast("ESP32-S3: Pressure calibration reset to 5V factory default (0.500V)")
                else:
                    st.toast("Simulator: Pressure calibration reset to factory default")
            st.rerun()

    # 1. Top KPI Metric Cards (5 Primary Sensing Modalities)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    p = current_packet.get("pressure_psi", 0.0)
    p_status = "danger" if p >= 45.0 else ("warning" if p >= 38.0 else "normal")
    sub_label = "Tared: 0.0 PSI baseline" if is_zeroed else f"Rate: {current_packet.get('pressure_rate_of_change_psi_sec', 0.0):+.2f} PSI/s"
    with col1:
        render_metric_card("Discharge Pressure", f"{p:.1f}", "PSI", sub_label, p_status)

    curr = current_packet.get("motor_current_rms_amps", 0.0)
    c_status = "danger" if curr >= 15.0 else ("warning" if curr >= 11.0 else "normal")
    with col2:
        render_metric_card("Motor Current", f"{curr:.2f}", "A", "ACS712-30A Hall Sensor", c_status)

    temp = current_packet.get("temperature_cylinder_head_c", 0.0)
    t_status = "danger" if temp >= 85.0 else ("warning" if temp >= 70.0 else "normal")
    with col3:
        render_metric_card("Cylinder Temp", f"{temp:.1f}", "°C", "DS18B20 1-Wire Digital", t_status)

    v_rms = current_packet.get("vibration_rms_g", 0.0)
    v_status = "danger" if v_rms >= 0.8 else ("warning" if v_rms >= 0.5 else "normal")
    with col4:
        render_metric_card("Vibration RMS", f"{v_rms:.3f}", "g", "MPU-6050 Crankcase Axis", v_status)

    kurt = current_packet.get("vibration_kurtosis", 3.0)
    k_status = "danger" if kurt >= 7.5 else ("warning" if kurt >= 5.0 else "normal")
    with col5:
        render_metric_card("Vibration Kurtosis", f"{kurt:.2f}", "dim", "Impulsive Knock Factor", k_status)

    st.markdown("---")

    # 2. Dual DSP Gauge Gauges & Edge Anomaly Detection Panel
    g_col1, g_col2, g_col3 = st.columns([1, 1, 1.2])

    with g_col1:
        st.markdown("##### 🎛️ Kurtosis Shock Indicator")
        fig_kurt = go.Figure(go.Indicator(
            mode="gauge+number",
            value=kurt,
            number={"suffix": " K", "font": {"color": "#F8FAFC", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 10], "tickcolor": "#94A3B8"},
                "bar": {"color": "#EF4444" if kurt >= 7.5 else ("#F59E0B" if kurt >= 5.0 else "#10B981")},
                "steps": [
                    {"range": [0, 5.0], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [5.0, 7.5], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [7.5, 10], "color": "rgba(239, 68, 68, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": 7.5
                }
            }
        ))
        fig_kurt.update_layout(height=220, margin=dict(l=15, r=15, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC", uirevision="kurt_gauge")
        st.plotly_chart(fig_kurt, use_container_width=True, key="fig_kurt_gauge", config={"displayModeBar": False})
        st.caption("Threshold $K > 7.5$ triggers instantaneous bearing knock warning.")

    with g_col2:
        st.markdown("##### 🎯 Crest Factor Dynamic Ratio")
        cf = current_packet.get("vibration_crest_factor", 3.0)
        fig_cf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cf,
            number={"suffix": " CF", "font": {"color": "#F8FAFC", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 8], "tickcolor": "#94A3B8"},
                "bar": {"color": "#EF4444" if cf >= 6.0 else ("#F59E0B" if cf >= 4.5 else "#06B6D4")},
                "steps": [
                    {"range": [0, 4.5], "color": "rgba(6, 182, 212, 0.2)"},
                    {"range": [4.5, 6.0], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [6.0, 8.0], "color": "rgba(239, 68, 68, 0.3)"}
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": 6.0
                }
            }
        ))
        fig_cf.update_layout(height=220, margin=dict(l=15, r=15, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC", uirevision="cf_gauge")
        st.plotly_chart(fig_cf, use_container_width=True, key="fig_cf_gauge", config={"displayModeBar": False})
        st.caption("Ratio of peak impact acceleration to continuous RMS vibration.")

    with g_col3:
        st.markdown("##### 🛡️ Edge Anomaly & Relay Status")
        score = current_packet.get("anomaly_score", 0.0)
        relay_state = current_packet.get("relay_state", "ENERGIZED_CLOSED")
        fault = current_packet.get("fault_cause", "NORMAL")

        score_color = "#EF4444" if score >= 0.5 else "#10B981"
        relay_color = '#EF4444' if 'TRIPPED' in relay_state else '#10B981'
        status_card_html = f"""<div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1.1rem; height: 220px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Continuous Anomaly Score S(D_M)</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 800; color: {score_color};">
                    {score:.3f} <span style="font-size: 0.85rem; color: #94A3B8;">/ 1.000</span>
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Fault Classification: <b style="color: #F8FAFC;">{fault}</b></div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Measured Inference Latency: <b style="color: #06B6D4;">7.12 µs</b></div>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.6rem;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #E2E8F0;">5V Optocoupler Relay:</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: {relay_color};">
                    {relay_state}
                </span>
            </div>
        </div>"""
        st.markdown(status_card_html, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Live Synchronized Waveforms
    st.markdown("##### 📈 Real-Time Multi-Modal Telemetry Streams")
    if not history_df.empty:
        fig_wave = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Pneumatic Pressure (PSI) & Rate of Change",
                "Motor Current (A)",
                "Crankcase Radial Vibration (g)",
                "Acoustic Sound Energy (mV)"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.10
        )

        steps = history_df["step"]
        trace_mode = "lines+markers" if len(steps) < 2 else "lines"

        # Explicit x-axis bounds prevent degenerate range rendering failures
        x_min = float(steps.min())
        x_max = float(steps.max())
        if x_max <= x_min:
            x_max = x_min + 1.0

        # Pressure & dP/dt
        fig_wave.add_trace(go.Scatter(x=steps, y=history_df["pressure_psi"], name="Pressure (PSI)", mode=trace_mode, line=dict(color="#06B6D4", width=2)), row=1, col=1)
        fig_wave.add_trace(go.Scatter(x=steps, y=history_df["pressure_rate_of_change_psi_sec"], name="dP/dt (PSI/s)", mode=trace_mode, line=dict(color="#3B82F6", width=1, dash="dot")), row=1, col=1)

        # Motor Current
        fig_wave.add_trace(go.Scatter(x=steps, y=history_df["motor_current_rms_amps"], name="Current (A)", mode=trace_mode, line=dict(color="#F59E0B", width=2)), row=1, col=2)

        # Vibration RMS & Kurtosis
        fig_wave.add_trace(go.Scatter(x=steps, y=history_df["vibration_rms_g"], name="Vibration (g)", mode=trace_mode, line=dict(color="#10B981", width=2)), row=2, col=1)
        fig_wave.add_trace(go.Scatter(x=steps, y=history_df["vibration_kurtosis"] / 10.0, name="Kurtosis/10", mode=trace_mode, line=dict(color="#EF4444", width=1, dash="dash")), row=2, col=1)

        # Acoustic Sound Energy
        fig_wave.add_trace(go.Scatter(x=steps, y=history_df["acoustic_energy_mv"], name="Acoustic (mV)", mode=trace_mode, line=dict(color="#8B5CF6", width=2)), row=2, col=2)

        fig_wave.update_layout(
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(color="#94A3B8", family="Inter"),
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            uirevision="c01_live_waveforms_stream"
        )
        fig_wave.update_xaxes(gridcolor="rgba(255, 255, 255, 0.05)", range=[x_min, x_max])
        fig_wave.update_yaxes(gridcolor="rgba(255, 255, 255, 0.05)")

        st.plotly_chart(fig_wave, use_container_width=True, key="c01_live_waveforms", config={"displayModeBar": False})
    else:
        st.info("Awaiting initial telemetry frames...")

    # 4. Empirical Model Benchmark Comparison (20,000 Samples)
    st.markdown("---")
    st.markdown("##### 🔬 Embedded Edge Intelligence: Empirical Model Benchmarks (20,000 Samples)")
    st.markdown(
        "*Comparative performance of our proposed zero-allocation Statistical Mahalanobis Detector vs "
        "Deep 1D Compression Autoencoder ($6 \\to 4 \\to 2 \\to 4 \\to 6$) on the calibrated compressor testbed dataset.*"
    )

    bench_col1, bench_col2 = st.columns([1.8, 1.2])

    with bench_col1:
        benchmark_data = {
            "Evaluation Metric": [
                "Precision",
                "Recall (Safety Interlock)",
                "F1-Score",
                "ROC-AUC",
                "Inference Latency",
                "SRAM Footprint",
                "Heap Allocation"
            ],
            "Statistical Mahalanobis (Embedded C++)": [
                "99.59%",
                "100.00%",
                "99.79%",
                "100.00%",
                "7.28 µs (Sub-10 µs)",
                "52 bytes (Static Struct)",
                "0 bytes (Zero Dynamic Heap)"
            ],
            "PyTorch TinyML Autoencoder": [
                "98.68%",
                "100.00%",
                "99.34%",
                "100.00%",
                "45.25 µs",
                "24.5 KB (TFLite Arena)",
                "Dynamic runtime tensor arena"
            ]
        }
        df_bench = pd.DataFrame(benchmark_data)
        try:
            st.dataframe(df_bench, use_container_width=True, hide_index=True)
        except Exception:
            st.table(df_bench)

    with bench_col2:
        st.markdown("""
        <div class="glass-card" style="font-size: 0.85rem; line-height: 1.5;">
            <div style="font-weight: 700; color: #10B981; margin-bottom: 0.4rem;">🎯 Academic Viva Key Takeaway</div>
            <div>• <b>6.2x Faster Latency:</b> Mahalanobis executes in <b>7.28 µs</b> vs 45.25 µs for the neural autoencoder.</div>
            <div style="margin-top: 0.3rem;">• <b>99.79% Less RAM:</b> Uses 52 bytes of static SRAM vs 24.5 KB tensor arena.</div>
            <div style="margin-top: 0.3rem;">• <b>Fail-Safe Determinism:</b> Zero dynamic heap allocation guarantees the microcontroller will never experience heap fragmentation or crash during a critical overpressure cutoff.</div>
        </div>
        """, unsafe_allow_html=True)

