"""
header.py
=========
Global top control and branding header for the Next-Gen Monitoring Dashboard.
"""

import streamlit as st
import textwrap


def clean_html(raw: str) -> str:
    """Removes all leading/trailing indentation and empty lines so Streamlit markdown never creates code blocks."""
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def render_header(is_live_serial: bool = False, is_tripped: bool = False, has_anomaly: bool = False):
    """Renders the top executive control bar with asset details and system status."""
    if is_tripped:
        status_color = "#EF4444"
        status_label = "SAFETY RELAY ISOLATED"
        dot_class = "led-red"
    elif has_anomaly:
        status_color = "#F59E0B"
        status_label = "ADVISORY WARNING"
        dot_class = "led-amber"
    elif is_live_serial:
        status_color = "#06B6D4"
        status_label = "HARDWARE TELEMETRY (ESP32-S3)"
        dot_class = "led-green"
    else:
        status_color = "#10B981"
        status_label = "SIMULATION RUNNER (DETERMINISTIC)"
        dot_class = "led-green"

    html = f"""
    <div style="background: rgba(15, 23, 42, 0.90); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-bottom: 1px solid rgba(148, 163, 184, 0.18); padding: 0.85rem 1.4rem; margin-bottom: 1.25rem; border-radius: 12px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.45);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%); width: 42px; height: 42px; border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 16px rgba(2, 132, 199, 0.35); border: 1px solid rgba(255, 255, 255, 0.15);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                </svg>
            </div>
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em; display: flex; align-items: center;">
                    AIR-GUARD INDUSTRIAL APM 
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; color: #38BDF8; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35); padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.6rem; letter-spacing: 0.05em;">
                        ENTERPRISE SUITE v2.4
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 1px;">
                    Turbomachinery Health Monitoring, Edge DSP & Digital Twin Prescriptive Intelligence
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <div style="text-align: right;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B;">
                    TARGET ASSET
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; font-weight: 600; color: #E2E8F0;">
                    CMP-150-RECIP (150 PSI / 12V DC)
                </div>
            </div>
            <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid {status_color}45; padding: 0.35rem 0.85rem; border-radius: 6px; display: flex; align-items: center; gap: 0.5rem;">
                <span class="led-dot {dot_class}"></span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 700; color: {status_color}; letter-spacing: 0.03em;">
                    {status_label}
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)
