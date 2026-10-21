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
    <div style="background: var(--header-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid var(--border-subtle); padding: 0.85rem 1.4rem; margin-bottom: 1.25rem; border-radius: 12px; display: flex; align-items: center; justify-content: space-between; box-shadow: var(--card-shadow);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="background: linear-gradient(135deg, #1A73E8 0%, #0D47A1 100%); width: 42px; height: 42px; border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(26, 115, 232, 0.35); border: 1px solid rgba(255, 255, 255, 0.2);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                </svg>
            </div>
            <div>
                <div style="font-family: var(--font-heading); font-size: 1.2rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.01em; display: flex; align-items: center;">
                    AIR-GUARD INDUSTRIAL APM 
                    <span style="font-family: var(--font-mono); font-size: 0.68rem; font-weight: 700; color: var(--google-blue); background: rgba(26, 115, 232, 0.12); border: 1px solid rgba(26, 115, 232, 0.3); padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.6rem; letter-spacing: 0.05em;">
                        ENTERPRISE SUITE v2.4
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 1px;">
                    Turbomachinery Health Monitoring, Edge DSP & Digital Twin Prescriptive Intelligence
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <div style="text-align: right;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">
                    TARGET ASSET
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.84rem; font-weight: 600; color: var(--text-primary);">
                    CMP-150-RECIP (150 PSI / 12V DC)
                </div>
            </div>
            <div style="background: var(--bg-card); border: 1px solid {status_color}55; padding: 0.35rem 0.85rem; border-radius: 6px; display: flex; align-items: center; gap: 0.5rem; box-shadow: var(--card-shadow);">
                <span class="led-dot {dot_class}"></span>
                <span style="font-family: var(--font-mono); font-size: 0.74rem; font-weight: 700; color: {status_color}; letter-spacing: 0.03em;">
                    {status_label}
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)
