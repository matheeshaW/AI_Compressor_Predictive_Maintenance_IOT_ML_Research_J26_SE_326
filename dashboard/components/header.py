"""
header.py
=========
Global top control and branding header for the Next-Gen Monitoring Dashboard.
"""

import streamlit as st
import textwrap


def render_header(is_live_serial: bool = False, is_tripped: bool = False):
    """Renders the top executive control bar with asset details and system status."""
    status_color = "#EF4444" if is_tripped else ("#06B6D4" if is_live_serial else "#10B981")
    status_label = "CRITICAL TRIP" if is_tripped else ("LIVE SERIAL (ESP32-S3)" if is_live_serial else "DETERMINISTIC SIMULATION")
    dot_class = "led-red" if is_tripped else "led-green"

    html = textwrap.dedent(f"""
    <div style="
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding: 0.9rem 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    ">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="
                background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
                width: 44px;
                height: 44px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4rem;
                box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
            ">
                ⚡
            </div>
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em;">
                    AIR-GUARD AI <span style="font-size: 0.75rem; font-weight: 600; color: #06B6D4; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">J26-SE-326</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8;">
                    Predictive Maintenance & Digital Twin Intelligence Platform — SLIIT Research
                </div>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <div style="text-align: right;">
                <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B;">
                    MONITORED ASSET
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 600; color: #E2E8F0;">
                    12V DC Metal Compressor (150 PSI)
                </div>
            </div>

            <div style="
                background: rgba(17, 24, 39, 0.8);
                border: 1px solid {status_color}40;
                padding: 0.35rem 0.85rem;
                border-radius: 9999px;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                <span class="led-dot {dot_class}"></span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {status_color};">
                    {status_label}
                </span>
            </div>
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)
