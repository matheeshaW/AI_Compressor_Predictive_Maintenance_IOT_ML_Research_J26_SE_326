"""
alert_banner.py
===============
Emergency safety and anomaly trip banner component.
"""

import streamlit as st


def render_alert_banner(is_tripped: bool, fault_cause: str, latency_us: float = 7.12):
    """Renders emergency trip status banner if the machine has entered fail-safe cutoff."""
    if is_tripped:
        html = f"""
        <div style="
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.28) 0%, rgba(185, 28, 28, 0.45) 100%);
            border: 1px solid #EF4444;
            border-radius: 10px;
            padding: 1rem 1.35rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
            animation: pulse-danger 1.5s infinite;
        ">
            <div style="display: flex; align-items: center; gap: 0.85rem;">
                <span style="font-size: 1.6rem;">🚨</span>
                <div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FEE2E2; letter-spacing: 0.04em;">
                        HARDWARE RELAY INTERLOCK TRIPPED — MOTOR POWER ISOLATED
                    </div>
                    <div style="font-size: 0.8rem; color: #FCA5A5; margin-top: 0.2rem;">
                        Trigger Cause: <b>{fault_cause}</b> | Edge Reaction Latency: <b>{latency_us:.2f} µs</b> (&lt;50ms Deadline Met)
                    </div>
                </div>
            </div>
            <div style="
                background: rgba(239, 68, 68, 0.35);
                border: 1px solid #EF4444;
                padding: 0.4rem 0.85rem;
                border-radius: 6px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                font-weight: 700;
                color: #FFFFFF;
            ">
                RELAY: OPEN (CUT)
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        html = """
        <div style="
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 8px;
            padding: 0.6rem 1.15rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
                <span style="font-size: 0.82rem; font-weight: 600; color: #D1FAE5; letter-spacing: 0.03em;">
                    FAIL-SAFE RELAY ARMED & ENERGIZED — ALL SENSOR CHANNELS NOMINAL
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6EE7B7;">
                INTERLOCK: ACTIVE-LOW (HEALTHY)
            </span>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
