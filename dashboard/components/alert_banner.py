"""
alert_banner.py
===============
Emergency safety and anomaly trip banner component.
Adaptive styling for Google Material Light and Google Console Dark themes.
"""

import streamlit as st


def clean_html(raw: str) -> str:
    """Removes all leading/trailing indentation and empty lines so Streamlit markdown never creates code blocks."""
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def render_alert_banner(is_tripped: bool, fault_cause: str, is_anomalous: bool = False, latency_us: float = 7.12):
    """Renders emergency trip status banner if the machine has entered fail-safe cutoff, or advisory predictive warning banner."""
    if is_tripped:
        html = f"""
        <div style="background: rgba(217, 48, 37, 0.16); border: 1.5px solid var(--neon-rose); border-radius: 10px; padding: 0.95rem 1.35rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 16px rgba(217, 48, 37, 0.25);">
            <div style="display: flex; align-items: center; gap: 0.9rem;">
                <span style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 800; background: var(--neon-rose); color: #FFFFFF; padding: 0.25rem 0.6rem; border-radius: 4px; letter-spacing: 0.08em;">
                    EMERGENCY TRIP
                </span>
                <div>
                    <div style="font-size: 0.98rem; font-weight: 700; color: var(--neon-rose); letter-spacing: 0.02em;">
                        HARDWARE RELAY INTERLOCK OPEN — MOTOR CIRCUIT ISOLATED
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem;">
                        Trigger Cause: <b style="color: var(--neon-rose);">{fault_cause}</b> | Edge Reaction Latency: <b style="color: var(--google-blue);">{latency_us:.2f} µs</b> (&lt;50ms Hard Cutoff Met)
                    </div>
                </div>
            </div>
            <div style="background: rgba(217, 48, 37, 0.25); border: 1px solid var(--neon-rose); padding: 0.35rem 0.85rem; border-radius: 6px; font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; color: var(--neon-rose);">
                RELAY: OPEN (ISOLATED)
            </div>
        </div>
        """
    elif is_anomalous:
        html = f"""
        <div style="background: rgba(249, 171, 0, 0.14); border: 1.5px solid var(--neon-amber); border-radius: 8px; padding: 0.75rem 1.25rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 10px rgba(249, 171, 0, 0.15);">
            <div style="display: flex; align-items: center; gap: 0.85rem;">
                <span style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 800; background: var(--neon-amber); color: #202124; padding: 0.2rem 0.55rem; border-radius: 4px; letter-spacing: 0.06em;">
                    ADVISORY
                </span>
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: var(--neon-amber); letter-spacing: 0.02em;">
                        PREDICTIVE ANOMALY DETECTED — COMPRESSOR RUNNING (INTERLOCK ARMED)
                    </div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.15rem;">
                        Classification: <b>{fault_cause}</b> | Continuous Multi-Modal Edge Tracking Active
                    </div>
                </div>
            </div>
            <div style="background: rgba(249, 171, 0, 0.2); border: 1px solid var(--neon-amber); padding: 0.35rem 0.75rem; border-radius: 6px; font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; color: var(--neon-amber);">
                RELAY: CLOSED (ARMED)
            </div>
        </div>
        """
    else:
        html = """
        <div style="background: rgba(30, 142, 62, 0.10); border: 1px solid rgba(30, 142, 62, 0.3); border-radius: 8px; padding: 0.6rem 1.15rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--neon-emerald); box-shadow: 0 0 8px var(--neon-emerald);"></span>
                <span style="font-size: 0.82rem; font-weight: 600; color: var(--neon-emerald); letter-spacing: 0.03em;">
                    FAIL-SAFE RELAY ARMED & ENERGIZED — ALL SENSOR CHANNELS NOMINAL
                </span>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; color: var(--neon-emerald);">
                INTERLOCK: CLOSED (HEALTHY)
            </span>
        </div>
        """
    st.markdown(clean_html(html), unsafe_allow_html=True)
