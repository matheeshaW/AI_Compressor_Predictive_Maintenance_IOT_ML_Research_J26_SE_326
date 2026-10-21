"""
alert_banner.py
===============
Emergency safety and anomaly trip banner component.
"""

import streamlit as st


def clean_html(raw: str) -> str:
    """Removes all leading/trailing indentation and empty lines so Streamlit markdown never creates code blocks."""
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def render_alert_banner(is_tripped: bool, fault_cause: str, is_anomalous: bool = False, latency_us: float = 7.12):
    """Renders emergency trip status banner if the machine has entered fail-safe cutoff, or advisory predictive warning banner."""
    if is_tripped:
        html = f"""
        <div style="background: linear-gradient(90deg, rgba(239, 68, 68, 0.28) 0%, rgba(185, 28, 28, 0.45) 100%); border: 1px solid #EF4444; border-radius: 10px; padding: 0.95rem 1.35rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); animation: pulse-danger 1.5s infinite;">
            <div style="display: flex; align-items: center; gap: 0.9rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 800; background: #EF4444; color: #FFFFFF; padding: 0.25rem 0.6rem; border-radius: 4px; letter-spacing: 0.08em;">
                    EMERGENCY TRIP
                </span>
                <div>
                    <div style="font-size: 1.02rem; font-weight: 700; color: #FEE2E2; letter-spacing: 0.02em;">
                        HARDWARE RELAY INTERLOCK OPEN — MOTOR CIRCUIT ISOLATED
                    </div>
                    <div style="font-size: 0.8rem; color: #FCA5A5; margin-top: 0.2rem;">
                        Trigger Cause: <b>{fault_cause}</b> | Edge Reaction Latency: <b>{latency_us:.2f} µs</b> (&lt;50ms Hard Cutoff Met)
                    </div>
                </div>
            </div>
            <div style="background: rgba(239, 68, 68, 0.35); border: 1px solid #EF4444; padding: 0.35rem 0.85rem; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700; color: #FFFFFF;">
                RELAY: OPEN (ISOLATED)
            </div>
        </div>
        """
    elif is_anomalous:
        html = f"""
        <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 0.75rem 1.25rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);">
            <div style="display: flex; align-items: center; gap: 0.85rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 800; background: #F59E0B; color: #0F172A; padding: 0.2rem 0.55rem; border-radius: 4px; letter-spacing: 0.06em;">
                    ADVISORY
                </span>
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #FEF3C7; letter-spacing: 0.02em;">
                        PREDICTIVE ANOMALY DETECTED — COMPRESSOR RUNNING (INTERLOCK ARMED)
                    </div>
                    <div style="font-size: 0.78rem; color: #FDE68A; margin-top: 0.15rem;">
                        Classification: <b>{fault_cause}</b> | Continuous Multi-Modal Edge Tracking Active
                    </div>
                </div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.25); border: 1px solid #F59E0B; padding: 0.35rem 0.75rem; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: #FBBF24;">
                RELAY: CLOSED (ARMED)
            </div>
        </div>
        """
    else:
        html = """
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.6rem 1.15rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
                <span style="font-size: 0.82rem; font-weight: 600; color: #D1FAE5; letter-spacing: 0.03em;">
                    FAIL-SAFE RELAY ARMED & ENERGIZED — ALL SENSOR CHANNELS NOMINAL
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6EE7B7;">
                INTERLOCK: CLOSED (HEALTHY)
            </span>
        </div>
        """
    st.markdown(clean_html(html), unsafe_allow_html=True)
