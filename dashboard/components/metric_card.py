"""
metric_card.py
==============
Industrial Glassmorphism KPI Metric Card Component.
"""

import streamlit as st


def render_metric_card(title: str, value: str, unit: str = "", subtext: str = "", status: str = "normal"):
    """
    Renders a high-contrast industrial glassmorphism metric card.
    status: 'normal' (emerald), 'warning' (amber), 'danger' (rose), 'info' (cyan)
    """
    color_map = {
        "normal": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "info": "#06B6D4"
    }
    accent_color = color_map.get(status, "#06B6D4")

    html = f"""
    <div style="
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-top: 3px solid {accent_color};
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.3);
    ">
        <div style="
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        ">
            {title}
        </div>
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.75rem;
            font-weight: 700;
            color: #F8FAFC;
            display: flex;
            align-items: baseline;
            gap: 0.3rem;
        ">
            <span>{value}</span>
            <span style="font-size: 0.85rem; font-weight: 500; color: #64748B;">{unit}</span>
        </div>
        <div style="
            font-size: 0.75rem;
            color: {accent_color};
            margin-top: 0.35rem;
            font-weight: 500;
        ">
            {subtext}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
