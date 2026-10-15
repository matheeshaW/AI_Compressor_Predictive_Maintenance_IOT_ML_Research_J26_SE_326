"""
dashboard/views/c03_digital_twin_view.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Interactive Monitoring & Subsystem Health Dashboard View
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
Design Science Research (DSR) Unified Platform View Controller
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Ensure root paths resolve cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
C03_ROOT = REPO_ROOT / "c03_digital_twin_behavioral"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(C03_ROOT) not in sys.path:
    sys.path.insert(0, str(C03_ROOT))

from c03_digital_twin_behavioral.src.twin_engine import IndustrialDigitalTwinEngine
from c03_digital_twin_behavioral.src.clustering.cluster_evaluator import ClusterEvaluator
from c03_digital_twin_behavioral.src.clustering.state_discovery import OperatingStateLabel


class DigitalTwinViewController:
    """
    State manager and business logic controller for the Component 03 Digital Twin View.
    Decoupled from Streamlit UI rendering to facilitate automated unit testing and headless validation.
    """

    def __init__(self, asset_id: str = "COMP-RECIP-01"):
        self.asset_id = asset_id
        self.engine = IndustrialDigitalTwinEngine(
            asset_id=asset_id,
            buffer_capacity=2048,
            target_sampling_rate_hz=25.0,
            deviation_threshold=3.0,
            persistence_threshold=5,
            nominal_compression_rate_psi_s=1.35,
        )
        self.is_initialized = False
        self._history: List[Dict[str, Any]] = []

    def initialize_default(self) -> None:
        """Initializes the twin models using synthetic calibrated baseline data."""
        baseline_df = ClusterEvaluator.generate_synthetic_compressor_data(n_samples_per_regime=150, random_state=42)
        self.engine.initialize_with_baseline(baseline_df)
        self.is_initialized = True

    def process_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Steps the digital twin engine and records history for time-series charts."""
        if not self.is_initialized:
            self.initialize_default()

        payload = self.engine.step(telemetry)
        self._history.append({
            "cycle": payload["metadata"]["twin_cycle_index"],
            "timestamp": payload["metadata"]["timestamp"],
            "pressure_psi": payload["telemetry_snapshot"]["pressure_psi"],
            "motor_current_rms_a": payload["telemetry_snapshot"]["motor_current_rms_a"],
            "state_label": payload["operating_state"]["state_label"],
            "deviation_index": payload["behavioral_intelligence"]["deviation_index"],
            "persistence_counter": payload["behavioral_intelligence"]["persistence_counter"],
            "drift_detected": payload["behavioral_intelligence"]["drift_detected"],
            "composite_degradation_score": payload["degradation_metrics"]["composite_degradation_score"],
            "twin_health_status": payload["metadata"]["twin_health_status"],
        })

        if len(self._history) > 200:
            self._history.pop(0)

        return payload

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def get_radar_comparison_data(self, latest_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts multi-variable normalized features comparing current observation
        against the active operating state's learned centroid envelope.
        """
        b_info = latest_payload.get("behavioral_intelligence", {})
        centroid = b_info.get("state_cluster_centroid", {})
        variance = b_info.get("state_cluster_variance", {})
        snap = latest_payload.get("telemetry_snapshot", {})

        metrics = ["pressure_psi", "motor_current_rms_a", "vibration_x_rms_ms2", "acoustic_energy_mv", "vibration_kurtosis"]
        radar_data = []

        for m in metrics:
            val = float(snap.get(m, 0.0))
            c_val = float(centroid.get(m, val))
            v_val = float(variance.get(m, 1.0))
            std_val = np.sqrt(max(1e-4, v_val))

            radar_data.append({
                "metric": m,
                "current_val": round(val, 2),
                "centroid_val": round(c_val, 2),
                "upper_bound_3sigma": round(c_val + 3.0 * std_val, 2),
                "lower_bound_3sigma": round(max(0.0, c_val - 3.0 * std_val), 2),
            })

        return {
            "state_label": latest_payload.get("operating_state", {}).get("state_label", "UNKNOWN"),
            "features": radar_data,
        }

    def get_subsystems_summary(self, latest_payload: Dict[str, Any]) -> Dict[str, Any]:
        return latest_payload.get("subsystems", {})


def render_digital_twin_view(controller: Optional[DigitalTwinViewController] = None, latest_payload: Optional[Dict[str, Any]] = None):
    """
    Renders the Component 03 Digital Twin View in Streamlit with modern industrial aesthetics.
    Gracefully handles environment if Streamlit is not installed.
    """
    try:
        import streamlit as st
    except ImportError:
        print("[WARNING] Streamlit is not installed. Digital Twin View cannot render GUI directly.")
        return

    st.markdown("""
        <style>
        .kpi-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            margin-bottom: 12px;
        }
        .status-badge-healthy {
            background-color: #10B981;
            color: #042F2E;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .status-badge-warning {
            background-color: #F59E0B;
            color: #451A03;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .status-badge-critical {
            background-color: #EF4444;
            color: #450A0A;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # 1. Header & Identity
    col_hdr_left, col_hdr_right = st.columns([3, 1])
    with col_hdr_left:
        st.title("Component 03: Industrial Digital Twin & Behavioral Intelligence")
        st.caption("Researcher: Kahakotuwa K.N (IT23432284) | Project: J26-SE-326 | SLIIT AIMS Cluster")
    with col_hdr_right:
        status = latest_payload.get("metadata", {}).get("twin_health_status", "NOMINAL") if latest_payload else "HEALTHY"
        badge_cls = "status-badge-healthy" if status == "HEALTHY" else ("status-badge-warning" if status in ["WARNING", "DEGRADED"] else "status-badge-critical")
        st.markdown(f"<div style='text-align: right; margin-top: 15px;'><span class='{badge_cls}'>{status}</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    if not latest_payload:
        st.info("No active telemetry stream connected. Run simulation or ingest live stream to populate twin.")
        return

    # 2. KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Discovered State", latest_payload["operating_state"]["state_label"])
    with kpi2:
        st.metric("State Confidence", f"{latest_payload['operating_state']['state_confidence'] * 100:.1f}%")
    with kpi3:
        st.metric("Mahalanobis Distance", f"{latest_payload['behavioral_intelligence']['deviation_index']:.2f} σ")
    with kpi4:
        st.metric("Persistence Counter", f"{latest_payload['behavioral_intelligence']['persistence_counter']} / 5")
    with kpi5:
        st.metric("Composite Degradation", f"{latest_payload['degradation_metrics']['composite_degradation_score'] * 100:.1f}%")

    # 3. Subsystem Decomposition
    st.subheader("Physical Sub-Assembly Health Decomposition")
    subsystems = latest_payload.get("subsystems", {})
    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)

    with sub_col1:
        m = subsystems.get("electric_motor", {})
        st.markdown("**Electric Motor**")
        st.progress(float(m.get("health_score", 1.0)))
        st.caption(f"Status: {m.get('status', 'NOMINAL')} | Temp: {m.get('operating_temperature_c', 0.0)}°C")

    with sub_col2:
        p = subsystems.get("cylinder_piston_assembly", {})
        st.markdown("**Cylinder & Piston**")
        st.progress(float(p.get("health_score", 1.0)))
        st.caption(f"Status: {p.get('status', 'OPTIMAL')} | Rate: {p.get('compression_rate_psi_s', 0.0)} PSI/s")

    with sub_col3:
        v = subsystems.get("suction_discharge_valves", {})
        st.markdown("**Suction/Discharge Valves**")
        st.progress(float(v.get("health_score", 1.0)))
        st.caption(f"Status: {v.get('status', 'NORMAL')} | Leak: {'YES' if v.get('valve_leakage_detected') else 'NO'}")

    with sub_col4:
        b = subsystems.get("main_bearing_assembly", {})
        st.markdown("**Main Bearing Assembly**")
        st.progress(float(b.get("health_score", 1.0)))
        st.caption(f"Status: {b.get('status', 'NOMINAL')} | Kurtosis: {b.get('impact_kurtosis', 3.0)}")

    # 4. Downstream Payload Inspector
    with st.expander("Downstream C04 RL Prescriptive Maintenance Contract Payload", expanded=False):
        st.json(latest_payload.get("downstream_contract_payload", {}))


if __name__ == "__main__":
    # Headless CLI verification runner
    print("Initializing DigitalTwinViewController in headless verification mode...")
    ctrl = DigitalTwinViewController()
    ctrl.initialize_default()

    sample_packet = {
        "pressure_psi": 75.0,
        "delta_pressure_dt": 1.35,
        "motor_current_rms_a": 18.2,
        "temperature_celsius": 62.0,
        "vibration_x_rms_ms2": 4.8,
        "vibration_kurtosis": 3.2,
        "acoustic_energy_mv": 260.0,
    }
    result = ctrl.process_telemetry(sample_packet)
    print(f"Cycle: {result['metadata']['twin_cycle_index']}")
    print(f"Discovered State: {result['operating_state']['state_label']}")
    print(f"Health Status: {result['metadata']['twin_health_status']}")
    print(f"Mahalanobis Distance: {result['behavioral_intelligence']['deviation_index']} sigma")
    print(f"Subsystems: {list(result['subsystems'].keys())}")
    print("Headless verification test completed successfully!")
