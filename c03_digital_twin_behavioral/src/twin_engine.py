"""
Digital twin engine orchestrating telemetry synchronization, operational state clustering,
behavioral anomaly profiling, and subsystem health decomposition.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from .sync.stream_buffer import CircularTelemetryBuffer, TelemetryFrame
from .sync.time_series_aligner import TimeSeriesAligner, AlignedSnapshot
from .clustering.state_discovery import (
    OperatingStateLabel,
    ClusteringAlgorithm,
    StateAssignment,
    CompressorStateClusterer,
    DEFAULT_FEATURE_COLUMNS,
)
from .behavioral.state_behavioral_profiler import (
    StateProfile,
    BehavioralAssessment,
    StateBehavioralProfiler,
)
from .behavioral.degradation_tracker import (
    DeviationClassification,
    DegradationMetrics,
    DegradationTracker,
)


class IndustrialDigitalTwinEngine:
    """
    Master Real-Time Digital Twin Engine for Industrial Air Compressors.
    Synchronizes multi-modal physical sensor streams, discovers operational states,
    tracks state-conditioned behavioral envelopes, and computes degradation indicators.
    Produces schema-compliant digital twin state representations and downstream RL payloads.
    """

    def __init__(
        self,
        asset_id: str = "COMP-IND-01",
        asset_type: str = "12V_RECIPROCATING_PISTON",
        twin_version: str = "1.0.0",
        buffer_capacity: int = 2048,
        target_sampling_rate_hz: float = 25.0,
        deviation_threshold: float = 3.0,
        persistence_threshold: int = 5,
        nominal_compression_rate_psi_s: float = 1.4,
        ambient_temperature_c: float = 25.0,
    ):
        """
        Args:
            asset_id: Unique industrial compressor physical asset identifier.
            asset_type: Mechanical compressor classification from schema enum.
            twin_version: Semantic version of the digital twin state model.
            buffer_capacity: Bounded ring buffer packet capacity.
            target_sampling_rate_hz: Uniform digital twin time-grid frequency (25 Hz).
            deviation_threshold: Statistical sigma threshold for behavioral envelopes (3.0).
            persistence_threshold: Consecutive windows required to confirm persistent degradation (5).
            nominal_compression_rate_psi_s: Healthy nameplate compression rate in loaded regime.
            ambient_temperature_c: Reference baseline ambient temperature.
        """
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.twin_version = twin_version
        self.sampling_rate_hz = target_sampling_rate_hz
        self.delta_t_sec = 1.0 / target_sampling_rate_hz

        # Core sub-modules
        self.buffer = CircularTelemetryBuffer(capacity=buffer_capacity)
        self.aligner = TimeSeriesAligner(target_freq_hz=target_sampling_rate_hz)
        self.clusterer: Optional[CompressorStateClusterer] = None
        self.profiler: Optional[StateBehavioralProfiler] = None
        self.tracker = DegradationTracker(
            deviation_threshold=deviation_threshold,
            persistence_threshold=persistence_threshold,
            nominal_compression_rate_psi_s=nominal_compression_rate_psi_s,
            ambient_temperature_c=ambient_temperature_c,
        )

        # Operational registers
        self.twin_cycle_index: int = 0
        self.is_initialized: bool = False
        self._last_snapshot: Optional[AlignedSnapshot] = None
        self._last_state: Optional[StateAssignment] = None
        self._last_assessment: Optional[BehavioralAssessment] = None
        self._last_behavioral_dict: Dict[str, Any] = {}
        self._last_degradation_dict: Dict[str, Any] = {}
        self._last_state_payload: Optional[Dict[str, Any]] = None

    def initialize_with_baseline(
        self,
        baseline_data: pd.DataFrame,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters: int = 5,
        random_state: int = 42,
    ) -> "IndustrialDigitalTwinEngine":
        """
        Fits unsupervised state discovery clusterer and state-conditioned behavioral profiler
        on historical baseline or benchmark compressor operational data.
        """
        self.clusterer = CompressorStateClusterer(
            algorithm=algorithm,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        self.clusterer.fit(baseline_data)

        # Assign predicted states to baseline data to train state-specific envelopes
        _, pred_labels, _ = self.clusterer.predict(baseline_data)
        baseline_with_states = baseline_data.copy()
        baseline_with_states["predicted_state"] = pred_labels

        self.profiler = StateBehavioralProfiler()
        self.profiler.fit(baseline_with_states, state_column="predicted_state")

        self.is_initialized = True
        return self

    def load_pretrained_models(
        self,
        clusterer_path: Union[str, Path],
        profiler_path: Union[str, Path],
    ) -> "IndustrialDigitalTwinEngine":
        """
        Loads pre-trained clusterer and profiler models from disk.
        """
        self.clusterer = CompressorStateClusterer.load_model(clusterer_path)
        self.profiler = StateBehavioralProfiler.load_profiles(profiler_path)
        self.is_initialized = True
        return self

    def ingest_frame(self, frame: TelemetryFrame) -> None:
        """
        Pushes a single TelemetryFrame into the circular ring buffer.
        """
        self.buffer.push(frame)

    def ingest_dict(self, data: Dict[str, Any]) -> None:
        """
        Pushes raw telemetry dictionary into the circular ring buffer.
        """
        self.buffer.push_dict(data)

    def step(
        self,
        telemetry: Optional[Union[Dict[str, Any], TelemetryFrame]] = None,
        c02_predictive_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes one complete digital twin behavioral evaluation cycle:
        1. Ingests telemetry observation (if supplied).
        2. Synchronizes multi-rate signals onto the 25 Hz time grid.
        3. Classifies operational regime with posterior confidence.
        4. Evaluates state-conditioned regularized Mahalanobis distance.
        5. Updates multi-window persistence filtering and degradation indicators.
        6. Decomposes physical sub-assembly health.
        7. Returns complete dictionary conforming to digital_twin_state_schema.json.
        """
        if not self.is_initialized:
            raise RuntimeError("IndustrialDigitalTwinEngine is not initialized. Call initialize_with_baseline() first.")

        self.twin_cycle_index += 1

        if telemetry is not None:
            if isinstance(telemetry, TelemetryFrame):
                self.ingest_frame(telemetry)
            elif isinstance(telemetry, dict):
                self.ingest_dict(telemetry)

        # 1. Resample and extract latest aligned 25 Hz snapshot
        snapshot = self.aligner.extract_latest_snapshot(self.buffer)
        if snapshot is None:
            # Fallback if buffer has <2 frames: synthesize from incoming or default
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            now_iso = datetime.now(timezone.utc).isoformat()
            if isinstance(telemetry, dict):
                p = float(telemetry.get("pressure_psi", 0.0))
                dp = float(telemetry.get("delta_pressure_dt", 0.0))
                t = float(telemetry.get("temperature_celsius", 25.0))
                i = float(telemetry.get("motor_current_rms_a", 0.0))
                vx = float(telemetry.get("vibration_x_rms_ms2", 0.0))
                vy = float(telemetry.get("vibration_y_rms_ms2", 0.0))
                vz = float(telemetry.get("vibration_z_rms_ms2", 0.0))
                k = float(telemetry.get("vibration_kurtosis", 3.0))
                cf = float(telemetry.get("vibration_crest_factor", 1.4))
                a = float(telemetry.get("acoustic_energy_mv", 0.0))
            else:
                p, dp, t, i, vx, vy, vz, k, cf, a = 25.0, 0.0, 35.0, 4.5, 0.8, 0.8, 0.7, 3.0, 1.4, 70.0

            snapshot = AlignedSnapshot(
                timestamp_ms=now_ms,
                timestamp_iso=now_iso,
                temperature_celsius=t,
                pressure_psi=p,
                delta_pressure_dt=dp,
                motor_current_rms_a=i,
                vibration_x_rms_ms2=vx,
                vibration_y_rms_ms2=vy,
                vibration_z_rms_ms2=vz,
                vibration_kurtosis=k,
                vibration_crest_factor=cf,
                acoustic_energy_mv=a,
                sample_window_size=512,
            )

        self._last_snapshot = snapshot

        # 2. Operational state discovery & posterior confidence
        assignment = self.clusterer.predict_single(snapshot, delta_t_sec=self.delta_t_sec)
        self._last_state = assignment

        # 3. State-conditioned behavioral profiling
        assessment = self.profiler.assess_deviation(snapshot, assignment.state_label)
        self._last_assessment = assessment

        # 4. Multi-window temporal persistence filtering & degradation tracking
        b_dict, m_dict = self.tracker.update(assessment, snapshot, delta_t_sec=self.delta_t_sec)
        self._last_behavioral_dict = b_dict
        self._last_degradation_dict = m_dict

        # 5. Physical sub-assembly decomposition
        subsystems_dict = self._evaluate_subsystems(snapshot, assignment, assessment, m_dict)

        # 6. Overall twin health status
        composite_score = float(m_dict["composite_degradation_score"])
        is_leak_state = (assignment.state_label == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value)
        valves_leaking = bool(subsystems_dict.get("suction_discharge_valves", {}).get("valve_leakage_detected", False))
        is_bearing_impacting = (subsystems_dict.get("main_bearing_assembly", {}).get("status") == "IMPACTING")
        is_thermal_critical = (snapshot.temperature_celsius > 75.0 or subsystems_dict.get("electric_motor", {}).get("status") in ("ELEVATED_TEMP", "CURRENT_OVERLOAD"))
        is_overpressure_critical = (snapshot.pressure_psi >= 115.0 and snapshot.delta_pressure_dt > 1.0)

        # Critical physical breakdown or extreme operating limit trip
        if is_bearing_impacting or is_thermal_critical or is_overpressure_critical:
            twin_status = "CRITICAL"
        # Air leak is an operational pneumatic degradation (matches C02 Degradation Detected)
        elif is_leak_state and composite_score <= 0.85:
            twin_status = "DEGRADED"
        elif b_dict["deviation_classification"] == DeviationClassification.CRITICAL_DRIFT.value:
            twin_status = "CRITICAL"
        elif b_dict["drift_detected"] or is_leak_state or valves_leaking:
            twin_status = "DEGRADED"
        elif b_dict["deviation_classification"] == DeviationClassification.TRANSIENT_PERTURBATION.value:
            twin_status = "WARNING"
        elif composite_score < 0.25:
            twin_status = "HEALTHY"
        elif composite_score < 0.60:
            twin_status = "DEGRADED"
        else:
            twin_status = "CRITICAL"

        # Ensure ISO-8601 UTC format with trailing Z if not present
        ts_iso = snapshot.timestamp_iso
        if "+" in ts_iso:
            ts_iso = ts_iso.split("+")[0] + "Z"
        elif not ts_iso.endswith("Z"):
            ts_iso = ts_iso + "Z"

        twin_id = self.asset_id if self.asset_id.startswith("TWIN-") else f"TWIN-{self.asset_id}"

        # 7. Formulate full digital twin state schema payload strictly matching digital_twin_state_schema.json
        payload = {
            "metadata": {
                "twin_id": twin_id,
                "physical_asset_id": self.asset_id,
                "asset_type": self.asset_type,
                "twin_version": self.twin_version,
                "timestamp": ts_iso,
                "sync_epoch_ms": int(snapshot.timestamp_ms),
                "twin_cycle_index": self.twin_cycle_index,
                "twin_health_status": twin_status,
            },
            "telemetry_snapshot": snapshot.to_dict(),
            "synchronized_telemetry": snapshot.to_dict(),
            "operating_state": assignment.to_dict(),
            "behavioral_intelligence": b_dict,
            "degradation_metrics": m_dict,
            "subsystems": subsystems_dict,
            "downstream_contract_payload": self._format_downstream_payload(
                snapshot=snapshot,
                assignment=assignment,
                b_dict=b_dict,
                m_dict=m_dict,
                ts_iso=ts_iso,
                twin_id=twin_id,
            ),
        }

        # Add optional C02 predictive diagnostics if available
        if c02_predictive_context:
            payload["predictive_diagnostics"] = c02_predictive_context
        else:
            # Nominal fallback diagnostics
            payload["predictive_diagnostics"] = {
                "predicted_fault": "AIR_LEAK" if assignment.state_label == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value else "NORMAL",
                "fault_probability": 0.85 if assignment.state_label == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value else 0.05,
                "estimated_rul_hours": 450.0 if not b_dict["drift_detected"] else 72.0,
                "rul_confidence_interval_95": [400.0, 500.0] if not b_dict["drift_detected"] else [50.0, 95.0],
                "health_index": round(float(np.clip(1.0 - composite_score, 0.0, 1.0)), 4),
            }

        self._last_state_payload = payload
        return payload

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Returns the most recent complete digital twin state payload snapshot conforming to
        digital_twin_state_schema.json. If no step has been executed yet, steps with nominal healthy packet.
        """
        if self._last_state_payload is not None:
            return self._last_state_payload
        return self.step()

    def get_c04_prescriptive_payload(self) -> Dict[str, Any]:
        """
        Extracts standardized payload for Component 04 RL Maintenance Optimization
        strictly conforming to shared/interfaces/c03_to_c04_twin_state_contract.json.
        """
        if self._last_snapshot is None or self._last_state is None:
            raise RuntimeError("Engine has not executed any steps yet.")

        return self._format_downstream_payload(
            self._last_snapshot,
            self._last_state,
            self._last_behavioral_dict,
            self._last_degradation_dict,
        )

    def _evaluate_subsystems(
        self,
        snapshot: AlignedSnapshot,
        assignment: StateAssignment,
        assessment: BehavioralAssessment,
        degradation_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Decomposes asset health into 4 physical sub-assemblies modeled in the digital twin schematic.
        """
        comp_score = float(degradation_metrics["composite_degradation_score"])
        eff_ratio = float(degradation_metrics["pressure_efficiency_ratio"])

        # 1. Electric Motor Subsystem
        motor_temp = snapshot.temperature_celsius
        motor_curr = snapshot.motor_current_rms_a
        if motor_curr > 22.0:
            motor_status = "CURRENT_OVERLOAD"
            motor_health = max(0.0, 1.0 - comp_score - 0.3)
        elif motor_temp > 75.0:
            motor_status = "ELEVATED_TEMP"
            motor_health = max(0.0, 1.0 - comp_score - 0.2)
        elif comp_score > 0.7:
            motor_status = "CRITICAL"
            motor_health = max(0.0, 1.0 - comp_score)
        else:
            motor_status = "NOMINAL"
            motor_health = round(1.0 - 0.5 * comp_score, 3)

        # 2. Cylinder & Piston Assembly
        compression_rate = snapshot.delta_pressure_dt
        if eff_ratio < 0.6 and assignment.state_label == OperatingStateLabel.PRESSURIZING_LOADED.value:
            piston_status = "RING_BLOWBY"
            piston_health = round(eff_ratio, 3)
        elif assignment.state_label == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value:
            piston_status = "PRESSURE_LOSS"
            piston_health = max(0.2, round(eff_ratio - 0.2, 3))
        else:
            piston_status = "OPTIMAL" if eff_ratio >= 0.85 else "PRESSURE_LOSS"
            piston_health = round(eff_ratio, 3)

        # 3. Suction & Discharge Valves
        is_leak = assignment.state_label == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value
        if snapshot.acoustic_energy_mv > 500.0 and is_leak:
            valve_status = "LEAKAGE_DETECTED"
            valve_health = 0.35
        elif snapshot.vibration_kurtosis > 4.5:
            valve_status = "VALVE_CHATTER"
            valve_health = 0.55
        else:
            valve_status = "NORMAL"
            valve_health = round(1.0 - 0.4 * comp_score, 3)

        # 4. Main Bearing Assembly
        vib_x = snapshot.vibration_x_rms_ms2
        kurt = snapshot.vibration_kurtosis
        if kurt > 5.5 and vib_x > 4.5:
            bearing_status = "IMPACTING"
            bearing_health = 0.25
        elif vib_x > 4.0:
            bearing_status = "WEAR_DETECTED"
            bearing_health = 0.50
        else:
            bearing_status = "NOMINAL"
            bearing_health = round(max(0.0, 1.0 - (vib_x / 8.0)), 3)

        return {
            "electric_motor": {
                "health_score": round(float(np.clip(motor_health, 0.0, 1.0)), 3),
                "status": motor_status,
                "operating_temperature_c": round(motor_temp, 2),
                "rms_current_a": round(motor_curr, 3),
            },
            "cylinder_piston_assembly": {
                "health_score": round(float(np.clip(piston_health, 0.0, 1.0)), 3),
                "status": piston_status,
                "compression_rate_psi_s": round(compression_rate, 3),
                "seal_integrity_ratio": round(float(np.clip(eff_ratio, 0.0, 1.0)), 3),
            },
            "suction_discharge_valves": {
                "health_score": round(float(np.clip(valve_health, 0.0, 1.0)), 3),
                "status": valve_status,
                "flow_efficiency_ratio": round(float(np.clip(eff_ratio, 0.0, 1.0)), 3),
                "valve_leakage_detected": is_leak,
            },
            "main_bearing_assembly": {
                "health_score": round(float(np.clip(bearing_health, 0.0, 1.0)), 3),
                "status": bearing_status,
                "vibration_rms_ms2": round(vib_x, 2),
                "impact_kurtosis": round(kurt, 2),
            },
        }

    def _format_downstream_payload(
        self,
        snapshot: AlignedSnapshot,
        assignment: StateAssignment,
        b_dict: Dict[str, Any],
        m_dict: Dict[str, Any],
        ts_iso: Optional[str] = None,
        twin_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Structures payload conforming strictly to shared/interfaces/c03_to_c04_twin_state_contract.json.
        """
        if ts_iso is None:
            ts_iso = snapshot.timestamp_iso
            if "+" in ts_iso:
                ts_iso = ts_iso.split("+")[0] + "Z"
            elif not ts_iso.endswith("Z"):
                ts_iso = ts_iso + "Z"

        if twin_id is None:
            twin_id = self.asset_id if self.asset_id.startswith("TWIN-") else f"TWIN-{self.asset_id}"

        return {
            "timestamp": ts_iso,
            "twin_id": twin_id,
            "operating_state": {
                "cluster_id": assignment.cluster_id,
                "state_label": assignment.state_label,
                "state_confidence": assignment.state_confidence,
            },
            "behavioral_intelligence": {
                "deviation_index": b_dict["deviation_index"],
                "persistence_counter": b_dict["persistence_counter"],
                "drift_detected": b_dict["drift_detected"],
            },
            "degradation_metrics": {
                "composite_degradation_score": m_dict["composite_degradation_score"],
                "pressure_efficiency_ratio": m_dict["pressure_efficiency_ratio"],
                "thermal_dissipation_factor": m_dict.get("thermal_dissipation_factor", 0.0),
            },
        }
