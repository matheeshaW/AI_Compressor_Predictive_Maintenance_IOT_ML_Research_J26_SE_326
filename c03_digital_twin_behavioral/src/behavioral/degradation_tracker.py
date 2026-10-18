"""
Multi-window temporal persistence filtering and mechanical/thermodynamic degradation tracking.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .state_behavioral_profiler import BehavioralAssessment
from ..clustering.state_discovery import OperatingStateLabel


class DeviationClassification(str, Enum):
    """
    Categorical qualification of observed behavioral deviation conforming to
    c03_digital_twin_behavioral/schemas/digital_twin_state_schema.json.
    """
    NOMINAL = "NOMINAL"
    TRANSIENT_PERTURBATION = "TRANSIENT_PERTURBATION"
    PERSISTENT_DEGRADATION = "PERSISTENT_DEGRADATION"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"


@dataclass
class DegradationMetrics:
    """
    Thermodynamic and mechanical degradation indicators conforming to digital twin state schema.
    """
    composite_degradation_score: float  # [0.0, 1.0] (0.0 = healthy, 1.0 = failed)
    pressure_efficiency_ratio: float    # [0.0, 1.0] (measured dP/dt vs nominal)
    thermal_dissipation_factor: float   # °C / Amp
    mechanical_vibration_severity: float # ISO 10816-3 normalized severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_degradation_score": round(float(self.composite_degradation_score), 4),
            "pressure_efficiency_ratio": round(float(self.pressure_efficiency_ratio), 4),
            "thermal_dissipation_factor": round(float(self.thermal_dissipation_factor), 3),
            "mechanical_vibration_severity": round(float(self.mechanical_vibration_severity), 3),
        }


class DegradationTracker:
    """
    Dynamic behavioral boundary tracking engine.
    Distinguishes transient operational noise (e.g. motor start surges, momentary valve clicks)
    from persistent irreversible mechanical degradation via multi-window persistence filtering.
    Calculates composite degradation scores and thermodynamic efficiency indices.
    """

    def __init__(
        self,
        deviation_threshold: float = 3.0,
        persistence_threshold: int = 5,
        nominal_compression_rate_psi_s: float = 1.4,
        ambient_temperature_c: float = 25.0,
        ema_smoothing_alpha: float = 0.1,
    ):
        """
        Args:
            deviation_threshold: Sigma/distance threshold defining the normal operational envelope (default 3.0).
            persistence_threshold: Consecutive anomalous windows required before declaring persistent drift (default 5).
            nominal_compression_rate_psi_s: Baseline pressurization rate during healthy loaded pumping.
            ambient_temperature_c: Reference ambient baseline temperature in Celsius.
            ema_smoothing_alpha: Exponential moving average coefficient for degradation score smoothing.
        """
        if deviation_threshold <= 0:
            raise ValueError("deviation_threshold must be strictly positive.")
        if persistence_threshold < 1:
            raise ValueError("persistence_threshold must be at least 1.")

        self.deviation_threshold = float(deviation_threshold)
        self.persistence_threshold = int(persistence_threshold)
        self.nominal_compression_rate = max(0.1, float(nominal_compression_rate_psi_s))
        self.ambient_temperature_c = float(ambient_temperature_c)
        self.ema_alpha = float(ema_smoothing_alpha)

        # Persistence filtering state
        self.persistence_counter: int = 0
        self.total_evaluations: int = 0
        self.drift_detected: bool = False
        self.current_classification = DeviationClassification.NOMINAL

        # Moving degradation indicators
        self._composite_score: float = 0.0
        self._pressure_efficiency: float = 1.0
        self._recent_deviations: deque = deque(maxlen=50)

    def update(
        self,
        assessment: BehavioralAssessment,
        telemetry: Union[Dict[str, Any], Any],
        delta_t_sec: float = 0.04,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Ingests a behavioral assessment and telemetry snapshot.
        Updates persistence counters, filters out transient noise,
        and computes degradation indicators.

        Returns:
            Tuple of:
                - behavioral_intelligence_dict: Conforms to schema 'behavioral_intelligence' object.
                - degradation_metrics_dict: Conforms to schema 'degradation_metrics' object.
        """
        self.total_evaluations += 1
        d_m = float(assessment.deviation_index)
        self._recent_deviations.append(d_m)

        # 1. Update temporal persistence counter
        if d_m > self.deviation_threshold:
            self.persistence_counter += 1
        else:
            self.persistence_counter = max(0, self.persistence_counter - 1)

        self.drift_detected = bool(self.persistence_counter >= self.persistence_threshold)

        # 2. Categorize deviation classification
        if d_m >= (2.0 * self.deviation_threshold):
            self.current_classification = DeviationClassification.CRITICAL_DRIFT
        elif self.drift_detected:
            self.current_classification = DeviationClassification.PERSISTENT_DEGRADATION
        elif self.persistence_counter > 0:
            self.current_classification = DeviationClassification.TRANSIENT_PERTURBATION
        else:
            self.current_classification = DeviationClassification.NOMINAL

        # 3. Calculate thermodynamic & mechanical indicators
        dp_dt = float(
            telemetry.get("delta_pressure_dt", 0.0)
            if isinstance(telemetry, dict)
            else getattr(telemetry, "delta_pressure_dt", 0.0)
        )
        temp_c = float(
            telemetry.get("temperature_celsius", self.ambient_temperature_c)
            if isinstance(telemetry, dict)
            else getattr(telemetry, "temperature_celsius", self.ambient_temperature_c)
        )
        current_a = float(
            telemetry.get("motor_current_rms_a", 0.0)
            if isinstance(telemetry, dict)
            else getattr(telemetry, "motor_current_rms_a", 0.0)
        )
        vib_x = float(
            telemetry.get("vibration_x_rms_ms2", 0.0)
            if isinstance(telemetry, dict)
            else getattr(telemetry, "vibration_x_rms_ms2", 0.0)
        )

        # Pressure efficiency ratio (updated dynamically when loaded)
        if assessment.state_label == OperatingStateLabel.PRESSURIZING_LOADED.value and dp_dt > 0.1:
            instant_eff = float(np.clip(dp_dt / self.nominal_compression_rate, 0.0, 1.0))
            self._pressure_efficiency = 0.9 * self._pressure_efficiency + 0.1 * instant_eff
        elif assessment.state_label == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value:
            self._pressure_efficiency = max(0.0, self._pressure_efficiency - 0.02)

        # Thermal dissipation factor: delta T / motor current (°C/A)
        temp_rise = max(0.0, temp_c - self.ambient_temperature_c)
        thermal_factor = float(temp_rise / max(1.0, current_a))

        # ISO 10816-3 mechanical vibration severity index (4.5 m/s² permissible baseline threshold)
        vib_severity = float(vib_x / 4.5)

        # Composite degradation score: weighted combination in [0.0, 1.0]
        drift_penalty = float(np.clip(d_m / (2.0 * self.deviation_threshold), 0.0, 1.0))
        eff_loss = float(1.0 - self._pressure_efficiency)
        vib_penalty = float(np.clip(vib_severity - 1.0, 0.0, 1.0)) if vib_severity > 1.0 else 0.0

        raw_degradation = 0.45 * drift_penalty + 0.35 * eff_loss + 0.20 * vib_penalty
        self._composite_score = (
            self.ema_alpha * raw_degradation + (1.0 - self.ema_alpha) * self._composite_score
        )
        self._composite_score = float(np.clip(self._composite_score, 0.0, 1.0))

        # Formulate payload strictly matching digital_twin_state_schema.json
        behavioral_dict = {
            "deviation_index": round(d_m, 4),
            "deviation_threshold": self.deviation_threshold,
            "persistence_counter": self.persistence_counter,
            "persistence_threshold": self.persistence_threshold,
            "deviation_classification": self.current_classification.value,
            "drift_detected": self.drift_detected,
            "state_cluster_centroid": {k: round(v, 4) for k, v in assessment.state_cluster_centroid.items()},
            "state_cluster_variance": {k: round(v, 4) for k, v in assessment.state_cluster_variance.items()},
        }

        metrics_obj = DegradationMetrics(
            composite_degradation_score=self._composite_score,
            pressure_efficiency_ratio=self._pressure_efficiency,
            thermal_dissipation_factor=thermal_factor,
            mechanical_vibration_severity=vib_severity,
        )

        return behavioral_dict, metrics_obj.to_dict()

    def reset(self) -> None:
        """
        Resets persistence counters and tracking registers.
        """
        self.persistence_counter = 0
        self.drift_detected = False
        self.current_classification = DeviationClassification.NOMINAL
        self._composite_score = 0.0
        self._pressure_efficiency = 1.0
        self._recent_deviations.clear()
