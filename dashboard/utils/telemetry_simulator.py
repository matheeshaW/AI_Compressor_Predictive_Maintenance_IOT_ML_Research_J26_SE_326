"""
telemetry_simulator.py
======================
High-fidelity physical telemetry generator simulating the 12V DC Metal Air Compressor
testbed across Nominal, Leak, Bearing Knock, and Overpressure regimes.
"""

import time
import math
import random
from typing import Dict, Any, List


class CompressorTelemetrySimulator:
    """Simulates realistic 25 Hz / 1 Hz multi-modal telemetry conforming to the C01 contract."""

    def __init__(self, device_id: str = "ESP32S3-COMP-01"):
        self.device_id = device_id
        self.step_idx = 0
        self.pressure = 24.0  # PSI
        self.temperature = 34.0  # Celsius
        self.relay_state = "ENERGIZED_CLOSED"
        self.active_regime = "NOMINAL"
        self.anomaly_score = 0.05
        self.fault_cause = "NORMAL"

    def set_fault_regime(self, regime: str):
        """Injects a specific operational regime or fault mode."""
        self.active_regime = regime
        # Automatically re-arm relay so the simulator executes the new regime immediately
        self.relay_state = "ENERGIZED_CLOSED"

        if regime == "NOMINAL":
            self.fault_cause = "NORMAL"
            self.anomaly_score = 0.05
            self.pressure = max(self.pressure, 26.0)
        elif regime == "AIR_LEAK":
            self.fault_cause = "PNEUMATIC_AIR_LEAK"
            self.anomaly_score = 0.75
            self.pressure = max(self.pressure, 25.0)
        elif regime == "BEARING_VALVE_KNOCK":
            self.fault_cause = "BEARING_VALVE_KNOCK"
            self.anomaly_score = 0.88
            self.pressure = max(self.pressure, 26.0)
        elif regime == "CRITICAL_OVERPRESSURE":
            self.fault_cause = "CRITICAL_OVERPRESSURE"
            self.anomaly_score = 0.95
            self.pressure = max(self.pressure, 34.0)
        elif regime == "THERMAL_OVERHEAT":
            self.fault_cause = "THERMAL_OVERHEAT"
            self.anomaly_score = 0.85
            self.temperature = max(self.temperature, 68.0)


    def manual_start(self):
        """Manually arms the relay and starts the compressor."""
        self.relay_state = "ENERGIZED_CLOSED"
        self.fault_cause = "NORMAL"
        self.active_regime = "NOMINAL"
        self.anomaly_score = 0.05

    def manual_stop(self):
        """Manually trips the relay (E-Stop)."""
        self.relay_state = "TRIPPED_OPEN"
        self.fault_cause = "MANUAL_ESTOP"
        self.anomaly_score = 0.0

    def reset_interlock(self):
        """Resets emergency interlock and returns machine to nominal armed state."""
        self.relay_state = "ENERGIZED_CLOSED"
        self.fault_cause = "NORMAL"
        self.active_regime = "NOMINAL"
        self.anomaly_score = 0.05

    def step(self) -> Dict[str, Any]:
        """Advances the physical simulation by one time step (~1.0s or frame step)."""
        self.step_idx += 1
        t = self.step_idx

        # If relay tripped, motor stops and pressure slowly bleeds
        if self.relay_state == "TRIPPED_OPEN":
            self.pressure = max(0.0, self.pressure - 0.4)
            self.temperature = max(28.0, self.temperature - 0.15)
            motor_current = 0.0
            vibration_rms = 0.02
            vibration_kurtosis = 3.0
            crest_factor = 1.0
            acoustic_energy = 15.0
            dp_dt = -0.4
            anomaly_score = 0.0 if self.fault_cause == "MANUAL_ESTOP" else 0.98
            return self._build_packet(t, motor_current, dp_dt, vibration_rms, vibration_kurtosis, crest_factor, acoustic_energy, anomaly_score)

        # Active physical dynamics based on regime
        if self.active_regime == "NOMINAL":
            # Normal steady state: oscillating around 28-32 PSI
            target_p = 30.0 + 2.5 * math.sin(t * 0.1)
            self.pressure += (target_p - self.pressure) * 0.15 + random.uniform(-0.1, 0.1)
            self.temperature = min(58.0, self.temperature + 0.05)
            dp_dt = 0.12 * math.cos(t * 0.1)
            motor_current = 8.2 + 0.11 * self.pressure + random.uniform(-0.15, 0.15)
            vibration_rms = 0.35 + 0.02 * math.sin(t * 0.2) + random.uniform(-0.02, 0.02)
            vibration_kurtosis = 3.05 + random.uniform(-0.15, 0.2)
            crest_factor = 3.1 + random.uniform(-0.1, 0.15)
            acoustic_energy = 112.0 + random.uniform(-5.0, 5.0)
            anomaly_score = 0.08 + 0.05 * random.random()
            self.fault_cause = "NORMAL"

        elif self.active_regime == "AIR_LEAK":
            # Significant pneumatic pressure decay despite high motor current
            self.pressure = max(8.0, self.pressure - 0.6 + random.uniform(-0.1, 0.1))
            self.temperature = min(68.0, self.temperature + 0.12)
            dp_dt = -0.6
            motor_current = 9.8 + random.uniform(-0.2, 0.2)  # High load fighting leak
            vibration_rms = 0.42 + random.uniform(-0.03, 0.03)
            vibration_kurtosis = 3.4 + random.uniform(-0.2, 0.3)
            crest_factor = 3.4 + random.uniform(-0.1, 0.2)
            acoustic_energy = 185.0 + random.uniform(-10.0, 10.0)  # Hissing noise
            anomaly_score = min(0.85, 0.55 + 0.02 * (t % 20))
            self.fault_cause = "PNEUMATIC_AIR_LEAK"

        elif self.active_regime == "BEARING_VALVE_KNOCK":
            # Normal pressure but severe impulsive vibration spikes (high Kurtosis)
            self.pressure = 30.0 + random.uniform(-0.2, 0.2)
            self.temperature = min(72.0, self.temperature + 0.18)
            dp_dt = 0.02
            motor_current = 9.1 + random.uniform(-0.3, 0.3)
            vibration_rms = 0.68 + random.uniform(-0.05, 0.05)
            vibration_kurtosis = 8.4 + random.uniform(-0.5, 0.8)  # Acute Kurtosis spike > 7.5!
            crest_factor = 6.8 + random.uniform(-0.3, 0.5)        # Severe Crest Factor > 6.0!
            acoustic_energy = 240.0 + random.uniform(-15.0, 15.0)
            anomaly_score = 0.92
            self.fault_cause = "BEARING_VALVE_KNOCK"

            # Auto-trip test: if persistent knock, trip relay!
            if random.random() > 0.45:
                self.relay_state = "TRIPPED_OPEN"

        elif self.active_regime == "CRITICAL_OVERPRESSURE":
            # Pressure ramps up toward 45 PSI threshold
            self.pressure += 1.8
            dp_dt = 1.8
            motor_current = 10.5 + 0.15 * self.pressure
            vibration_rms = 0.55 + 0.01 * self.pressure
            vibration_kurtosis = 3.8
            crest_factor = 3.6
            acoustic_energy = 160.0
            anomaly_score = min(1.0, 0.6 + 0.02 * (self.pressure - 30.0))
            self.fault_cause = "PRESSURE_SURGE"

            # 45 PSI Emergency Cutoff Trigger
            if self.pressure >= 45.0:
                self.relay_state = "TRIPPED_OPEN"
                self.fault_cause = "CRITICAL_OVERPRESSURE"
                anomaly_score = 1.0

        elif self.active_regime == "THERMAL_OVERHEAT":
            self.temperature += 2.2
            dp_dt = 0.05
            motor_current = 9.5
            vibration_rms = 0.38
            vibration_kurtosis = 3.1
            crest_factor = 3.2
            acoustic_energy = 120.0
            anomaly_score = 0.88
            self.fault_cause = "THERMAL_OVERHEAT"
            if self.temperature >= 85.0:
                self.relay_state = "TRIPPED_OPEN"

        return self._build_packet(t, motor_current, dp_dt, vibration_rms, vibration_kurtosis, crest_factor, acoustic_energy, anomaly_score)

    def _build_packet(self, t: int, current: float, dp_dt: float, v_rms: float, kurt: float, cf: float, acoustic: float, score: float) -> Dict[str, Any]:
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {
            "timestamp_iso": iso_time,
            "device_id": self.device_id,
            "step": t,
            "motor_current_rms_amps": round(current, 2),
            "pressure_psi": round(self.pressure, 2),
            "pressure_rate_of_change_psi_sec": round(dp_dt, 3),
            "temperature_cylinder_head_c": round(self.temperature, 1),
            "vibration_rms_g": round(v_rms, 3),
            "vibration_kurtosis": round(kurt, 2),
            "vibration_crest_factor": round(cf, 2),
            "acoustic_energy_mv": round(acoustic, 1),
            "relay_state": self.relay_state,
            "anomaly_score": round(score, 3),
            "fault_cause": self.fault_cause,
            "trip_relay": (self.relay_state == "TRIPPED_OPEN")
        }
