"""
degradation_simulator.py
========================
Physical Degradation Dynamics and Failure Transition Engine for Industrial Air Compressors.
Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np


class CompressorDegradationSimulator:
    """
    Simulates thermo-mechanical degradation dynamics, multi-subsystem wear accumulation,
    and catastrophic failure hazard transitions for industrial reciprocating compressors.
    
    Subsystems modeled:
    1. Pneumatic Sealing & Couplers (air leaks, pressure drop)
    2. Valve Train & Head Gasket (reed valve fluttering, backflow, pressure efficiency decay)
    3. Mechanical Bearings & Crankshaft (friction, motor current surge, vibration kurtosis)
    """

    # Subsystem degradation weights forming the composite degradation index
    WEIGHT_PNEUMATIC = 0.30
    WEIGHT_VALVE = 0.40
    WEIGHT_BEARING = 0.30

    # Operational Regime Multipliers (amplification on baseline wear rate)
    REGIME_WEAR_MULTIPLIERS = {
        0: 0.25,  # IDLE_UNLOADED: Motor unloaded, minimal frictional wear
        1: 2.20,  # PRESSURIZING_LOADED: Full compression stroke, maximum thermal-mechanical load
        2: 0.60,  # HOLDING_PRESSURE: Cutoff standby, intermittent top-up
        3: 1.00,  # BLOWDOWN_DISCHARGE: Condensate venting, standard cycling
        4: 3.50,  # ANOMALOUS_LEAK_REGIME: Continuous pumping against severe leakage, extreme stress
    }

    def __init__(
        self,
        base_wear_rate: float = 0.0018,
        noise_std: float = 0.0004,
        hazard_baseline: float = 0.0005,
        hazard_steepness: float = 5.2,
        seed: Optional[int] = None,
    ):
        """
        Args:
            base_wear_rate: Average degradation increment per inspection step under normal load.
            noise_std: Gaussian stochastic variation in degradation increment.
            hazard_baseline: Baseline stochastic failure hazard rate lambda_0.
            hazard_steepness: Exponential sensitivity factor kappa for wear-to-failure hazard.
            seed: Optional random seed for reproducible simulation.
        """
        self.base_wear_rate = base_wear_rate
        self.noise_std = noise_std
        self.hazard_baseline = hazard_baseline
        self.hazard_steepness = hazard_steepness

        self.rng = np.random.default_rng(seed)

        # State tracking
        self.operating_hours: float = 0.0
        self.hours_since_last_maint: float = 0.0
        self.consecutive_anomaly_steps: int = 0
        self.total_failures: int = 0

        # Subsystem degradation levels [0.0 = pristine, 1.0 = total failure]
        self.pneumatic_wear: float = 0.0
        self.valve_wear: float = 0.0
        self.bearing_wear: float = 0.0

        # Current regime
        self.current_regime: int = 1  # default to PRESSURIZING_LOADED

    def reset(self, initial_wear: Optional[float] = None) -> Dict[str, Any]:
        """Resets the simulator state to pristine or specified initial condition."""
        if initial_wear is None:
            # Minor random bench run-in wear
            self.pneumatic_wear = float(self.rng.uniform(0.01, 0.05))
            self.valve_wear = float(self.rng.uniform(0.01, 0.04))
            self.bearing_wear = float(self.rng.uniform(0.01, 0.03))
        else:
            w = max(0.0, min(1.0, initial_wear))
            self.pneumatic_wear = w
            self.valve_wear = w
            self.bearing_wear = w

        self.operating_hours = 0.0
        self.hours_since_last_maint = 0.0
        self.consecutive_anomaly_steps = 0
        self.total_failures = 0
        self.current_regime = 1

        return self.get_telemetry()

    @property
    def composite_degradation(self) -> float:
        """Calculates normalized composite degradation score D_t in [0.0, 1.0]."""
        comp = (
            self.WEIGHT_PNEUMATIC * self.pneumatic_wear
            + self.WEIGHT_VALVE * self.valve_wear
            + self.WEIGHT_BEARING * self.bearing_wear
        )
        return float(np.clip(comp, 0.0, 1.0))

    def step_dynamics(
        self,
        hours_elapsed: float = 4.0,
        regime: Optional[int] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Simulates natural equipment degradation over an operational step (Do-Nothing operation).
        
        Returns:
            failed: Boolean indicating whether an unscheduled catastrophic failure occurred.
            info: Diagnostics dictionary detailing wear increments and failure cause.
        """
        if regime is not None:
            self.current_regime = regime

        self.operating_hours += hours_elapsed
        self.hours_since_last_maint += hours_elapsed

        regime_mult = self.REGIME_WEAR_MULTIPLIERS.get(self.current_regime, 1.0)
        
        # Thermal acceleration factor based on current bearing wear
        thermal_factor = 1.0 + 1.2 * (self.bearing_wear ** 1.8)

        # 1. Subsystem wear increments
        noise = self.rng.normal(0.0, self.noise_std, size=3)
        delta_p = (self.base_wear_rate * 1.1 * regime_mult * (1.0 + 0.5 * self.pneumatic_wear)) + noise[0]
        delta_v = (self.base_wear_rate * 1.3 * regime_mult * thermal_factor) + noise[1]
        delta_b = (self.base_wear_rate * 0.9 * regime_mult * (1.0 + 0.8 * self.bearing_wear)) + noise[2]

        self.pneumatic_wear = float(np.clip(self.pneumatic_wear + max(0.0, delta_p), 0.0, 1.0))
        self.valve_wear = float(np.clip(self.valve_wear + max(0.0, delta_v), 0.0, 1.0))
        self.bearing_wear = float(np.clip(self.bearing_wear + max(0.0, delta_b), 0.0, 1.0))

        comp = self.composite_degradation

        # Update anomaly persistence
        if comp > 0.40:
            self.consecutive_anomaly_steps += 1
        else:
            self.consecutive_anomaly_steps = max(0, self.consecutive_anomaly_steps - 1)

        # 2. Catastrophic Breakdown Hazard Check
        # Hazard increases exponentially with composite degradation
        hazard_prob = 1.0 - math.exp(-self.hazard_baseline * math.exp(self.hazard_steepness * comp))
        hazard_prob = min(hazard_prob, 0.99)

        catastrophic_failure = False
        failure_cause = "NORMAL"

        if comp >= 1.0:
            catastrophic_failure = True
            failure_cause = "END_OF_LIFE_EXHAUSTION"
        elif self.rng.random() < hazard_prob:
            catastrophic_failure = True
            if self.valve_wear >= max(self.pneumatic_wear, self.bearing_wear):
                failure_cause = "VALVE_SHATTER_BREAKDOWN"
            elif self.bearing_wear >= self.pneumatic_wear:
                failure_cause = "BEARING_SEIZURE_MOTOR_BURNOUT"
            else:
                failure_cause = "PRESSURE_LINE_BURST_FAILURE"

        if catastrophic_failure:
            self.total_failures += 1

        info = {
            "catastrophic_failure": catastrophic_failure,
            "failure_cause": failure_cause,
            "hazard_probability": hazard_prob,
            "composite_degradation": comp,
            "pneumatic_wear": self.pneumatic_wear,
            "valve_wear": self.valve_wear,
            "bearing_wear": self.bearing_wear,
            "operating_hours": self.operating_hours,
        }

        return catastrophic_failure, info

    def apply_maintenance(self, action: int) -> Dict[str, Any]:
        """
        Executes physical maintenance action and updates subsystem degradation.
        
        Actions:
            1: Minor Pneumatic Service (Filters, fittings, leak traps) -> restores pneumatic wear by 80%
            2: Medium Mechanical Overhaul (Reed valves, gaskets, lubricant flush) -> restores valve wear by 90%, bearing by 40%
            3: Major Overhaul / Component Replacement (Motor rewind, cylinder block) -> full reset
        """
        prev_comp = self.composite_degradation
        action_name = "NONE"

        if action == 1:
            action_name = "MINOR_PNEUMATIC_SERVICE"
            # Restores pneumatic seals and cleans air path
            self.pneumatic_wear = float(np.clip(self.pneumatic_wear * 0.20 + self.rng.normal(0, 0.01), 0.01, 1.0))
            self.valve_wear = float(np.clip(self.valve_wear * 0.90, 0.01, 1.0))
            self.hours_since_last_maint = 0.0

        elif action == 2:
            action_name = "MEDIUM_MECHANICAL_OVERHAUL"
            # Overhauls valve head, resets valve wear and part of bearing wear
            self.valve_wear = float(np.clip(self.valve_wear * 0.10 + self.rng.normal(0, 0.01), 0.01, 1.0))
            self.pneumatic_wear = float(np.clip(self.pneumatic_wear * 0.30, 0.01, 1.0))
            self.bearing_wear = float(np.clip(self.bearing_wear * 0.60, 0.01, 1.0))
            self.hours_since_last_maint = 0.0

        elif action == 3:
            action_name = "MAJOR_COMPONENT_REPLACEMENT"
            # Pristine factory overhaul
            self.pneumatic_wear = float(self.rng.uniform(0.01, 0.03))
            self.valve_wear = float(self.rng.uniform(0.01, 0.03))
            self.bearing_wear = float(self.rng.uniform(0.01, 0.03))
            self.hours_since_last_maint = 0.0
            self.consecutive_anomaly_steps = 0

        new_comp = self.composite_degradation

        return {
            "action": action,
            "action_name": action_name,
            "degradation_reduction": max(0.0, prev_comp - new_comp),
            "new_composite_degradation": new_comp,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Synthesizes physical sensor telemetry and predictive metrics conditioned
        on the underlying degradation state and operating regime.
        """
        comp = self.composite_degradation
        
        # Pressure delivery efficiency (drops as leaks and valve wear progress)
        eta_p = float(np.clip(1.0 - (0.45 * self.pneumatic_wear + 0.35 * self.valve_wear), 0.20, 1.0))
        delivered_pressure_bar = float(6.5 * eta_p + self.rng.normal(0, 0.08))
        delivered_pressure_psi = delivered_pressure_bar * 14.5038

        # Motor Current (rises as bearings wear and compression efficiency drops)
        motor_current_a = float(5.2 + (4.8 * self.bearing_wear ** 1.5) + (1.2 * self.valve_wear) + self.rng.normal(0, 0.12))
        thermal_dissipation = float(1.0 + 0.85 * self.bearing_wear + 0.35 * (1.0 - eta_p))
        head_temp_c = float(32.0 + 45.0 * comp + self.rng.normal(0, 0.8))

        # Vibration metrics
        vib_kurtosis = float(3.0 + 8.5 * (self.valve_wear ** 2) + 6.0 * (self.bearing_wear ** 2) + self.rng.normal(0, 0.2))
        vib_rms_g = float(0.85 + 2.4 * comp + self.rng.normal(0, 0.05))

        # Behavioral deviation index (Mahalanobis distance equivalent from C03)
        deviation_index = float(0.4 + 4.8 * (comp ** 1.6) + self.rng.normal(0, 0.15))

        # RUL estimation (in hours, based on max operational life of 1200 hrs)
        max_life_hours = 1200.0
        rul_hours = max(0.0, (1.0 - comp) * max_life_hours + self.rng.normal(0, 15.0))
        normalized_rul = float(np.clip(rul_hours / max_life_hours, 0.0, 1.0))

        # Anomaly score in [0.0, 1.0]
        anomaly_score = float(np.clip(0.05 + 0.90 * (comp ** 1.4) + self.rng.normal(0, 0.02), 0.0, 1.0))

        return {
            "composite_degradation": comp,
            "pneumatic_wear": self.pneumatic_wear,
            "valve_wear": self.valve_wear,
            "bearing_wear": self.bearing_wear,
            "pressure_efficiency": eta_p,
            "delivered_pressure_bar": delivered_pressure_bar,
            "delivered_pressure_psi": delivered_pressure_psi,
            "motor_current_a": motor_current_a,
            "thermal_dissipation": thermal_dissipation,
            "head_temp_c": head_temp_c,
            "vibration_kurtosis": vib_kurtosis,
            "vibration_rms_g": vib_rms_g,
            "deviation_index": deviation_index,
            "anomaly_score": anomaly_score,
            "rul_hours": rul_hours,
            "normalized_rul": normalized_rul,
            "operating_regime": self.current_regime,
            "consecutive_anomaly_steps": self.consecutive_anomaly_steps,
            "hours_since_last_maint": self.hours_since_last_maint,
            "operating_hours": self.operating_hours,
        }
