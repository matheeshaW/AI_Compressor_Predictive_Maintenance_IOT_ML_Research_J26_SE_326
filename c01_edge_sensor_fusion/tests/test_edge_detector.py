#!/usr/bin/env python3
"""
test_edge_detector.py
Unit tests and mathematical verification for C01 Edge Anomaly Detector & Interlock
Author: Weerakoon W.M.M.B (IT23155152) - J26-SE-326
"""

import os
import json
import unittest
import numpy as np
import pandas as pd


class PythonEdgeDetector:
    """Python reference implementation mirroring C++ StatisticalAnomalyDetector."""

    def __init__(self, thresholds_path: str):
        with open(thresholds_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.features = cfg["features"]
        stats = cfg["feature_baseline_statistics"]
        self.means = np.array([stats[col]["mean"] for col in self.features])
        self.stds = np.array([stats[col]["std"] for col in self.features])
        self.std_inv = 1.0 / np.maximum(self.stds, 1e-4)

        th = cfg["thresholds"]
        self.tau_dm = th["mahalanobis_threshold_dm"]
        self.hard_p = th["pressure_safety_cutoff_psi"]
        self.hard_kurt = th["kurtosis_critical_knock"]
        self.hard_cf = th["crest_factor_severe_impact"]
        self.hard_temp = th["temperature_thermal_trip_celsius"]
        self.debounce_thresh = th["persistence_debounce_frames"]
        self.debounce_counter = 0

    def evaluate(self, x_dict: dict, current_pressure: float, current_temp: float) -> dict:
        res = {
            "is_anomalous": False,
            "anomaly_score": 0.0,
            "mahalanobis_distance": 0.0,
            "trip_relay": False,
            "fault_cause": "NORMAL"
        }

        # 1. Immediate Critical Hardware Checks
        if current_pressure >= self.hard_p:
            res["is_anomalous"] = True
            res["anomaly_score"] = 1.0
            res["mahalanobis_distance"] = 12.0
            res["trip_relay"] = True
            res["fault_cause"] = "CRITICAL_OVERPRESSURE"
            return res

        if current_temp >= self.hard_temp:
            res["is_anomalous"] = True
            res["anomaly_score"] = 0.95
            res["trip_relay"] = True
            res["fault_cause"] = "THERMAL_OVERHEAT"
            return res

        # 2. Compute weighted distance
        x_vec = np.array([x_dict.get(col, 0.0) for col in self.features])
        z = (x_vec - self.means) * self.std_inv
        dm = float(np.sqrt(np.sum(z**2)))
        res["mahalanobis_distance"] = round(dm, 3)
        res["anomaly_score"] = round(dm / (dm + self.tau_dm), 3)

        # 3. Fault classification
        kurt = x_dict.get("vibration_kurtosis", 3.0)
        cf = x_dict.get("vibration_crest_factor", 1.5)
        acute_knock = (kurt >= self.hard_kurt or cf >= self.hard_cf)
        stat_outlier = (dm >= self.tau_dm)

        if acute_knock or stat_outlier:
            self.debounce_counter += 1
            if self.debounce_counter >= self.debounce_thresh:
                res["is_anomalous"] = True
                if acute_knock:
                    res["fault_cause"] = "BEARING_VALVE_KNOCK"
                    if kurt >= (self.hard_kurt * 1.3):
                        res["trip_relay"] = True
                elif current_pressure >= (self.hard_p - 3.0):
                    res["fault_cause"] = "PRESSURE_SURGE"
                else:
                    res["fault_cause"] = "MULTIMODAL_ANOMALY"
        else:
            if self.debounce_counter > 0:
                self.debounce_counter -= 1

        return res


class TestEdgeAnomalyDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.thresh_path = os.path.join(base_dir, "edge_models", "anomaly_thresholds.json")
        cls.detector = PythonEdgeDetector(cls.thresh_path)
        cls.contract_path = os.path.abspath(
            os.path.join(base_dir, "..", "shared", "interfaces", "c01_to_c02_telemetry_contract.json")
        )

    def setUp(self):
        self.detector.debounce_counter = 0

    def test_nominal_operating_condition(self):
        """Under nominal steady state, the detector outputs NORMAL and trips nothing."""
        nominal_x = {
            "vibration_composite_rms_ms2": 9.81,
            "vibration_kurtosis": 3.0,
            "vibration_crest_factor": 1.6,
            "pressure_psi": 32.1,
            "motor_current_rms_a": 4.8,
            "acoustic_energy_mv": 85.0
        }
        res = self.detector.evaluate(nominal_x, current_pressure=32.1, current_temp=28.5)
        self.assertFalse(res["is_anomalous"])
        self.assertFalse(res["trip_relay"])
        self.assertLess(res["anomaly_score"], 0.5)
        self.assertEqual(res["fault_cause"], "NORMAL")

    def test_critical_pressure_emergency_trip(self):
        """Pressure exceeding 45 PSI immediately cuts off relay without debounce waiting."""
        x = {
            "vibration_composite_rms_ms2": 9.81,
            "vibration_kurtosis": 3.0,
            "vibration_crest_factor": 1.5,
            "pressure_psi": 46.2,
            "motor_current_rms_a": 7.5,
            "acoustic_energy_mv": 90.0
        }
        res = self.detector.evaluate(x, current_pressure=46.2, current_temp=30.0)
        self.assertTrue(res["is_anomalous"])
        self.assertTrue(res["trip_relay"])
        self.assertEqual(res["fault_cause"], "CRITICAL_OVERPRESSURE")
        self.assertAlmostEqual(res["anomaly_score"], 1.0)

    def test_bearing_knock_impulse_detection(self):
        """Acute Kurtosis spike (>7.5) triggers BEARING_VALVE_KNOCK."""
        knock_x = {
            "vibration_composite_rms_ms2": 13.5,
            "vibration_kurtosis": 8.8,
            "vibration_crest_factor": 6.8,
            "pressure_psi": 31.8,
            "motor_current_rms_a": 5.2,
            "acoustic_energy_mv": 240.0
        }
        # Debounce frame 1
        res1 = self.detector.evaluate(knock_x, current_pressure=31.8, current_temp=31.0)
        # Debounce frame 2
        res2 = self.detector.evaluate(knock_x, current_pressure=31.8, current_temp=31.0)

        self.assertTrue(res2["is_anomalous"])
        self.assertEqual(res2["fault_cause"], "BEARING_VALVE_KNOCK")
        self.assertGreater(res2["anomaly_score"], 0.6)

    def test_thermal_overheat_fail_safe(self):
        """Temperature exceeding 85.0 deg C triggers emergency thermal trip."""
        x = {
            "vibration_composite_rms_ms2": 9.81,
            "vibration_kurtosis": 3.0,
            "vibration_crest_factor": 1.5,
            "pressure_psi": 32.0,
            "motor_current_rms_a": 4.8,
            "acoustic_energy_mv": 85.0
        }
        res = self.detector.evaluate(x, current_pressure=32.0, current_temp=87.5)
        self.assertTrue(res["is_anomalous"])
        self.assertTrue(res["trip_relay"])
        self.assertEqual(res["fault_cause"], "THERMAL_OVERHEAT")

    def test_telemetry_contract_with_anomaly_score(self):
        """Verifies that an edge frame with anomaly_score complies with JSON schema."""
        from jsonschema import validate
        with open(self.contract_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        sample_packet = {
            "device_id": "ESP32S3-COMPRESSOR-TESTBED-01",
            "timestamp": "2026-10-12T10:00:00.000Z",
            "sample_rate_hz": 25.0,
            "features": {
                "temperature_celsius": 28.5,
                "pressure_psi": 32.1,
                "delta_pressure_dt": 0.05,
                "motor_current_rms_a": 4.82,
                "vibration_x_rms_ms2": 9.81,
                "vibration_y_rms_ms2": 0.15,
                "vibration_z_rms_ms2": 0.12,
                "vibration_kurtosis": 3.02,
                "vibration_crest_factor": 1.55,
                "acoustic_energy_mv": 86.5
            },
            "edge_status": {
                "anomaly_flag": False,
                "anomaly_score": 0.12,
                "relay_latched": False,
                "sram_free_kb": 312.5
            }
        }
        validate(instance=sample_packet, schema=schema)


if __name__ == "__main__":
    unittest.main()
