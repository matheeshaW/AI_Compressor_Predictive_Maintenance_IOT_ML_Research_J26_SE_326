#!/usr/bin/env python3
"""
test_feature_extractor.py
Unit tests and mathematical verification for C01 DSP Feature Extraction algorithms
Author: Weerakoon W.M.M.B (IT23155152) - J26-SE-326
"""

import os
import unittest
import numpy as np
import pandas as pd


class PythonFeatureExtractor:
    """Python reference implementation mirroring C++ MultiModalFeatureExtractor."""

    @staticmethod
    def compute_rms(signal: np.ndarray) -> float:
        if len(signal) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(signal))))

    @staticmethod
    def compute_peak_to_peak(signal: np.ndarray) -> tuple:
        if len(signal) == 0:
            return 0.0, 0.0
        p2p = float(np.ptp(signal))
        peak_abs = float(np.max(np.abs(signal)))
        return p2p, peak_abs

    @staticmethod
    def compute_kurtosis(signal: np.ndarray) -> float:
        """Computes Pearson's Kurtosis (Normal distribution = 3.0)."""
        n = len(signal)
        if n < 4:
            return 3.0
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-5:
            return 3.0
        z = (signal - mean) / std
        return float(np.mean(np.power(z, 4)))

    @staticmethod
    def compute_crest_factor(signal: np.ndarray, zero_mean: bool = True) -> float:
        """Crest Factor: ratio of peak dynamic deviation to RMS of the AC vibration component."""
        if len(signal) == 0:
            return 0.0
        sig = signal - np.mean(signal) if zero_mean else signal
        rms = PythonFeatureExtractor.compute_rms(sig)
        if rms < 1e-4:
            return 0.0
        peak_abs = float(np.max(np.abs(sig)))
        return float(peak_abs / rms)

    @staticmethod
    def compute_delta_p_dt(pressures: np.ndarray, times_sec: np.ndarray) -> float:
        if len(pressures) < 2 or (times_sec[-1] - times_sec[0]) < 1e-3:
            return 0.0
        return float((pressures[-1] - pressures[0]) / (times_sec[-1] - times_sec[0]))


class TestDSPFeatureExtractor(unittest.TestCase):

    def test_sine_wave_metrics(self):
        """A pure sine wave with amplitude A=10 has theoretical RMS = 10 / sqrt(2) ~= 7.071,
        P2P = 20, Crest Factor = sqrt(2) ~= 1.414, and Kurtosis = 1.5."""
        t = np.linspace(0, 1.0, 1000)
        signal = 10.0 * np.sin(2 * np.pi * 50 * t)

        rms = PythonFeatureExtractor.compute_rms(signal)
        p2p, peak_abs = PythonFeatureExtractor.compute_peak_to_peak(signal)
        cf = PythonFeatureExtractor.compute_crest_factor(signal)
        kurt = PythonFeatureExtractor.compute_kurtosis(signal)

        self.assertAlmostEqual(rms, 10.0 / np.sqrt(2), delta=0.1)
        self.assertAlmostEqual(p2p, 20.0, delta=0.1)
        self.assertAlmostEqual(cf, np.sqrt(2), delta=0.05)
        self.assertAlmostEqual(kurt, 1.5, delta=0.1)

    def test_gaussian_normal_kurtosis(self):
        """A healthy Gaussian noise signal should exhibit mesokurtic Kurtosis ~ 3.0."""
        np.random.seed(42)
        normal_vibration = np.random.normal(loc=9.81, scale=0.4, size=2000)
        kurt = PythonFeatureExtractor.compute_kurtosis(normal_vibration)
        self.assertAlmostEqual(kurt, 3.0, delta=0.25)

    def test_impulsive_knocking_kurtosis_spike(self):
        """Anomalous mechanical knock / valve slap produces heavy-tailed spikes (Kurtosis > 5.0)."""
        np.random.seed(42)
        vibration = np.random.normal(loc=9.81, scale=0.3, size=1000)
        # Inject periodic high-amplitude shock pulses
        vibration[::100] += np.random.choice([6.0, -6.0], size=10)

        kurt = PythonFeatureExtractor.compute_kurtosis(vibration)
        cf = PythonFeatureExtractor.compute_crest_factor(vibration)

        self.assertGreater(kurt, 5.0, "Kurtosis should spike above 5.0 under impulsive mechanical knocking")
        self.assertGreater(cf, 4.0, "Crest factor should elevate significantly during impulsive impacts")

    def test_pressure_rate_of_change(self):
        """A linear compression ramp from 0 to 45 PSI over 10 seconds has dP/dt = 4.5 PSI/s."""
        times = np.linspace(0, 10.0, 100)
        pressures = 4.5 * times

        dp_dt = PythonFeatureExtractor.compute_delta_p_dt(pressures, times)
        self.assertAlmostEqual(dp_dt, 4.5, places=2)

    def test_real_baseline_dataset_features(self):
        """Verifies feature extraction against actual captured testbed CSV data."""
        raw_csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "baseline_compressor_runs.csv"
        )
        self.assertTrue(os.path.exists(raw_csv_path), "baseline_compressor_runs.csv must exist")

        df = pd.read_csv(raw_csv_path)
        self.assertGreater(len(df), 10, "Captured CSV must contain sufficient rows")

        vib_x = df["vib_x"].values
        rms_x = PythonFeatureExtractor.compute_rms(vib_x)
        kurt_x = PythonFeatureExtractor.compute_kurtosis(vib_x)
        cf_x = PythonFeatureExtractor.compute_crest_factor(vib_x)

        self.assertGreater(rms_x, 8.0, "Earth gravity baseline vibration RMS should be around ~9.8 m/s^2")
        self.assertGreater(kurt_x, 1.0, "Kurtosis should be a valid positive number")
        self.assertGreater(cf_x, 0.5, "Crest factor should be a valid positive number")

    def test_pipeline_and_contract_validation(self):
        """Verifies that MultiModalDSPPipeline generates all required fields from the shared JSON contract."""
        import sys
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from dsp_pipeline import MultiModalDSPPipeline

        raw_csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "baseline_compressor_runs.csv"
        )
        df_raw = pd.read_csv(raw_csv_path)

        pipeline = MultiModalDSPPipeline()
        df_features = pipeline.process_dataframe(df_raw)

        contract_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "shared", "interfaces", "c01_to_c02_telemetry_contract.json")
        )
        is_valid, msg = pipeline.validate_schema(df_features, contract_path)
        self.assertTrue(is_valid, f"Contract schema validation failed: {msg}")
        self.assertEqual(len(df_features), len(df_raw))


if __name__ == "__main__":
    unittest.main()

