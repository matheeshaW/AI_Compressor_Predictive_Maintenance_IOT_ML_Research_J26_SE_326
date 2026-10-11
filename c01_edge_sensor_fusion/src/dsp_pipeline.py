"""
c01_edge_sensor_fusion/src/dsp_pipeline.py
Component 01: Multi-Modal Sliding-Window DSP Feature Extraction Pipeline
Researcher: Weerakoon W.M.M.B (IT23155152) - J26-SE-326

Mirrors the embedded C++ MultiModalFeatureExtractor logic in pure Python/NumPy/Pandas
for offline algorithm verification, data science exploratory analysis, and dataset export.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class MultiModalDSPPipeline:
    """
    Sliding-window DSP feature extractor matching embedded ESP32-S3 firmware.
    
    Default parameters (at 25 Hz sampling rate, 40ms interval):
      - Vibration window: 64 samples (~2.56 seconds)
      - Pressure window: 16 samples (~0.64 seconds)
      - Current window: 32 samples (~1.28 seconds)
      - Acoustic window: 32 samples (~1.28 seconds)
    """

    def __init__(
        self,
        vib_window: int = 64,
        press_window: int = 16,
        curr_window: int = 32,
        sound_window: int = 32,
    ):
        self.vib_window = vib_window
        self.press_window = press_window
        self.curr_window = curr_window
        self.sound_window = sound_window

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a raw multi-modal telemetry DataFrame and appends sliding-window DSP metrics.
        Returns a new DataFrame containing both aligned sensor states and extracted features.
        """
        df = df.copy()

        # Sort by timestamp_ms if not monotonic
        if not df["timestamp_ms"].is_monotonic_increasing:
            df = df.sort_values("timestamp_ms").reset_index(drop=True)

        n = len(df)
        out = pd.DataFrame(index=df.index)

        # Retain metadata and labels
        out["timestamp_iso"] = df["timestamp_iso"]
        out["timestamp_ms"] = df["timestamp_ms"]
        out["temperature_celsius"] = df["temp_c"]
        out["pressure_psi"] = df["press_psi"]
        out["anomaly_flag"] = df["anomaly"]
        out["relay_latched"] = df["relay"]

        # -------------------------------------------------------------
        # 1. Vibration DSP Features (MPU-6050 3-Axis)
        # -------------------------------------------------------------
        # RMS = sqrt(mean(x^2))
        vib_x = df["vib_x"].values
        vib_y = df["vib_y"].values
        vib_z = df["vib_z"].values

        vib_x_rms = np.zeros(n)
        vib_y_rms = np.zeros(n)
        vib_z_rms = np.zeros(n)
        vib_kurtosis = np.full(n, 3.0)
        vib_crest_factor = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.vib_window + 1)
            wx = vib_x[start : i + 1]
            wy = vib_y[start : i + 1]
            wz = vib_z[start : i + 1]

            vib_x_rms[i] = np.sqrt(np.mean(wx**2))
            vib_y_rms[i] = np.sqrt(np.mean(wy**2))
            vib_z_rms[i] = np.sqrt(np.mean(wz**2))

            if len(wx) >= 16:
                mean_x = np.mean(wx)
                std_x = np.std(wx)
                if std_x > 1e-4:
                    # Pearson Kurtosis = E[(X - mu)^4] / sigma^4
                    z = (wx - mean_x) / std_x
                    vib_kurtosis[i] = np.mean(z**4)

                    # Dynamic AC Crest Factor = max|X - mu| / sigma
                    peak_dev = np.max(np.abs(wx - mean_x))
                    vib_crest_factor[i] = peak_dev / std_x
                else:
                    vib_kurtosis[i] = 3.0
                    vib_crest_factor[i] = 0.0

        out["vibration_x_rms_ms2"] = np.round(vib_x_rms, 3)
        out["vibration_y_rms_ms2"] = np.round(vib_y_rms, 3)
        out["vibration_z_rms_ms2"] = np.round(vib_z_rms, 3)
        out["vibration_composite_rms_ms2"] = np.round(
            np.sqrt(vib_x_rms**2 + vib_y_rms**2 + vib_z_rms**2), 3
        )
        out["vibration_kurtosis"] = np.round(vib_kurtosis, 2)
        out["vibration_crest_factor"] = np.round(vib_crest_factor, 2)

        # -------------------------------------------------------------
        # 2. Thermodynamic Pneumatic Features (Delta P / Delta t)
        # -------------------------------------------------------------
        press_vals = df["press_psi"].values
        time_ms = df["timestamp_ms"].values
        delta_p_dt = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.press_window + 1)
            if i - start >= 1:
                dt_sec = (time_ms[i] - time_ms[start]) / 1000.0
                if dt_sec > 0.01:
                    delta_p_dt[i] = (press_vals[i] - press_vals[start]) / dt_sec

        out["delta_pressure_dt"] = np.round(delta_p_dt, 3)

        # -------------------------------------------------------------
        # 3. Electrical Motor Current Features (True RMS & Peak)
        # -------------------------------------------------------------
        curr_vals = df["curr_a"].values
        curr_rms = np.zeros(n)
        curr_peak = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.curr_window + 1)
            wc = curr_vals[start : i + 1]
            curr_rms[i] = np.sqrt(np.mean(wc**2))
            curr_peak[i] = np.max(np.abs(wc))

        out["motor_current_rms_a"] = np.round(curr_rms, 3)
        out["motor_current_peak_a"] = np.round(curr_peak, 3)

        # -------------------------------------------------------------
        # 4. Acoustic Energy Features (RMS Energy & Peak Burst)
        # -------------------------------------------------------------
        sound_vals = df["sound_mv"].values
        sound_rms = np.zeros(n)
        sound_peak = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.sound_window + 1)
            ws = sound_vals[start : i + 1]
            sound_rms[i] = np.sqrt(np.mean(ws**2))
            sound_peak[i] = np.max(ws)

        out["acoustic_energy_mv"] = np.round(sound_rms, 1)
        out["acoustic_peak_mv"] = np.round(sound_peak, 1)

        return out

    def validate_schema(self, df_processed: pd.DataFrame, schema_path: str) -> Tuple[bool, str]:
        """Validates that extracted features satisfy c01_to_c02_telemetry_contract.json"""
        if not os.path.exists(schema_path):
            return False, f"Contract schema file not found at {schema_path}"

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        required_features = schema["properties"]["features"]["required"]
        missing = [rf for rf in required_features if rf not in df_processed.columns]
        if missing:
            return False, f"Missing required contract features: {missing}"

        return True, "All required schema contract features verified successfully."


def process_and_export(raw_csv_path: str, processed_csv_path: str, contract_json_path: str) -> pd.DataFrame:
    """End-to-end batch processing utility."""
    print(f"[INGEST] Reading raw telemetry from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)
    print(f"[INFO] Loaded {len(df_raw)} records across {len(df_raw.columns)} modalities.")

    pipeline = MultiModalDSPPipeline()
    df_features = pipeline.process_dataframe(df_raw)

    valid, msg = pipeline.validate_schema(df_features, contract_json_path)
    print(f"[CONTRACT VERIFICATION] {msg}")
    if not valid:
        raise ValueError(msg)

    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    df_features.to_csv(processed_csv_path, index=False)
    print(f"[EXPORT] Successfully exported {len(df_features)} processed records to {processed_csv_path}")

    return df_features


if __name__ == "__main__":
    import sys

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(base_dir, "data", "raw", "baseline_compressor_runs.csv")
    out_path = os.path.join(base_dir, "data", "processed", "sliding_window_features.csv")
    schema_path = os.path.abspath(
        os.path.join(base_dir, "..", "shared", "interfaces", "c01_to_c02_telemetry_contract.json")
    )

    process_and_export(raw_path, out_path, schema_path)
