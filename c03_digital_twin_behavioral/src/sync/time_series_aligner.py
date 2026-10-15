"""
src/sync/time_series_aligner.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Multi-rate time-series temporal alignment, resampling, interpolation, and feature windowing.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
from .stream_buffer import CircularTelemetryBuffer, TelemetryFrame


@dataclass
class AlignedSnapshot:
    """
    Standardized synchronized telemetry snapshot conforming to
    c03_digital_twin_behavioral/schemas/digital_twin_state_schema.json.
    """
    timestamp_ms: int
    timestamp_iso: str
    temperature_celsius: float
    pressure_psi: float
    delta_pressure_dt: float
    motor_current_rms_a: float
    vibration_x_rms_ms2: float
    vibration_y_rms_ms2: float
    vibration_z_rms_ms2: float
    vibration_kurtosis: float
    vibration_crest_factor: float
    acoustic_energy_mv: float
    sample_window_size: int
    is_stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_celsius": round(self.temperature_celsius, 2),
            "pressure_psi": round(self.pressure_psi, 2),
            "delta_pressure_dt": round(self.delta_pressure_dt, 3),
            "motor_current_rms_a": round(self.motor_current_rms_a, 3),
            "vibration_x_rms_ms2": round(self.vibration_x_rms_ms2, 2),
            "vibration_y_rms_ms2": round(self.vibration_y_rms_ms2, 2),
            "vibration_z_rms_ms2": round(self.vibration_z_rms_ms2, 2),
            "vibration_kurtosis": round(self.vibration_kurtosis, 2),
            "vibration_crest_factor": round(self.vibration_crest_factor, 2),
            "acoustic_energy_mv": round(self.acoustic_energy_mv, 1),
            "sample_window_size": self.sample_window_size,
        }


class TimeSeriesAligner:
    """
    Resamples, temporally aligns, and interpolates multi-rate sensor telemetry streams.
    Synchronizes 25-50 Hz mechanical/pneumatic signals with 1 Hz thermodynamic signals
    into a uniform time grid for digital twin state tracking.
    """

    # Feature definitions for alignment strategy
    CONTINUOUS_FEATURES = [
        "pressure_psi",
        "delta_pressure_dt",
        "motor_current_rms_a",
        "vibration_x_rms_ms2",
        "vibration_y_rms_ms2",
        "vibration_z_rms_ms2",
        "acoustic_energy_mv",
    ]

    STEP_FEATURES = [
        "temperature_celsius",
        "vibration_kurtosis",
        "vibration_crest_factor",
    ]

    BOOLEAN_FEATURES = [
        "anomaly_flag",
        "relay_latched",
    ]

    def __init__(
        self,
        target_freq_hz: float = 25.0,
        max_interpolation_gap_sec: float = 5.0,
        default_window_size: int = 50,
    ):
        """
        Args:
            target_freq_hz: Resampling grid frequency (Hz). Default 25.0 Hz = 40ms interval.
            max_interpolation_gap_sec: Maximum time gap where interpolation is permitted.
            default_window_size: Number of samples per feature extraction window.
        """
        if target_freq_hz <= 0:
            raise ValueError("Target frequency must be strictly positive.")

        self.target_freq_hz = target_freq_hz
        self.interval_ms = int(1000.0 / target_freq_hz)
        self.max_gap_ms = int(max_interpolation_gap_sec * 1000.0)
        self.default_window_size = default_window_size

    def align_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw unaligned DataFrame with irregular timestamps and resamples
        onto an exact uniform time grid.
        """
        if df_raw.empty or len(df_raw) < 2:
            return pd.DataFrame()

        df = df_raw.copy().sort_values("timestamp_ms").reset_index(drop=True)
        t_start = df["timestamp_ms"].iloc[0]
        t_end = df["timestamp_ms"].iloc[-1]

        if t_end <= t_start:
            return pd.DataFrame()

        # Construct uniform regular time grid bounded within [t_start, t_end]
        grid_timestamps = np.arange(t_start, t_end + 1, self.interval_ms, dtype=np.int64)
        aligned_df = pd.DataFrame({"timestamp_ms": grid_timestamps})

        # Merge on exact or nearest timestamp
        merged = pd.merge_asof(
            aligned_df,
            df,
            on="timestamp_ms",
            direction="nearest",
            tolerance=self.max_gap_ms
        )

        # Apply Linear Interpolation for continuous physical signals
        for col in self.CONTINUOUS_FEATURES:
            if col in df.columns:
                # Interpolate using timestamp index
                s = pd.Series(df[col].values, index=df["timestamp_ms"])
                # Reindex with union to interpolate
                combined_index = np.unique(np.concatenate([df["timestamp_ms"].values, grid_timestamps]))
                s_reindexed = s.reindex(combined_index).interpolate(method="index")
                merged[col] = s_reindexed.loc[grid_timestamps].values

        # Apply Zero-Order Hold (Forward-Fill) for slow thermodynamic and discrete parameters
        for col in self.STEP_FEATURES + self.BOOLEAN_FEATURES:
            if col in df.columns:
                s = pd.Series(df[col].values, index=df["timestamp_ms"])
                combined_index = np.unique(np.concatenate([df["timestamp_ms"].values, grid_timestamps]))
                s_reindexed = s.reindex(combined_index).ffill().bfill().infer_objects(copy=False)
                merged[col] = s_reindexed.loc[grid_timestamps].values

        # Ensure forward-fill / backward-fill on boundaries
        merged = merged.ffill().bfill().infer_objects(copy=False)

        # Compute dynamic first derivatives (dP/dt in PSI/sec) if not already present
        dt_seconds = self.interval_ms / 1000.0
        if "delta_pressure_dt" in merged.columns and not merged["delta_pressure_dt"].isna().all() and (merged["delta_pressure_dt"].abs() > 1e-4).any():
            pass  # Precomputed at the edge DSP or direct telemetry
        elif "pressure_psi" in merged.columns and len(merged) > 1:
            merged["delta_pressure_dt"] = np.gradient(merged["pressure_psi"].values, dt_seconds)
        else:
            merged["delta_pressure_dt"] = 0.0

        return merged

    def align_buffer(self, buffer: CircularTelemetryBuffer) -> pd.DataFrame:
        """
        Extracts all frames from buffer and returns a resampled uniform DataFrame.
        """
        df_raw = buffer.to_dataframe()
        return self.align_dataframe(df_raw)

    def extract_latest_snapshot(
        self,
        buffer: CircularTelemetryBuffer,
        window_size: Optional[int] = None
    ) -> Optional[AlignedSnapshot]:
        """
        Computes the most recent synchronized telemetry snapshot from the buffer.
        """
        win_size = window_size or self.default_window_size
        aligned_df = self.align_buffer(buffer)

        if aligned_df.empty or len(aligned_df) < 2:
            # Fallback to most recent raw frame if buffer has at least 1 item
            recent = buffer.get_recent(1)
            if not recent:
                return None
            f = recent[0]
            return AlignedSnapshot(
                timestamp_ms=f.timestamp_ms,
                timestamp_iso=f.timestamp_iso,
                temperature_celsius=f.temperature_celsius,
                pressure_psi=f.pressure_psi,
                delta_pressure_dt=f.delta_pressure_dt,
                motor_current_rms_a=f.motor_current_rms_a,
                vibration_x_rms_ms2=f.vibration_x_rms_ms2,
                vibration_y_rms_ms2=f.vibration_y_rms_ms2,
                vibration_z_rms_ms2=f.vibration_z_rms_ms2,
                vibration_kurtosis=f.vibration_kurtosis,
                vibration_crest_factor=f.vibration_crest_factor,
                acoustic_energy_mv=f.acoustic_energy_mv,
                sample_window_size=win_size,
                is_stale=False,
            )

        latest_row = aligned_df.iloc[-1]
        time_since_latest_ms = buffer.get_recent(1)[0].timestamp_ms - latest_row["timestamp_ms"]
        is_stale = bool(abs(time_since_latest_ms) > self.max_gap_ms)

        return AlignedSnapshot(
            timestamp_ms=int(latest_row["timestamp_ms"]),
            timestamp_iso=str(latest_row.get("timestamp_iso", "")),
            temperature_celsius=float(latest_row.get("temperature_celsius", 25.0)),
            pressure_psi=float(latest_row.get("pressure_psi", 0.0)),
            delta_pressure_dt=float(latest_row.get("delta_pressure_dt", 0.0)),
            motor_current_rms_a=float(latest_row.get("motor_current_rms_a", 0.0)),
            vibration_x_rms_ms2=float(latest_row.get("vibration_x_rms_ms2", 0.0)),
            vibration_y_rms_ms2=float(latest_row.get("vibration_y_rms_ms2", 0.0)),
            vibration_z_rms_ms2=float(latest_row.get("vibration_z_rms_ms2", 0.0)),
            vibration_kurtosis=float(latest_row.get("vibration_kurtosis", 3.0)),
            vibration_crest_factor=float(latest_row.get("vibration_crest_factor", 1.4)),
            acoustic_energy_mv=float(latest_row.get("acoustic_energy_mv", 0.0)),
            sample_window_size=win_size,
            is_stale=is_stale,
        )

    def extract_feature_matrix(
        self,
        aligned_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Converts aligned DataFrame into a normalized feature matrix for clustering.
        """
        cols = feature_columns or [
            "pressure_psi",
            "delta_pressure_dt",
            "motor_current_rms_a",
            "vibration_x_rms_ms2",
            "temperature_celsius",
            "acoustic_energy_mv",
        ]
        available_cols = [c for c in cols if c in aligned_df.columns]
        if not available_cols or aligned_df.empty:
            return np.empty((0, len(cols)))

        return aligned_df[available_cols].to_numpy(dtype=np.float64)

    def generate_sliding_windows(
        self,
        aligned_df: pd.DataFrame,
        window_size: int = 50,
        stride: int = 10,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generates overlapping 3D sliding window tensors (N_windows, window_size, N_features)
        alongside window center timestamps.
        """
        matrix = self.extract_feature_matrix(aligned_df, feature_columns)
        n_samples = len(matrix)

        if n_samples < window_size:
            return np.empty((0, window_size, matrix.shape[1])), []

        windows = []
        timestamps = []
        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size
            windows.append(matrix[start:end])
            mid_idx = start + (window_size // 2)
            timestamps.append(int(aligned_df["timestamp_ms"].iloc[mid_idx]))

        return np.array(windows, dtype=np.float64), timestamps
