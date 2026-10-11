"""
src/sync/stream_buffer.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
High-performance thread-safe circular ring buffer for asynchronous multi-modal telemetry.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class TelemetryFrame:
    """
    Immutable representation of a single multi-sensor telemetry observation.
    Compatible with C01 telemetry contract and serial stream outputs.
    """
    timestamp_ms: int
    timestamp_iso: str = ""
    pressure_psi: float = 0.0
    temperature_celsius: float = 0.0
    motor_current_rms_a: float = 0.0
    vibration_x_rms_ms2: float = 0.0
    vibration_y_rms_ms2: float = 0.0
    vibration_z_rms_ms2: float = 0.0
    vibration_kurtosis: float = 3.0
    vibration_crest_factor: float = 1.4
    acoustic_energy_mv: float = 0.0
    anomaly_flag: bool = False
    relay_latched: bool = False
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp_iso:
            self.timestamp_iso = datetime.now(timezone.utc).isoformat()


class CircularTelemetryBuffer:
    """
    Thread-safe, fixed-capacity circular ring buffer with chronological order preservation,
    duplicate suppression, and sliding-window slicing for industrial compressor digital twins.
    """

    def __init__(self, capacity: int = 2048, max_reorder_window_ms: int = 500):
        """
        Args:
            capacity: Maximum number of frames retained in memory before FIFO eviction.
            max_reorder_window_ms: Allowed jitter window in ms to re-sort out-of-order packets.
        """
        if capacity <= 0:
            raise ValueError("Buffer capacity must be a positive integer.")

        self._capacity = capacity
        self._max_reorder_window_ms = max_reorder_window_ms
        self._buffer: List[TelemetryFrame] = []
        self._lock = threading.Lock()
        self._total_pushed: int = 0
        self._dropped_count: int = 0
        self._duplicate_count: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self._capacity

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def duplicate_count(self) -> int:
        with self._lock:
            return self._duplicate_count

    def push(self, frame: TelemetryFrame) -> bool:
        """
        Pushes a new TelemetryFrame into the buffer.
        Handles out-of-order packets and maintains strictly non-decreasing timestamps.

        Returns:
            bool: True if inserted successfully, False if rejected as duplicate/outdated.
        """
        with self._lock:
            self._total_pushed += 1

            # Case 1: Empty buffer
            if not self._buffer:
                self._buffer.append(frame)
                return True

            latest_ms = self._buffer[-1].timestamp_ms

            # Case 2: In-order normal arrival (monotonically increasing)
            if frame.timestamp_ms > latest_ms:
                if len(self._buffer) >= self._capacity:
                    self._buffer.pop(0)
                    self._dropped_count += 1
                self._buffer.append(frame)
                return True

            # Case 3: Duplicate timestamp check
            if frame.timestamp_ms == latest_ms:
                self._duplicate_count += 1
                return False

            # Case 4: Out-of-order packet arriving within allowed jitter window
            if latest_ms - frame.timestamp_ms <= self._max_reorder_window_ms:
                # Binary search / linear insertion from tail
                idx = len(self._buffer) - 1
                while idx >= 0 and self._buffer[idx].timestamp_ms > frame.timestamp_ms:
                    idx -= 1

                # Check if duplicate found at insertion boundary
                if idx >= 0 and self._buffer[idx].timestamp_ms == frame.timestamp_ms:
                    self._duplicate_count += 1
                    return False

                # Insert at sorted position
                self._buffer.insert(idx + 1, frame)
                if len(self._buffer) > self._capacity:
                    self._buffer.pop(0)
                    self._dropped_count += 1
                return True

            # Case 5: Packet arrived too late (older than max_reorder_window_ms)
            self._dropped_count += 1
            return False

    def push_dict(self, data: Dict[str, Any]) -> bool:
        """
        Convenience method to ingest raw dictionary from JSON contract or CSV row.
        """
        # Resolve timestamp
        t_ms = data.get("timestamp_ms")
        if t_ms is None:
            t_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        t_iso = data.get("timestamp_iso", "")
        features = data.get("features", {})
        edge_status = data.get("edge_status", {})

        # Extract multi-modal parameters with fallbacks
        frame = TelemetryFrame(
            timestamp_ms=int(t_ms),
            timestamp_iso=str(t_iso),
            pressure_psi=float(features.get("pressure_psi", data.get("press_psi", 0.0))),
            temperature_celsius=float(features.get("temperature_celsius", data.get("temp_c", 0.0))),
            motor_current_rms_a=float(features.get("motor_current_rms_a", data.get("curr_a", 0.0))),
            vibration_x_rms_ms2=float(features.get("vibration_x_rms_ms2", data.get("vib_x", 0.0))),
            vibration_y_rms_ms2=float(features.get("vibration_y_rms_ms2", data.get("vib_y", 0.0))),
            vibration_z_rms_ms2=float(features.get("vibration_z_rms_ms2", data.get("vib_z", 0.0))),
            vibration_kurtosis=float(features.get("vibration_kurtosis", data.get("kurtosis", 3.0))),
            vibration_crest_factor=float(features.get("vibration_crest_factor", data.get("crest_factor", 1.4))),
            acoustic_energy_mv=float(features.get("acoustic_energy_mv", data.get("sound_mv", 0.0))),
            anomaly_flag=bool(edge_status.get("anomaly_flag", data.get("anomaly", False))),
            relay_latched=bool(edge_status.get("relay_latched", data.get("relay", False))),
            raw_payload=data,
        )
        return self.push(frame)

    def get_recent(self, n: int) -> List[TelemetryFrame]:
        """Returns the most recent n frames (chronological)."""
        with self._lock:
            if n <= 0:
                return []
            return list(self._buffer[-n:])

    def get_window(self, start_ms: int, end_ms: int) -> List[TelemetryFrame]:
        """Returns all frames where start_ms <= timestamp_ms <= end_ms."""
        with self._lock:
            return [f for f in self._buffer if start_ms <= f.timestamp_ms <= end_ms]

    def get_all(self) -> List[TelemetryFrame]:
        """Returns a copy of all current frames in chronological order."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Empties the buffer and resets counters."""
        with self._lock:
            self._buffer.clear()
            self._dropped_count = 0
            self._duplicate_count = 0

    def time_span_seconds(self) -> float:
        """Returns the time duration currently stored in the buffer in seconds."""
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return (self._buffer[-1].timestamp_ms - self._buffer[0].timestamp_ms) / 1000.0

    def estimated_sampling_rate_hz(self) -> float:
        """Computes the empirical sample rate in Hz across the current buffer."""
        with self._lock:
            if len(self._buffer) < 5:
                return 0.0
            dt_list = [
                self._buffer[i].timestamp_ms - self._buffer[i - 1].timestamp_ms
                for i in range(1, len(self._buffer))
            ]
            median_dt_ms = float(np.median(dt_list))
            if median_dt_ms <= 0:
                return 0.0
            return 1000.0 / median_dt_ms

    def to_dataframe(self) -> pd.DataFrame:
        """Exports the buffer content into a clean pandas DataFrame."""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame(columns=[
                    "timestamp_ms", "timestamp_iso", "pressure_psi", "temperature_celsius",
                    "motor_current_rms_a", "vibration_x_rms_ms2", "vibration_y_rms_ms2",
                    "vibration_z_rms_ms2", "vibration_kurtosis", "vibration_crest_factor",
                    "acoustic_energy_mv", "anomaly_flag", "relay_latched"
                ])

            data = [
                {
                    "timestamp_ms": f.timestamp_ms,
                    "timestamp_iso": f.timestamp_iso,
                    "pressure_psi": f.pressure_psi,
                    "temperature_celsius": f.temperature_celsius,
                    "motor_current_rms_a": f.motor_current_rms_a,
                    "vibration_x_rms_ms2": f.vibration_x_rms_ms2,
                    "vibration_y_rms_ms2": f.vibration_y_rms_ms2,
                    "vibration_z_rms_ms2": f.vibration_z_rms_ms2,
                    "vibration_kurtosis": f.vibration_kurtosis,
                    "vibration_crest_factor": f.vibration_crest_factor,
                    "acoustic_energy_mv": f.acoustic_energy_mv,
                    "anomaly_flag": f.anomaly_flag,
                    "relay_latched": f.relay_latched,
                }
                for f in self._buffer
            ]
            return pd.DataFrame(data)
