"""
tests/test_sync_engine.py
Unit tests verifying CircularTelemetryBuffer and TimeSeriesAligner.
Component 03: Industrial Digital Twin & Behavioral Intelligence
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    class _DummyPytest:
        @staticmethod
        def fixture(func):
            return func
    pytest = _DummyPytest()

# Ensure src is on python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.sync.stream_buffer import CircularTelemetryBuffer, TelemetryFrame
from src.sync.time_series_aligner import TimeSeriesAligner, AlignedSnapshot


# ====================================================================
# TEST SUITE 1: CircularTelemetryBuffer Tests
# ====================================================================

def test_buffer_initialization():
    buf = CircularTelemetryBuffer(capacity=100)
    assert buf.capacity == 100
    assert buf.size == 0
    assert not buf.is_full
    assert buf.dropped_count == 0


def test_buffer_push_and_fifo_eviction():
    buf = CircularTelemetryBuffer(capacity=5)
    for i in range(10):
        frame = TelemetryFrame(
            timestamp_ms=1000 + (i * 40),
            pressure_psi=20.0 + i,
            motor_current_rms_a=5.0
        )
        assert buf.push(frame) is True

    assert buf.size == 5
    assert buf.is_full is True
    assert buf.total_pushed == 10
    assert buf.dropped_count == 5

    # Verify FIFO order preserved: oldest should now be i=5
    recent = buf.get_all()
    assert recent[0].timestamp_ms == 1000 + (5 * 40)
    assert recent[-1].timestamp_ms == 1000 + (9 * 40)


def test_buffer_out_of_order_jitter_insertion():
    """Verifies that packets arriving slightly out of order are sorted chronologically."""
    buf = CircularTelemetryBuffer(capacity=10, max_reorder_window_ms=500)

    # Push frames: t=100, t=180, t=260
    buf.push(TelemetryFrame(timestamp_ms=100, pressure_psi=10.0))
    buf.push(TelemetryFrame(timestamp_ms=180, pressure_psi=18.0))
    buf.push(TelemetryFrame(timestamp_ms=260, pressure_psi=26.0))

    # Delayed frame arriving out-of-order: t=140
    pushed = buf.push(TelemetryFrame(timestamp_ms=140, pressure_psi=14.0))
    assert pushed is True

    # Order should be: 100, 140, 180, 260
    all_frames = buf.get_all()
    timestamps = [f.timestamp_ms for f in all_frames]
    assert timestamps == [100, 140, 180, 260]
    pressures = [f.pressure_psi for f in all_frames]
    assert pressures == [10.0, 14.0, 18.0, 26.0]


def test_buffer_duplicate_rejection():
    """Verifies that duplicate packets are rejected without corrupting order."""
    buf = CircularTelemetryBuffer(capacity=10)
    buf.push(TelemetryFrame(timestamp_ms=100))
    buf.push(TelemetryFrame(timestamp_ms=140))

    # Exact duplicate
    res = buf.push(TelemetryFrame(timestamp_ms=140))
    assert res is False
    assert buf.duplicate_count == 1
    assert buf.size == 2


def test_buffer_push_dict_convenience():
    buf = CircularTelemetryBuffer(capacity=10)
    payload = {
        "timestamp_ms": 2500,
        "features": {
            "pressure_psi": 34.5,
            "temperature_celsius": 28.2,
            "motor_current_rms_a": 4.8,
            "vibration_x_rms_ms2": 9.81,
            "vibration_kurtosis": 3.1,
            "acoustic_energy_mv": 95.0,
        },
        "edge_status": {
            "anomaly_flag": False,
            "relay_latched": False
        }
    }
    assert buf.push_dict(payload) is True
    assert buf.size == 1
    f = buf.get_recent(1)[0]
    assert f.pressure_psi == 34.5
    assert f.temperature_celsius == 28.2
    assert f.motor_current_rms_a == 4.8


def test_buffer_statistics_and_dataframe():
    buf = CircularTelemetryBuffer(capacity=50)
    # 25 Hz sequence (interval 40ms)
    for i in range(25):
        buf.push(TelemetryFrame(
            timestamp_ms=i * 40,
            pressure_psi=30.0 + (i * 0.1),
            temperature_celsius=27.0
        ))

    assert buf.size == 25
    assert np.isclose(buf.time_span_seconds(), 24 * 0.04, atol=0.01)
    assert np.isclose(buf.estimated_sampling_rate_hz(), 25.0, atol=0.5)

    df = buf.to_dataframe()
    assert len(df) == 25
    assert "pressure_psi" in df.columns
    assert df["pressure_psi"].iloc[0] == 30.0


# ====================================================================
# TEST SUITE 2: TimeSeriesAligner Tests
# ====================================================================

def test_aligner_resampling_and_linear_interpolation():
    """
    Tests that irregular samples (e.g. at 0, 42, 85, 120 ms)
    are cleanly resampled onto a strict 25 Hz (40 ms) grid: 0, 40, 80, 120 ms.
    """
    aligner = TimeSeriesAligner(target_freq_hz=25.0)

    # Pressure linearly ramps from 10 to 40 PSI
    raw_data = pd.DataFrame({
        "timestamp_ms": [0, 50, 100, 150],
        "pressure_psi": [10.0, 20.0, 30.0, 40.0],
        "motor_current_rms_a": [2.0, 3.0, 4.0, 5.0],
        "temperature_celsius": [25.0, 25.0, 26.0, 26.0],
        "vibration_x_rms_ms2": [1.0, 1.0, 1.0, 1.0],
        "vibration_y_rms_ms2": [0.0, 0.0, 0.0, 0.0],
        "vibration_z_rms_ms2": [0.0, 0.0, 0.0, 0.0],
        "vibration_kurtosis": [3.0, 3.0, 3.0, 3.0],
        "vibration_crest_factor": [1.4, 1.4, 1.4, 1.4],
        "acoustic_energy_mv": [50.0, 50.0, 50.0, 50.0],
        "anomaly_flag": [False, False, False, False],
        "relay_latched": [False, False, False, False],
    })

    aligned = aligner.align_dataframe(raw_data)
    assert not aligned.empty

    # Grid should step exactly by 40 ms: 0, 40, 80, 120
    expected_ts = [0, 40, 80, 120]
    assert list(aligned["timestamp_ms"]) == expected_ts

    # Linear interpolation at t=40:
    # between t=0 (10.0) and t=50 (20.0) -> slope = 0.2 PSI/ms
    # at t=40 -> 10 + 40 * 0.2 = 18.0 PSI
    assert np.isclose(aligned["pressure_psi"].iloc[1], 18.0, atol=0.1)


def test_aligner_multi_rate_slow_temperature_hold():
    """
    Verifies that slow 1 Hz temperature is held constant (Zero-Order Hold)
    between updates while fast pressure updates at 25 Hz.
    """
    aligner = TimeSeriesAligner(target_freq_hz=25.0)

    # 1 second duration: fast pressure every 40 ms, slow temperature only updates at t=0 and t=1000
    records = []
    for t in range(0, 1040, 40):
        # Temp changes at t=600 ms
        temp = 25.0 if t < 600 else 30.0
        records.append({
            "timestamp_ms": t,
            "pressure_psi": 20.0 + (t * 0.01),
            "motor_current_rms_a": 5.0,
            "temperature_celsius": temp,
            "vibration_x_rms_ms2": 9.81,
            "vibration_y_rms_ms2": 0.1,
            "vibration_z_rms_ms2": 0.1,
            "vibration_kurtosis": 3.0,
            "vibration_crest_factor": 1.4,
            "acoustic_energy_mv": 75.0,
            "anomaly_flag": False,
            "relay_latched": False
        })
    df_raw = pd.DataFrame(records)

    aligned = aligner.align_dataframe(df_raw)
    assert len(aligned) == len(records)
    # Check that temperature remains steady before transition and steps after
    assert aligned.loc[aligned["timestamp_ms"] < 600, "temperature_celsius"].unique().tolist() == [25.0]
    assert aligned.loc[aligned["timestamp_ms"] >= 600, "temperature_celsius"].unique().tolist() == [30.0]


def test_aligner_computes_delta_pressure_derivative():
    aligner = TimeSeriesAligner(target_freq_hz=25.0)

    # Pressure increasing at exactly 2.5 PSI per second (0.1 PSI per 40ms step)
    records = []
    for i in range(50):
        t = i * 40
        records.append({
            "timestamp_ms": t,
            "pressure_psi": 10.0 + (i * 0.1),
            "motor_current_rms_a": 4.0,
            "temperature_celsius": 25.0,
            "vibration_x_rms_ms2": 9.81,
            "vibration_y_rms_ms2": 0.0,
            "vibration_z_rms_ms2": 0.0,
            "vibration_kurtosis": 3.0,
            "vibration_crest_factor": 1.4,
            "acoustic_energy_mv": 60.0,
            "anomaly_flag": False,
            "relay_latched": False
        })
    df_raw = pd.DataFrame(records)
    aligned = aligner.align_dataframe(df_raw)

    assert "delta_pressure_dt" in aligned.columns
    # Mid-array derivative should equal 0.1 PSI / 0.04 s = 2.5 PSI/sec
    mid_derivatives = aligned["delta_pressure_dt"].iloc[5:45]
    assert np.allclose(mid_derivatives, 2.5, atol=0.1)


def test_extract_latest_snapshot():
    aligner = TimeSeriesAligner(target_freq_hz=25.0)
    buf = CircularTelemetryBuffer(capacity=50)

    for i in range(20):
        buf.push(TelemetryFrame(
            timestamp_ms=1000 + (i * 40),
            pressure_psi=30.0 + (i * 0.5),
            temperature_celsius=29.0,
            motor_current_rms_a=5.2,
            vibration_x_rms_ms2=9.85,
            vibration_kurtosis=3.2,
            vibration_crest_factor=1.45,
            acoustic_energy_mv=80.0
        ))

    snapshot = aligner.extract_latest_snapshot(buf, window_size=50)
    assert snapshot is not None
    assert isinstance(snapshot, AlignedSnapshot)
    assert snapshot.pressure_psi >= 39.0
    assert snapshot.temperature_celsius == 29.0
    assert not snapshot.is_stale

    d = snapshot.to_dict()
    assert "pressure_psi" in d
    assert "delta_pressure_dt" in d
    assert "motor_current_rms_a" in d
    assert "sample_window_size" in d


def test_sliding_window_generation():
    aligner = TimeSeriesAligner(target_freq_hz=25.0)
    records = []
    for i in range(100):
        records.append({
            "timestamp_ms": i * 40,
            "pressure_psi": 20.0 + (i * 0.1),
            "delta_pressure_dt": 2.5,
            "motor_current_rms_a": 5.0,
            "vibration_x_rms_ms2": 9.8,
            "temperature_celsius": 26.0,
            "acoustic_energy_mv": 70.0,
        })
    df = pd.DataFrame(records)

    window_size = 30
    stride = 10
    windows, ts = aligner.generate_sliding_windows(df, window_size=window_size, stride=stride)

    assert len(windows) > 0
    # Expected shape: (N_windows, window_size, 6_features)
    assert windows.shape[1] == window_size
    assert windows.shape[2] == 6
    assert len(ts) == len(windows)


# ====================================================================
# Standalone execution
# ====================================================================

if __name__ == "__main__":
    print("Running Day 02 Synchronization Engine Test Suite...")

    test_buffer_initialization()
    print("  [PASS] test_buffer_initialization")

    test_buffer_push_and_fifo_eviction()
    print("  [PASS] test_buffer_push_and_fifo_eviction")

    test_buffer_out_of_order_jitter_insertion()
    print("  [PASS] test_buffer_out_of_order_jitter_insertion")

    test_buffer_duplicate_rejection()
    print("  [PASS] test_buffer_duplicate_rejection")

    test_buffer_push_dict_convenience()
    print("  [PASS] test_buffer_push_dict_convenience")

    test_buffer_statistics_and_dataframe()
    print("  [PASS] test_buffer_statistics_and_dataframe")

    test_aligner_resampling_and_linear_interpolation()
    print("  [PASS] test_aligner_resampling_and_linear_interpolation")

    test_aligner_multi_rate_slow_temperature_hold()
    print("  [PASS] test_aligner_multi_rate_slow_temperature_hold")

    test_aligner_computes_delta_pressure_derivative()
    print("  [PASS] test_aligner_computes_delta_pressure_derivative")

    test_extract_latest_snapshot()
    print("  [PASS] test_extract_latest_snapshot")

    test_sliding_window_generation()
    print("  [PASS] test_sliding_window_generation")

    print("\nALL 11 DAY 02 SYNCHRONIZATION TESTS PASSED CLEANLY!")
