"""
Simulation CLI: Replays multi-regime compressor telemetry and monitors digital twin states in real time.
"""

import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Generator
import numpy as np

# Ensure c03 module path is available
MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from src.twin_engine import IndustrialDigitalTwinEngine
from src.clustering.cluster_evaluator import ClusterEvaluator
from src.clustering.state_discovery import OperatingStateLabel


def generate_live_compressor_stream(
    total_cycles: int = 120,
    inject_spike_at: int = 40,
    inject_leak_at: int = 70,
    sampling_rate_hz: float = 25.0,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates realistic, continuous physical sensor telemetry transitioning through:
    1. Idle Unloaded -> 2. Pressurizing Loaded -> 3. Holding Pressure ->
    4. Transient Electrical/Mechanical Spike -> 5. Persistent Pneumatic Leak Degradation.
    """
    dt = 1.0 / sampling_rate_hz
    base_epoch = int(datetime.now(timezone.utc).timestamp() * 1000)

    # State variables
    pressure = 25.0
    temp = 32.0

    for step_i in range(total_cycles):
        now_ms = base_epoch + int(step_i * dt * 1000)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Nominal regime progression
        if step_i < 20:
            # Regime 1: Idle Unloaded (Low steady pressure, idle current ~4.5A)
            pressure = 25.0 + np.random.normal(0, 0.02)
            dp_dt = 0.0 + np.random.normal(0, 0.01)
            current = 4.5 + np.random.normal(0, 0.15)
            temp += 0.005
            vib_x = 0.8 + np.random.normal(0, 0.04)
            vib_y = 0.7 + np.random.normal(0, 0.04)
            vib_z = 0.6 + np.random.normal(0, 0.04)
            kurt = 2.9 + np.random.normal(0, 0.06)
            crest = 1.4 + np.random.normal(0, 0.03)
            acoustic = 70.0 + np.random.normal(0, 4.0)

        elif step_i < 60:
            # Regime 2: Pressurizing Loaded (Smoothly climbing to 118 PSI at ~1.4 PSI/s)
            dp_dt = 1.4 + np.random.normal(0, 0.02)
            pressure = 25.0 + (step_i - 20) * (93.0 / 40.0)
            current = 18.2 + np.random.normal(0, 0.25)
            temp += 0.03
            vib_x = 4.8 + np.random.normal(0, 0.12)
            vib_y = 3.9 + np.random.normal(0, 0.1)
            vib_z = 3.2 + np.random.normal(0, 0.08)
            kurt = 3.2 + np.random.normal(0, 0.08)
            crest = 1.6 + np.random.normal(0, 0.04)
            acoustic = 260.0 + np.random.normal(0, 8.0)

        elif step_i < 75:
            # Regime 3: Holding Pressure / Cut-out (Steady at 118 PSI, motor unloaded)
            pressure = 118.0 + np.random.normal(0, 0.02)
            dp_dt = 0.01 + np.random.normal(0, 0.01)
            current = 5.2 + np.random.normal(0, 0.15)
            temp = max(38.0, temp - 0.01)
            vib_x = 1.1 + np.random.normal(0, 0.04)
            vib_y = 0.9 + np.random.normal(0, 0.03)
            vib_z = 0.8 + np.random.normal(0, 0.03)
            kurt = 3.0 + np.random.normal(0, 0.05)
            crest = 1.42 + np.random.normal(0, 0.03)
            acoustic = 95.0 + np.random.normal(0, 5.0)

        else:
            # Regime 5: Persistent Pneumatic Leak & Valve Degradation
            leak_severity = min(1.0, (step_i - inject_leak_at) / 20.0)
            pressure = max(30.0, 118.0 - (step_i - 75) * 0.8)
            dp_dt = -0.45 + np.random.normal(0, 0.02)
            current = 13.5 + 2.5 * leak_severity + np.random.normal(0, 0.2)
            temp += 0.04 * leak_severity
            vib_x = 3.6 + 0.8 * leak_severity + np.random.normal(0, 0.1)
            vib_y = 2.9 + 0.6 * leak_severity + np.random.normal(0, 0.08)
            vib_z = 2.4 + 0.5 * leak_severity + np.random.normal(0, 0.08)
            kurt = 4.5 + 0.8 * leak_severity + np.random.normal(0, 0.12)
            crest = 1.8 + 0.4 * leak_severity + np.random.normal(0, 0.06)
            acoustic = 540.0 + 150.0 * leak_severity + np.random.normal(0, 15.0)

        # Inject single-frame transient perturbation at target step
        if step_i == inject_spike_at:
            current = 24.5  # Temporary electrical surge
            vib_x = 5.2    # Mechanical bump
            kurt = 6.8

        yield {
            "timestamp_ms": now_ms,
            "timestamp_iso": now_iso,
            "pressure_psi": round(float(np.clip(pressure, 0.0, 150.0)), 2),
            "delta_pressure_dt": round(float(dp_dt), 3),
            "motor_current_rms_a": round(float(np.clip(current, 0.0, 30.0)), 3),
            "temperature_celsius": round(float(np.clip(temp, 15.0, 95.0)), 2),
            "vibration_x_rms_ms2": round(float(np.clip(vib_x, 0.0, 25.0)), 2),
            "vibration_y_rms_ms2": round(float(np.clip(vib_y, 0.0, 25.0)), 2),
            "vibration_z_rms_ms2": round(float(np.clip(vib_z, 0.0, 25.0)), 2),
            "vibration_kurtosis": round(float(np.clip(kurt, 1.0, 15.0)), 2),
            "vibration_crest_factor": round(float(np.clip(crest, 1.0, 10.0)), 2),
            "acoustic_energy_mv": round(float(np.clip(acoustic, 0.0, 1000.0)), 1),
            "sample_window_size": 512,
        }


def run_live_simulation(
    cycles: int = 100,
    delay: float = 0.03,
    save_snapshot: bool = True,
) -> None:
    """
    Executes live multi-cycle digital twin simulation runner with real-time terminal monitor.
    """
    print("\n" + "=" * 96)
    print("  INDUSTRIAL AIR COMPRESSOR DIGITAL TWIN & BEHAVIORAL INTELLIGENCE ENGINE (C03)")
    print("  Live Telemetry Replay Simulation & Operating State Monitor")
    print("=" * 96)

    # 1. Initialize Twin Engine with Baseline Data
    print("[1/4] Generating healthy operating baseline and training twin models...")
    baseline_df = ClusterEvaluator.generate_synthetic_compressor_data(n_samples_per_regime=150, random_state=42)
    engine = IndustrialDigitalTwinEngine(
        asset_id="COMP-RECIP-01",
        buffer_capacity=2048,
        target_sampling_rate_hz=25.0,
        deviation_threshold=3.0,
        persistence_threshold=5,
        nominal_compression_rate_psi_s=1.35,
    )
    engine.initialize_with_baseline(baseline_df)
    print("      Twin Engine initialized: K-Means/GMM (K=5) + Mahalanobis Profiler + Persistence Tracker.")

    # 2. Setup Telemetry Stream
    print(f"[2/4] Streaming {cycles} cycles at 25 Hz uniform time grid (delay={delay}s per cycle)...")
    print("-" * 96)
    print(f"{'Cycle':>5} | {'Press':>6} | {'Curr':>5} | {'Temp':>5} | {'Discovered State':<21} | {'Conf':>5} | {'D_M':>5} | {'Pers':>4} | {'Drift?':<6} | {'Health':<8}")
    print("-" * 96)

    stream = generate_live_compressor_stream(
        total_cycles=cycles,
        inject_spike_at=30,
        inject_leak_at=55,
        sampling_rate_hz=25.0,
    )

    latest_payload: Dict[str, Any] = {}
    transient_spikes_observed = 0
    persistent_alerts_observed = 0

    for step_idx, telemetry in enumerate(stream, start=1):
        latest_payload = engine.step(telemetry)

        # Extract monitoring metrics
        p = telemetry["pressure_psi"]
        i = telemetry["motor_current_rms_a"]
        t = telemetry["temperature_celsius"]
        state_label = latest_payload["operating_state"]["state_label"]
        conf = latest_payload["operating_state"]["state_confidence"]
        d_m = latest_payload["behavioral_intelligence"]["deviation_index"]
        persist = latest_payload["behavioral_intelligence"]["persistence_counter"]
        drift = latest_payload["behavioral_intelligence"]["drift_detected"]
        health = latest_payload["metadata"]["twin_health_status"]

        if persist == 1 and not drift:
            transient_spikes_observed += 1
        if drift:
            persistent_alerts_observed += 1

        drift_str = "DRIFT!" if drift else "Nominal"
        print(f"{step_idx:5d} | {p:6.1f} | {i:5.1f} | {t:5.1f} | {state_label:<21} | {conf:5.2f} | {d_m:5.2f} | {persist:4d} | {drift_str:<6} | {health:<8}")

        if delay > 0.0:
            time.sleep(delay)

    print("-" * 96)
    print(f"[3/4] Simulation run complete ({cycles} cycles processed).")
    print(f"      Transient perturbations filtered: {transient_spikes_observed}")
    print(f"      Persistent degradation alerts:     {persistent_alerts_observed}")

    # 3. Export Snapshot Artifacts
    if save_snapshot and latest_payload:
        demo_dir = MODULE_ROOT / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = demo_dir / "latest_twin_state.json"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(latest_payload, f, indent=2)
        print(f"[4/4] Exported validated digital twin state: {snapshot_path}")

        contract_path = demo_dir / "latest_c04_payload.json"
        c04_payload = engine.get_c04_prescriptive_payload()
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(c04_payload, f, indent=2)
        print(f"      Exported C04 prescriptive RL contract payload: {contract_path}")

    print("=" * 96 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Digital Twin Live Telemetry Simulator")
    parser.add_argument("--cycles", type=int, default=80, help="Total cycles to simulate (default: 80)")
    parser.add_argument("--delay", type=float, default=0.01, help="Inter-cycle delay in seconds (default: 0.01s)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving JSON snapshot artifacts")
    args = parser.parse_args()

    run_live_simulation(
        cycles=args.cycles,
        delay=args.delay,
        save_snapshot=not args.no_save,
    )
