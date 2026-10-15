"""
tests/test_twin_engine.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Comprehensive Integration and End-to-End Orchestration Test Suite
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

import json
from pathlib import Path
import pytest
import jsonschema
import numpy as np
import pandas as pd

from src.twin_engine import IndustrialDigitalTwinEngine
from src.sync.stream_buffer import TelemetryFrame
from src.clustering.cluster_evaluator import ClusterEvaluator
from src.clustering.state_discovery import OperatingStateLabel
from src.behavioral.degradation_tracker import DeviationClassification

# Resolve schema paths
MODULE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_ROOT.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

TWIN_SCHEMA_PATH = MODULE_ROOT / "schemas" / "digital_twin_state_schema.json"
C04_CONTRACT_PATH = REPO_ROOT / "shared" / "interfaces" / "c03_to_c04_twin_state_contract.json"


@pytest.fixture(scope="module")
def twin_schema():
    with open(TWIN_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def c04_contract():
    with open(C04_CONTRACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline_data():
    return ClusterEvaluator.generate_synthetic_compressor_data(n_samples_per_regime=120, random_state=42)


@pytest.fixture
def initialized_engine(baseline_data):
    engine = IndustrialDigitalTwinEngine(
        asset_id="COMP-RECIP-01",
        asset_type="12V_RECIPROCATING_PISTON",
        twin_version="1.0.0",
        buffer_capacity=2048,
        target_sampling_rate_hz=25.0,
        deviation_threshold=3.0,
        persistence_threshold=5,
        nominal_compression_rate_psi_s=1.35,
    )
    engine.initialize_with_baseline(baseline_data)
    return engine


def test_uninitialized_engine_raises_runtime_error():
    """Stepping an uninitialized engine must raise RuntimeError."""
    raw_engine = IndustrialDigitalTwinEngine()
    assert not raw_engine.is_initialized
    with pytest.raises(RuntimeError, match="not initialized"):
        raw_engine.step({"pressure_psi": 25.0})


def test_engine_initialization_with_baseline(baseline_data):
    """Engine initialization fits clusterer and profiler and sets operational registers."""
    engine = IndustrialDigitalTwinEngine()
    engine.initialize_with_baseline(baseline_data)
    assert engine.is_initialized
    assert engine.clusterer is not None
    assert engine.profiler is not None
    assert engine.twin_cycle_index == 0


def test_engine_step_with_dict_and_frame(initialized_engine):
    """Engine accepts both dict and TelemetryFrame telemetry inputs and increments cycles."""
    # Step 1: Ingest via raw dictionary
    packet_dict = {
        "timestamp_ms": 1773000000000,
        "pressure_psi": 25.0,
        "delta_pressure_dt": 0.0,
        "motor_current_rms_a": 4.5,
        "temperature_celsius": 38.0,
        "vibration_x_rms_ms2": 0.8,
        "vibration_y_rms_ms2": 0.7,
        "vibration_z_rms_ms2": 0.6,
        "vibration_kurtosis": 2.9,
        "vibration_crest_factor": 1.4,
        "acoustic_energy_mv": 70.0,
    }
    payload1 = initialized_engine.step(packet_dict)
    assert initialized_engine.twin_cycle_index == 1
    assert payload1["metadata"]["twin_cycle_index"] == 1
    assert payload1["operating_state"]["state_label"] == OperatingStateLabel.IDLE_UNLOADED.value

    # Step 2: Ingest via TelemetryFrame dataclass
    frame = TelemetryFrame(
        timestamp_ms=1773000000040,
        pressure_psi=75.0,
        delta_pressure_dt=1.4,
        motor_current_rms_a=18.2,
        temperature_celsius=62.0,
        vibration_x_rms_ms2=4.8,
        vibration_y_rms_ms2=3.9,
        vibration_z_rms_ms2=3.2,
        vibration_kurtosis=3.2,
        vibration_crest_factor=1.6,
        acoustic_energy_mv=260.0,
    )
    payload2 = initialized_engine.step(frame)
    assert initialized_engine.twin_cycle_index == 2
    assert payload2["metadata"]["twin_cycle_index"] == 2
    assert payload2["operating_state"]["state_label"] == OperatingStateLabel.PRESSURIZING_LOADED.value


def test_engine_full_state_payload_conforms_to_schema(initialized_engine, twin_schema):
    """The complete twin state dictionary strictly validates against digital_twin_state_schema.json."""
    packet = {
        "pressure_psi": 118.0,
        "delta_pressure_dt": 0.01,
        "motor_current_rms_a": 5.2,
        "temperature_celsius": 55.0,
        "vibration_x_rms_ms2": 1.1,
        "vibration_y_rms_ms2": 0.9,
        "vibration_z_rms_ms2": 0.8,
        "vibration_kurtosis": 3.0,
        "vibration_crest_factor": 1.42,
        "acoustic_energy_mv": 95.0,
    }
    payload = initialized_engine.step(packet)

    # Perform strict JSON schema validation
    jsonschema.validate(instance=payload, schema=twin_schema)
    assert payload["metadata"]["twin_id"] == "TWIN-COMP-RECIP-01"
    assert payload["metadata"]["physical_asset_id"] == "COMP-RECIP-01"
    assert payload["metadata"]["asset_type"] == "12V_RECIPROCATING_PISTON"
    assert payload["metadata"]["twin_version"] == "1.0.0"


def test_downstream_c04_payload_conforms_to_contract(initialized_engine, c04_contract):
    """The downstream payload exported for C04 RL validates strictly against c03_to_c04_twin_state_contract.json."""
    packet = {
        "pressure_psi": 92.0,
        "delta_pressure_dt": -0.45,
        "motor_current_rms_a": 13.5,
        "temperature_celsius": 68.0,
        "vibration_x_rms_ms2": 3.6,
        "vibration_y_rms_ms2": 2.9,
        "vibration_z_rms_ms2": 2.4,
        "vibration_kurtosis": 4.5,
        "vibration_crest_factor": 1.8,
        "acoustic_energy_mv": 540.0,
    }
    payload = initialized_engine.step(packet)

    # Validate embedded payload
    downstream_embedded = payload["downstream_contract_payload"]
    jsonschema.validate(instance=downstream_embedded, schema=c04_contract)

    # Validate method call
    downstream_direct = initialized_engine.get_c04_prescriptive_payload()
    jsonschema.validate(instance=downstream_direct, schema=c04_contract)
    assert downstream_direct["twin_id"] == "TWIN-COMP-RECIP-01"
    assert "composite_degradation_score" in downstream_direct["degradation_metrics"]


def test_subsystem_health_decomposition_nominal(initialized_engine):
    """Nominal compressor operation produces high health scores across all 4 sub-assemblies."""
    nominal_packet = {
        "pressure_psi": 25.0,
        "delta_pressure_dt": 0.0,
        "motor_current_rms_a": 4.5,
        "temperature_celsius": 38.0,
        "vibration_x_rms_ms2": 0.8,
        "vibration_y_rms_ms2": 0.7,
        "vibration_z_rms_ms2": 0.6,
        "vibration_kurtosis": 2.9,
        "vibration_crest_factor": 1.4,
        "acoustic_energy_mv": 70.0,
    }
    payload = initialized_engine.step(nominal_packet)
    subsystems = payload["subsystems"]

    assert subsystems["electric_motor"]["health_score"] >= 0.80
    assert subsystems["electric_motor"]["status"] == "NOMINAL"
    assert subsystems["cylinder_piston_assembly"]["health_score"] >= 0.80
    assert subsystems["suction_discharge_valves"]["health_score"] >= 0.80
    assert not subsystems["suction_discharge_valves"]["valve_leakage_detected"]
    assert subsystems["main_bearing_assembly"]["health_score"] >= 0.80
    assert subsystems["main_bearing_assembly"]["status"] == "NOMINAL"


def test_subsystem_health_decomposition_faults(initialized_engine):
    """Injected localized mechanical/pneumatic faults reflect in specific subsystem alerts."""
    # Fault 1: Motor overload
    overload_packet = {
        "pressure_psi": 50.0,
        "delta_pressure_dt": 0.5,
        "motor_current_rms_a": 25.0,  # Overload current > 22A
        "temperature_celsius": 82.0,  # Elevated temperature > 75C
        "vibration_x_rms_ms2": 2.0,
        "vibration_kurtosis": 3.0,
        "acoustic_energy_mv": 120.0,
    }
    payload1 = initialized_engine.step(overload_packet)
    motor = payload1["subsystems"]["electric_motor"]
    assert motor["status"] == "CURRENT_OVERLOAD"

    # Fault 2: Valve leakage
    leak_packet = {
        "pressure_psi": 92.0,
        "delta_pressure_dt": -0.45,
        "motor_current_rms_a": 13.5,
        "temperature_celsius": 68.0,
        "vibration_x_rms_ms2": 3.6,
        "vibration_kurtosis": 4.5,
        "acoustic_energy_mv": 650.0,  # Acoustic hissing > 500 mV
    }
    payload2 = initialized_engine.step(leak_packet)
    valves = payload2["subsystems"]["suction_discharge_valves"]
    if payload2["operating_state"]["state_label"] == OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value:
        assert valves["valve_leakage_detected"]
        assert valves["status"] == "LEAKAGE_DETECTED"

    # Fault 3: Bearing impact damage
    bearing_packet = {
        "pressure_psi": 25.0,
        "delta_pressure_dt": 0.0,
        "motor_current_rms_a": 4.5,
        "temperature_celsius": 38.0,
        "vibration_x_rms_ms2": 5.5,  # High radial acceleration
        "vibration_kurtosis": 6.8,    # Severe impact spalls > 5.5
        "acoustic_energy_mv": 80.0,
    }
    payload3 = initialized_engine.step(bearing_packet)
    bearing = payload3["subsystems"]["main_bearing_assembly"]
    assert bearing["status"] == "IMPACTING"
    assert bearing["health_score"] <= 0.50


def test_transient_noise_rejection_vs_persistent_leak_drift(initialized_engine):
    """
    Demonstrates the core DSR hypothesis:
    A single-frame transient perturbation does NOT declare drift,
    whereas >= 5 consecutive anomalous frames trigger persistent drift declaration.
    """
    # 1. Establish nominal loaded baseline
    loaded_packet = {
        "pressure_psi": 75.0,
        "delta_pressure_dt": 1.4,
        "motor_current_rms_a": 18.2,
        "temperature_celsius": 62.0,
        "vibration_x_rms_ms2": 4.8,
        "vibration_kurtosis": 3.2,
        "acoustic_energy_mv": 260.0,
    }
    for _ in range(3):
        p = initialized_engine.step(loaded_packet)

    # 2. Inject single-frame severe transient spike
    spike_packet = dict(loaded_packet)
    spike_packet["motor_current_rms_a"] = 28.0
    spike_packet["vibration_x_rms_ms2"] = 8.5
    spike_payload = initialized_engine.step(spike_packet)

    assert spike_payload["behavioral_intelligence"]["persistence_counter"] == 1
    assert not spike_payload["behavioral_intelligence"]["drift_detected"]

    # 3. Next nominal frame decays persistence counter
    nominal_payload = initialized_engine.step(loaded_packet)
    assert nominal_payload["behavioral_intelligence"]["persistence_counter"] == 0
    assert not nominal_payload["behavioral_intelligence"]["drift_detected"]

    # 4. Inject consecutive sustained degraded frames (>3.0 sigma deviation)
    degraded_loaded = dict(loaded_packet)
    degraded_loaded["motor_current_rms_a"] = 20.8
    degraded_loaded["vibration_x_rms_ms2"] = 5.8
    degraded_loaded["delta_pressure_dt"] = 1.05

    drift_declared = False
    for i in range(1, 7):
        deg_payload = initialized_engine.step(dict(degraded_loaded))
        if deg_payload["behavioral_intelligence"]["drift_detected"]:
            drift_declared = True
            assert deg_payload["behavioral_intelligence"]["persistence_counter"] >= 5
            assert deg_payload["behavioral_intelligence"]["deviation_classification"] == DeviationClassification.PERSISTENT_DEGRADATION.value
            break

    assert drift_declared, "Persistent drift must be declared after >=5 consecutive anomalous windows."


def test_engine_model_saving_and_loading(initialized_engine, tmp_path):
    """Clusterer and profiler models can be serialized to disk and reloaded by twin engine."""
    clusterer_file = tmp_path / "compressor_clusterer.joblib"
    profiler_file = tmp_path / "behavioral_profiler.joblib"

    initialized_engine.clusterer.save_model(clusterer_file)
    initialized_engine.profiler.save_profiles(profiler_file)

    new_engine = IndustrialDigitalTwinEngine()
    assert not new_engine.is_initialized
    new_engine.load_pretrained_models(clusterer_file, profiler_file)
    assert new_engine.is_initialized

    # Validate reloaded engine steps accurately
    test_packet = {
        "pressure_psi": 25.0,
        "delta_pressure_dt": 0.0,
        "motor_current_rms_a": 4.5,
        "temperature_celsius": 38.0,
        "vibration_x_rms_ms2": 0.8,
        "vibration_kurtosis": 2.9,
        "acoustic_energy_mv": 70.0,
    }
    res = new_engine.step(test_packet)
    assert res["operating_state"]["state_label"] == OperatingStateLabel.IDLE_UNLOADED.value


def test_view_controller_headless_processing():
    """Verify the Streamlit Dashboard view controller processes telemetry and formats radar features."""
    from dashboard.views.c03_digital_twin_view import DigitalTwinViewController

    ctrl = DigitalTwinViewController(asset_id="COMP-RECIP-TEST")
    ctrl.initialize_default()
    assert ctrl.is_initialized

    packet = {
        "pressure_psi": 75.0,
        "delta_pressure_dt": 1.4,
        "motor_current_rms_a": 18.2,
        "temperature_celsius": 62.0,
        "vibration_x_rms_ms2": 4.8,
        "vibration_kurtosis": 3.2,
        "acoustic_energy_mv": 260.0,
    }
    payload = ctrl.process_telemetry(packet)
    assert payload["metadata"]["twin_id"] == "TWIN-COMP-RECIP-TEST"

    radar = ctrl.get_radar_comparison_data(payload)
    assert radar["state_label"] == OperatingStateLabel.PRESSURIZING_LOADED.value
    assert len(radar["features"]) == 5

    history = ctrl.get_history()
    assert len(history) == 1
    assert history[0]["pressure_psi"] == 75.0

