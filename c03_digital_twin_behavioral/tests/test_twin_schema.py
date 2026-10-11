"""
test_twin_schema.py
Unit tests verifying the formal Digital Twin JSON Schema and contract compatibility.
Component 03: Industrial Digital Twin & Behavioral Intelligence
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

import json
import os
try:
    import pytest
except ImportError:
    class _DummyPytest:
        @staticmethod
        def fixture(func):
            return func
    pytest = _DummyPytest()
import jsonschema
from jsonschema import validate, ValidationError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C03_DIR = os.path.join(REPO_ROOT, "c03_digital_twin_behavioral")
SCHEMA_PATH = os.path.join(C03_DIR, "schemas", "digital_twin_state_schema.json")
DOWNSTREAM_CONTRACT_PATH = os.path.join(REPO_ROOT, "shared", "interfaces", "c03_to_c04_twin_state_contract.json")


@pytest.fixture
def digital_twin_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def downstream_contract_schema():
    with open(DOWNSTREAM_CONTRACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_twin_state():
    return {
        "metadata": {
            "twin_id": "TWIN-COMPRESSOR-RECIP-01",
            "physical_asset_id": "ESP32S3-COMPRESSOR-TESTBED-01",
            "asset_type": "12V_RECIPROCATING_PISTON",
            "twin_version": "1.0.0",
            "timestamp": "2026-10-10T12:00:00Z",
            "sync_epoch_ms": 1791633600000
        },
        "telemetry_snapshot": {
            "temperature_celsius": 32.4,
            "pressure_psi": 38.5,
            "delta_pressure_dt": 0.45,
            "motor_current_rms_a": 5.2,
            "vibration_x_rms_ms2": 9.85,
            "vibration_y_rms_ms2": 0.32,
            "vibration_z_rms_ms2": 0.28,
            "vibration_kurtosis": 3.12,
            "vibration_crest_factor": 1.45,
            "acoustic_energy_mv": 85.0,
            "sample_window_size": 512
        },
        "predictive_diagnostics": {
            "predicted_fault": "NORMAL",
            "fault_probability": 0.08,
            "estimated_rul_hours": 720.0,
            "rul_confidence_interval_95": [680.0, 760.0],
            "health_index": 0.94
        },
        "operating_state": {
            "cluster_id": 1,
            "state_label": "PRESSURIZING_LOADED",
            "state_confidence": 0.96,
            "clustering_algorithm": "GAUSSIAN_MIXTURE_MODEL",
            "time_in_state_seconds": 45.0,
            "transition_count": 12
        },
        "behavioral_intelligence": {
            "deviation_index": 0.82,
            "deviation_threshold": 3.0,
            "persistence_counter": 0,
            "persistence_threshold": 5,
            "deviation_classification": "NOMINAL",
            "drift_detected": False,
            "state_cluster_centroid": {
                "pressure_psi": 35.0,
                "motor_current_rms_a": 5.0,
                "temperature_celsius": 31.0
            },
            "state_cluster_variance": {
                "pressure_psi": 4.5,
                "motor_current_rms_a": 0.3,
                "temperature_celsius": 1.2
            }
        },
        "degradation_metrics": {
            "composite_degradation_score": 0.08,
            "pressure_efficiency_ratio": 0.98,
            "thermal_dissipation_factor": 0.052,
            "mechanical_vibration_severity": 1.15
        },
        "subsystems": {
            "electric_motor": {
                "health_score": 0.96,
                "status": "NOMINAL",
                "operating_temperature_c": 32.4,
                "rms_current_a": 5.2
            },
            "cylinder_piston_assembly": {
                "health_score": 0.98,
                "status": "OPTIMAL",
                "compression_rate_psi_s": 0.45,
                "seal_integrity_ratio": 0.99
            },
            "suction_discharge_valves": {
                "health_score": 0.95,
                "status": "NORMAL",
                "flow_efficiency_ratio": 0.97,
                "valve_leakage_detected": False
            },
            "main_bearing_assembly": {
                "health_score": 0.94,
                "status": "NOMINAL",
                "vibration_rms_ms2": 9.85,
                "impact_kurtosis": 3.12
            }
        },
        "downstream_contract_payload": {
            "timestamp": "2026-10-10T12:00:00Z",
            "twin_id": "TWIN-COMPRESSOR-RECIP-01",
            "operating_state": {
                "cluster_id": 1,
                "state_label": "PRESSURIZING_LOADED",
                "state_confidence": 0.96
            },
            "behavioral_intelligence": {
                "deviation_index": 0.82,
                "persistence_counter": 0,
                "drift_detected": False
            },
            "degradation_metrics": {
                "composite_degradation_score": 0.08,
                "pressure_efficiency_ratio": 0.98,
                "thermal_dissipation_factor": 0.052
            }
        }
    }


@pytest.fixture
def valid_twin_state():
    return get_valid_twin_state()


def test_schema_file_exists():
    assert os.path.exists(SCHEMA_PATH), f"Schema file not found at {SCHEMA_PATH}"


def test_valid_twin_state_conforms_to_schema(digital_twin_schema, valid_twin_state):
    """Verifies that a valid Digital Twin state complies with the DSR schema."""
    validate(instance=valid_twin_state, schema=digital_twin_schema)


def test_downstream_payload_conforms_to_shared_contract(downstream_contract_schema, valid_twin_state):
    """Verifies that the embedded downstream payload strictly matches c03_to_c04_twin_state_contract.json."""
    payload = valid_twin_state["downstream_contract_payload"]
    validate(instance=payload, schema=downstream_contract_schema)


def test_missing_required_metadata_field_fails(digital_twin_schema, valid_twin_state):
    """Verifies that omitting a required field fails validation."""
    invalid_state = valid_twin_state.copy()
    del invalid_state["metadata"]["twin_id"]
    with pytest.raises(ValidationError):
        validate(instance=invalid_state, schema=digital_twin_schema)


def test_invalid_state_label_enum_fails(digital_twin_schema, valid_twin_state):
    """Verifies that an unrecognized operational state label is rejected."""
    invalid_state = valid_twin_state.copy()
    invalid_state["operating_state"]["state_label"] = "UNKNOWN_IMPOSSIBLE_STATE"
    with pytest.raises(ValidationError):
        validate(instance=invalid_state, schema=digital_twin_schema)


if __name__ == "__main__":
    import sys
    print("Running Digital Twin Schema Verification Tests...")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        dt_schema = json.load(f)
    with open(DOWNSTREAM_CONTRACT_PATH, "r", encoding="utf-8") as f:
        downstream_schema = json.load(f)

    # Instantiate valid state
    dummy = get_valid_twin_state()

    # Test 1: File exists
    test_schema_file_exists()
    print("  [PASS] Schema file exists")

    # Test 2: Valid instance
    test_valid_twin_state_conforms_to_schema(dt_schema, dummy)
    print("  [PASS] Valid twin state conforms to DSR schema")

    # Test 3: Downstream contract
    test_downstream_payload_conforms_to_shared_contract(downstream_schema, dummy)
    print("  [PASS] Downstream payload strictly conforms to c03_to_c04_twin_state_contract.json")

    # Test 4: Missing field
    try:
        inv1 = dummy.copy()
        del inv1["metadata"]["twin_id"]
        validate(instance=inv1, schema=dt_schema)
        print("  [FAIL] Missing twin_id did not raise error")
        sys.exit(1)
    except ValidationError:
        print("  [PASS] Missing required metadata.twin_id successfully rejected")

    # Test 5: Invalid enum
    try:
        inv2 = dummy.copy()
        inv2["operating_state"]["state_label"] = "UNKNOWN_IMPOSSIBLE_STATE"
        validate(instance=inv2, schema=dt_schema)
        print("  [FAIL] Invalid state label did not raise error")
        sys.exit(1)
    except ValidationError:
        print("  [PASS] Unrecognized state label enum successfully rejected")

    print("\nALL 5 DIGITAL TWIN SCHEMA TESTS PASSED SUCCESSFULLY!")

