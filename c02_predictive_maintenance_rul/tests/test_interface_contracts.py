"""
c02_predictive_maintenance_rul/tests/test_interface_contracts.py
Component 02: AI-Based Predictive Maintenance & RUL Forecasting
Automated Interface Contract Test Suite
Author: Herath H.M.N.P (IT23221178)
"""

import os
import sys
import json
import logging
import jsonschema
from datetime import datetime, timezone

# Add project root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("C02_ContractTest")

C01_SCHEMA_PATH = os.path.join(REPO_ROOT, "shared", "interfaces", "c01_to_c02_telemetry_contract.json")
C02_SCHEMA_PATH = os.path.join(REPO_ROOT, "shared", "interfaces", "c02_to_c03_prediction_contract.json")
SAMPLE_PAYLOAD_PATH = os.path.join(REPO_ROOT, "c02_predictive_maintenance_rul", "evaluation", "c02_to_c03_sample_payload.json")


def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def test_upstream_c01_contract():
    """Validates that C02 can ingest and validate C01 edge telemetry."""
    logger.info("--- TEST 1: Upstream C01 -> C02 Contract Validation ---")
    c01_schema = load_json(C01_SCHEMA_PATH)

    mock_c01_packet = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": "ESP32S3-COMPRESSOR-TESTBED-01",
        "sample_rate_hz": 25.0,
        "features": {
            "temperature_celsius": 41.8,
            "pressure_psi": 93.2,
            "delta_pressure_dt": -0.12,
            "motor_current_rms_a": 7.65,
            "vibration_x_rms_ms2": 1.42,
            "vibration_y_rms_ms2": 1.10,
            "vibration_z_rms_ms2": 0.95,
            "vibration_kurtosis": 3.08,
            "vibration_crest_factor": 2.78,
            "acoustic_energy_mv": 118.4
        },
        "edge_status": {
            "anomaly_flag": False,
            "anomaly_score": 0.08,
            "relay_latched": False,
            "sram_free_kb": 186.4
        }
    }

    jsonschema.validate(instance=mock_c01_packet, schema=c01_schema)
    logger.info("PASSED: Upstream C01 telemetry packet matches schema specification.")


def test_downstream_c02_to_c03_contract():
    """Validates that C02 generated payload complies with C02 -> C03 contract."""
    logger.info("--- TEST 2: Downstream C02 -> C03 Contract Validation ---")
    c02_schema = load_json(C02_SCHEMA_PATH)
    sample_payload = load_json(SAMPLE_PAYLOAD_PATH)

    jsonschema.validate(instance=sample_payload, schema=c02_schema)
    logger.info("PASSED: C02 sample prediction payload strictly conforms to C03 contract.")
    
    # Specific semantic assertions
    assert sample_payload["fault_prediction"]["predicted_class"] in [
        "NORMAL", "AIR_LEAK", "VALVE_FAULT", "BEARING_DEGRADATION", "OVERHEATING"
    ], "Invalid predicted_class"
    assert 0.0 <= sample_payload["fault_prediction"]["confidence_score"] <= 1.0, "Confidence score out of range"
    assert sample_payload["rul_estimation"]["estimated_rul_hours"] >= 0.0, "Negative RUL"
    assert 0.0 <= sample_payload["rul_estimation"]["health_index"] <= 1.0, "Health Index out of bounds"
    assert len(sample_payload["explainability"]["top_contributing_features"]) > 0, "No SHAP features found"
    logger.info(f"Verified {len(sample_payload['explainability']['top_contributing_features'])} SHAP features present.")


def test_negative_schema_validation():
    """Verifies that the schema validator catches invalid or corrupted payloads."""
    logger.info("--- TEST 3: Negative Schema Validation (Fault Injection) ---")
    c02_schema = load_json(C02_SCHEMA_PATH)
    
    corrupt_payload = {
        "timestamp": "invalid-timestamp",
        "asset_id": "TEST-01",
        "fault_prediction": {
            "predicted_class": "INVALID_CLASS",  # Not in allowed enum
            "confidence_score": 1.5               # Exceeds maximum 1.0
        }
    }

    try:
        jsonschema.validate(instance=corrupt_payload, schema=c02_schema)
        raise AssertionError("Validation should have failed for corrupted payload!")
    except jsonschema.exceptions.ValidationError:
        logger.info("PASSED: Schema validator correctly rejected invalid payload.")


def run_all_contract_tests():
    logger.info("=================================================================")
    logger.info("   COMPONENT 02: AUTOMATED INTERFACE CONTRACT TEST SUITE ")
    logger.info("=================================================================")
    test_upstream_c01_contract()
    test_downstream_c02_to_c03_contract()
    test_negative_schema_validation()
    logger.info("=================================================================")
    logger.info("   ALL 3 INTERFACE CONTRACT TESTS PASSED SUCCESSFULLY! (100%)    ")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_all_contract_tests()
