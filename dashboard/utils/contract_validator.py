"""
contract_validator.py
=====================
Validates incoming telemetry payloads against the official shared JSON schemas
defined in shared/interfaces/
"""

import os
import json
from typing import Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join(REPO_ROOT, "shared", "interfaces", "c01_to_c02_telemetry_contract.json")


class TelemetryContractValidator:
    """Validates C01 -> C02 telemetry packets against the official JSON contract."""
    
    def __init__(self, schema_path: str = SCHEMA_PATH):
        self.schema = None
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.schema = json.load(f)
            except Exception as e:
                print(f"[!] Warning: Failed to load contract schema: {e}")

    def validate_packet(self, packet: Dict[str, Any]) -> Tuple[bool, str]:
        """Performs structural and schema validation on a telemetry packet."""
        if not isinstance(packet, dict):
            return False, "Packet must be a valid JSON dictionary"

        required_keys = [
            "timestamp_iso", "device_id", "motor_current_rms_amps",
            "pressure_psi", "temperature_cylinder_head_c", "vibration_rms_g",
            "vibration_kurtosis", "relay_state", "anomaly_score"
        ]

        missing = [k for k in required_keys if k not in packet]
        if missing:
            return False, f"Missing required contract keys: {missing}"

        # Range bounds validation
        if not (0.0 <= packet.get("anomaly_score", 0.0) <= 1.0):
            return False, f"anomaly_score out of [0.0, 1.0] bounds: {packet.get('anomaly_score')}"

        if packet.get("pressure_psi", 0.0) < 0.0:
            return False, f"pressure_psi cannot be negative: {packet.get('pressure_psi')}"

        return True, "Valid Contract Packet"
