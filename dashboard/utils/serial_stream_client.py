"""
serial_stream_client.py
========================
Serial COM client reading live JSON telemetry packets from the physical
ESP32-S3 testbed microcontroller.
"""

import json
import time
from typing import Optional, Dict, Any, List

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SerialStreamClient:
    """Manages serial connection to ESP32-S3 and reads JSON telemetry frames."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False

    @staticmethod
    def list_available_ports() -> List[str]:
        """Returns a list of detected serial communication ports."""
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def connect(self, port_name: Optional[str] = None) -> bool:
        """Attempts to open the serial port."""
        if not SERIAL_AVAILABLE:
            return False

        target_port = port_name or self.port
        if not target_port:
            available = self.list_available_ports()
            if available:
                target_port = available[0]
            else:
                return False

        try:
            self.ser = serial.Serial(target_port, self.baudrate, timeout=self.timeout)
            self.port = target_port
            self.is_connected = True
            time.sleep(1.0)  # Allow bootloader settling
            return True
        except Exception as e:
            self.is_connected = False
            return False

    def read_packet(self) -> Optional[Dict[str, Any]]:
        """Reads and parses a telemetry packet (JSON or CSV) from the serial stream."""
        if not self.is_connected or not self.ser:
            return None

        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                return None

            # 1. JSON Packet format
            if line.startswith("{") and line.endswith("}"):
                data = json.loads(line)
                if "features" in data and "edge_status" in data:
                    f = data["features"]
                    s = data["edge_status"]
                    return {
                        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "device_id": data.get("device_id", "ESP32S3-COMP-01"),
                        "motor_current_rms_amps": round(f.get("motor_current_rms_a", 0.0), 2),
                        "pressure_psi": round(f.get("pressure_psi", 0.0), 2),
                        "pressure_rate_of_change_psi_sec": round(f.get("delta_pressure_dt", 0.0), 3),
                        "temperature_cylinder_head_c": round(f.get("temperature_celsius", 25.0), 1),
                        "vibration_rms_g": round(f.get("vibration_x_rms_ms2", 0.0) / 9.80665, 3),
                        "vibration_kurtosis": round(f.get("vibration_kurtosis", 3.0), 2),
                        "vibration_crest_factor": round(f.get("vibration_crest_factor", 1.5), 2),
                        "acoustic_energy_mv": round(f.get("acoustic_energy_mv", 0.0), 1),
                        "relay_state": "TRIPPED_OPEN" if s.get("relay_latched", False) else "ENERGIZED_CLOSED",
                        "anomaly_score": round(s.get("anomaly_score", 0.0), 3),
                        "fault_cause": s.get("fault_cause", "NORMAL"),
                        "trip_relay": bool(s.get("relay_latched", False) or s.get("anomaly_flag", False))
                    }
                return data

            # 2. Standard CSV Telemetry Packet (12 comma-separated values)
            # timestamp_ms,temp_c,press_psi,press_v,curr_a,curr_v,sound_mv,vib_x,vib_y,vib_z,anomaly,relay
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 12 and parts[0].isdigit():
                t_ms = int(parts[0])
                temp_c = float(parts[1])
                press_psi = float(parts[2])
                press_v = float(parts[3])
                curr_a = float(parts[4])
                curr_v = float(parts[5])
                sound_mv = float(parts[6])
                vib_x = float(parts[7])
                vib_y = float(parts[8])
                vib_z = float(parts[9])
                anomaly = int(parts[10]) == 1
                relay_latched = int(parts[11]) == 1

                # Calculate vibration composite vector
                v_composite = (vib_x**2 + vib_y**2 + vib_z**2)**0.5
                v_rms_g = v_composite / 9.80665

                # Kurtosis estimate: when anomaly occurs or vibration spikes
                if anomaly:
                    kurtosis = 8.8 if sound_mv > 200.0 else 7.6
                    crest_factor = 6.2
                    score = 0.94
                    fault_cause = "CRITICAL_OVERPRESSURE" if press_psi >= 45.0 else ("BEARING_VALVE_KNOCK" if sound_mv > 150.0 else "MULTIMODAL_ANOMALY")
                else:
                    kurtosis = 3.02
                    crest_factor = 2.1
                    score = 0.08
                    fault_cause = "NORMAL"

                return {
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "device_id": "ESP32S3-COMPRESSOR-TESTBED-01",
                    "timestamp_ms": t_ms,
                    "motor_current_rms_amps": round(curr_a, 2),
                    "pressure_psi": round(press_psi, 2),
                    "pressure_rate_of_change_psi_sec": 0.0,
                    "temperature_cylinder_head_c": round(temp_c, 1),
                    "vibration_rms_g": round(v_rms_g, 3),
                    "vibration_kurtosis": round(kurtosis, 2),
                    "vibration_crest_factor": round(crest_factor, 2),
                    "acoustic_energy_mv": round(sound_mv, 1),
                    "relay_state": "TRIPPED_OPEN" if relay_latched else "ENERGIZED_CLOSED",
                    "anomaly_score": round(score, 3),
                    "fault_cause": fault_cause,
                    "trip_relay": relay_latched or anomaly
                }
        except Exception:
            pass
        return None

    def close(self):
        """Closes the active serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False
