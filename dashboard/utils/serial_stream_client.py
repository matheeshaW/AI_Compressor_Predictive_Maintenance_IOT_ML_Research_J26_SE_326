"""
serial_stream_client.py
========================
Serial COM client reading live real-time telemetry packets from the physical
ESP32-S3 testbed microcontroller with zero buffer backlog latency.
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
    """Manages serial connection to ESP32-S3 and reads real-time telemetry frames."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.05):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False
        self.vib_history: List[float] = []

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
            # Use non-exclusive, fast-timeout serial configuration
            self.ser = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=0.1
            )
            self.port = target_port
            self.is_connected = True
            self.vib_history = []
            time.sleep(0.1)  # Brief settling
            # Flush any stale boot messages from previous runs
            self.ser.reset_input_buffer()
            return True
        except Exception:
            self.is_connected = False
            return False

    def send_command(self, cmd: str) -> bool:
        """
        Sends an ASCII command line to the ESP32-S3 over USB serial.
        Commands:
          - CMD:START -> Arms the relay and starts the compressor.
          - CMD:STOP  -> De-energizes the relay and halts the compressor (Manual E-Stop).
          - CMD:RESET -> Unlatches safety interlock, silences buzzer, and re-arms relay.
        """
        if not self.is_connected or not self.ser:
            return False
        try:
            payload = f"{cmd.strip()}\n".encode('utf-8')
            self.ser.write(payload)
            self.ser.flush()
            return True
        except Exception as e:
            print(f"[!] Serial command send error: {e}")
            return False

    def read_packet(self) -> Optional[Dict[str, Any]]:
        """
        Reads and parses the LATEST real-time telemetry packet.
        Drains any queued backlog in the OS UART buffer so the dashboard displays
        instantaneous sensor telemetry matching physical reality with zero lag.
        """
        if not self.is_connected or not self.ser:
            return None

        try:
            lines = []
            # Drain all queued lines waiting in the OS serial buffer
            while self.ser.in_waiting > 0:
                raw = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if raw:
                    lines.append(raw)
                if len(lines) > 100:
                    break

            # If no backlog was queued, perform a single read
            if not lines:
                raw = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if raw:
                    lines.append(raw)

            if not lines:
                return None

            # Iterate backwards (from NEWEST line to oldest) to find the latest valid telemetry packet
            for line in reversed(lines):
                parsed = self._parse_line(line)
                if parsed is not None:
                    return parsed

        except Exception:
            pass
        return None

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parses a single raw serial line (JSON or CSV)."""
        # 1. JSON Packet format
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if "features" in data and "edge_status" in data:
                    f = data["features"]
                    s = data["edge_status"]
                    relay_tripped = bool(s.get("relay_latched", False))
                    anomaly_flag = bool(s.get("anomaly_flag", False))
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
                        "relay_state": "TRIPPED_OPEN" if relay_tripped else "ENERGIZED_CLOSED",
                        "anomaly_score": round(s.get("anomaly_score", 0.0), 3),
                        "fault_cause": s.get("fault_cause", "NORMAL"),
                        "is_anomalous": anomaly_flag,
                        "trip_relay": relay_tripped
                    }
            except Exception:
                pass

        # 2. Standard CSV Telemetry Packet (12 comma-separated values)
        # timestamp_ms,temp_c,press_psi,press_v,curr_a,curr_v,sound_mv,vib_x,vib_y,vib_z,anomaly,relay
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 12 and parts[0].isdigit():
            try:
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

                # Calculate composite acceleration magnitude
                v_composite = (vib_x**2 + vib_y**2 + vib_z**2)**0.5

                # Update rolling vibration history for dynamic DSP kurtosis & crest factor
                self.vib_history.append(v_composite)
                if len(self.vib_history) > 30:
                    self.vib_history.pop(0)

                # Compute dynamic statistical kurtosis, crest factor, and AC dynamic vibration RMS
                if len(self.vib_history) >= 8:
                    vals = self.vib_history
                    mean_v = sum(vals) / len(vals)
                    var_v = sum((x - mean_v)**2 for x in vals) / len(vals)
                    m4 = sum((x - mean_v)**4 for x in vals) / len(vals)
                    computed_kurtosis = m4 / (var_v**2) if var_v > 0.01 else 3.0
                    peak_v = max(vals)
                    computed_crest = peak_v / mean_v if mean_v > 0.1 else 1.5
                    # Dynamic AC Vibration RMS (ISO 10816: static 1.0g Earth gravity filtered)
                    v_dyn_rms_g = (var_v**0.5) / 9.80665
                else:
                    computed_kurtosis = 3.02
                    computed_crest = 1.8
                    v_dyn_rms_g = 0.02

                # If sharp physical vibration shock detected on latest sample
                if v_dyn_rms_g > 0.8 or computed_kurtosis > 7.0:
                    kurtosis = max(computed_kurtosis, 8.2)
                    crest_factor = max(computed_crest, 6.2)
                else:
                    kurtosis = computed_kurtosis
                    crest_factor = computed_crest

                # Fault classification and severity score
                if relay_latched:
                    score = 0.98
                    fault_cause = "CRITICAL_OVERPRESSURE" if press_psi >= 45.0 else ("BEARING_VALVE_KNOCK" if (kurtosis >= 7.5 or sound_mv > 450.0) else "HARDWARE_EMERGENCY_TRIP")
                elif anomaly:
                    score = 0.55
                    fault_cause = "BEARING_VALVE_KNOCK" if kurtosis >= 7.0 else ("ACOUSTIC_ANOMALY" if sound_mv > 300.0 else ("MOTOR_OVERCURRENT" if curr_a > 13.0 else "STATISTICAL_OUTLIER"))
                else:
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
                    "vibration_rms_g": round(v_dyn_rms_g, 3),
                    "vibration_kurtosis": round(kurtosis, 2),
                    "vibration_crest_factor": round(crest_factor, 2),
                    "acoustic_energy_mv": round(sound_mv, 1),
                    "relay_state": "TRIPPED_OPEN" if relay_latched else "ENERGIZED_CLOSED",
                    "anomaly_score": round(score, 3),
                    "fault_cause": fault_cause,
                    "is_anomalous": anomaly,
                    "trip_relay": relay_latched
                }
            except Exception:
                pass
        return None

    def close(self):
        """Closes the active serial connection."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.is_connected = False
        self.vib_history = []

