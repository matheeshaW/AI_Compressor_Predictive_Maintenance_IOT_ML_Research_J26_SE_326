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
        """Reads and parses a single line JSON telemetry packet from the serial stream."""
        if not self.is_connected or not self.ser:
            return None

        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        except Exception:
            pass
        return None

    def close(self):
        """Closes the active serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False
