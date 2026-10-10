#!/usr/bin/env python3
"""
serial_telemetry_recorder.py
Component 01: High-Speed Serial Telemetry Recorder & Data Logger
Researcher: Weerakoon W.M.M.B (IT23155152) - J26-SE-326

Connects to ESP32-S3 over UART/USB-Serial, ingests dual-rate telemetry packets,
validates schema consistency, and streams data into structured CSV files for
offline machine learning model training and digital twin synchronization.
"""

import os
import sys
import time
import argparse
import random
from datetime import datetime, timezone

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

CSV_COLUMNS = [
    "timestamp_iso",
    "timestamp_ms",
    "temp_c",
    "press_psi",
    "press_v",
    "curr_a",
    "curr_v",
    "sound_mv",
    "vib_x",
    "vib_y",
    "vib_z",
    "anomaly",
    "relay"
]

def generate_simulated_packet(base_time_ms: int, state: str = "NORMAL") -> list:
    """Generates realistic compressor telemetry for offline testing and benchmarking."""
    now_iso = datetime.now(timezone.utc).isoformat()

    
    if state == "NORMAL":
        temp_c = 28.5 + random.uniform(-0.3, 0.4)
        press_psi = 32.0 + random.uniform(-0.5, 0.8)
        press_v = 0.33 + (press_psi / 72.52) * 2.64
        curr_a = 4.8 + random.uniform(-0.2, 0.3)
        curr_v = 1.65 + (curr_a * 0.066)
        sound_mv = random.uniform(60.0, 110.0)
        vib_x = 9.81 + random.uniform(-0.4, 0.4)
        vib_y = random.uniform(-0.3, 0.3)
        vib_z = random.uniform(-0.3, 0.3)
        anomaly = 0
        relay = 0
    elif state == "HIGH_PRESSURE":
        temp_c = 34.2 + random.uniform(0.0, 1.2)
        press_psi = 46.5 + random.uniform(0.0, 2.5)  # Breaches 45 PSI trip limit
        press_v = 0.33 + (press_psi / 72.52) * 2.64
        curr_a = 7.9 + random.uniform(0.0, 0.8)
        curr_v = 1.65 + (curr_a * 0.066)
        sound_mv = random.uniform(140.0, 240.0)
        vib_x = 9.81 + random.uniform(-1.2, 1.8)
        vib_y = random.uniform(-0.8, 0.8)
        vib_z = random.uniform(-0.8, 0.8)
        anomaly = 1
        relay = 1
    else:  # BEARING_VIBRATION
        temp_c = 31.0 + random.uniform(0.0, 0.5)
        press_psi = 30.0 + random.uniform(-1.0, 1.0)
        press_v = 0.33 + (press_psi / 72.52) * 2.64
        curr_a = 5.4 + random.uniform(-0.3, 0.5)
        curr_v = 1.65 + (curr_a * 0.066)
        sound_mv = random.uniform(180.0, 310.0)
        vib_x = 9.81 + random.uniform(-3.5, 4.2)
        vib_y = random.uniform(-2.2, 2.5)
        vib_z = random.uniform(-1.8, 2.0)
        anomaly = 1
        relay = 0

    return [
        now_iso,
        base_time_ms,
        round(temp_c, 2),
        round(press_psi, 2),
        round(press_v, 3),
        round(curr_a, 3),
        round(curr_v, 3),
        round(sound_mv, 1),
        round(vib_x, 2),
        round(vib_y, 2),
        round(vib_z, 2),
        anomaly,
        relay
    ]

def record_telemetry(port: str, baud: int, output_path: str, duration_sec: int, simulate: bool):
    """Logs telemetry data to the specified output CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    print("=================================================================")
    print("  J26-SE-326: Component 01 Serial Telemetry Recorder            ")
    print(f"  Target File: {output_path}")
    print(f"  Mode: {'SIMULATED' if simulate else f'HARDWARE ({port} @ {baud})'}")
    print(f"  Capture Duration: {duration_sec if duration_sec > 0 else 'Continuous'} sec")
    print("=================================================================\n")

    ser = None
    if not simulate:
        if not HAS_SERIAL:
            print("[ERROR] pyserial is not installed. Run `pip install pyserial` or use `--simulate`.")
            sys.exit(1)
        try:
            ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(2.0)  # Allow MCU reboot
            print(f"[OK] Connected to hardware port {port} at {baud} baud.")
        except Exception as e:
            print(f"[ERROR] Could not open port {port}: {e}")
            print("[INFO] Falling back to simulation mode.")
            simulate = True

    start_time = time.time()
    packet_count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(CSV_COLUMNS) + "\n")
            f.flush()

        try:
            base_ms = 0
            while True:
                if duration_sec > 0 and (time.time() - start_time) >= duration_sec:
                    print(f"\n[INFO] Target duration {duration_sec}s reached. Stopping capture.")
                    break

                if simulate:
                    time.sleep(0.04)  # 25 Hz rate
                    base_ms += 40
                    # Introduce occasional anomalies
                    state = "HIGH_PRESSURE" if (150 < packet_count < 180) else "NORMAL"
                    record = generate_simulated_packet(base_ms, state=state)
                    line = ",".join(map(str, record))
                else:
                    raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw_line or "timestamp" in raw_line.lower() or "=" in raw_line:
                        continue
                    parts = raw_line.split(",")
                    if len(parts) < 11:
                        continue
                    now_iso = datetime.utcnow().isoformat() + "Z"
                    line = f"{now_iso},{raw_line}"

                f.write(line + "\n")
                f.flush()
                packet_count += 1

                if packet_count % 25 == 0:  # Print status once per second (every 25 frames)
                    fields = line.split(",")
                    print(f"[{packet_count:05d}] Temp: {fields[2]}°C | Pres: {fields[3]} PSI | Curr: {fields[5]}A | Sound: {fields[7]}mV | VibX: {fields[8]} m/s² | Anom: {fields[11]}")

        except KeyboardInterrupt:
            print(f"\n[INFO] User interrupted capture after {packet_count} packets.")
        finally:
            if ser and ser.is_open:
                ser.close()

    print(f"\n[COMPLETE] Successfully logged {packet_count} packets to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="J26-SE-326 Component 01 Telemetry Recorder")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port (e.g., /dev/ttyUSB0 or COM8)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--output", default="../data/raw/baseline_compressor_runs.csv", help="Output CSV path")
    parser.add_argument("--duration", type=int, default=10, help="Capture duration in seconds (0 = infinite)")
    parser.add_argument("--simulate", action="store_true", help="Generate simulated packets without hardware")

    args = parser.parse_args()
    record_telemetry(args.port, args.baud, args.output, args.duration, args.simulate)

if __name__ == "__main__":
    main()
