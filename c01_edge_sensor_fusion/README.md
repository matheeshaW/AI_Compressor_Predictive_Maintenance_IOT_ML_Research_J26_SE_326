# Component 01: Adaptive Multi-Sensor Fusion & Edge Anomaly Detection

**Researcher:** Weerakoon W.M.M.B (`IT23155152`)  
**Target Milestone:** Progress Presentation 1 (PP1) — >70% Functional Completion

---

## 1. Scope & Objectives
Component 01 is the physical and embedded edge intelligence layer. It directly interfaces heterogeneous sensors on an ESP32-S3 microcontroller, executes real-time digital signal processing (DSP), computes time-domain statistical features, detects operational anomalies on-device, and actuates physical fail-safe cut-offs.

- **SO1:** Multi-sensor acquisition (DS18B20 Temp, MPU-6050 Vibration, G1/4 Pressure, ACS712-30A Current, MAX4466 Acoustic via ADS1115).
- **SO2:** Embedded DSP feature extraction (RMS, Peak-to-Peak, Kurtosis, Crest Factor, $\Delta P/\Delta t$).
- **SO3:** Multi-sensor fusion across mechanical, electrical, thermodynamic, and acoustic domains.
- **SO4:** Lightweight on-device anomaly detection and sub-50ms relay emergency cut-off.
- **SO5:** Edge computing benchmark (latency, memory footprint, >85% bandwidth reduction).

---

## 2. Directory Structure
```text
c01_edge_sensor_fusion/
├── firmware/
│   ├── esp32s3_multimodal_acquisition/
│   │   └── esp32s3_multimodal_acquisition.ino
│   └── dsp/
│       ├── feature_extractor.h
│       └── feature_extractor.cpp
├── edge_models/
│   ├── statistical_detector.h
│   └── anomaly_thresholds.json
├── data_logger/
│   ├── serial_telemetry_recorder.py
│   └── requirements.txt
├── benchmarks/
│   └── benchmark_edge_latency.py
└── tests/
    └── test_feature_extractor.py
```

---

## 3. Hardware Testbed (Verified Rev 3.0 Architecture)
- **MCU:** ESP32-S3 DevKitC-1 N16R8 (Xtensa LX7 @ 240MHz, 8MB PSRAM, 16MB Flash)
- **Power Management:** MB-102 HW-131 on right edge; Row A (Cols 1–21) as master signal bus.
- **Sensors:**
  - `GPIO 4`: DS18B20 1-Wire Digital Temperature
  - `GPIO 8 (SDA) / GPIO 9 (SCL)`: Shared I2C bus for MPU-6050 (`0x68`) and ADS1115 (`0x48`)
  - `ADS1115 A0`: G1/4 Pressure Transducer (0–5 bar / 0–72.5 PSI)
  - `ADS1115 A1`: ACS712-30A Hall-Effect Motor Current Sensor
  - `ADS1115 A2`: MAX4466 Acoustic Microphone with Adjustable Gain
  - `GPIO 5`: 5V Optocoupler Relay Interlock
  - `GPIO 6`: Active Piezo Warning Buzzer
  - `GPIO 7`: Green Status / Heartbeat LED
