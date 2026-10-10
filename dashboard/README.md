# Unified Monitoring Dashboard

This directory houses the interactive, unified real-time monitoring and visualization dashboard for the J26-SE-326 project.

---

## 1. Role in the Architecture
The dashboard serves as the central visual integration layer, bringing together outputs from all four research components:
- **Tab 1: C01 Edge Telemetry & Hardware Health**
  - Live waveform visualizer (Temperature, 3-Axis Vibration, Pneumatic Pressure, Motor Current, Acoustic Spectrum).
  - On-device DSP feature gauges (RMS, Kurtosis, Crest Factor, $\Delta P/\Delta t$).
  - Edge Anomaly Detection alert status and 5V Fail-Safe Relay latch indicator.
- **Tab 2: C02 Predictive Maintenance & RUL Forecasting**
  - Fault classification probabilities (Normal, Air Leak, Valve Fault, Bearing Degradation, Overheating).
  - Remaining Useful Life (RUL) forecasting curve with 95% confidence bounds.
  - Interactive SHAP waterfall and summary plots explaining feature contributions to maintenance personnel.
- **Tab 3: C03 Industrial Digital Twin**
  - Real-time compressor operating state indicator (Idle, Pressurizing, Holding, Discharge, Anomaly).
  - 2D/3D state schematic showing active physical subsystem states.
  - State-specific behavioral profile tracking (deviation index, persistence counter, and degradation drift).
- **Tab 4: C04 RL Prescriptive Maintenance Policies**
  - Prescriptive maintenance recommendation feed (e.g. "Optimal Action: Perform Minor Service in 12 operating hours").
  - Comparative policy visualizer (DQN policy vs Run-to-Failure vs Fixed Periodic).
  - Cumulative downtime hours avoided and operational cost savings.

---

## 2. Technology Stack
- **Framework:** Streamlit (Python) for rapid, interactive research dashboard development.
- **Plotting & Visuals:** Plotly, Altair, and Matplotlib.
- **Data Ingestion:** Reads live serial data / CSV logs from C01 or loads model artifacts from C02, C03, and C04.

---

## 3. Quick Start
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
