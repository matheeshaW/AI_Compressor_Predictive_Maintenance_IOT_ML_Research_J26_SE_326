# Industrial AI Compressor Intelligence Platform — Next-Gen Monitoring Dashboard

> **Component:** Unified Interactive Monitoring Dashboard  
> **Status:** Production Architecture (Redesigned from Scratch)  
> **Architecture:** Modular Multi-View Engine with Industrial Glassmorphism Design  

---

## 1. Overview & Redesign Vision

This dashboard is an entirely **new, modern, industrial-grade monitoring platform** designed from the ground up to replace earlier basic prototypes. It serves as the unified real-time mission control interface for project **J26-SE-326**, integrating physical telemetry, deep learning diagnostics, digital twin behavioral modeling, and reinforcement learning recommendations into an executive-level, visually stunning application.

### Key Architectural Enhancements over Legacy Prototype
1. **Modular Multi-View Architecture:** Decoupled into specialized view controllers (`views/`) and reusable UI components (`components/`) rather than a single monolithic script.
2. **Live Data & Contract Compliance:** Natively consumes and validates against the project's shared JSON schemas in `shared/interfaces/` (`c01_to_c02`, `c02_to_c03`, `c03_to_c04`).
3. **Dual Telemetry Modes:** Toggle seamlessly between **Live Serial Streaming** (connecting directly to the ESP32-S3 testbed over USB/UART) and **Deterministic Replay** (using benchmark datasets like MetroPT-3 and physical testbed captures).
4. **State-of-the-Art Industrial Aesthetics:** Modern dark-mode UI with tailored HSL color palettes, frosted glass cards (glassmorphism), neon status indicators, responsive charts, and Google Fonts typography (Inter & Outfit).

---

## 2. Directory Structure

```text
dashboard/
├── README.md
├── app.py                            # Main application entry point & layout coordinator
├── requirements.txt                  # Dashboard dependencies (Streamlit, Plotly, Altair, etc.)
│
├── views/                            # Component-specific interactive views
│   ├── c01_edge_telemetry_view.py     # Live multi-modal waveforms, DSP feature gauges, relay interlock
│   ├── c02_predictive_view.py         # AI fault classifications, RUL forecast curves, SHAP waterfall
│   ├── c03_digital_twin_view.py       # Twin operational states, 2D/3D component schematic, drift tracker
│   └── c04_prescriptive_view.py       # RL maintenance recommendations, DQN policy actions, ROI curves
│
├── components/                       # Reusable UI widgets & design system
│   ├── header.py                      # Global system status bar, asset selector, mode toggle
│   ├── metric_card.py                 # Industrial KPI metric cards with trend indicators
│   ├── alert_banner.py                # Fail-safe emergency trip & anomaly notifications
│   └── radar_chart.py                 # Multi-dimensional fault probability visualizations
│
├── styles/                           # Design tokens & styling
│   └── theme.css                      # Industrial dark-theme CSS, glassmorphism, animations
│
└── utils/                            # Ingestion & data handlers
    ├── serial_stream_client.py        # Serial COM reader for live ESP32-S3 telemetry
    ├── telemetry_simulator.py         # Replay runner for benchmark datasets & recorded CSVs
    └── contract_validator.py          # Validates payloads against shared JSON interface schemas
```

---

## 3. Four Dedicated Intelligence Views

### View 1: C01 Edge Telemetry & Hardware Health
- **Live Signal Waveforms:** High-refresh-rate multi-sensor plots for Temperature (DS18B20), 3-Axis Vibration (MPU-6050), Pneumatic Pressure (G1/4), Motor Current (ACS712), and Acoustics (MAX4466).
- **DSP Feature Gauges:** Circular dials and dynamic level meters for RMS vibration, Kurtosis impact factor, Crest factor, and $\Delta P/\Delta t$.
- **Emergency Interlock Panel:** Live status indicator showing the 5V relay latch status, trip latency benchmark (<50 ms), and audible alarm activation state.

### View 2: C02 Predictive Maintenance & RUL Forecasting
- **Fault Diagnostic Radar:** Real-time probability distribution across health conditions (`NORMAL`, `AIR_LEAK`, `VALVE_FAULT`, `BEARING_DEGRADATION`, `OVERHEATING`).
- **RUL Degradation Trajectory:** Historical and projected Remaining Useful Life (RUL) curve with dynamic 95% confidence interval bands.
- **Interactive SHAP Explainability:** TreeSHAP / KernelSHAP waterfall plots attributing predictions to specific sensor deviations (e.g. current spike + pressure decay).

### View 3: C03 Industrial Digital Twin & Behavioral Intelligence
- **Operational Regime Clustering:** Unsupervised cluster scatter visualizer (K-Means / GMM) showing active operational states (`IDLE`, `PRESSURIZING`, `HOLDING`, `BLOWDOWN`, `ANOMALOUS_LEAK`).
- **2D/3D Compressor Asset Schematic:** Interactive diagram highlighting degrading physical sub-assemblies (cylinder head, valves, crankcase, motor).
- **Behavioral Envelope Tracking:** Dynamic normal operating boundary vs actual observation, tracking persistent deviation counters and thermal/pneumatic drift trends.

### View 4: C04 RL Prescriptive Maintenance Policies
- **Prescriptive Action Feed:** Real-time action recommendations generated by the Deep Q-Network (`OPERATE_NORMALLY`, `SCHEDULE_MINOR_SERVICE`, `PERFORM_DEEP_INSPECTION`, `EMERGENCY_OVERHAUL`).
- **Policy Comparison Matrix:** Side-by-side performance comparison against conventional baselines (Run-to-Failure vs Fixed Periodic).
- **Executive ROI & Cost Curves:** Cumulative operational maintenance cost curves, downtime hours avoided, and projected financial savings.

---

## 4. Setup & Running

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
