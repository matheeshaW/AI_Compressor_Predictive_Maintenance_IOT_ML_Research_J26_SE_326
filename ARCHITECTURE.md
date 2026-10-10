# System Architecture: J26-SE-326

## 1. High-Level Architectural Pipeline

The research architecture bridges physical asset telemetry with predictive, behavioral, and prescriptive intelligence:

```
+----------------------------------------------------------------------------------------------------+
|                                    PHYSICAL AIR COMPRESSOR SYSTEM                                  |
|   - 12V Reciprocating Piston Compressor / Industrial Rotary Screw Machines                         |
|   - Multi-Sensor Suite: DS18B20 Temp, MPU-6050 Vibration, ADS1115 (Pressure, Current, Acoustic)     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|  COMPONENT 01: Adaptive Multi-Sensor Fusion & Edge Anomaly Detection                                |
|  Researcher: Weerakoon W.M.M.B (IT23155152)                                                        |
|  - Microcontroller: ESP32-S3 DevKitC-1 (Xtensa Dual-Core LX7 @ 240MHz, 8MB PSRAM, 16MB Flash)      |
|  - Dual-Rate Acquisition: 25-50 Hz mechanical vibration & acoustics; 1-10 Hz thermal/pressure     |
|  - DSP Feature Extraction: Sliding window (512 samples) -> RMS, Kurtosis, Crest Factor, dP/dt      |
|  - Edge Anomaly Detection: Statistical Mahalanobis / Quantized TinyML Autoencoder                  |
|  - Fail-Safe Interlock: <50ms Relay trip on acute overpressure / severe mechanical vibration      |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼ [JSON Contract: c01_to_c02_telemetry_contract.json]
+----------------------------------------------------------------------------------------------------+
|  COMPONENT 02: AI-Based Predictive Maintenance & RUL Forecasting                                   |
|  Researcher: Herath H.M.N.P (IT23221178)                                                           |
|  - Multivariate Datasets: MetroPT-3 (1.5M points) + Laboratory Prototype Telemetry Streams         |
|  - Temporal Modeling: Recurrent LSTM Networks capturing sequential degradation trends               |
|  - Ensemble Comparison: Gradient Boosted Trees (XGBoost) for tabular feature-based classification   |
|  - Remaining Useful Life (RUL): Piecewise linear degradation curves & regression metrics (MAE/RMSE)|
|  - Explainable AI (XAI): TreeSHAP & KernelSHAP waterfall attributions for maintenance diagnosis    |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼ [JSON Contract: c02_to_c03_prediction_contract.json]
+----------------------------------------------------------------------------------------------------+
|  COMPONENT 03: Industrial Digital Twin & Behavioral Intelligence                                    |
|  Researcher: Kahakotuwa K.N (IT23432284)                                                           |
|  - Design Science Research (DSR) Artifact                                                          |
|  - Time-Series Synchronization Engine: Resampling, buffer alignment, missing value interpolation   |
|  - Operating State Discovery: Unsupervised clustering (K-Means / GMM) -> Idle, Pumping, Leak, etc.  |
|  - State-Specific Profiling: Dynamic operating envelopes, persistent deviation vs transient noise   |
|  - Digital Twin State Schema: Real-time virtual representation tracking health & degradation drift  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼ [JSON Contract: c03_to_c04_twin_state_contract.json]
+----------------------------------------------------------------------------------------------------+
|  COMPONENT 04: Reinforcement Learning-Based Maintenance Optimization                                |
|  Researcher: Liyanage H.G.W.R (IT23163690)                                                         |
|  - Formal Markov Decision Process (MDP): State S (health, RUL, twin state), Action A, Reward R     |
|  - Custom Simulation Environment: OpenAI Gymnasium compliant CompressorMaintenanceEnv               |
|  - Conventional Baselines: Run-to-Failure (Corrective), Fixed Periodic Maintenance                  |
|  - Prescriptive Agent: Deep Q-Network (DQN) with experience replay buffer & epsilon-greedy policy   |
|  - Multi-Objective Optimization: Trade-off between downtime cost, repair cost, and failure risk     |
+----------------------------------------------------------------------------------------------------+
```

## 2. Shared Interface Contracts

To guarantee that each component can be tested independently while remaining ready for seamless end-to-end integration, components exchange standardized JSON schemas defined in [`shared/interfaces/`](shared/interfaces/):

1. **`c01_to_c02_telemetry_contract.json`**:
   Streams preprocessed multi-sensor feature vectors (RMS vibration, temperature, pneumatic pressure, RMS current, acoustic energy) alongside edge anomaly flags.
2. **`c02_to_c03_prediction_contract.json`**:
   Carries predicted fault classifications, confidence probabilities, estimated Remaining Useful Life (RUL), and primary contributing SHAP feature attributions.
3. **`c03_to_c04_twin_state_contract.json`**:
   Exposes the synchronized Digital Twin state representation: active operating cluster, state confidence, behavioral deviation index, persistence counter, and operational degradation drift.
