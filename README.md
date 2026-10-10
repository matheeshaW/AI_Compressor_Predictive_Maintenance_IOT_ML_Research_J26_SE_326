# AI-Powered Predictive Maintenance and Intelligent Health Monitoring System for Industrial Air Compressors

[![Project ID](https://img.shields.io/badge/Project_ID-J26--SE--326-blue.svg)](https://github.com/matheeshaW/AI_Compressor_Predictive_Maintenance_IOT_ML_Research_J26_SE_326)
[![Module](https://img.shields.io/badge/Module-IT4010_Research_Project-brightgreen.svg)]()
[![Degree](https://img.shields.io/badge/Degree-BSc_(Hons)_Software_Engineering-orange.svg)]()
[![Research Cluster](https://img.shields.io/badge/Cluster-AIMS_Autonomous_Intelligent_Machines_&_Systems-purple.svg)]()
[![License](https://img.shields.io/badge/License-Academic_Use_Only-lightgrey.svg)]()

---

## 1. Project Overview

Industrial air compressors serve as critical baseline utility assets across heavy manufacturing, food processing, pharmaceutical, and automotive industries. Unexpected compressor failures trigger unplanned plant downtime, severe production disruption, elevated maintenance costs, and energy inefficiencies.

This research project develops an end-to-end, integrated four-tier intelligence framework:
$$\text{Physical Multi-Sensor Edge Acquisition} \longrightarrow \text{Predictive Maintenance \& RUL} \longrightarrow \text{Behavioral Digital Twin} \longrightarrow \text{RL Prescriptive Maintenance Policy}$$

---

## 2. Four Research Components & Student Assignments

| Component | Responsibility / Title | Researcher | Student ID | Directory |
|---|---|---|---|---|
| **C01** | **Adaptive Multi-Sensor Fusion & Edge Anomaly Detection** | **Weerakoon W.M.M.B** | `IT23155152` | [`c01_edge_sensor_fusion/`](c01_edge_sensor_fusion/) |
| **C02** | **AI-Based Predictive Maintenance & RUL Forecasting** | **Herath H.M.N.P** | `IT23221178` | [`c02_predictive_maintenance_rul/`](c02_predictive_maintenance_rul/) |
| **C03** | **Industrial Digital Twin & Behavioral Intelligence** | **Kahakotuwa K.N** | `IT23432284` | [`c03_digital_twin_behavioral/`](c03_digital_twin_behavioral/) |
| **C04** | **Reinforcement Learning-Based Maintenance Optimization** | **Liyanage H.G.W.R** | `IT23163690` | [`c04_rl_maintenance_optimization/`](c04_rl_maintenance_optimization/) |

---

## 3. System Architecture & Component Interactions

```
+----------------------------------------------------------------------------------------------------+
|                                    J26-SE-326 SYSTEM PIPELINE                                      |
+----------------------------------------------------------------------------------------------------+
|  [C01: Edge Sensing & DSP]        -->  [C02: Fault Prediction & RUL]                                |
|  - Physical Multi-Modal Testbed        - MetroPT-3 Multivariate Ingestion                           |
|  - ESP32-S3 Dual-Rate DSP              - XGBoost vs LSTM Comparison                                 |
|  - Lightweight Edge Anomaly            - RUL Degradation Formulation                                |
|  - Sub-50ms Relay Interlock            - SHAP Feature Explainability                                |
+------------------------------------+---------------------------------------------------------------+
                  |                                                  |
                  v                                                  v
+------------------------------------+---------------------------------------------------------------+
|  [C03: Behavioral Digital Twin]    -->  [C04: RL Maintenance Optimization]                           |
|  - Time-Series Sync Engine              - Markov Decision Process (MDP) Formulated                  |
|  - Unsupervised State Clustering       - Custom Gymnasium Compressor Maintenance Env                |
|  - State Behavioral Profiling          - DQN Agent vs Heuristic Baselines                           |
|  - Deviation & Degradation Tracking    - Cost, Risk, & Downtime Trade-Off Evaluation                |
+----------------------------------------------------------------------------------------------------+
```

Detailed architectural blueprints and interface contracts can be inspected in [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`shared/interfaces/`](shared/interfaces/).

---

## 4. Repository Structure

```text
AI_Compressor_Predictive_Maintenance_IOT_ML_Research_J26_SE_326/
├── .github/                      # GitHub workflows and PR templates
├── c01_edge_sensor_fusion/       # Member 1: Embedded C++, DSP, TinyML, Serial logger
├── c02_predictive_maintenance_rul/# Member 2: MetroPT-3 pipelines, XGBoost, LSTM, SHAP
├── c03_digital_twin_behavioral/  # Member 3: Time-series sync, clustering, digital twin engine
├── c04_rl_maintenance_optimization/# Member 4: Gymnasium simulation env, DQN agent, baselines
├── shared/                       # Cross-component JSON schema contracts & test data
└── docs/                         # Milestone deliverables, diagrams, and benchmarks
```

---

## 5. Git Strategy & Contribution Workflow

To maintain code quality and preserve clear attribution for SLIIT assessments, all members must adhere to this workflow:

1. **Branching Model:**
   - `main`: Stable, tagged milestone releases (e.g. `v0.7.0-pp1-release`). Protected from direct pushes.
   - `develop`: Integration branch for active peer reviews.
   - Feature branches: `feature/c0<N>-<feature-name>` (e.g., `feature/c01-dual-rate-dsp-pipeline`).
2. **Conventional Commits:**
   - Format: `<type>(<scope>): <imperative summary>`
   - Allowed types: `feat`, `fix`, `test`, `docs`, `chore`, `perf`.
   - Allowed scopes: `c01`, `c02`, `c03`, `c04`, `shared`, `repo`.
   - Example: `feat(c02): train baseline XGBoost classifier on MetroPT-3 features`
3. **Pull Requests:**
   - Open PRs against `develop` using `.github/PULL_REQUEST_TEMPLATE.md`.
   - Require at least one group member review before merging.

---

## 6. Academic Milestone Status

| Milestone | Due Date | Target Status | Individual Progress Target |
|---|---|---|---|
| Milestone 1–5: Proposal Stage & Presentation | September 2026 | **Completed** | 100% |
| Milestone 6: Individual Project Proposal Report | 2026-09-28 | **Completed & Submitted** | 100% |
| **Milestone 7: Progress Presentation 1 (PP1)** | **2026-10-22 to 2026-10-27** | **In Active Execution** | **>70% Functional Completion** |
| Milestone 8: Semester 1 Checklist Submission | 2026-10-30 | Upcoming | 100% |