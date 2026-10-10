# Component 03: Industrial Digital Twin & Behavioral Intelligence

**Researcher:** Kahakotuwa K.N (`IT23432284`)  
**Target Milestone:** Progress Presentation 1 (PP1) — >70% Functional Completion  
**Methodology:** Design Science Research (DSR)

---

## 1. Scope & Objectives
Component 03 develops a continuously synchronized digital twin artifact that maintains a real-time behavioral representation of compressor operation, discovers operating regimes through unsupervised learning, and tracks persistent degradation drift.

- **SO1:** Formalize digital twin state schema and interface specification.
- **SO2:** Implement time-series synchronization, temporal alignment, and stream buffering.
- **SO3:** Discover operational states (Idle, Pumping, Pressurized, Discharge) via unsupervised clustering (K-Means / GMM).
- **SO4:** Develop state-specific behavioral baselines and track persistent deviation vs transient noise.
- **SO5:** Expose structured behavioral context via `shared/interfaces/c03_to_c04_twin_state_contract.json`.

---

## 2. Directory Structure
```text
c03_digital_twin_behavioral/
├── schemas/
│   └── digital_twin_state_schema.json
├── src/
│   ├── sync/
│   │   ├── time_series_aligner.py
│   │   └── stream_buffer.py
│   ├── clustering/
│   │   ├── state_discovery.py
│   │   └── cluster_evaluator.py
│   ├── behavioral/
│   │   ├── state_behavioral_profiler.py
│   │   └── degradation_tracker.py
│   └── twin_engine.py
├── demo/
│   └── simulate_twin_telemetry.py
├── notebooks/
│   └── 01_state_clustering_evaluation.ipynb
├── requirements.txt
└── tests/
    └── test_sync_engine.py
```
