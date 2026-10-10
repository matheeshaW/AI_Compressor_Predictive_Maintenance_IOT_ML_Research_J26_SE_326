# Component 02: AI-Based Predictive Maintenance & RUL Forecasting

**Researcher:** Herath H.M.N.P (`IT23221178`)  
**Target Milestone:** Progress Presentation 1 (PP1) — >70% Functional Completion

---

## 1. Scope & Objectives
Component 02 develops machine learning and deep learning pipelines to predict developing compressor faults, estimate Remaining Useful Life (RUL), and interpret predictions using SHAP (SHapley Additive exPlanations).

- **SO1:** Preprocess multivariate compressor time-series without temporal leakage.
- **SO2:** Develop and evaluate temporal deep learning models (LSTM).
- **SO3:** Develop and evaluate ensemble machine learning models (XGBoost).
- **SO4:** Formulate degradation targets and forecast Remaining Useful Life (RUL).
- **SO5:** Generate SHAP explainability waterfall and summary plots for maintenance diagnosis.
- **SO6:** Benchmark and export comparative metrics (F1-score, MAE, RMSE, latency).

---

## 2. Directory Structure
```text
c02_predictive_maintenance_rul/
├── data/
│   ├── download_datasets.py
│   └── sample_metropt3_snippet.csv
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaner.py
│   │   └── sequence_window_generator.py
│   ├── models/
│   │   ├── xgboost_baseline.py
│   │   └── lstm_temporal_model.py
│   ├── rul/
│   │   └── rul_target_builder.py
│   └── xai/
│       └── shap_explainer.py
├── evaluation/
│   ├── evaluate_models.py
│   └── metrics_summary.csv
├── notebooks/
│   ├── 01_metropt3_eda_and_cleaning.ipynb
│   ├── 02_model_training_and_comparison.ipynb
│   └── 03_shap_explainability_analysis.ipynb
├── requirements.txt
└── tests/
    └── test_preprocessing.py
```

---

## 3. Datasets Used
- **MetroPT-3 Dataset:** Primary multivariate benchmark with 1.5M+ records containing pressure, oil temperature, motor current, and air leak failure events.
- **Laboratory Prototype Streams:** Ingests live telemetry conforming to `shared/interfaces/c01_to_c02_telemetry_contract.json`.
