# Component 01: Machine Learning & DSP Notebooks

This directory contains Python Jupyter Notebooks used for offline data exploration, DSP algorithm validation, and anomaly detection model training prior to embedded edge quantization:

1. **`01_sensor_eda_and_dsp_validation.ipynb`**
   - Exploratory data analysis (EDA) of 5-modal sensor telemetry.
   - Verification of time-domain feature extraction formulas (RMS, Kurtosis, Crest Factor, $\Delta P/\Delta t$).
   - Spectral analysis (Fast Fourier Transform / power spectral density) of vibration and acoustics.

2. **`02_edge_anomaly_detection_training.ipynb`**
   - Training unsupervised anomaly detection models (1D-CNN Autoencoder, Isolation Forest, Mahalanobis Distance).
   - Evaluation metrics: ROC-AUC, Precision, Recall, and reconstruction error distributions.
   - Quantization via TensorFlow Lite (INT8) and export to C++ header array (`edge_models/tinyml_anomaly_model.h`).
