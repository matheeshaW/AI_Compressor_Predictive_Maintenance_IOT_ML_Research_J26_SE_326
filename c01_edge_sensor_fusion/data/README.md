# Component 01: Telemetry Data Directory

This directory stores sensor streams recorded from the ESP32-S3 testbed:

- `raw/`: Raw high-speed serial telemetry CSV captures (`timestamp`, `temp_c`, `vib_x`, `vib_y`, `vib_z`, `press_psi`, `curr_a`, `mic_mv`, `anomaly_flag`).
- `processed/`: Sliding-window feature extractions (`rms_vib`, `kurtosis_vib`, `crest_factor`, `delta_p_dt`, `rms_curr`).

---

## Guidelines
- Do NOT commit multi-gigabyte raw files directly to Git. Commit lightweight sample snippets (`<5 MB`) for automated testing.
- Use `data_logger/serial_telemetry_recorder.py` to capture live testbed streams.
