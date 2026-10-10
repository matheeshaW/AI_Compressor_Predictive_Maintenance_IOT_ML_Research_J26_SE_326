# Shared Interfaces and Sample Data

This directory houses the formal, versioned JSON Schema interface contracts that govern all inter-component communications within the J26-SE-326 pipeline:

1. **`interfaces/c01_to_c02_telemetry_contract.json`**  
   Preprocessed multi-sensor feature packets streamed from Component 01 (Edge) to Component 02 (Predictive Layer).
2. **`interfaces/c02_to_c03_prediction_contract.json`**  
   Predicted fault classifications, RUL estimations, and SHAP explainability attributions from Component 02 to Component 03 (Digital Twin).
3. **`interfaces/c03_to_c04_twin_state_contract.json`**  
   Synchronized operational regimes, state confidence, and behavioral deviation indices passed from Component 03 to Component 04 (RL Maintenance Optimization).

---

## Contract Compliance Rule
Before opening a Pull Request merging any component feature into `develop`, the contributor must verify that their component's inputs and outputs validate against these schemas using `jsonschema` or equivalent validation tools.
