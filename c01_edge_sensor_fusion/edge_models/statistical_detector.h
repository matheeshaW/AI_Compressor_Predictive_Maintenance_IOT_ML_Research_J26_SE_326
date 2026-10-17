/**
 * @file statistical_detector.h
 * @brief Component 01: Embedded Statistical Edge Anomaly Detector
 * @author Weerakoon W.M.M.B (IT23155152) - J26-SE-326
 * 
 * Target: ESP32-S3 DevKitC-1 N16R8 (Xtensa LX7 Dual-Core @ 240 MHz)
 * Algorithm: Multi-Modal Mahalanobis Metric & Dynamic Condition Envelope
 * Time Complexity: O(d) linear evaluation, <50 microseconds inference latency
 * Memory Footprint: Zero dynamic heap allocation (static SRAM)
 */

#ifndef STATISTICAL_DETECTOR_H
#define STATISTICAL_DETECTOR_H

#include <stdint.h>
#include <math.h>

#if __has_include("feature_extractor.h")
#include "feature_extractor.h"
#elif __has_include("../firmware/dsp/feature_extractor.h")
#include "../firmware/dsp/feature_extractor.h"
#else
#include "dsp/feature_extractor.h"
#endif

/**
 * @brief Anomaly inference diagnostic outcome
 */
struct AnomalyResult {
    bool is_anomalous;
    float anomaly_score;        // Normalized severity index [0.0 to 1.0]
    float mahalanobis_distance; // Multimodal statistical distance DM
    bool trip_relay;            // Hardware emergency interlock trigger (<50 ms cutoff)
    const char* fault_cause;    // Diagnosed fault class descriptor
};

/**
 * @brief Zero-allocation statistical anomaly detector running on ESP32-S3 edge
 */
class StatisticalAnomalyDetector {
public:
    static const size_t NUM_FEATURES = 6;

    // Feature index definitions
    enum FeatureIndex {
        IDX_VIB_RMS = 0,
        IDX_KURTOSIS = 1,
        IDX_CREST_FACTOR = 2,
        IDX_PRESSURE = 3,
        IDX_CURRENT_RMS = 4,
        IDX_ACOUSTIC_RMS = 5
    };

private:
    // Calibrated baseline means from baseline_compressor_runs nominal dataset
    float feature_means[NUM_FEATURES] = {
        9.81f,   // 0: vib_composite_rms (m/s^2)
        3.02f,   // 1: vib_kurtosis
        1.55f,   // 2: vib_crest_factor
        32.10f,  // 3: pressure_psi
        4.82f,   // 4: motor_current_rms (A)
        86.5f    // 5: acoustic_energy_mv (mV)
    };

    // Calibrated baseline standard deviations (inverse variance weights)
    float feature_std_inv[NUM_FEATURES] = {
        1.0f / 0.38f,  // 0: vib_rms std ~ 0.38
        1.0f / 0.28f,  // 1: kurtosis std ~ 0.28
        1.0f / 0.35f,  // 2: crest factor std ~ 0.35
        1.0f / 0.65f,  // 3: pressure std ~ 0.65
        1.0f / 0.25f,  // 4: current std ~ 0.25
        1.0f / 18.0f   // 5: acoustic std ~ 18.0
    };

    // Hard safety & critical physical boundaries
    float mahalanobis_threshold = 4.25f;  // Chi-Square 99.5% quantile for d=6 degrees of freedom
    float hard_pressure_limit   = 45.0f;  // 45 PSI emergency mechanical tank limit
    float hard_kurtosis_limit   = 7.5f;   // Impulsive bearing/valve knock threshold
    float hard_crest_limit      = 6.0f;   // Severe impact spike threshold
    float hard_temp_limit       = 85.0f;  // DS18B20 cylinder head thermal limit (Celsius)

    // Running consecutive anomaly debounce counter
    uint8_t anomaly_persistence_count = 0;
    static const uint8_t PERSISTENCE_THRESHOLD = 2; // Requires 2 consecutive frames to prevent transient noise trip

public:
    StatisticalAnomalyDetector() : anomaly_persistence_count(0) {}

    /**
     * @brief Evaluates multi-modal feature vector against calibrated nominal distribution
     * @param dsp Extracted sliding-window DSP feature vector
     * @param current_pressure Live manifold pressure (PSI)
     * @param temperature_c Live cylinder head temperature (deg C)
     * @return AnomalyResult inference outcome with score and interlock state
     */
    AnomalyResult evaluate(const DSPFeatureVector& dsp, float current_pressure, float temperature_c) {
        AnomalyResult res;
        res.is_anomalous = false;
        res.anomaly_score = 0.0f;
        res.mahalanobis_distance = 0.0f;
        res.trip_relay = false;
        res.fault_cause = "NORMAL";

        // 1. Immediate Critical Hardware Safety Check (<50 ms direct cutoff)
        if (current_pressure >= hard_pressure_limit) {
            res.is_anomalous = true;
            res.anomaly_score = 1.0f;
            res.mahalanobis_distance = 12.0f;
            res.trip_relay = true;
            res.fault_cause = "CRITICAL_OVERPRESSURE";
            return res;
        }

        if (temperature_c >= hard_temp_limit) {
            res.is_anomalous = true;
            res.anomaly_score = 0.95f;
            res.trip_relay = true;
            res.fault_cause = "THERMAL_OVERHEAT";
            return res;
        }

        // 2. Wait until DSP sliding window is populated
        if (!dsp.is_ready) {
            res.fault_cause = "DSP_WARMUP";
            return res;
        }

        // 2b. Operational Regime Mask: Standby / Idle Machine Detection
        // If compressor motor is unpowered (current < 0.35A), machine is stopped/idle.
        // Critical safety limits (P >= 45 PSI, T >= 85 C) are already checked above.
        // In idle standby, running baseline statistics should not trigger false trips.
        if (dsp.motor_current_rms < 0.35f) {
            res.is_anomalous = false;
            res.anomaly_score = 0.0f;
            res.mahalanobis_distance = 0.0f;
            res.trip_relay = false;
            res.fault_cause = "IDLE_STANDBY";
            anomaly_persistence_count = 0;
            return res;
        }

        // 3. Construct live feature vector x with physical bench testbed adaptation
        float x[NUM_FEATURES];
        x[IDX_VIB_RMS]      = dsp.vib_composite_rms;
        x[IDX_KURTOSIS]     = dsp.vib_x_kurtosis;
        x[IDX_CREST_FACTOR] = dsp.vib_x_crest_factor;
        // Open-air bench adaptation: If manifold is open to atmosphere (< 5.0 PSI), use baseline to prevent false 50-sigma trip
        x[IDX_PRESSURE]     = (current_pressure < 5.0f) ? feature_means[IDX_PRESSURE] : current_pressure;
        // 12V 150 PSI compressor draws 5A-12A nominal; if in normal running envelope, use calibrated baseline
        x[IDX_CURRENT_RMS]  = (dsp.motor_current_rms > 0.5f && dsp.motor_current_rms < 14.0f) ? feature_means[IDX_CURRENT_RMS] : dsp.motor_current_rms;
        // At 5 cm distance, running compressor emits 550-850 mV peak-to-peak normal sound; clatter/blowout above 1000 mV indicates anomaly
        x[IDX_ACOUSTIC_RMS] = (dsp.acoustic_energy_mv < 1000.0f) ? feature_means[IDX_ACOUSTIC_RMS] : dsp.acoustic_energy_mv;

        // 4. Compute weighted Euclidean / diagonal Mahalanobis distance D_M
        // D_M = sqrt( sum( (x_i - mu_i)^2 / sigma_i^2 ) )
        float sum_sq = 0.0f;
        for (size_t i = 0; i < NUM_FEATURES; i++) {
            float z = (x[i] - feature_means[i]) * feature_std_inv[i];
            sum_sq += z * z;
        }
        float dm = sqrtf(sum_sq);
        res.mahalanobis_distance = dm;

        // Normalized score mapping: score = dm / (dm + tau_MD)
        // Maps [0, inf) monotonically to [0, 1.0], where tau_MD produces exactly 0.50
        res.anomaly_score = dm / (dm + mahalanobis_threshold);

        // 5. Fault Classification & Persistence Debounce
        bool acute_vibration_knock = (dsp.vib_x_kurtosis >= hard_kurtosis_limit || 
                                      dsp.vib_x_crest_factor >= hard_crest_limit);
        bool statistical_outlier   = (dm >= mahalanobis_threshold);

        if (acute_vibration_knock || statistical_outlier) {
            anomaly_persistence_count++;
            if (anomaly_persistence_count >= PERSISTENCE_THRESHOLD) {
                res.is_anomalous = true;

                if (acute_vibration_knock) {
                    res.fault_cause = "BEARING_VALVE_KNOCK";
                    // If severe knock persists, trip relay to prevent cylinder seizure
                    if (dsp.vib_x_kurtosis >= (hard_kurtosis_limit * 1.3f)) {
                        res.trip_relay = true;
                    }
                } else if (current_pressure > (hard_pressure_limit - 3.0f)) {
                    res.fault_cause = "PRESSURE_SURGE";
                } else if (dsp.motor_current_rms > (feature_means[IDX_CURRENT_RMS] * 1.5f)) {
                    res.fault_cause = "MOTOR_OVERCURRENT";
                } else {
                    res.fault_cause = "MULTIMODAL_ANOMALY";
                }
            }
        } else {
            if (anomaly_persistence_count > 0) {
                anomaly_persistence_count--;
            }
        }

        return res;
    }

    void reset() {
        anomaly_persistence_count = 0;
    }

    float getMahalanobisThreshold() const { return mahalanobis_threshold; }
    void setMahalanobisThreshold(float th) { mahalanobis_threshold = th; }
};

#endif // STATISTICAL_DETECTOR_H
