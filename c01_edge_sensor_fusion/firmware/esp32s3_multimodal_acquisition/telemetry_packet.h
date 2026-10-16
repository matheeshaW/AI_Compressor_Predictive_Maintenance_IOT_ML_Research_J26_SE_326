#ifndef TELEMETRY_PACKET_H
#define TELEMETRY_PACKET_H

#include <Arduino.h>

#include "feature_extractor.h"

/**
 * @brief Multi-Modal Telemetry Frame
 * Encapsulates the synchronized physical state across all 5 modalities
 * and conforms to the shared interface contract (c01_to_c02_telemetry_contract.json)
 */
struct MultiModalTelemetry {
    uint32_t timestamp_ms;
    float temperature_c;
    float pressure_psi;
    float pressure_volts;
    float motor_current_a;
    float motor_current_volts;
    float acoustic_mv;
    float accel_x_ms2;
    float accel_y_ms2;
    float accel_z_ms2;
    bool anomaly_flag;
    float anomaly_score;
    bool relay_latched;
    const char* fault_cause;

    // Sliding-Window DSP Statistical Metrics
    bool dsp_ready;
    float dsp_vib_x_rms;
    float dsp_vib_composite_rms;
    float dsp_vib_kurtosis;
    float dsp_vib_crest_factor;
    float dsp_delta_p_dt;
    float dsp_motor_current_rms;
    float dsp_acoustic_energy_mv;

    MultiModalTelemetry() :
        timestamp_ms(0), temperature_c(25.0f), pressure_psi(0.0f), pressure_volts(0.33f),
        motor_current_a(0.0f), motor_current_volts(1.65f), acoustic_mv(0.0f),
        accel_x_ms2(9.81f), accel_y_ms2(0.0f), accel_z_ms2(0.0f),
        anomaly_flag(false), anomaly_score(0.0f), relay_latched(false), fault_cause("NORMAL"),
        dsp_ready(false), dsp_vib_x_rms(0.0f), dsp_vib_composite_rms(0.0f),
        dsp_vib_kurtosis(3.0f), dsp_vib_crest_factor(0.0f), dsp_delta_p_dt(0.0f),
        dsp_motor_current_rms(0.0f), dsp_acoustic_energy_mv(0.0f) {}

    /**
     * @brief Formats raw telemetry packet into standardized comma-separated string
     */
    String toCSV() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "%lu,%.2f,%.2f,%.3f,%.3f,%.3f,%.1f,%.2f,%.2f,%.2f,%d,%d",
                 (unsigned long)timestamp_ms,
                 temperature_c,
                 pressure_psi,
                 pressure_volts,
                 motor_current_a,
                 motor_current_volts,
                 acoustic_mv,
                 accel_x_ms2,
                 accel_y_ms2,
                 accel_z_ms2,
                 anomaly_flag ? 1 : 0,
                 relay_latched ? 1 : 0);
        return String(buf);
    }

    /**
     * @brief CSV Header line matching the raw serialized format
     */
    static const char* getCSVHeader() {
        return "timestamp_ms,temp_c,press_psi,press_v,curr_a,curr_v,sound_mv,vib_x,vib_y,vib_z,anomaly,relay";
    }

    /**
     * @brief Extended DSP feature CSV string
     */
    String toDSPCSV() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "%lu,%.2f,%.2f,%.3f,%.3f,%.1f,%.2f,%.2f,%.2f,%d,%d",
                 (unsigned long)timestamp_ms,
                 temperature_c,
                 pressure_psi,
                 dsp_delta_p_dt,
                 dsp_motor_current_rms,
                 dsp_acoustic_energy_mv,
                 dsp_vib_x_rms,
                 dsp_vib_kurtosis,
                 dsp_vib_crest_factor,
                 anomaly_flag ? 1 : 0,
                 relay_latched ? 1 : 0);
        return String(buf);
    }

    static const char* getDSPCSVHeader() {
        return "timestamp_ms,temp_c,press_psi,delta_p_dt,curr_rms,sound_rms,vib_x_rms,vib_kurt,vib_crest,anomaly,relay";
    }

    /**
     * @brief Outputs structured JSON matching c01_to_c02_telemetry_contract.json
     */
    String toJSON(const char* device_id = "ESP32S3-COMPRESSOR-TESTBED-01") const {
#if defined(ESP32)
        float sram_kb = (float)ESP.getFreeHeap() / 1024.0f;
#else
        float sram_kb = 320.0f;
#endif
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"device_id\":\"%s\",\"timestamp_ms\":%lu,\"features\":{"
                 "\"temperature_celsius\":%.2f,\"pressure_psi\":%.2f,\"delta_pressure_dt\":%.3f,"
                 "\"motor_current_rms_a\":%.3f,\"acoustic_energy_mv\":%.1f,"
                 "\"vibration_x_rms_ms2\":%.2f,\"vibration_y_rms_ms2\":%.2f,\"vibration_z_rms_ms2\":%.2f,"
                 "\"vibration_kurtosis\":%.2f,\"vibration_crest_factor\":%.2f},"
                 "\"edge_status\":{\"anomaly_flag\":%s,\"anomaly_score\":%.3f,\"fault_cause\":\"%s\",\"relay_latched\":%s,\"sram_free_kb\":%.1f}}",
                 device_id,
                 (unsigned long)timestamp_ms,
                 temperature_c,
                 pressure_psi,
                 dsp_ready ? dsp_delta_p_dt : 0.0f,
                 dsp_ready ? dsp_motor_current_rms : motor_current_a,
                 dsp_ready ? dsp_acoustic_energy_mv : acoustic_mv,
                 dsp_ready ? dsp_vib_x_rms : accel_x_ms2,
                 accel_y_ms2,
                 accel_z_ms2,
                 dsp_ready ? dsp_vib_kurtosis : 3.0f,
                 dsp_ready ? dsp_vib_crest_factor : 0.0f,
                 anomaly_flag ? "true" : "false",
                 anomaly_score,
                 fault_cause ? fault_cause : "NORMAL",
                 relay_latched ? "true" : "false",
                 sram_kb);
        return String(buf);
    }
};

#endif // TELEMETRY_PACKET_H
