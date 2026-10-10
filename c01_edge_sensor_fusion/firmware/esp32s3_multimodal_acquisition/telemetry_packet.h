#ifndef TELEMETRY_PACKET_H
#define TELEMETRY_PACKET_H

#include <Arduino.h>

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
    bool relay_latched;

    /**
     * @brief Formats telemetry packet into a standardized comma-separated string
     */
    String toCSV() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "%lu,%.2f,%.2f,%.3f,%.3f,%.3f,%.1f,%.2f,%.2f,%.2f,%d,%d",
                 timestamp_ms,
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
     * @brief CSV Header line matching the serialized format
     */
    static const char* getCSVHeader() {
        return "timestamp_ms,temp_c,press_psi,press_v,curr_a,curr_v,sound_mv,vib_x,vib_y,vib_z,anomaly,relay";
    }

    /**
     * @brief Outputs structured JSON matching c01_to_c02_telemetry_contract.json
     */
    String toJSON(const char* device_id = "ESP32S3-COMPRESSOR-TESTBED-01") const {
        char buf[384];
        snprintf(buf, sizeof(buf),
                 "{\"device_id\":\"%s\",\"timestamp_ms\":%lu,\"features\":{"
                 "\"temperature_celsius\":%.2f,\"pressure_psi\":%.2f,"
                 "\"motor_current_rms_a\":%.3f,\"acoustic_energy_mv\":%.1f,"
                 "\"vibration_x_rms_ms2\":%.2f,\"vibration_y_rms_ms2\":%.2f,\"vibration_z_rms_ms2\":%.2f},"
                 "\"edge_status\":{\"anomaly_flag\":%s,\"relay_latched\":%s}}",
                 device_id,
                 timestamp_ms,
                 temperature_c,
                 pressure_psi,
                 motor_current_a,
                 acoustic_mv,
                 accel_x_ms2,
                 accel_y_ms2,
                 accel_z_ms2,
                 anomaly_flag ? "true" : "false",
                 relay_latched ? "true" : "false");
        return String(buf);
    }
};

#endif // TELEMETRY_PACKET_H
