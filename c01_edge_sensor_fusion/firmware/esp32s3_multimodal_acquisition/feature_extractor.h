#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <stdint.h>
#include <math.h>
#include <stddef.h>

/**
 * @brief Zero-allocation circular ring buffer for embedded sliding window DSP
 * @tparam T Data type (e.g., float)
 * @tparam Capacity Fixed maximum window capacity (e.g. 64 or 128 samples)
 */
template <typename T, size_t Capacity>
class CircularBuffer {
private:
    T buffer[Capacity];
    size_t head = 0;
    size_t count = 0;

public:
    CircularBuffer() : head(0), count(0) {}

    void push(T value) {
        buffer[head] = value;
        head = (head + 1) % Capacity;
        if (count < Capacity) {
            count++;
        }
    }

    void clear() {
        head = 0;
        count = 0;
    }

    size_t size() const { return count; }
    size_t capacity() const { return Capacity; }
    bool isFull() const { return count == Capacity; }

    T get(size_t index) const {
        if (index >= count) return T();
        size_t actualIndex;
        if (count < Capacity) {
            actualIndex = index;
        } else {
            actualIndex = (head + index) % Capacity;
        }
        return buffer[actualIndex];
    }
};

/**
 * @brief Extracted multi-modal feature vector representing compressor mechanical & physical condition
 * Conforms directly to shared interface contracts (c01_to_c02_telemetry_contract.json)
 */
struct DSPFeatureVector {
    // Mechanical Vibration Metrics (ISO 10816 standards)
    float vib_x_rms;          // Vibration Root Mean Square (m/s^2)
    float vib_y_rms;
    float vib_z_rms;
    float vib_composite_rms;  // sqrt(x^2 + y^2 + z^2)
    float vib_x_kurtosis;     // 4th statistical moment: impulsiveness metric for bearing/valve knock
    float vib_x_crest_factor; // Ratio of Peak to RMS: early impact indicator
    float vib_x_peak_to_peak; // Total excursion min to max (m/s^2)

    // Thermodynamic Pneumatic Metrics
    float pressure_current_psi;
    float pressure_mean_psi;
    float delta_pressure_dt;  // Rate of pressure change (PSI per second)

    // Electrical Motor Metrics
    float motor_current_rms;  // True RMS current (Amperes)
    float motor_current_peak; // Peak startup / compression stroke current (Amperes)

    // Acoustic Metrics
    float acoustic_energy_mv; // Windowed acoustic RMS energy (mV)
    float acoustic_peak_mv;   // Max acoustic peak-to-peak burst (mV)

    bool is_ready;            // True when sliding window buffer is sufficiently populated
};

/**
 * @brief Sliding-window embedded DSP feature extractor for industrial air compressors
 */
class MultiModalFeatureExtractor {
public:
    static const size_t VIBRATION_WINDOW_SIZE = 64;  // ~2.5 seconds at 25 Hz
    static const size_t PRESSURE_WINDOW_SIZE  = 16;  // ~0.64 seconds at 25 Hz
    static const size_t CURRENT_WINDOW_SIZE   = 32;  // ~1.28 seconds at 25 Hz
    static const size_t ACOUSTIC_WINDOW_SIZE  = 32;

private:
    CircularBuffer<float, VIBRATION_WINDOW_SIZE> vibXBuffer;
    CircularBuffer<float, VIBRATION_WINDOW_SIZE> vibYBuffer;
    CircularBuffer<float, VIBRATION_WINDOW_SIZE> vibZBuffer;

    CircularBuffer<float, PRESSURE_WINDOW_SIZE> pressureBuffer;
    CircularBuffer<uint32_t, PRESSURE_WINDOW_SIZE> pressureTimeBuffer;

    CircularBuffer<float, CURRENT_WINDOW_SIZE> currentBuffer;
    CircularBuffer<float, ACOUSTIC_WINDOW_SIZE> acousticBuffer;

public:
    MultiModalFeatureExtractor();

    void pushVibration(float x, float y, float z);
    void pushPressure(float psi, uint32_t timestamp_ms);
    void pushCurrent(float current_a);
    void pushAcoustic(float sound_mv);

    DSPFeatureVector computeFeatures();

    // Core statistical functions
    static float computeMean(const float* data, size_t n);
    static float computeRMS(const float* data, size_t n);
    static float computePeakToPeak(const float* data, size_t n, float& peak_abs);
    static float computeKurtosis(const float* data, size_t n, float mean, float std_dev);
    static float computeStdDev(const float* data, size_t n, float mean);
};

#endif // FEATURE_EXTRACTOR_H
