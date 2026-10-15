#include "feature_extractor.h"

MultiModalFeatureExtractor::MultiModalFeatureExtractor() {
}

void MultiModalFeatureExtractor::pushVibration(float x, float y, float z) {
    vibXBuffer.push(x);
    vibYBuffer.push(y);
    vibZBuffer.push(z);
}

void MultiModalFeatureExtractor::pushPressure(float psi, uint32_t timestamp_ms) {
    pressureBuffer.push(psi);
    pressureTimeBuffer.push(timestamp_ms);
}

void MultiModalFeatureExtractor::pushCurrent(float current_a) {
    currentBuffer.push(current_a);
}

void MultiModalFeatureExtractor::pushAcoustic(float sound_mv) {
    acousticBuffer.push(sound_mv);
}

float MultiModalFeatureExtractor::computeMean(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / static_cast<float>(n);
}

float MultiModalFeatureExtractor::computeRMS(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    float sumSq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sumSq += data[i] * data[i];
    }
    return sqrtf(sumSq / static_cast<float>(n));
}

float MultiModalFeatureExtractor::computeStdDev(const float* data, size_t n, float mean) {
    if (n <= 1) return 0.0f;
    float sumSqDiff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = data[i] - mean;
        sumSqDiff += diff * diff;
    }
    return sqrtf(sumSqDiff / static_cast<float>(n));
}

float MultiModalFeatureExtractor::computePeakToPeak(const float* data, size_t n, float& peak_abs) {
    if (n == 0) {
        peak_abs = 0.0f;
        return 0.0f;
    }
    float minVal = data[0];
    float maxVal = data[0];
    float maxAbs = fabsf(data[0]);

    for (size_t i = 1; i < n; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
        float absV = fabsf(data[i]);
        if (absV > maxAbs) maxAbs = absV;
    }
    peak_abs = maxAbs;
    return (maxVal - minVal);
}

float MultiModalFeatureExtractor::computeKurtosis(const float* data, size_t n, float mean, float std_dev) {
    if (n < 4 || std_dev < 1e-5f) {
        return 3.0f; // Default mesokurtic baseline for healthy Gaussian process
    }
    float sumFourth = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float z = (data[i] - mean) / std_dev;
        sumFourth += z * z * z * z;
    }
    return sumFourth / static_cast<float>(n);
}

DSPFeatureVector MultiModalFeatureExtractor::computeFeatures() {
    DSPFeatureVector feats;
    size_t nVib = vibXBuffer.size();

    feats.is_ready = (nVib >= 16); // Minimum 16 samples for valid statistical estimates

    if (nVib == 0) {
        feats.vib_x_rms = 0.0f;
        feats.vib_y_rms = 0.0f;
        feats.vib_z_rms = 0.0f;
        feats.vib_composite_rms = 0.0f;
        feats.vib_x_kurtosis = 3.0f;
        feats.vib_x_crest_factor = 0.0f;
        feats.vib_x_peak_to_peak = 0.0f;
        feats.pressure_current_psi = 0.0f;
        feats.pressure_mean_psi = 0.0f;
        feats.delta_pressure_dt = 0.0f;
        feats.motor_current_rms = 0.0f;
        feats.motor_current_peak = 0.0f;
        feats.acoustic_energy_mv = 0.0f;
        feats.acoustic_peak_mv = 0.0f;
        return feats;
    }

    // 1. Vibration Analysis on X, Y, Z
    float arrX[VIBRATION_WINDOW_SIZE];
    float arrY[VIBRATION_WINDOW_SIZE];
    float arrZ[VIBRATION_WINDOW_SIZE];

    for (size_t i = 0; i < nVib; i++) {
        arrX[i] = vibXBuffer.get(i);
        arrY[i] = vibYBuffer.get(i);
        arrZ[i] = vibZBuffer.get(i);
    }

    feats.vib_x_rms = computeRMS(arrX, nVib);
    feats.vib_y_rms = computeRMS(arrY, nVib);
    feats.vib_z_rms = computeRMS(arrZ, nVib);
    feats.vib_composite_rms = sqrtf(feats.vib_x_rms * feats.vib_x_rms +
                                    feats.vib_y_rms * feats.vib_y_rms +
                                    feats.vib_z_rms * feats.vib_z_rms);

    float meanX = computeMean(arrX, nVib);
    float stdX  = computeStdDev(arrX, nVib, meanX);
    feats.vib_x_kurtosis = computeKurtosis(arrX, nVib, meanX, stdX);

    // Dynamic AC Crest Factor: Peak dynamic excursion divided by AC RMS (std_dev)
    float peakDevX = 0.0f;
    for (size_t i = 0; i < nVib; i++) {
        float dev = fabsf(arrX[i] - meanX);
        if (dev > peakDevX) peakDevX = dev;
    }
    feats.vib_x_crest_factor = (stdX > 1e-4f) ? (peakDevX / stdX) : 0.0f;


    // 2. Pressure & Delta P / Delta T
    size_t nPress = pressureBuffer.size();
    if (nPress > 0) {
        float arrP[PRESSURE_WINDOW_SIZE];
        for (size_t i = 0; i < nPress; i++) {
            arrP[i] = pressureBuffer.get(i);
        }
        feats.pressure_current_psi = arrP[nPress - 1];
        feats.pressure_mean_psi = computeMean(arrP, nPress);

        if (nPress >= 2) {
            uint32_t tFirst = pressureTimeBuffer.get(0);
            uint32_t tLast  = pressureTimeBuffer.get(nPress - 1);
            float dt_sec = static_cast<float>(tLast - tFirst) / 1000.0f;
            if (dt_sec > 0.01f) {
                feats.delta_pressure_dt = (arrP[nPress - 1] - arrP[0]) / dt_sec;
            } else {
                feats.delta_pressure_dt = 0.0f;
            }
        } else {
            feats.delta_pressure_dt = 0.0f;
        }
    } else {
        feats.pressure_current_psi = 0.0f;
        feats.pressure_mean_psi = 0.0f;
        feats.delta_pressure_dt = 0.0f;
    }

    // 3. Electrical Motor Current
    size_t nCurr = currentBuffer.size();
    if (nCurr > 0) {
        float arrI[CURRENT_WINDOW_SIZE];
        for (size_t i = 0; i < nCurr; i++) {
            arrI[i] = currentBuffer.get(i);
        }
        feats.motor_current_rms = computeRMS(arrI, nCurr);
        float peakAbsI = 0.0f;
        computePeakToPeak(arrI, nCurr, peakAbsI);
        feats.motor_current_peak = peakAbsI;
    } else {
        feats.motor_current_rms = 0.0f;
        feats.motor_current_peak = 0.0f;
    }

    // 4. Acoustic Energy
    size_t nSound = acousticBuffer.size();
    if (nSound > 0) {
        float arrS[ACOUSTIC_WINDOW_SIZE];
        for (size_t i = 0; i < nSound; i++) {
            arrS[i] = acousticBuffer.get(i);
        }
        feats.acoustic_energy_mv = computeRMS(arrS, nSound);
        float peakAbsS = 0.0f;
        computePeakToPeak(arrS, nSound, peakAbsS);
        feats.acoustic_peak_mv = peakAbsS;
    } else {
        feats.acoustic_energy_mv = 0.0f;
        feats.acoustic_peak_mv = 0.0f;
    }

    return feats;
}
