/**
 * @file esp32s3_multimodal_acquisition.ino
 * @brief Component 01: Multi-Modal Edge Acquisition & Fail-Safe Firmware
 * @author Weerakoon W.M.M.B (IT23155152) - J26-SE-326
 * 
 * Target Board: ESP32-S3 DevKitC-1 N16R8 (Xtensa LX7 Dual-Core @ 240 MHz)
 * Hardware Architecture: Rev 3.0 Left-Edge Row A Master Signal Bus
 */

#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADS1X15.h>

#include "sensor_config.h"
#include "telemetry_packet.h"

// Peripheral Instances
OneWire oneWire(PIN_DS18B20);
DallasTemperature tempSensor(&oneWire);
Adafruit_MPU6050 mpu;
Adafruit_ADS1115 ads;

// Operational Flags
bool mpu_ready = false;
bool ads_ready = false;
bool temp_ready = false;

// Dynamic Telemetry State
MultiModalTelemetry currentFrame;
float cachedTemperature = 25.0f;
bool emergencyCutoffLatched = false;

// Non-blocking Timing Trackers
uint32_t lastFastSampleTime = 0;
uint32_t lastSlowSampleTime = 0;
uint32_t lastHeartbeatTime  = 0;
bool ledHeartbeatState      = false;

void setup() {
    Serial.begin(115200);
    Serial0.begin(115200);
    delay(1000);

    Serial.println("\n===============================================================");
    Serial.println("  J26-SE-326: ESP32-S3 Multi-Modal Industrial Acquisition    ");
    Serial.println("  Component 01: Edge Sensing, Dual-Rate DSP & Fail-Safe Layer  ");
    Serial.println("  Researcher: Weerakoon W.M.M.B (IT23155152)                   ");
    Serial.println("===============================================================");

    // 1. Configure Actuator & Alarm Pins
    pinMode(PIN_LED_GREEN, OUTPUT);
    pinMode(PIN_BUZZER, OUTPUT);
    pinMode(PIN_RELAY, OUTPUT);

    // Initial state: Relay ON (energized/closed for compressor motor), LED ON
    digitalWrite(PIN_RELAY, LOW); // Active-low optocoupler: LOW = closed/conducting
    digitalWrite(PIN_LED_GREEN, HIGH);

    // Startup Diagnostic Beep Sequence (2 chirps)
    digitalWrite(PIN_BUZZER, HIGH); delay(50); digitalWrite(PIN_BUZZER, LOW); delay(80);
    digitalWrite(PIN_BUZZER, HIGH); delay(50); digitalWrite(PIN_BUZZER, LOW);

    // 2. Initialize Shared I2C Bus (SDA = GPIO 8, SCL = GPIO 9)
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    Wire.setClock(400000); // 400 kHz Fast-Mode I2C

    // 3. Initialize MPU-6050 Accelerometer
    if (mpu.begin(ADDR_MPU6050, &Wire)) {
        mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
        mpu_ready = true;
        Serial.println("[OK] MPU-6050 Vibration Sensor online at 0x68");
    } else {
        Serial.println("[WARN] MPU-6050 Not Found at 0x68. Check I2C bus.");
    }

    // 4. Initialize ADS1115 Precision ADC
    if (ads.begin(ADDR_ADS1115, &Wire)) {
        ads.setGain(GAIN_ONE); // +/- 4.096V range (1 bit = 0.125 mV)
        ads_ready = true;
        Serial.println("[OK] ADS1115 16-Bit ADC online at 0x48");
    } else {
        Serial.println("[WARN] ADS1115 Not Found at 0x48. Check I2C bus.");
    }

    // 5. Initialize DS18B20 1-Wire Temperature Sensor
    tempSensor.begin();
    tempSensor.setWaitForConversion(false); // Non-blocking conversion mode
    tempSensor.requestTemperatures();
    temp_ready = true;
    Serial.println("[OK] DS18B20 1-Wire Temperature Sensor online on GPIO 4");

    Serial.println("\n[SYSTEM] Starting Dual-Rate Telemetry Stream (25 Hz fast / 1 Hz slow)...");
    Serial.println(MultiModalTelemetry::getCSVHeader());
    Serial0.println(MultiModalTelemetry::getCSVHeader());
    Serial.println("===============================================================\n");
}

void loop() {
    uint32_t now = millis();

    // -------------------------------------------------------------
    // TASK 1: LED Heartbeat (500 ms)
    // -------------------------------------------------------------
    if (now - lastHeartbeatTime >= HEARTBEAT_INTERVAL_MS) {
        lastHeartbeatTime = now;
        ledHeartbeatState = !ledHeartbeatState;
        digitalWrite(PIN_LED_GREEN, ledHeartbeatState ? HIGH : LOW);
    }

    // -------------------------------------------------------------
    // TASK 2: Slow Loop (1 Hz) - Thermodynamic Acquisition (DS18B20)
    // -------------------------------------------------------------
    if (now - lastSlowSampleTime >= SLOW_SAMPLE_INTERVAL_MS) {
        lastSlowSampleTime = now;
        if (temp_ready) {
            float t = tempSensor.getTempCByIndex(0);
            if (t > -50.0f && t < 125.0f) {
                cachedTemperature = t;
            }
            tempSensor.requestTemperatures(); // Trigger next conversion asynchronously
        }
    }

    // -------------------------------------------------------------
    // TASK 3: Fast Loop (25 Hz / 40 ms) - Mechanical & Acoustic
    // -------------------------------------------------------------
    if (now - lastFastSampleTime >= FAST_SAMPLE_INTERVAL_MS) {
        lastFastSampleTime = now;

        currentFrame.timestamp_ms = now;
        currentFrame.temperature_c = cachedTemperature;

        // 3a. Read MPU-6050 Vibration
        if (mpu_ready) {
            sensors_event_t a, g, temp;
            mpu.getEvent(&a, &g, &temp);
            currentFrame.accel_x_ms2 = a.acceleration.x;
            currentFrame.accel_y_ms2 = a.acceleration.y;
            currentFrame.accel_z_ms2 = a.acceleration.z;
        } else {
            currentFrame.accel_x_ms2 = 0.0f;
            currentFrame.accel_y_ms2 = 0.0f;
            currentFrame.accel_z_ms2 = GRAVITY_EARTH;
        }

        // 3b. Read ADS1115 Analog Modalities
        if (ads_ready) {
            // Channel A0: G1/4 Pressure Transducer
            int16_t rawP = ads.readADC_SingleEnded(ADC_CH_PRESSURE);
            currentFrame.pressure_volts = ads.computeVolts(rawP);
            if (currentFrame.pressure_volts >= 0.30f) {
                currentFrame.pressure_psi = ((currentFrame.pressure_volts - PRESSURE_ZERO_OFFSET_V) / PRESSURE_SPAN_VOLTS) * PRESSURE_MAX_PSI;
                if (currentFrame.pressure_psi < 0.0f) currentFrame.pressure_psi = 0.0f;
            } else {
                currentFrame.pressure_psi = 0.0f;
            }

            // Channel A1: ACS712-30A Current Sensor
            int16_t rawI = ads.readADC_SingleEnded(ADC_CH_CURRENT);
            currentFrame.motor_current_volts = ads.computeVolts(rawI);
            currentFrame.motor_current_a = (currentFrame.motor_current_volts - CURRENT_ZERO_OFFSET_V) / CURRENT_SENSITIVITY_V_A;
            if (currentFrame.motor_current_a < 0.0f) currentFrame.motor_current_a = -currentFrame.motor_current_a; // RMS magnitude

            // Channel A2: MAX4466 Acoustic Microphone (Peak-to-Peak Window)
            float minV = 4.0f, maxV = 0.0f;
            for (int i = 0; i < ACOUSTIC_WINDOW_SAMPLES; i++) {
                float v = ads.computeVolts(ads.readADC_SingleEnded(ADC_CH_ACOUSTIC));
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
            currentFrame.acoustic_mv = (maxV - minV) * 1000.0f;
        }

        // ---------------------------------------------------------
        // 3c. Fail-Safe Interlock Protection (<50 ms Emergency Trip)
        // ---------------------------------------------------------
        if (currentFrame.pressure_psi >= PRESSURE_SAFETY_LIMIT_PSI) {
            emergencyCutoffLatched = true;
            digitalWrite(PIN_RELAY, HIGH); // De-energize relay to trip motor
            digitalWrite(PIN_BUZZER, HIGH); // Continuous alarm tone
            currentFrame.anomaly_flag = true;
        } else {
            if (!emergencyCutoffLatched) {
                currentFrame.anomaly_flag = false;
            }
        }
        currentFrame.relay_latched = emergencyCutoffLatched;

        // ---------------------------------------------------------
        // 3d. Stream Structured CSV Telemetry Packet
        // ---------------------------------------------------------
        String csvLine = currentFrame.toCSV();
        Serial.println(csvLine);
        Serial0.println(csvLine);
    }
}
