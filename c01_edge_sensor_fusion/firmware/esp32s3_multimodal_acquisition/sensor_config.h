#ifndef SENSOR_CONFIG_H
#define SENSOR_CONFIG_H

#include <Arduino.h>

// ====================================================================
// HARDWARE PIN DEFINITIONS (Rev 3.0 Left-Edge Row A Master Signal Bus)
// ====================================================================
#define PIN_DS18B20      4    // Hole 18A: DS18B20 1-Wire Digital Temperature
#define PIN_RELAY        5    // Hole 17A: 5V Optocoupler Fail-Safe Relay
#define PIN_BUZZER       6    // Hole 16A: Active Piezo Warning Alarm Buzzer
#define PIN_LED_GREEN    7    // Hole 15A: Heartbeat & Status Green LED
#define PIN_I2C_SDA      8    // Hole 10A: Shared I2C Data (SDA)
#define PIN_I2C_SCL      9    // Hole 7A:  Shared I2C Clock (SCL)

// ====================================================================
// I2C PERIPHERAL ADDRESSES
// ====================================================================
#define ADDR_MPU6050     0x68 // InvenSense 6-DOF Accelerometer / Gyroscope
#define ADDR_ADS1115     0x48 // Texas Instruments 16-Bit Precision ADC

// ====================================================================
// ADS1115 ANALOG CHANNEL MAPPING
// ====================================================================
#define ADC_CH_PRESSURE  0    // Channel A0: Piezoresistive G1/4 Pressure Transducer
#define ADC_CH_CURRENT   1    // Channel A1: ACS712-30A Hall-Effect Motor Current Sensor
#define ADC_CH_ACOUSTIC  2    // Channel A2: MAX4466 Electret Microphone with Gain

// ====================================================================
// DUAL-RATE ACQUISITION TIMING CONSTANTS (Non-Blocking)
// ====================================================================
#define FAST_SAMPLE_INTERVAL_MS   40   // 25 Hz fast acquisition loop (Vibration, Pressure, Acoustics)
#define SLOW_SAMPLE_INTERVAL_MS   1000 // 1 Hz slow thermodynamic loop (DS18B20 Temperature)
#define HEARTBEAT_INTERVAL_MS     500  // 500 ms LED toggle
#define ACOUSTIC_WINDOW_SAMPLES   32   // Fast burst samples for peak-to-peak sound energy

// ====================================================================
// TRANSDUCER CALIBRATION & PHYSICAL CONSTANTS
// ====================================================================
// G1/4 Pressure Transducer (0 - 0.5 MPa / 0 - 72.52 PSI)
// 3.3V ratiometric excitation: atmospheric baseline ~0.330V, span ~2.64V
#define PRESSURE_ZERO_OFFSET_V    0.330f
#define PRESSURE_SPAN_VOLTS       2.640f
#define PRESSURE_MAX_PSI          72.52f
#define PRESSURE_SAFETY_LIMIT_PSI 45.00f // Automated emergency trip threshold

// ACS712-30A Hall-Effect Current Sensor
// 3.3V ratiometric midpoint: Vcc/2 = 1.650V, sensitivity = 66 mV/A
#define CURRENT_ZERO_OFFSET_V     1.650f
#define CURRENT_SENSITIVITY_V_A   0.066f

// MPU-6050 Accelerometer
#define GRAVITY_EARTH             9.80665f

#endif // SENSOR_CONFIG_H
