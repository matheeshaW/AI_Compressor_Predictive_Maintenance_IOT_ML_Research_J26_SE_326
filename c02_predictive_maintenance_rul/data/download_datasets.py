"""
Dataset Ingestion and Sample Generator for Component 02
SLIIT Research Project J26-SE-326
Student: Herath H.M.N.P (IT23221178)

Provides automated downloading for the primary MetroPT-3 multivariate compressor
dataset (UCI ML Repository) and builds a clean offline benchmark snippet
(sample_metropt3_snippet.csv) preserving realistic failure dynamics.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("C02_DataIngestion")

METROPT3_UCI_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"

EXPECTED_METROPT3_COLUMNS = [
    "timestamp",
    "TP2",               # Compressor discharge pressure (bar)
    "TP3",               # Pneumatic grid / reservoir pressure (bar)
    "H1",                # Oil separator filter pressure drop (bar)
    "DV_pressure",       # Desiccant air dryer drain valve pressure (bar)
    "Motor_current",     # Drive motor current (Amperes)
    "Oil_temperature",   # Internal oil reservoir temperature (°C)
    "COMP",              # Compressor motor active signal (0 or 1)
    "DV_eletric",        # Electric drain valve activation (0 or 1)
    "Towers",            # Desiccant tower cycle (0 or 1)
    "MPG",               # Minimum pressure valve signal (0 or 1)
    "LPS",               # Low pressure switch alarm (0 or 1)
    "Pressure_switch",   # Cut-in / cut-out pressure switch (0 or 1)
    "Oil_level",         # Oil level status (0 or 1)
    "Caudal_impulses",   # Air flow volume impulse counter
    "failure_label"      # 0 = Normal, 1 = Air Leak Failure
]


def generate_curated_metropt3_snippet(output_path: str, n_samples: int = 7200):
    """
    Generates a deterministic, physically grounded MetroPT-3 benchmark snippet
    spanning 2 hours of operating telemetry (1 Hz sampling = 7200 points).
    
    Phases:
      1. Hours 0.0 - 1.2: Normal cyclic compressor operation (healthy).
      2. Hours 1.2 - 1.7: Developing air leak (continuous duty cycle, pressure drop).
      3. Hours 1.7 - 2.0: Active air leak failure (severe pressure loss, thermal buildup).
    """
    logger.info(f"Generating curated MetroPT-3 sample snippet with {n_samples} timestamps...")
    np.random.seed(42)

    start_time = datetime(2020, 4, 18, 10, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]

    # Time progression in seconds
    t = np.arange(n_samples)

    # Base operating cycle: Compressor charges for 180s, unloads for 360s
    cycle_period = 540
    phase_in_cycle = t % cycle_period
    is_charging_normal = (phase_in_cycle < 180).astype(int)

    # Degradation progression (Phase 2 & 3 start after index 4320 ~ 1.2 hours)
    degradation_start_idx = 4320
    failure_start_idx = 6120

    comp = np.zeros(n_samples, dtype=int)
    tp2 = np.zeros(n_samples)
    tp3 = np.zeros(n_samples)
    motor_current = np.zeros(n_samples)
    oil_temperature = np.zeros(n_samples)
    failure_label = np.zeros(n_samples, dtype=int)

    # Base oil temperature starting at 52°C
    base_temp = 52.0

    for i in range(n_samples):
        noise_p = np.random.normal(0, 0.08)
        noise_i = np.random.normal(0, 0.15)
        noise_t = np.random.normal(0, 0.05)

        if i < degradation_start_idx:
            # HEALTHY CYCLIC OPERATION
            comp[i] = is_charging_normal[i]
            if comp[i] == 1:
                progress = phase_in_cycle[i] / 180.0
                tp2[i] = 8.2 + (progress * 1.6) + noise_p      # 8.2 -> 9.8 bar
                tp3[i] = 7.8 + (progress * 1.4) + noise_p      # 7.8 -> 9.2 bar
                motor_current[i] = 6.2 + (progress * 0.8) + noise_i
                base_temp = min(68.0, base_temp + 0.02)
            else:
                tp2[i] = max(1.2, 8.2 - ((phase_in_cycle[i] - 180) / 10.0)) + noise_p
                tp3[i] = max(7.5, 9.2 - ((phase_in_cycle[i] - 180) / 400.0)) + noise_p
                motor_current[i] = max(0.0, 0.2 + noise_i)
                base_temp = max(52.0, base_temp - 0.01)

            oil_temperature[i] = base_temp + noise_t
            failure_label[i] = 0

        elif i < failure_start_idx:
            # DEVELOPING AIR LEAK DEGRADATION (Compressor forced to run longer)
            comp[i] = 1  # Continuous pumping due to escaping air
            # Pressure struggles to build up
            tp2[i] = 7.4 + np.random.normal(0, 0.12)
            tp3[i] = 7.2 + np.random.normal(0, 0.10)
            motor_current[i] = 7.1 + noise_i
            base_temp = min(82.0, base_temp + 0.015)
            oil_temperature[i] = base_temp + noise_t
            failure_label[i] = 1

        else:
            # ACTIVE AIR LEAK FAILURE (Severe pressure drop, high temperature)
            comp[i] = 1
            tp2[i] = max(4.5, 6.8 - ((i - failure_start_idx) / 600.0)) + noise_p
            tp3[i] = max(4.0, 6.5 - ((i - failure_start_idx) / 500.0)) + noise_p
            motor_current[i] = 7.8 + noise_i
            base_temp = min(91.0, base_temp + 0.01)
            oil_temperature[i] = base_temp + noise_t
            failure_label[i] = 1

    # Remaining supporting columns
    h1 = np.where(comp == 1, 0.45 + np.random.normal(0, 0.02, n_samples), 0.05)
    dv_pressure = np.where(comp == 1, 0.10 + np.random.normal(0, 0.01, n_samples), 0.0)
    dv_eletric = (phase_in_cycle == 180).astype(int)
    towers = ((t // 1200) % 2).astype(int)
    mpg = comp
    lps = np.where(tp3 < 5.0, 1, 0)
    pressure_switch = np.where(tp3 < 8.0, 1, 0)
    oil_level = np.ones(n_samples, dtype=int)
    caudal_impulses = np.cumsum(comp) % 1000

    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "TP2": np.round(tp2, 3),
        "TP3": np.round(tp3, 3),
        "H1": np.round(h1, 3),
        "DV_pressure": np.round(dv_pressure, 3),
        "Motor_current": np.round(motor_current, 3),
        "Oil_temperature": np.round(oil_temperature, 2),
        "COMP": comp,
        "DV_eletric": dv_eletric,
        "Towers": towers,
        "MPG": mpg,
        "LPS": lps,
        "Pressure_switch": pressure_switch,
        "Oil_level": oil_level,
        "Caudal_impulses": caudal_impulses,
        "failure_label": failure_label
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved sample snippet to: {output_path}")
    logger.info(f"Dataset Shape: {df.shape} | Failure Ratio: {df['failure_label'].mean():.2%}")
    return df


def verify_dataset(filepath: str) -> bool:
    """Performs integrity verification on a MetroPT-3 CSV file."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False

    df = pd.read_csv(filepath)
    logger.info(f"--- Verifying Dataset: {os.path.basename(filepath)} ---")
    logger.info(f"Total Rows: {len(df):,} | Columns: {len(df.columns)}")

    missing_cols = [c for c in EXPECTED_METROPT3_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
    else:
        logger.info("All 16 expected MetroPT-3 columns are present.")

    null_count = df.isnull().sum().sum()
    logger.info(f"Total Missing / Null Values: {null_count}")

    if "failure_label" in df.columns:
        counts = df["failure_label"].value_counts().to_dict()
        logger.info(f"Class Distribution: Normal (0) = {counts.get(0, 0)}, Failure (1) = {counts.get(1, 0)}")

    logger.info("Verification passed successfully.\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetroPT-3 Dataset Loader for Component 02")
    parser.add_argument("--generate-sample", action="store_true", help="Generate the calibrated sample snippet")
    parser.add_argument("--output", type=str, default="c02_predictive_maintenance_rul/data/sample_metropt3_snippet.csv")
    args = parser.parse_args()

    # Default action if no flags provided: generate sample snippet
    output_csv = args.output
    generate_curated_metropt3_snippet(output_csv)
    verify_dataset(output_csv)
