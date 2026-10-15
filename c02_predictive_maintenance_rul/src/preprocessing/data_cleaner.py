"""
Data Cleaning and Scaling Engine for Component 02
SLIIT Research Project J26-SE-326
Student: Herath H.M.N.P (IT23221178)

Ensures leakage-free normalization, missing value imputation,
and timestamp chronological alignment for compressor telemetry.
"""

import os
import json
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("C02_DataCleaner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Continuous analog sensor channels to normalize
ANALOG_SENSOR_COLS = [
    "TP2",               # Discharge pressure (bar)
    "TP3",               # Pneumatic reservoir pressure (bar)
    "H1",                # Oil separator pressure drop (bar)
    "DV_pressure",       # Drain valve pressure (bar)
    "Motor_current",     # Motor electrical current (A)
    "Oil_temperature"    # Internal lubrication temp (°C)
]

# Discrete digital control state channels (kept unscaled 0/1)
DIGITAL_STATE_COLS = [
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"
]


class DataCleaner:
    """
    Cleans, validates, and standardizes multivariate compressor telemetry.
    Guarantees no temporal data leakage by fitting parameters exclusively on training periods.
    """

    def __init__(self, analog_cols: Optional[List[str]] = None):
        self.analog_cols = analog_cols or ANALOG_SENSOR_COLS
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.scaler_params = {}

    def load_and_sort(self, filepath: str) -> pd.DataFrame:
        """Loads CSV and guarantees chronological sorting without duplicates."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")

        df = pd.read_csv(filepath)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by="timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        logger.info(f"Loaded {len(df):,} records chronologically sorted from {filepath}")
        return df

    def handle_missing_and_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values with forward-fill and clips extreme non-physical spikes."""
        df_clean = df.copy()

        # Forward fill then backward fill for continuous temporal continuity
        df_clean = df_clean.ffill().bfill()

        # Physical safety bounds for industrial air compressors
        bounds = {
            "TP2": (0.0, 16.0),            # Bar (atmospheric to max pressure head)
            "TP3": (0.0, 16.0),
            "Motor_current": (0.0, 35.0),   # Amperes (stall current threshold)
            "Oil_temperature": (-10.0, 130.0) # °C (lubricant flashpoint range)
        }

        for col, (lower, upper) in bounds.items():
            if col in df_clean.columns:
                outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
                if outliers > 0:
                    logger.warning(f"Clipping {outliers} outlier values outside physical bounds in {col}")
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

        return df_clean

    def fit_scaler(self, train_df: pd.DataFrame):
        """Fits StandardScaler ONLY on training data to prevent temporal leakage."""
        cols_to_scale = [c for c in self.analog_cols if c in train_df.columns]
        self.scaler.fit(train_df[cols_to_scale])
        self.is_fitted = True

        self.scaler_params = {
            col: {
                "mean": float(self.scaler.mean_[i]),
                "std": float(self.scaler.scale_[i])
            }
            for i, col in enumerate(cols_to_scale)
        }
        logger.info("StandardScaler fitted successfully on training split only.")

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms analog features using the fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before transforming.")

        df_scaled = df.copy()
        cols_to_scale = [c for c in self.analog_cols if c in df.columns]
        df_scaled[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df_scaled

    def save_scaler_params(self, output_json_path: str):
        """Exports scaler parameters to JSON for production dashboard & contract inference."""
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(self.scaler_params, f, indent=2)
        logger.info(f"Scaler parameters saved to: {output_json_path}")
