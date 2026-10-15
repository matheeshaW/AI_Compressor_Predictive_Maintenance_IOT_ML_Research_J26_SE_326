"""
Sequence Window and Feature Generator for Component 02
SLIIT Research Project J26-SE-326
Student: Herath H.M.N.P (IT23221178)

Generates:
  1. 2D Tabular Rolling-Window Features for XGBoost Baseline.
  2. 3D Sequential Window Tensors for LSTM Temporal Network.
Guarantees strictly chronological Train/Validation/Test splits without temporal overlap.
"""

import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("C02_SequenceGenerator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SequenceWindowGenerator:
    """
    Constructs leakage-free tabular and sequential modeling representations
    from multivariate compressor telemetry.
    """

    def __init__(self, window_size: int = 60, step_size: int = 1):
        """
        Args:
            window_size: Number of temporal steps per window (60s at 1 Hz).
            step_size: Stride between successive sliding windows.
        """
        self.window_size = window_size
        self.step_size = step_size

    def chronological_split(
        self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically: [Train (70%) | Validation (15%) | Test (15%)].
        Never uses random shuffling on temporal data!
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        # Leave a safety buffer of window_size to prevent edge window overlap
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        logger.info(
            f"Chronological Split complete -> Train: {len(train_df):,} | "
            f"Val: {len(val_df):,} | Test: {len(test_df):,}"
        )
        return train_df, val_df, test_df

    def create_lstm_sequences(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "failure_label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D tensors for LSTM:
            X shape: (num_samples, window_size, num_features)
            y shape: (num_samples,) [Label at the end of each window]
        """
        features = df[feature_cols].values
        targets = df[target_col].values if target_col in df.columns else np.zeros(len(df))

        X_list = []
        y_list = []

        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            X_list.append(features[i : i + self.window_size, :])
            # Target is the label at the latest time step of the window
            y_list.append(targets[i + self.window_size - 1])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(f"Generated LSTM sequences: X={X.shape}, y={y.shape}")
        return X, y

    def extract_tabular_window_features(
        self, df: pd.DataFrame, analog_cols: List[str], target_col: str = "failure_label"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts 2D rolling statistical features for XGBoost:
          - Mean, Standard Deviation, Min, Max
          - Delta Pressure Rate (dP/dt)
          - Compressor Duty Cycle (% of window COMP == 1)
        """
        records = []
        targets = []

        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            window = df.iloc[i : i + self.window_size]
            row_feat = {}

            # Statistical aggregations across analog channels
            for col in analog_cols:
                if col in window.columns:
                    vals = window[col].values
                    row_feat[f"{col}_mean"] = float(np.mean(vals))
                    row_feat[f"{col}_std"] = float(np.std(vals))
                    row_feat[f"{col}_min"] = float(np.min(vals))
                    row_feat[f"{col}_max"] = float(np.max(vals))

            # Rate of change: delta P over window
            if "TP2" in window.columns:
                row_feat["TP2_delta"] = float(window["TP2"].iloc[-1] - window["TP2"].iloc[0])
            if "TP3" in window.columns:
                row_feat["TP3_delta"] = float(window["TP3"].iloc[-1] - window["TP3"].iloc[0])

            # Operational duty cycle (% time compressor was running)
            if "COMP" in window.columns:
                row_feat["COMP_duty_cycle"] = float(np.mean(window["COMP"].values))

            records.append(row_feat)
            if target_col in window.columns:
                targets.append(window[target_col].iloc[-1])
            else:
                targets.append(0)

        X_tab = pd.DataFrame(records)
        y_tab = pd.Series(targets, name=target_col)

        logger.info(f"Generated XGBoost tabular features: X={X_tab.shape}, y={y_tab.shape}")
        return X_tab, y_tab
