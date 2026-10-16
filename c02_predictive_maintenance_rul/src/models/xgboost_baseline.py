"""
Baseline XGBoost Classifier and Regressor for Component 02
SLIIT Research Project J26-SE-326
Student: Herath H.M.N.P (IT23221178)

Establishes the non-deep learning empirical baseline for:
  1. Compressor Fault Classification (Normal vs Air Leak).
  2. Remaining Useful Life (RUL) regression forecasting.
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from c02_predictive_maintenance_rul.src.preprocessing.data_cleaner import DataCleaner, ANALOG_SENSOR_COLS
from c02_predictive_maintenance_rul.src.preprocessing.sequence_window_generator import SequenceWindowGenerator

logger = logging.getLogger("C02_XGBoostBaseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class XGBoostBaselineModel:
    """
    Implements, trains, and evaluates baseline XGBoost models on tabular time-series features.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Baseline classifier hyperparameters
        self.clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        )
        self.reg = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="rmse"
        )
        self.is_trained = False

    def train_classifier(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ):
        """Trains the fault classification model."""
        logger.info(f"Training XGBoost Classifier on {len(X_train)} samples ({X_train.shape[1]} features)...")
        self.clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self.is_trained = True
        logger.info("XGBoost Classifier training completed successfully.")

    def evaluate_classifier(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluates classification performance on unseen test data."""
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(np.mean(y_pred == y_test)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 1.0
        }

        logger.info("=== XGBoost Classification Performance ===")
        for k, v in metrics.items():
            logger.info(f"  {k.upper()}: {v:.4f}")

        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred, zero_division=0))
        return metrics

    def train_and_eval_rul_regressor(
        self, X_train: pd.DataFrame, y_train_rul: pd.Series,
        X_test: pd.DataFrame, y_test_rul: pd.Series
    ) -> Dict[str, float]:
        """Trains and evaluates baseline RUL regressor."""
        logger.info(f"Training XGBoost RUL Regressor on {len(X_train)} samples...")
        self.reg.fit(X_train, y_train_rul, verbose=False)

        rul_pred = self.reg.predict(X_test)
        mae = float(mean_absolute_error(y_test_rul, rul_pred))
        rmse = float(root_mean_squared_error(y_test_rul, rul_pred))

        rul_metrics = {"rul_mae_hours": mae, "rul_rmse_hours": rmse}
        logger.info("=== XGBoost RUL Regression Performance ===")
        logger.info(f"  MAE:  {mae:.2f} hours")
        logger.info(f"  RMSE: {rmse:.2f} hours")
        return rul_metrics

    def save_models(self, clf_path: str, reg_path: str):
        """Serializes trained models to standard JSON format."""
        os.makedirs(os.path.dirname(os.path.abspath(clf_path)), exist_ok=True)
        self.clf.save_model(clf_path)
        self.reg.save_model(reg_path)
        logger.info(f"Models saved: {clf_path} and {reg_path}")


def run_xgboost_training_pipeline():
    """End-to-end execution of the XGBoost baseline training and evaluation."""
    dataset_path = "c02_predictive_maintenance_rul/data/sample_metropt3_snippet.csv"
    metrics_path = "c02_predictive_maintenance_rul/evaluation/metrics_summary.csv"
    clf_model_path = "c02_predictive_maintenance_rul/src/models/xgboost_fault_model.json"
    reg_model_path = "c02_predictive_maintenance_rul/src/models/xgboost_rul_model.json"

    # 1. Ingestion & Cleaning
    cleaner = DataCleaner()
    df_raw = cleaner.load_and_sort(dataset_path)
    df_clean = cleaner.handle_missing_and_outliers(df_raw)

    # 2. Chronological Split (70% train, 15% val, 15% test)
    generator = SequenceWindowGenerator(window_size=60, step_size=1)
    train_df, val_df, test_df = generator.chronological_split(df_clean, train_ratio=0.70, val_ratio=0.15)

    # 3. Fit scaler on train only and transform
    cleaner.fit_scaler(train_df)
    train_scaled = cleaner.transform_features(train_df)
    val_scaled = cleaner.transform_features(val_df)
    test_scaled = cleaner.transform_features(test_df)

    # 4. Extract Tabular Features for XGBoost
    X_train, y_train = generator.extract_tabular_window_features(train_scaled, ANALOG_SENSOR_COLS)
    X_val, y_val = generator.extract_tabular_window_features(val_scaled, ANALOG_SENSOR_COLS)
    X_test, y_test = generator.extract_tabular_window_features(test_scaled, ANALOG_SENSOR_COLS)

    # 5. Formulate Synthetic Piecewise Linear RUL Target (in hours) for baseline demonstration
    # Normal operation = 100 hours constant; Degrading = linear drop to 0
    y_train_rul = pd.Series(np.where(y_train == 0, 100.0, np.linspace(100.0, 0.0, len(y_train))))
    y_test_rul = pd.Series(np.where(y_test == 0, 100.0, np.linspace(100.0, 0.0, len(y_test))))

    # 6. Train and Evaluate
    model = XGBoostBaselineModel()
    model.train_classifier(X_train, y_train, X_val, y_val)
    clf_metrics = model.evaluate_classifier(X_test, y_test)
    rul_metrics = model.train_and_eval_rul_regressor(X_train, y_train_rul, X_test, y_test_rul)

    # 7. Save Models
    model.save_models(clf_model_path, reg_model_path)

    # 8. Export Metrics to CSV
    combined_metrics = {
        "model_name": "XGBoost_Baseline",
        "timestamp": pd.Timestamp.now().isoformat(),
        **clf_metrics,
        **rul_metrics
    }
    metrics_df = pd.DataFrame([combined_metrics])
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    if os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(metrics_path, index=False)

    logger.info(f"Evaluation metrics appended to: {metrics_path}")


if __name__ == "__main__":
    run_xgboost_training_pipeline()
