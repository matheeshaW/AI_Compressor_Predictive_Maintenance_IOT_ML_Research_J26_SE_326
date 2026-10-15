"""
PyTorch Multi-Task LSTM Temporal Network for Component 02
SLIIT Research Project J26-SE-326
Student: Herath H.M.N.P (IT23221178)

Ingests 3D sequential sliding windows (batch, 60s, 7 features) and predicts:
  1. Fault Probability (Classification Head: Normal vs Air Leak).
  2. Remaining Useful Life in Hours (Regression Head: Multi-task prognosis).
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error
)

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from c02_predictive_maintenance_rul.src.preprocessing.data_cleaner import DataCleaner, ANALOG_SENSOR_COLS
from c02_predictive_maintenance_rul.src.preprocessing.sequence_window_generator import SequenceWindowGenerator
from c02_predictive_maintenance_rul.src.rul.rul_target_builder import RULTargetBuilder

logger = logging.getLogger("C02_LSTMModel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MultiTaskCompressorLSTM(nn.Module):
    """
    2-Layer Recurrent LSTM with dual task-specific heads.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(MultiTaskCompressorLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        # Classification Head (Binary Fault Probability: Normal vs Air Leak)
        self.fc_clf = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Regression Head (Remaining Useful Life in Hours)
        self.fc_reg = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, sequence_length=60, input_dim=7]
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the latent state of the final time step
        last_hidden = self.dropout(lstm_out[:, -1, :])

        fault_prob = self.fc_clf(last_hidden).squeeze(-1)
        rul_pred = self.fc_reg(last_hidden).squeeze(-1)
        return fault_prob, rul_pred


class LSTMTemporalTrainer:
    """Handles data preparation, training loops, evaluation, and checkpointing."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MultiTaskCompressorLSTM(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clf_criterion = nn.BCELoss()
        self.reg_criterion = nn.MSELoss()
        logger.info(f"Initialized Multi-Task LSTM on device: {self.device}")

    def train_epochs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15
    ):
        """Executes multi-task training loop across epochs."""
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y_clf, batch_y_rul in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y_clf = batch_y_clf.to(self.device)
                batch_y_rul = batch_y_rul.to(self.device)

                self.optimizer.zero_grad()
                pred_clf, pred_rul = self.model(batch_x)

                # Combined Multi-Task Loss: Classification BCE + Scaled Regression MSE
                loss_clf = self.clf_criterion(pred_clf, batch_y_clf)
                loss_reg = self.reg_criterion(pred_rul, batch_y_rul)
                loss = loss_clf + (0.005 * loss_reg)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == epochs:
                logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - Multi-Task Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluates both heads on unseen chronological test split."""
        self.model.eval()
        all_true_clf, all_pred_clf, all_prob_clf = [], [], []
        all_true_rul, all_pred_rul = [], []

        with torch.no_grad():
            for batch_x, batch_y_clf, batch_y_rul in test_loader:
                batch_x = batch_x.to(self.device)
                prob_clf, pred_rul = self.model(batch_x)

                all_prob_clf.extend(prob_clf.cpu().numpy())
                all_pred_clf.extend((prob_clf.cpu().numpy() >= 0.5).astype(int))
                all_true_clf.extend(batch_y_clf.numpy())

                all_pred_rul.extend(pred_rul.cpu().numpy())
                all_true_rul.extend(batch_y_rul.numpy())

        all_true_clf = np.array(all_true_clf)
        all_pred_clf = np.array(all_pred_clf)
        all_prob_clf = np.array(all_prob_clf)
        all_true_rul = np.array(all_true_rul)
        all_pred_rul = np.array(all_pred_rul)

        mae = float(mean_absolute_error(all_true_rul, all_pred_rul))
        rmse = float(root_mean_squared_error(all_true_rul, all_pred_rul))

        metrics = {
            "accuracy": float(np.mean(all_pred_clf == all_true_clf)),
            "precision": float(precision_score(all_true_clf, all_pred_clf, zero_division=0)),
            "recall": float(recall_score(all_true_clf, all_pred_clf, zero_division=0)),
            "f1_score": float(f1_score(all_true_clf, all_pred_clf, zero_division=0)),
            "roc_auc": float(roc_auc_score(all_true_clf, all_prob_clf)) if len(np.unique(all_true_clf)) > 1 else 1.0,
            "rul_mae_hours": mae,
            "rul_rmse_hours": rmse
        }

        logger.info("\n=== PyTorch Multi-Task LSTM Test Performance ===")
        for k, v in metrics.items():
            logger.info(f"  {k.upper()}: {v:.4f}")
        return metrics

    def save_checkpoint(self, output_path: str):
        """Saves model state dict."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"Trained LSTM checkpoint saved to: {output_path}")


def run_lstm_training_pipeline():
    """End-to-end execution of Day 4 LSTM temporal baseline."""
    dataset_path = "c02_predictive_maintenance_rul/data/sample_metropt3_snippet.csv"
    metrics_path = "c02_predictive_maintenance_rul/evaluation/metrics_summary.csv"
    model_checkpoint_path = "c02_predictive_maintenance_rul/src/models/lstm_model.pt"

    # 1. Clean & Sort
    cleaner = DataCleaner()
    df_raw = cleaner.load_and_sort(dataset_path)
    df_clean = cleaner.handle_missing_and_outliers(df_raw)

    # 2. Add Piecewise Linear RUL target column to the dataset
    rul_builder = RULTargetBuilder(max_rul_hours=100.0)
    df_clean["rul_hours"], df_clean["health_index"] = rul_builder.build_piecewise_linear_target(df_clean)

    # 3. Chronological Split (70/15/15)
    generator = SequenceWindowGenerator(window_size=60, step_size=1)
    train_df, val_df, test_df = generator.chronological_split(df_clean, train_ratio=0.70, val_ratio=0.15)

    # 4. Scale features on train split only
    cleaner.fit_scaler(train_df)
    train_scaled = cleaner.transform_features(train_df)
    val_scaled = cleaner.transform_features(val_df)
    test_scaled = cleaner.transform_features(test_df)

    # 5. Build 3D LSTM Tensors
    feature_cols = ANALOG_SENSOR_COLS + ["COMP"]
    X_train, y_train_clf = generator.create_lstm_sequences(train_scaled, feature_cols, target_col="failure_label")
    _, y_train_rul = generator.create_lstm_sequences(train_scaled, feature_cols, target_col="rul_hours")

    X_val, y_val_clf = generator.create_lstm_sequences(val_scaled, feature_cols, target_col="failure_label")
    _, y_val_rul = generator.create_lstm_sequences(val_scaled, feature_cols, target_col="rul_hours")

    X_test, y_test_clf = generator.create_lstm_sequences(test_scaled, feature_cols, target_col="failure_label")
    _, y_test_rul = generator.create_lstm_sequences(test_scaled, feature_cols, target_col="rul_hours")

    # PyTorch DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_clf), torch.from_numpy(y_train_rul)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_clf), torch.from_numpy(y_val_rul)),
        batch_size=64, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test_clf), torch.from_numpy(y_test_rul)),
        batch_size=64, shuffle=False
    )

    # 6. Train Multi-Task LSTM
    trainer = LSTMTemporalTrainer(input_dim=len(feature_cols), hidden_dim=64, lr=0.001)
    trainer.train_epochs(train_loader, val_loader, epochs=15)

    # 7. Evaluate on Test Split
    lstm_metrics = trainer.evaluate(test_loader)
    trainer.save_checkpoint(model_checkpoint_path)

    # 8. Append to metrics_summary.csv for scientific comparison against XGBoost
    metrics_row = {
        "model_name": "LSTM_Temporal_MultiTask",
        "timestamp": pd.Timestamp.now().isoformat(),
        **lstm_metrics
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(metrics_path, mode="a", header=False, index=False)
    logger.info(f"Evaluation metrics appended to: {metrics_path}")


if __name__ == "__main__":
    run_lstm_training_pipeline()
