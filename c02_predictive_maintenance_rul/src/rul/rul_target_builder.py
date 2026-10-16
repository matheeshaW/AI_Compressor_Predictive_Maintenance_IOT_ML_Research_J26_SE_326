"""
Remaining Useful Life (RUL) Target Builder for Component 02
SLIIT Research Project J26-SE-326
Student: Herath H.M.N.P (IT23221178)

Constructs mathematically rigorous piecewise-linear RUL degradation targets
and normalized Health Indices (HI) from compressor run-to-failure telemetry.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("C02_RULTargetBuilder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RULTargetBuilder:
    """
    Formulates piecewise-linear RUL targets for degradation forecasting.
    Conforms to established prognostic standards (Heimes et al., Saxena et al.).
    """

    def __init__(self, max_rul_hours: float = 100.0, sampling_rate_hz: float = 1.0):
        """
        Args:
            max_rul_hours: Plateau threshold for healthy baseline equipment life (hours).
            sampling_rate_hz: Sampling frequency of raw telemetry (1 Hz = 1 sample/sec).
        """
        self.max_rul_hours = max_rul_hours
        self.sampling_rate_hz = sampling_rate_hz

    def build_piecewise_linear_target(
        self, df: pd.DataFrame, failure_label_col: str = "failure_label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a piecewise-linear countdown RUL target in operating hours.

        Logic:
          - During normal operation (label 0): RUL = max_rul_hours (plateau).
          - During degradation / failure (label 1): RUL counts down linearly to 0.

        Returns:
            rul_hours: Continuous RUL target in hours.
            health_index: Normalized health score between 1.0 (pristine) and 0.0 (failed).
        """
        n = len(df)
        labels = df[failure_label_col].values if failure_label_col in df.columns else np.zeros(n)
        rul_hours = np.zeros(n, dtype=np.float32)

        # Identify the transition point where degradation starts
        degradation_indices = np.where(labels == 1)[0]
        if len(degradation_indices) == 0:
            logger.warning("No failure events detected; assigning constant maximum RUL.")
            rul_hours[:] = self.max_rul_hours
        else:
            first_fail_idx = degradation_indices[0]
            last_fail_idx = degradation_indices[-1]
            fail_span_seconds = max(1, last_fail_idx - first_fail_idx)

            # Healthy period: constant maximum RUL
            rul_hours[:first_fail_idx] = self.max_rul_hours

            # Degradation period: linear decline from max_rul down to 0
            decay = np.linspace(self.max_rul_hours, 0.0, n - first_fail_idx)
            rul_hours[first_fail_idx:] = decay

        # Compute normalized Health Index (HI)
        health_index = np.clip(rul_hours / self.max_rul_hours, 0.0, 1.0)

        logger.info(
            f"Built RUL targets: Max={rul_hours.max():.1f}h, Min={rul_hours.min():.1f}h | "
            f"Health Index range: [{health_index.min():.2f}, {health_index.max():.2f}]"
        )
        return rul_hours, health_index
