"""
src/behavioral/state_behavioral_profiler.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
State-conditioned statistical envelope profiling and Mahalanobis distance estimation.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
try:
    import joblib
except ImportError:
    import pickle

    class _JoblibFallback:
        @staticmethod
        def dump(value, filename, **kwargs):
            with open(filename, "wb") as f:
                pickle.dump(value, f)

        @staticmethod
        def load(filename, **kwargs):
            with open(filename, "rb") as f:
                return pickle.load(f)

    joblib = _JoblibFallback()

import numpy as np
import pandas as pd

from ..clustering.state_discovery import (
    OperatingStateLabel,
    DEFAULT_FEATURE_COLUMNS,
)


@dataclass
class StateProfile:
    """
    Statistical distribution parameters for a single operational compressor regime.
    """
    state_label: str
    sample_count: int
    centroid: Dict[str, float]
    std_dev: Dict[str, float]
    variance: Dict[str, float]
    covariance_matrix: np.ndarray
    inv_covariance_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_label": self.state_label,
            "sample_count": self.sample_count,
            "centroid": {k: round(v, 4) for k, v in self.centroid.items()},
            "std_dev": {k: round(v, 4) for k, v in self.std_dev.items()},
            "variance": {k: round(v, 4) for k, v in self.variance.items()},
        }


@dataclass
class BehavioralAssessment:
    """
    Evaluated distance and deviation of a live observation relative to its active state envelope.
    """
    state_label: str
    deviation_index: float  # Mahalanobis distance D_M
    feature_z_scores: Dict[str, float]
    dominant_deviating_feature: str
    state_cluster_centroid: Dict[str, float]
    state_cluster_variance: Dict[str, float]


class StateBehavioralProfiler:
    """
    Learns state-conditioned multivariate baseline envelopes for each operational regime.
    Computes regularized Mahalanobis distance and feature-specific Z-scores to quantify
    behavioral deviation independent of cross-regime operational variance.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        regularization_eps: float = 1e-4,
    ):
        """
        Args:
            feature_columns: List of feature channels to profile (default: 6D physical vector).
            regularization_eps: Ridge shrinkage factor added to covariance diagonals for numerical stability.
        """
        self.feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
        self.regularization_eps = max(1e-6, regularization_eps)
        self.profiles: Dict[str, StateProfile] = {}
        self.is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        state_column: str = "state_label",
    ) -> "StateBehavioralProfiler":
        """
        Learns baseline means, variances, and covariance matrices conditioned on each state regime.

        Args:
            data: Historical baseline dataset containing features and state labels.
            state_column: Column name containing operational state strings.
        """
        if state_column not in data.columns:
            raise ValueError(f"State column '{state_column}' not found in training dataset.")

        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")

        self.profiles.clear()
        grouped = data.groupby(state_column)

        for state, group in grouped:
            state_str = str(state)
            n_samples = len(group)
            if n_samples < 2:
                continue

            X_mat = group[self.feature_columns].values.astype(float)
            mean_vec = np.mean(X_mat, axis=0)
            var_vec = np.var(X_mat, axis=0, ddof=1)
            std_vec = np.std(X_mat, axis=0, ddof=1)

            # Empirical covariance with Ledoit-Wolf-style ridge regularization
            cov_mat = np.cov(X_mat, rowvar=False)
            if cov_mat.ndim == 0:
                cov_mat = np.array([[cov_mat]])
            elif cov_mat.ndim == 1:
                cov_mat = np.diag(cov_mat)

            # Regularize covariance diagonal to guarantee invertibility
            reg_eye = np.eye(len(self.feature_columns)) * self.regularization_eps
            reg_cov = cov_mat + reg_eye

            try:
                inv_cov = np.linalg.inv(reg_cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(reg_cov)

            centroid_dict = {col: float(mean_vec[i]) for i, col in enumerate(self.feature_columns)}
            std_dict = {col: float(std_vec[i]) for i, col in enumerate(self.feature_columns)}
            var_dict = {col: float(var_vec[i]) for i, col in enumerate(self.feature_columns)}

            profile = StateProfile(
                state_label=state_str,
                sample_count=n_samples,
                centroid=centroid_dict,
                std_dev=std_dict,
                variance=var_dict,
                covariance_matrix=reg_cov,
                inv_covariance_matrix=inv_cov,
            )
            self.profiles[state_str] = profile

        if not self.profiles:
            raise ValueError("Failed to fit profiles: no valid state groups found.")

        self.is_fitted = True
        return self

    def assess_deviation(
        self,
        features: Union[Dict[str, Any], Any],
        state_label: str,
    ) -> BehavioralAssessment:
        """
        Computes Mahalanobis distance and feature Z-scores for an observation
        evaluated strictly against the learned envelope of its active operating state.

        Args:
            features: Dictionary or object with telemetry feature attributes.
            state_label: Currently active operational state string.
        """
        if not self.is_fitted:
            raise RuntimeError("StateBehavioralProfiler is not fitted. Call fit() first.")

        profile = self.profiles.get(state_label)
        # Fallback to nearest or first profile if state is novel
        if profile is None:
            profile = next(iter(self.profiles.values()))

        x_vec = np.zeros(len(self.feature_columns), dtype=float)
        z_scores: Dict[str, float] = {}

        for i, col in enumerate(self.feature_columns):
            if isinstance(features, dict):
                val = float(features.get(col, profile.centroid[col]))
            else:
                val = float(getattr(features, col, profile.centroid[col]))
            x_vec[i] = val

            mu = profile.centroid[col]
            sigma = max(profile.std_dev[col], 1e-6)
            z_scores[col] = round(abs(val - mu) / sigma, 3)

        # Mahalanobis Distance: D_M = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
        diff = x_vec - np.array([profile.centroid[col] for col in self.feature_columns])
        dist_sq = float(diff.T @ profile.inv_covariance_matrix @ diff)
        mahalanobis_dist = float(np.sqrt(max(0.0, dist_sq)))

        # Identify feature driving the largest deviation
        dominant_feature = max(z_scores.keys(), key=lambda k: z_scores[k])

        return BehavioralAssessment(
            state_label=state_label,
            deviation_index=round(mahalanobis_dist, 4),
            feature_z_scores=z_scores,
            dominant_deviating_feature=dominant_feature,
            state_cluster_centroid=profile.centroid,
            state_cluster_variance=profile.variance,
        )

    def get_profile(self, state_label: str) -> Optional[StateProfile]:
        """
        Retrieves profile record for requested state.
        """
        return self.profiles.get(state_label)

    def save_profiles(self, filepath: Union[str, Path]) -> None:
        """
        Persists learned profiles to disk.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "feature_columns": self.feature_columns,
            "regularization_eps": self.regularization_eps,
            "profiles": self.profiles,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load_profiles(cls, filepath: Union[str, Path]) -> "StateBehavioralProfiler":
        """
        Loads saved profiles from disk.
        """
        payload = joblib.load(filepath)
        profiler = cls(
            feature_columns=payload["feature_columns"],
            regularization_eps=payload["regularization_eps"],
        )
        profiler.profiles = payload["profiles"]
        profiler.is_fitted = payload["is_fitted"]
        return profiler
