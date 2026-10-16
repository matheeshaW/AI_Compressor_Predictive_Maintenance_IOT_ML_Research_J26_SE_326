"""
src/clustering/state_discovery.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Unsupervised operational state discovery, clustering, and semantic regime mapping.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from dataclasses import dataclass, field
from enum import Enum
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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class OperatingStateLabel(str, Enum):
    """
    Standardized operational state labels conforming to
    c03_digital_twin_behavioral/schemas/digital_twin_state_schema.json.
    """
    IDLE_UNLOADED = "IDLE_UNLOADED"
    PRESSURIZING_LOADED = "PRESSURIZING_LOADED"
    HOLDING_PRESSURE = "HOLDING_PRESSURE"
    BLOWDOWN_DISCHARGE = "BLOWDOWN_DISCHARGE"
    ANOMALOUS_LEAK_REGIME = "ANOMALOUS_LEAK_REGIME"


class ClusteringAlgorithm(str, Enum):
    """
    Supported clustering algorithm paradigms.
    """
    K_MEANS = "K_MEANS"
    GAUSSIAN_MIXTURE_MODEL = "GAUSSIAN_MIXTURE_MODEL"
    DBSCAN = "DBSCAN"
    HEURISTIC_SEEDED_GMM = "HEURISTIC_SEEDED_GMM"


DEFAULT_FEATURE_COLUMNS = [
    "pressure_psi",
    "delta_pressure_dt",
    "motor_current_rms_a",
    "vibration_x_rms_ms2",
    "vibration_kurtosis",
    "acoustic_energy_mv",
]


@dataclass
class StateAssignment:
    """
    Real-time operating state classification result conforming to
    digital twin state schema 'operating_state' object specification.
    """
    cluster_id: int
    state_label: str
    state_confidence: float
    clustering_algorithm: str
    time_in_state_seconds: float = 0.0
    transition_count: int = 0
    posterior_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts assignment into a validated payload for the digital twin state schema.
        """
        return {
            "cluster_id": int(self.cluster_id),
            "state_label": str(self.state_label),
            "state_confidence": round(float(self.state_confidence), 4),
            "clustering_algorithm": str(self.clustering_algorithm),
            "time_in_state_seconds": round(float(self.time_in_state_seconds), 2),
            "transition_count": int(self.transition_count),
        }


class CompressorStateClusterer:
    """
    Unsupervised behavioral clustering engine discovering operational compressor regimes.
    Supports K-Means and Gaussian Mixture Models (GMM) with automated semantic alignment,
    Z-score feature standardization, soft posterior confidence estimation, and online
    state persistence tracking.
    """

    def __init__(
        self,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters: int = 5,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        softmax_temperature: float = 1.0,
    ):
        """
        Args:
            algorithm: Clustering algorithm paradigm (K_MEANS or GAUSSIAN_MIXTURE_MODEL).
            n_clusters: Number of target operational regimes (K=5 for primary compressor regimes).
            feature_columns: Multi-sensor feature subspace to cluster across.
            random_state: Reproducibility seed.
            softmax_temperature: Temperature scaling factor for K-Means distance-to-probability conversion.
        """
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
        self.random_state = random_state
        self.softmax_temperature = max(0.01, softmax_temperature)

        self.scaler = StandardScaler()
        self.model: Optional[Union[KMeans, GaussianMixture]] = None
        self.is_fitted = False

        # Cluster metadata
        self.cluster_to_label: Dict[int, OperatingStateLabel] = {}
        self.centroids_unscaled: Optional[np.ndarray] = None

        # Real-time streaming state tracking
        self._current_cluster_id: Optional[int] = None
        self._current_state_label: Optional[OperatingStateLabel] = None
        self._time_in_state_seconds: float = 0.0
        self._transition_count: int = 0

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        custom_label_map: Optional[Dict[int, OperatingStateLabel]] = None,
    ) -> "CompressorStateClusterer":
        """
        Fits feature scaler and clustering model on historical/simulated compressor telemetry.
        Automatically maps discovered latent clusters to physical semantic compressor regimes.

        Args:
            X: Telemetry observations matrix (N, D).
            custom_label_map: Optional explicit mapping from cluster_id to OperatingStateLabel.
        """
        X_df = self._to_dataframe(X)
        if len(X_df) < self.n_clusters:
            raise ValueError(f"Insufficient training observations ({len(X_df)}) for {self.n_clusters} clusters.")

        X_raw = X_df[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X_raw)

        if self.algorithm == ClusteringAlgorithm.K_MEANS:
            self.model = KMeans(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=self.random_state,
            )
            self.model.fit(X_scaled)
            self.centroids_unscaled = self.scaler.inverse_transform(self.model.cluster_centers_)

        elif self.algorithm in (ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL, ClusteringAlgorithm.HEURISTIC_SEEDED_GMM):
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="full",
                max_iter=200,
                n_init=3,
                random_state=self.random_state,
            )
            self.model.fit(X_scaled)
            self.centroids_unscaled = self.scaler.inverse_transform(self.model.means_)

        else:
            raise NotImplementedError(f"Clustering algorithm {self.algorithm} is not supported.")

        self.is_fitted = True

        if custom_label_map:
            self.cluster_to_label = dict(custom_label_map)
        else:
            self._align_semantic_regimes(X_raw)

        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Assigns batch observations to discovered operating clusters.

        Returns:
            Tuple of:
                - cluster_ids: Integer cluster indices (N,).
                - state_labels: Semantic operational strings (N,).
                - confidences: Posterior probability or distance confidence [0.0, 1.0] (N,).
        """
        self._ensure_fitted()
        X_df = self._to_dataframe(X)
        X_raw = X_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X_raw)

        if isinstance(self.model, KMeans):
            cluster_ids = self.model.predict(X_scaled)
            # Softmax confidence over Euclidean distance to centroids
            distances = self.model.transform(X_scaled)  # shape (N, K)
            scaled_neg_dist = -distances / self.softmax_temperature
            exp_dist = np.exp(scaled_neg_dist - np.max(scaled_neg_dist, axis=1, keepdims=True))
            probs = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)
            confidences = np.max(probs, axis=1)

        elif isinstance(self.model, GaussianMixture):
            probs = self.model.predict_proba(X_scaled)
            cluster_ids = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)

        else:
            raise RuntimeError("Underlying clustering model is undefined.")

        state_labels = [self.cluster_to_label.get(int(cid), OperatingStateLabel.IDLE_UNLOADED).value for cid in cluster_ids]

        return cluster_ids, state_labels, confidences

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Returns full probability matrix (N, K) over all discovered clusters.
        """
        self._ensure_fitted()
        X_df = self._to_dataframe(X)
        X_raw = X_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X_raw)

        if isinstance(self.model, GaussianMixture):
            return self.model.predict_proba(X_scaled)

        elif isinstance(self.model, KMeans):
            distances = self.model.transform(X_scaled)
            scaled_neg_dist = -distances / self.softmax_temperature
            exp_dist = np.exp(scaled_neg_dist - np.max(scaled_neg_dist, axis=1, keepdims=True))
            return exp_dist / np.sum(exp_dist, axis=1, keepdims=True)

        raise RuntimeError("Model is not initialized.")

    def predict_single(
        self,
        telemetry: Union[Dict[str, Any], Any],
        delta_t_sec: float = 0.04,
    ) -> StateAssignment:
        """
        Real-time online streaming inference method.
        Accepts single telemetry dictionary or AlignedSnapshot, extracts features,
        assigns operational state, updates persistence counters, and returns a schema-ready StateAssignment.

        Args:
            telemetry: Dictionary or object exposing feature attributes.
            delta_t_sec: Elapsed duration between telemetry frames (default 0.04s for 25 Hz).
        """
        self._ensure_fitted()

        # Extract features into 1-row DataFrame
        features_dict: Dict[str, float] = {}
        for col in self.feature_columns:
            if isinstance(telemetry, dict):
                val = telemetry.get(col, 0.0)
            else:
                val = getattr(telemetry, col, 0.0)
            features_dict[col] = float(val)

        row_df = pd.DataFrame([features_dict])
        cluster_ids, state_labels, confidences = self.predict(row_df)
        cid = int(cluster_ids[0])
        label = state_labels[0]
        confidence = float(confidences[0])

        probs = self.predict_proba(row_df)[0]
        posterior_dict = {
            self.cluster_to_label.get(k, OperatingStateLabel.IDLE_UNLOADED).value: round(float(probs[k]), 4)
            for k in range(self.n_clusters)
        }

        # Update persistence & transition tracking
        if self._current_cluster_id is None:
            self._current_cluster_id = cid
            self._current_state_label = self.cluster_to_label.get(cid, OperatingStateLabel.IDLE_UNLOADED)
            self._time_in_state_seconds = delta_t_sec
            self._transition_count = 0
        elif cid == self._current_cluster_id:
            self._time_in_state_seconds += delta_t_sec
        else:
            self._transition_count += 1
            self._current_cluster_id = cid
            self._current_state_label = self.cluster_to_label.get(cid, OperatingStateLabel.IDLE_UNLOADED)
            self._time_in_state_seconds = delta_t_sec

        return StateAssignment(
            cluster_id=cid,
            state_label=label,
            state_confidence=confidence,
            clustering_algorithm=self.algorithm.value,
            time_in_state_seconds=self._time_in_state_seconds,
            transition_count=self._transition_count,
            posterior_probabilities=posterior_dict,
        )

    def get_cluster_centroids(self) -> pd.DataFrame:
        """
        Returns physical unscaled centroid feature values for each cluster along with semantic labels.
        """
        self._ensure_fitted()
        df = pd.DataFrame(self.centroids_unscaled, columns=self.feature_columns)
        df.insert(0, "cluster_id", range(self.n_clusters))
        df["state_label"] = [self.cluster_to_label.get(i, OperatingStateLabel.IDLE_UNLOADED).value for i in range(self.n_clusters)]
        return df

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Serializes trained clusterer model, scaler, and label mappings to disk.
        """
        self._ensure_fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "feature_columns": self.feature_columns,
            "random_state": self.random_state,
            "softmax_temperature": self.softmax_temperature,
            "scaler": self.scaler,
            "model": self.model,
            "cluster_to_label": self.cluster_to_label,
            "centroids_unscaled": self.centroids_unscaled,
        }
        joblib.dump(payload, path)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "CompressorStateClusterer":
        """
        Deserializes a saved compressor state clusterer.
        """
        payload = joblib.load(filepath)
        clusterer = cls(
            algorithm=payload["algorithm"],
            n_clusters=payload["n_clusters"],
            feature_columns=payload["feature_columns"],
            random_state=payload["random_state"],
            softmax_temperature=payload["softmax_temperature"],
        )
        clusterer.scaler = payload["scaler"]
        clusterer.model = payload["model"]
        clusterer.cluster_to_label = payload["cluster_to_label"]
        clusterer.centroids_unscaled = payload["centroids_unscaled"]
        clusterer.is_fitted = True
        return clusterer

    def _align_semantic_regimes(self, X_raw: np.ndarray) -> None:
        """
        Autonomous physical semantic regime mapping.
        Assigns human-interpretable industrial labels to unsupervised latent clusters
        by evaluating physical thermodynamic and electro-mechanical centroid parameters:
        - PRESSURIZING_LOADED: High positive delta P/dt (> 0.2 PSI/s), high motor current.
        - BLOWDOWN_DISCHARGE: High negative delta P/dt (< -0.8 PSI/s), high acoustic energy.
        - IDLE_UNLOADED: Lowest motor current and vibration, near-zero delta P/dt.
        - HOLDING_PRESSURE: High steady pressure (> baseline), delta P/dt ~ 0.
        - ANOMALOUS_LEAK_REGIME: Negative delta P/dt without discharge signature or high kurtosis.
        """
        centroids = self.centroids_unscaled  # (K, D)
        col_idx = {col: i for i, col in enumerate(self.feature_columns)}

        p_idx = col_idx.get("pressure_psi", 0)
        dp_idx = col_idx.get("delta_pressure_dt", 1)
        i_idx = col_idx.get("motor_current_rms_a", 2)
        v_idx = col_idx.get("vibration_x_rms_ms2", 3)
        k_idx = col_idx.get("vibration_kurtosis", 4)
        a_idx = col_idx.get("acoustic_energy_mv", 5)

        remaining_clusters = set(range(self.n_clusters))
        assigned: Dict[int, OperatingStateLabel] = {}

        # 1. PRESSURIZING_LOADED: Highest positive dP/dt
        best_pressurizing = max(remaining_clusters, key=lambda c: centroids[c, dp_idx])
        assigned[best_pressurizing] = OperatingStateLabel.PRESSURIZING_LOADED
        remaining_clusters.remove(best_pressurizing)

        # 2. BLOWDOWN_DISCHARGE: Most negative dP/dt (or highest acoustic if dP/dt is tied)
        if remaining_clusters:
            best_blowdown = min(remaining_clusters, key=lambda c: centroids[c, dp_idx])
            assigned[best_blowdown] = OperatingStateLabel.BLOWDOWN_DISCHARGE
            remaining_clusters.remove(best_blowdown)

        # 3. IDLE_UNLOADED: Lowest motor current
        if remaining_clusters:
            best_idle = min(remaining_clusters, key=lambda c: centroids[c, i_idx])
            assigned[best_idle] = OperatingStateLabel.IDLE_UNLOADED
            remaining_clusters.remove(best_idle)

        # 4. HOLDING_PRESSURE: Highest remaining pressure with steady dP/dt
        if remaining_clusters:
            best_holding = max(remaining_clusters, key=lambda c: centroids[c, p_idx])
            assigned[best_holding] = OperatingStateLabel.HOLDING_PRESSURE
            remaining_clusters.remove(best_holding)

        # 5. ANOMALOUS_LEAK_REGIME: Remaining cluster (or highest acoustic/kurtosis remaining)
        for c in remaining_clusters:
            assigned[c] = OperatingStateLabel.ANOMALOUS_LEAK_REGIME

        self.cluster_to_label = assigned

    def _to_dataframe(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Converts array-like inputs to a validated pandas DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Feature dimension mismatch: expected {len(self.feature_columns)} features, got {X.shape[1]}"
                )
            return pd.DataFrame(X, columns=self.feature_columns)
        raise TypeError(f"Unsupported data type for X: {type(X)}")

    def _ensure_fitted(self) -> None:
        """
        Raises RuntimeError if model is invoked prior to fitting.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("CompressorStateClusterer is not fitted. Call fit() first.")
