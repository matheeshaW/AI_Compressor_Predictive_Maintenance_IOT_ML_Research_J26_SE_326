"""
Clustering evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
and synthetic telemetry generation for compressor state clustering benchmarks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .state_discovery import ClusteringAlgorithm, DEFAULT_FEATURE_COLUMNS, OperatingStateLabel


@dataclass
class ClusterEvaluationMetrics:
    """
    Standardized evaluation record for a specific cluster configuration K.
    """
    k: int
    algorithm: str
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_index: float
    inertia: Optional[float] = None
    bic: Optional[float] = None
    aic: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "algorithm": self.algorithm,
            "silhouette_score": round(self.silhouette_score, 4),
            "davies_bouldin_index": round(self.davies_bouldin_index, 4),
            "calinski_harabasz_index": round(self.calinski_harabasz_index, 2),
            "inertia": round(self.inertia, 2) if self.inertia is not None else None,
            "bic": round(self.bic, 2) if self.bic is not None else None,
            "aic": round(self.aic, 2) if self.aic is not None else None,
        }


class ClusterEvaluator:
    """
    Quantitative validation suite for assessing unsupervised cluster separation and compactness.
    Calculates Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index,
    and Information Criteria (BIC/AIC/Inertia) across candidate K values.
    """

    @staticmethod
    def evaluate_k_range(
        X: Union[np.ndarray, pd.DataFrame],
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        k_values: Optional[List[int]] = None,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> List[ClusterEvaluationMetrics]:
        """
        Sweeps across candidate cluster counts K and computes all validation metrics.

        Args:
            X: Telemetry observations matrix or DataFrame.
            algorithm: K_MEANS or GAUSSIAN_MIXTURE_MODEL.
            k_values: List of candidate cluster integers (defaults to [2, 3, 4, 5, 6]).
            feature_columns: Feature names if X is a DataFrame.
            random_state: Reproducibility seed.
        """
        if k_values is None:
            k_values = [2, 3, 4, 5, 6]

        cols = feature_columns or DEFAULT_FEATURE_COLUMNS
        if isinstance(X, pd.DataFrame):
            X_raw = X[cols].values
        else:
            X_raw = np.asarray(X)

        if len(X_raw) < max(k_values):
            raise ValueError(f"Number of samples ({len(X_raw)}) must be greater than max K ({max(k_values)}).")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        results: List[ClusterEvaluationMetrics] = []

        for k in k_values:
            inertia_val = None
            bic_val = None
            aic_val = None

            if algorithm == ClusteringAlgorithm.K_MEANS:
                model = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=10,
                    random_state=random_state,
                )
                labels = model.fit_predict(X_scaled)
                inertia_val = float(model.inertia_)

            elif algorithm in (ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL, ClusteringAlgorithm.HEURISTIC_SEEDED_GMM):
                model = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    max_iter=150,
                    n_init=3,
                    random_state=random_state,
                )
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                bic_val = float(model.bic(X_scaled))
                aic_val = float(model.aic(X_scaled))

            else:
                raise NotImplementedError(f"Algorithm {algorithm} not supported for metric sweep.")

            # Calculate standard unsupervised clustering metrics
            sil = float(silhouette_score(X_scaled, labels))
            db = float(davies_bouldin_score(X_scaled, labels))
            ch = float(calinski_harabasz_score(X_scaled, labels))

            metric = ClusterEvaluationMetrics(
                k=k,
                algorithm=algorithm.value,
                silhouette_score=sil,
                davies_bouldin_index=db,
                calinski_harabasz_index=ch,
                inertia=inertia_val,
                bic=bic_val,
                aic=aic_val,
            )
            results.append(metric)

        return results

    @staticmethod
    def to_dataframe(metrics_list: List[ClusterEvaluationMetrics]) -> pd.DataFrame:
        """
        Converts a list of evaluation metric objects into a clean comparative DataFrame.
        """
        records = [m.to_dict() for m in metrics_list]
        return pd.DataFrame(records)

    @staticmethod
    def find_optimal_k(
        metrics_list: List[ClusterEvaluationMetrics],
        criterion: str = "silhouette",
    ) -> int:
        """
        Identifies the mathematically optimal K according to the chosen metric criterion:
        - "silhouette": Maximizes silhouette score.
        - "davies_bouldin": Minimizes Davies-Bouldin index.
        - "calinski_harabasz": Maximizes Calinski-Harabasz score.
        - "bic": Minimizes Bayesian Information Criterion (for GMM).
        """
        if not metrics_list:
            raise ValueError("Metrics list is empty.")

        crit = criterion.lower()
        if crit == "silhouette":
            best = max(metrics_list, key=lambda m: m.silhouette_score)
        elif crit in ("davies_bouldin", "db"):
            best = min(metrics_list, key=lambda m: m.davies_bouldin_index)
        elif crit in ("calinski_harabasz", "ch"):
            best = max(metrics_list, key=lambda m: m.calinski_harabasz_index)
        elif crit == "bic":
            bics = [m for m in metrics_list if m.bic is not None]
            if not bics:
                raise ValueError("No BIC values found in metrics list.")
            best = min(bics, key=lambda m: m.bic if m.bic is not None else float("inf"))
        else:
            raise ValueError(f"Unknown optimization criterion: {criterion}")

        return best.k

    @staticmethod
    def generate_synthetic_compressor_data(
        n_samples_per_regime: int = 150,
        noise_level: float = 0.05,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generates realistic multi-modal industrial compressor telemetry simulating the 5
        operational regimes with calibrated electro-mechanical and thermodynamic characteristics:
        1. IDLE_UNLOADED
        2. PRESSURIZING_LOADED
        3. HOLDING_PRESSURE
        4. BLOWDOWN_DISCHARGE
        5. ANOMALOUS_LEAK_REGIME

        Useful for benchmarking, synthetic data simulations, and unit tests.
        """
        rng = np.random.RandomState(random_state)
        records: List[Dict[str, Any]] = []

        base_time_ms = 1773000000000
        sample_interval_ms = 40  # 25 Hz

        regimes = [
            # Regime 0: IDLE_UNLOADED (Motor idling or minimal load, pressure low/settled)
            {
                "regime": OperatingStateLabel.IDLE_UNLOADED.value,
                "p_mean": 25.0, "p_std": 2.0,
                "dp_mean": 0.0, "dp_std": 0.02,
                "i_mean": 4.5, "i_std": 0.3,
                "v_mean": 0.8, "v_std": 0.1,
                "k_mean": 2.9, "k_std": 0.1,
                "a_mean": 70.0, "a_std": 10.0,
                "temp_mean": 38.0, "temp_std": 1.0,
            },
            # Regime 1: PRESSURIZING_LOADED (Active pumping, rising pressure, high current, high vibration)
            {
                "regime": OperatingStateLabel.PRESSURIZING_LOADED.value,
                "p_mean": 75.0, "p_std": 15.0,
                "dp_mean": 1.4, "dp_std": 0.2,
                "i_mean": 18.2, "i_std": 0.8,
                "v_mean": 4.8, "v_std": 0.4,
                "k_mean": 3.2, "k_std": 0.2,
                "a_mean": 260.0, "a_std": 25.0,
                "temp_mean": 62.0, "temp_std": 3.0,
            },
            # Regime 2: HOLDING_PRESSURE (Receiver pressurized at cut-out limit, compressor unloaded)
            {
                "regime": OperatingStateLabel.HOLDING_PRESSURE.value,
                "p_mean": 118.0, "p_std": 1.5,
                "dp_mean": 0.01, "dp_std": 0.03,
                "i_mean": 5.2, "i_std": 0.4,
                "v_mean": 1.1, "v_std": 0.15,
                "k_mean": 3.0, "k_std": 0.1,
                "a_mean": 95.0, "a_std": 15.0,
                "temp_mean": 55.0, "temp_std": 2.0,
            },
            # Regime 3: BLOWDOWN_DISCHARGE (Rapid pressure vent/dump through purge valve)
            {
                "regime": OperatingStateLabel.BLOWDOWN_DISCHARGE.value,
                "p_mean": 60.0, "p_std": 20.0,
                "dp_mean": -3.2, "dp_std": 0.4,
                "i_mean": 3.0, "i_std": 0.5,
                "v_mean": 2.2, "v_std": 0.3,
                "k_mean": 3.6, "k_std": 0.3,
                "a_mean": 780.0, "a_std": 60.0,
                "temp_mean": 48.0, "temp_std": 2.0,
            },
            # Regime 4: ANOMALOUS_LEAK_REGIME (Abnormal pressure decay with acoustic hiss and cycling)
            {
                "regime": OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value,
                "p_mean": 92.0, "p_std": 6.0,
                "dp_mean": -0.45, "dp_std": 0.1,
                "i_mean": 13.5, "i_std": 1.2,
                "v_mean": 3.6, "v_std": 0.35,
                "k_mean": 4.5, "k_std": 0.4,
                "a_mean": 540.0, "a_std": 45.0,
                "temp_mean": 68.0, "temp_std": 3.0,
            },
        ]

        curr_time = base_time_ms
        for cfg in regimes:
            for _ in range(n_samples_per_regime):
                rec = {
                    "timestamp_ms": curr_time,
                    "pressure_psi": float(rng.normal(cfg["p_mean"], cfg["p_std"])),
                    "delta_pressure_dt": float(rng.normal(cfg["dp_mean"], cfg["dp_std"])),
                    "motor_current_rms_a": max(0.0, float(rng.normal(cfg["i_mean"], cfg["i_std"]))),
                    "vibration_x_rms_ms2": max(0.0, float(rng.normal(cfg["v_mean"], cfg["v_std"]))),
                    "vibration_kurtosis": max(1.0, float(rng.normal(cfg["k_mean"], cfg["k_std"]))),
                    "acoustic_energy_mv": max(0.0, float(rng.normal(cfg["a_mean"], cfg["a_std"]))),
                    "temperature_celsius": float(rng.normal(cfg["temp_mean"], cfg["temp_std"])),
                    "true_regime": cfg["regime"],
                }
                records.append(rec)
                curr_time += sample_interval_ms

        return pd.DataFrame(records)
