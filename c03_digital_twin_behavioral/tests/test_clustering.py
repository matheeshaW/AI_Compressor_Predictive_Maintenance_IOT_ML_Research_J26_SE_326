"""
Unit tests for unsupervised state discovery, semantic regime mapping,
and clustering evaluation metrics.
"""

import math
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.clustering.state_discovery import (
    OperatingStateLabel,
    ClusteringAlgorithm,
    StateAssignment,
    CompressorStateClusterer,
    DEFAULT_FEATURE_COLUMNS,
)
from src.clustering.cluster_evaluator import (
    ClusterEvaluationMetrics,
    ClusterEvaluator,
)
from src.sync.time_series_aligner import AlignedSnapshot


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """
    Fixture providing 500 samples across the 5 canonical compressor operating regimes.
    """
    return ClusterEvaluator.generate_synthetic_compressor_data(
        n_samples_per_regime=100,
        noise_level=0.04,
        random_state=42,
    )


def test_synthetic_data_generation(synthetic_data: pd.DataFrame):
    """
    Verify synthetic generator creates balanced, physically realistic multi-modal telemetry.
    """
    df = synthetic_data
    assert len(df) == 500
    assert "true_regime" in df.columns
    for col in DEFAULT_FEATURE_COLUMNS:
        assert col in df.columns
        assert not df[col].isna().any(), f"Column {col} contains NaN values"

    unique_regimes = df["true_regime"].unique()
    assert len(unique_regimes) == 5
    assert OperatingStateLabel.IDLE_UNLOADED.value in unique_regimes
    assert OperatingStateLabel.PRESSURIZING_LOADED.value in unique_regimes
    assert OperatingStateLabel.HOLDING_PRESSURE.value in unique_regimes
    assert OperatingStateLabel.BLOWDOWN_DISCHARGE.value in unique_regimes
    assert OperatingStateLabel.ANOMALOUS_LEAK_REGIME.value in unique_regimes


def test_kmeans_clustering_fit_and_predict(synthetic_data: pd.DataFrame):
    """
    Verify K-Means model convergence, cluster assignments, and softmax confidence scores.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.K_MEANS,
        n_clusters=5,
        random_state=42,
    )
    assert not clusterer.is_fitted

    clusterer.fit(synthetic_data)
    assert clusterer.is_fitted
    assert clusterer.centroids_unscaled is not None
    assert clusterer.centroids_unscaled.shape == (5, len(DEFAULT_FEATURE_COLUMNS))

    cids, labels, confs = clusterer.predict(synthetic_data)
    assert len(cids) == len(synthetic_data)
    assert len(labels) == len(synthetic_data)
    assert len(confs) == len(synthetic_data)

    # Check boundaries
    assert set(cids).issubset({0, 1, 2, 3, 4})
    assert all(0.0 <= c <= 1.0 for c in confs)
    valid_labels = {lbl.value for lbl in OperatingStateLabel}
    assert all(l in valid_labels for l in labels)


def test_gmm_clustering_fit_and_predict(synthetic_data: pd.DataFrame):
    """
    Verify Gaussian Mixture Model convergence and Bayesian posterior probability calibration.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(synthetic_data)
    assert clusterer.is_fitted

    probs = clusterer.predict_proba(synthetic_data)
    assert probs.shape == (len(synthetic_data), 5)
    # Probabilities for each row must sum to 1.0 within numerical epsilon
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    cids, labels, confs = clusterer.predict(synthetic_data)
    assert len(cids) == len(synthetic_data)
    assert np.all(confs >= 0.2)  # Confidence in a 5-cluster problem should exceed uniform 1/5 for assigned max


def test_semantic_regime_alignment(synthetic_data: pd.DataFrame):
    """
    Verify automated physical semantic alignment assigns appropriate compressor regimes
    based on thermodynamic and electrical centroid properties.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(synthetic_data)

    centroids_df = clusterer.get_cluster_centroids()
    assert len(centroids_df) == 5
    assigned_labels = centroids_df["state_label"].tolist()

    # Must map all 5 distinct operational regimes
    assert len(set(assigned_labels)) == 5
    assert OperatingStateLabel.PRESSURIZING_LOADED.value in assigned_labels
    assert OperatingStateLabel.BLOWDOWN_DISCHARGE.value in assigned_labels
    assert OperatingStateLabel.IDLE_UNLOADED.value in assigned_labels

    # Verify physical sanity:
    # 1. PRESSURIZING_LOADED centroid must have positive delta_pressure_dt
    pressurizing_row = centroids_df[centroids_df["state_label"] == OperatingStateLabel.PRESSURIZING_LOADED.value].iloc[0]
    assert pressurizing_row["delta_pressure_dt"] > 0.3

    # 2. BLOWDOWN_DISCHARGE centroid must have negative delta_pressure_dt
    blowdown_row = centroids_df[centroids_df["state_label"] == OperatingStateLabel.BLOWDOWN_DISCHARGE.value].iloc[0]
    assert blowdown_row["delta_pressure_dt"] < -0.5

    # 3. IDLE_UNLOADED centroid must have lower motor current than PRESSURIZING_LOADED
    idle_row = centroids_df[centroids_df["state_label"] == OperatingStateLabel.IDLE_UNLOADED.value].iloc[0]
    assert idle_row["motor_current_rms_a"] < pressurizing_row["motor_current_rms_a"]


def test_predict_single_with_dict_and_snapshot(synthetic_data: pd.DataFrame):
    """
    Verify online single-packet prediction with both dictionary and AlignedSnapshot objects.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(synthetic_data)

    # 1. Using dictionary input
    sample_dict = {
        "pressure_psi": 85.0,
        "delta_pressure_dt": 1.3,
        "motor_current_rms_a": 18.0,
        "vibration_x_rms_ms2": 4.5,
        "vibration_kurtosis": 3.1,
        "acoustic_energy_mv": 250.0,
    }
    assignment1 = clusterer.predict_single(sample_dict, delta_t_sec=0.04)
    assert isinstance(assignment1, StateAssignment)
    assert assignment1.state_label in {l.value for l in OperatingStateLabel}
    assert 0.0 <= assignment1.state_confidence <= 1.0
    assert assignment1.clustering_algorithm == ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL.value
    assert assignment1.time_in_state_seconds == pytest.approx(0.04)

    # Validate dictionary conforms to schema
    payload = assignment1.to_dict()
    assert "cluster_id" in payload
    assert "state_label" in payload
    assert "state_confidence" in payload
    assert "clustering_algorithm" in payload
    assert "time_in_state_seconds" in payload
    assert "transition_count" in payload

    # 2. Using AlignedSnapshot input
    snapshot = AlignedSnapshot(
        timestamp_ms=1773000000000,
        timestamp_iso="2026-03-08T12:00:00Z",
        temperature_celsius=62.0,
        pressure_psi=85.0,
        delta_pressure_dt=1.3,
        motor_current_rms_a=18.0,
        vibration_x_rms_ms2=4.5,
        vibration_y_rms_ms2=4.2,
        vibration_z_rms_ms2=3.8,
        vibration_kurtosis=3.1,
        vibration_crest_factor=1.4,
        acoustic_energy_mv=250.0,
        sample_window_size=512,
    )
    assignment2 = clusterer.predict_single(snapshot, delta_t_sec=0.04)
    assert isinstance(assignment2, StateAssignment)
    assert assignment2.state_label == assignment1.state_label


def test_online_persistence_and_transition_tracking(synthetic_data: pd.DataFrame):
    """
    Verify cumulative time-in-state calculation and transition counter increments.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(synthetic_data)

    pressurizing_sample = {
        "pressure_psi": 80.0,
        "delta_pressure_dt": 1.5,
        "motor_current_rms_a": 18.5,
        "vibration_x_rms_ms2": 4.8,
        "vibration_kurtosis": 3.2,
        "acoustic_energy_mv": 260.0,
    }

    blowdown_sample = {
        "pressure_psi": 60.0,
        "delta_pressure_dt": -3.5,
        "motor_current_rms_a": 3.0,
        "vibration_x_rms_ms2": 2.2,
        "vibration_kurtosis": 3.6,
        "acoustic_energy_mv": 780.0,
    }

    # Step 1: 1st frame in pressurizing
    res1 = clusterer.predict_single(pressurizing_sample, delta_t_sec=0.04)
    assert res1.time_in_state_seconds == pytest.approx(0.04)
    assert res1.transition_count == 0

    # Step 2: 2nd frame in same state -> accumulates time
    res2 = clusterer.predict_single(pressurizing_sample, delta_t_sec=0.04)
    assert res2.time_in_state_seconds == pytest.approx(0.08)
    assert res2.transition_count == 0

    # Step 3: 3rd frame in same state
    res3 = clusterer.predict_single(pressurizing_sample, delta_t_sec=0.04)
    assert res3.time_in_state_seconds == pytest.approx(0.12)
    assert res3.transition_count == 0

    # Step 4: Transition to blowdown
    res4 = clusterer.predict_single(blowdown_sample, delta_t_sec=0.04)
    assert res4.transition_count == 1
    assert res4.time_in_state_seconds == pytest.approx(0.04)


def test_model_serialization_and_deserialization(synthetic_data: pd.DataFrame):
    """
    Verify model persistence (save_model and load_model) retains identical predictions.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(synthetic_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_cluster_model.joblib"
        clusterer.save_model(model_path)
        assert model_path.exists()

        loaded_clusterer = CompressorStateClusterer.load_model(model_path)
        assert loaded_clusterer.is_fitted
        assert loaded_clusterer.n_clusters == 5
        assert loaded_clusterer.algorithm == ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL

        # Check identical predictions on synthetic data
        orig_cids, orig_labels, orig_confs = clusterer.predict(synthetic_data)
        loaded_cids, loaded_labels, loaded_confs = loaded_clusterer.predict(synthetic_data)

        np.testing.assert_array_equal(orig_cids, loaded_cids)
        assert orig_labels == loaded_labels
        np.testing.assert_allclose(orig_confs, loaded_confs, atol=1e-6)


def test_cluster_evaluator_metrics_sweep(synthetic_data: pd.DataFrame):
    """
    Verify ClusterEvaluator computes Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics across K.
    """
    k_candidates = [2, 3, 4, 5]
    metrics = ClusterEvaluator.evaluate_k_range(
        synthetic_data,
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        k_values=k_candidates,
        random_state=42,
    )

    assert len(metrics) == len(k_candidates)

    for m in metrics:
        assert isinstance(m, ClusterEvaluationMetrics)
        assert m.k in k_candidates
        # Silhouette score must be within [-1.0, 1.0]
        assert -1.0 <= m.silhouette_score <= 1.0
        # Davies-Bouldin index is non-negative
        assert m.davies_bouldin_index >= 0.0
        # Calinski-Harabasz index is strictly positive
        assert m.calinski_harabasz_index > 0.0
        # GMM metrics include BIC and AIC
        assert m.bic is not None
        assert m.aic is not None

    df_metrics = ClusterEvaluator.to_dataframe(metrics)
    assert len(df_metrics) == len(k_candidates)
    assert "silhouette_score" in df_metrics.columns
    assert "davies_bouldin_index" in df_metrics.columns
    assert "calinski_harabasz_index" in df_metrics.columns


def test_cluster_evaluator_find_optimal_k(synthetic_data: pd.DataFrame):
    """
    Verify optimal K selection across different mathematical optimization criteria.
    """
    metrics = ClusterEvaluator.evaluate_k_range(
        synthetic_data,
        algorithm=ClusteringAlgorithm.K_MEANS,
        k_values=[3, 4, 5, 6],
        random_state=42,
    )

    opt_sil = ClusterEvaluator.find_optimal_k(metrics, criterion="silhouette")
    assert opt_sil in [3, 4, 5, 6]

    opt_db = ClusterEvaluator.find_optimal_k(metrics, criterion="davies_bouldin")
    assert opt_db in [3, 4, 5, 6]

    opt_ch = ClusterEvaluator.find_optimal_k(metrics, criterion="calinski_harabasz")
    assert opt_ch in [3, 4, 5, 6]


def test_unfitted_model_raises_runtime_error(synthetic_data: pd.DataFrame):
    """
    Verify attempting to predict on an unfitted clusterer raises RuntimeError.
    """
    unfitted = CompressorStateClusterer(algorithm=ClusteringAlgorithm.K_MEANS, n_clusters=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict(synthetic_data)

    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.predict_single({"pressure_psi": 50.0})

    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted.get_cluster_centroids()
