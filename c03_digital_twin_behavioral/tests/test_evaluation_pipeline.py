"""
tests/test_evaluation_pipeline.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Unit tests for the quantitative clustering evaluation and temporal stability pipeline.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate_clustering_stability import (
    run_clustering_metric_sweep,
    compute_state_transition_matrix,
    simulate_temporal_drift_run,
)
from src.clustering.state_discovery import (
    CompressorStateClusterer,
    ClusteringAlgorithm,
)
from src.clustering.cluster_evaluator import ClusterEvaluator
from src.behavioral.state_behavioral_profiler import StateBehavioralProfiler
from src.behavioral.degradation_tracker import (
    DeviationClassification,
    DegradationTracker,
)


@pytest.fixture
def sample_telemetry() -> pd.DataFrame:
    """
    Generates balanced synthetic compressor telemetry.
    """
    return ClusterEvaluator.generate_synthetic_compressor_data(
        n_samples_per_regime=50,
        noise_level=0.03,
        random_state=42,
    )


def test_clustering_metric_sweep_produces_valid_csv(sample_telemetry: pd.DataFrame):
    """
    Verify metric sweep across K in [2..6] exports valid metrics table and confirms K=5 optimality.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_metrics.csv"
        df_res = run_clustering_metric_sweep(sample_telemetry, csv_path)

        assert csv_path.exists()
        assert len(df_res) == 10  # 5 K values * 2 algorithms (K-Means and GMM)
        assert "silhouette_score" in df_res.columns
        assert "davies_bouldin_index" in df_res.columns
        assert "calinski_harabasz_index" in df_res.columns

        # Verify all silhouette scores are in valid [-1, 1] range
        assert (df_res["silhouette_score"] >= -1.0).all()
        assert (df_res["silhouette_score"] <= 1.0).all()

        # K=5 must achieve superior silhouette compared to K=2
        sub_k5 = df_res[df_res["k"] == 5]["silhouette_score"].mean()
        sub_k2 = df_res[df_res["k"] == 2]["silhouette_score"].mean()
        assert sub_k5 > sub_k2


def test_markov_state_transition_matrix_is_stochastic():
    """
    Verify computed transition probability matrix is a valid stochastic matrix (row sums == 1.0).
    """
    labels = ["IDLE", "IDLE", "LOADED", "LOADED", "HOLDING", "BLOWDOWN", "IDLE"]
    prob_mat, state_names = compute_state_transition_matrix(labels)

    assert prob_mat.shape == (len(state_names), len(state_names))
    row_sums = np.sum(prob_mat, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_temporal_drift_simulation_filters_noise_and_detects_leak(sample_telemetry: pd.DataFrame):
    """
    Verify that temporal simulation ignores transient shock spikes and detects true degradation.
    """
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(sample_telemetry)

    _, pred_labels, _ = clusterer.predict(sample_telemetry)
    sample_telemetry["predicted_state"] = pred_labels

    profiler = StateBehavioralProfiler()
    profiler.fit(sample_telemetry, state_column="predicted_state")

    tracker = DegradationTracker(deviation_threshold=3.0, persistence_threshold=5)

    df_drift = simulate_temporal_drift_run(
        clusterer=clusterer,
        profiler=profiler,
        tracker=tracker,
        n_nominal_cycles=60,
        n_leak_cycles=40,
    )

    assert len(df_drift) == 100

    # Phase 1: Nominal operation should never declare persistent drift
    df_nominal = df_drift[df_drift["phase"] == "NOMINAL"]
    assert not df_nominal["drift_detected"].any()

    # The transient shock spikes (at t=40) must be classified as TRANSIENT_PERTURBATION
    transient_rows = df_nominal[df_nominal["classification"] == DeviationClassification.TRANSIENT_PERTURBATION.value]
    assert len(transient_rows) > 0

    # Phase 2: Developing leak must trigger PERSISTENT_DEGRADATION and drift_detected == True
    df_leak = df_drift[df_drift["phase"] == "DEVELOPING_LEAK"]
    assert df_leak["drift_detected"].iloc[-1] is True or df_leak["drift_detected"].iloc[-1] == True
    assert DeviationClassification.PERSISTENT_DEGRADATION.value in df_leak["classification"].values


def test_plots_exist():
    """
    Verify evaluation plots directory contains all 4 generated PNG artifacts.
    """
    plots_dir = Path(__file__).resolve().parent.parent / "evaluation" / "plots"
    expected_files = [
        "clustering_metrics_sweep.png",
        "pca_cluster_projection.png",
        "temporal_drift_trajectory.png",
        "state_transition_heatmap.png",
    ]

    for fname in expected_files:
        p = plots_dir / fname
        assert p.exists(), f"Expected plot {fname} does not exist."
        assert p.stat().st_size > 1000, f"Plot {fname} appears empty."
