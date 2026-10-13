"""
tests/test_behavioral.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Unit tests for state-conditioned behavioral profiling, Mahalanobis distance,
transient noise filtering, and persistent degradation tracking.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.clustering.state_discovery import OperatingStateLabel
from src.clustering.cluster_evaluator import ClusterEvaluator
from src.behavioral.state_behavioral_profiler import (
    StateProfile,
    BehavioralAssessment,
    StateBehavioralProfiler,
)
from src.behavioral.degradation_tracker import (
    DeviationClassification,
    DegradationMetrics,
    DegradationTracker,
)


@pytest.fixture
def synthetic_training_data() -> pd.DataFrame:
    """
    Provides 500 multi-modal observations across all 5 operational regimes.
    """
    df = ClusterEvaluator.generate_synthetic_compressor_data(
        n_samples_per_regime=100,
        noise_level=0.03,
        random_state=42,
    )
    df["state_label"] = df["true_regime"]
    return df


def test_profiler_fit_and_profiles_created(synthetic_training_data: pd.DataFrame):
    """
    Verify profiler calculates valid centroids, standard deviations, and invertible covariances
    for all 5 operating states.
    """
    profiler = StateBehavioralProfiler()
    assert not profiler.is_fitted

    profiler.fit(synthetic_training_data, state_column="state_label")
    assert profiler.is_fitted
    assert len(profiler.profiles) == 5

    for label in OperatingStateLabel:
        prof = profiler.get_profile(label.value)
        assert prof is not None
        assert prof.sample_count == 100
        assert len(prof.centroid) == 6
        assert len(prof.variance) == 6
        assert prof.covariance_matrix.shape == (6, 6)
        assert prof.inv_covariance_matrix.shape == (6, 6)
        # Covariance diagonal must be positive
        assert np.all(np.diag(prof.covariance_matrix) > 0)


def test_mahalanobis_distance_nominal_vs_perturbed(synthetic_training_data: pd.DataFrame):
    """
    Verify Mahalanobis distance is near zero at centroid and scales with statistical perturbation.
    """
    profiler = StateBehavioralProfiler()
    profiler.fit(synthetic_training_data, state_column="state_label")

    target_state = OperatingStateLabel.PRESSURIZING_LOADED.value
    profile = profiler.get_profile(target_state)
    assert profile is not None

    # 1. Observation exactly at centroid -> Mahalanobis distance ~ 0
    nominal_obs = dict(profile.centroid)
    assess_nom = profiler.assess_deviation(nominal_obs, target_state)
    assert assess_nom.deviation_index < 0.1
    assert all(z < 0.1 for z in assess_nom.feature_z_scores.values())

    # 2. Perturbed observation (elevated current and vibration)
    perturbed_obs = dict(profile.centroid)
    perturbed_obs["motor_current_rms_a"] += 4.0 * profile.std_dev["motor_current_rms_a"]
    perturbed_obs["vibration_x_rms_ms2"] += 4.0 * profile.std_dev["vibration_x_rms_ms2"]

    assess_pert = profiler.assess_deviation(perturbed_obs, target_state)
    assert assess_pert.deviation_index > 3.0
    assert assess_pert.feature_z_scores["motor_current_rms_a"] >= 3.9


def test_state_conditioned_isolation_prevents_false_alarms(synthetic_training_data: pd.DataFrame):
    """
    Verify state-conditioned profiling correctly prevents false alarms:
    An 18.2A current is nominal for PRESSURIZING_LOADED, but highly anomalous for IDLE_UNLOADED.
    """
    profiler = StateBehavioralProfiler()
    profiler.fit(synthetic_training_data, state_column="state_label")

    loaded_profile = profiler.get_profile(OperatingStateLabel.PRESSURIZING_LOADED.value)
    idle_profile = profiler.get_profile(OperatingStateLabel.IDLE_UNLOADED.value)

    loaded_obs = dict(loaded_profile.centroid)

    # Evaluated in its proper context (PRESSURIZING_LOADED) -> nominal
    assess_in_context = profiler.assess_deviation(loaded_obs, OperatingStateLabel.PRESSURIZING_LOADED.value)
    assert assess_in_context.deviation_index < 0.5

    # Evaluated in wrong context (IDLE_UNLOADED) -> massive anomalous deviation
    assess_out_of_context = profiler.assess_deviation(loaded_obs, OperatingStateLabel.IDLE_UNLOADED.value)
    assert assess_out_of_context.deviation_index > 10.0


def test_transient_noise_rejection():
    """
    Verify multi-window persistence filter ignores momentary 1-2 frame perturbation spikes.
    """
    tracker = DegradationTracker(deviation_threshold=3.0, persistence_threshold=5)

    dummy_centroid = {"pressure_psi": 80.0, "delta_pressure_dt": 1.2}
    dummy_variance = {"pressure_psi": 4.0, "delta_pressure_dt": 0.04}

    # Spike 1: anomalous D_M = 3.5
    assessment_spike = BehavioralAssessment(
        state_label=OperatingStateLabel.PRESSURIZING_LOADED.value,
        deviation_index=3.5,
        feature_z_scores={"pressure_psi": 3.2},
        dominant_deviating_feature="pressure_psi",
        state_cluster_centroid=dummy_centroid,
        state_cluster_variance=dummy_variance,
    )
    b_dict, m_dict = tracker.update(assessment_spike, {"delta_pressure_dt": 1.2, "temperature_celsius": 60.0})
    assert tracker.persistence_counter == 1
    assert not tracker.drift_detected
    assert b_dict["deviation_classification"] == DeviationClassification.TRANSIENT_PERTURBATION.value

    # Spike 2: anomalous D_M = 3.6
    b_dict2, _ = tracker.update(assessment_spike, {"delta_pressure_dt": 1.2, "temperature_celsius": 60.0})
    assert tracker.persistence_counter == 2
    assert not tracker.drift_detected
    assert b_dict2["deviation_classification"] == DeviationClassification.TRANSIENT_PERTURBATION.value


def test_persistent_degradation_trigger():
    """
    Verify persistence counter triggers PERSISTENT_DEGRADATION once reaching persistence threshold (5 frames).
    """
    tracker = DegradationTracker(deviation_threshold=3.0, persistence_threshold=5)

    dummy_centroid = {"pressure_psi": 80.0}
    dummy_variance = {"pressure_psi": 4.0}
    assessment = BehavioralAssessment(
        state_label=OperatingStateLabel.PRESSURIZING_LOADED.value,
        deviation_index=3.4,
        feature_z_scores={"pressure_psi": 3.1},
        dominant_deviating_feature="pressure_psi",
        state_cluster_centroid=dummy_centroid,
        state_cluster_variance=dummy_variance,
    )

    # Feed 4 consecutive anomalous frames -> not yet drift
    for i in range(4):
        b_dict, _ = tracker.update(assessment, {"delta_pressure_dt": 1.1})
        assert not tracker.drift_detected
        assert b_dict["deviation_classification"] == DeviationClassification.TRANSIENT_PERTURBATION.value

    # 5th frame reaches persistence threshold -> triggers PERSISTENT_DEGRADATION
    b_dict5, _ = tracker.update(assessment, {"delta_pressure_dt": 1.1})
    assert tracker.drift_detected
    assert tracker.persistence_counter == 5
    assert b_dict5["deviation_classification"] == DeviationClassification.PERSISTENT_DEGRADATION.value
    assert b_dict5["drift_detected"] is True


def test_critical_drift_trigger():
    """
    Verify extreme deviations exceeding 2x threshold (D_M >= 6.0) immediately trigger CRITICAL_DRIFT.
    """
    tracker = DegradationTracker(deviation_threshold=3.0, persistence_threshold=5)

    severe_assessment = BehavioralAssessment(
        state_label=OperatingStateLabel.PRESSURIZING_LOADED.value,
        deviation_index=7.2,
        feature_z_scores={"pressure_psi": 6.8},
        dominant_deviating_feature="pressure_psi",
        state_cluster_centroid={"pressure_psi": 80.0},
        state_cluster_variance={"pressure_psi": 4.0},
    )

    b_dict, _ = tracker.update(severe_assessment, {"delta_pressure_dt": 0.4})
    assert b_dict["deviation_classification"] == DeviationClassification.CRITICAL_DRIFT.value


def test_recovery_and_persistence_decay():
    """
    Verify counter decays cleanly when healthy frames return after transient perturbations.
    """
    tracker = DegradationTracker(deviation_threshold=3.0, persistence_threshold=5)

    bad_assess = BehavioralAssessment(
        state_label=OperatingStateLabel.PRESSURIZING_LOADED.value,
        deviation_index=3.5,
        feature_z_scores={},
        dominant_deviating_feature="",
        state_cluster_centroid={},
        state_cluster_variance={},
    )
    good_assess = BehavioralAssessment(
        state_label=OperatingStateLabel.PRESSURIZING_LOADED.value,
        deviation_index=1.2,
        feature_z_scores={},
        dominant_deviating_feature="",
        state_cluster_centroid={},
        state_cluster_variance={},
    )

    # 2 anomalous frames
    tracker.update(bad_assess, {})
    tracker.update(bad_assess, {})
    assert tracker.persistence_counter == 2

    # 2 healthy frames -> decays back to 0
    tracker.update(good_assess, {})
    assert tracker.persistence_counter == 1
    b_dict, _ = tracker.update(good_assess, {})
    assert tracker.persistence_counter == 0
    assert b_dict["deviation_classification"] == DeviationClassification.NOMINAL.value


def test_degradation_metrics_computation():
    """
    Verify composite degradation score, pressure efficiency, and thermal indicators remain bounded and valid.
    """
    tracker = DegradationTracker(
        deviation_threshold=3.0,
        nominal_compression_rate_psi_s=1.4,
        ambient_temperature_c=25.0,
    )

    assessment = BehavioralAssessment(
        state_label=OperatingStateLabel.PRESSURIZING_LOADED.value,
        deviation_index=2.0,
        feature_z_scores={"pressure_psi": 1.5},
        dominant_deviating_feature="pressure_psi",
        state_cluster_centroid={},
        state_cluster_variance={},
    )

    telemetry = {
        "delta_pressure_dt": 0.7,  # Half of nominal 1.4 -> ~50% efficiency
        "temperature_celsius": 65.0,
        "motor_current_rms_a": 18.0,
        "vibration_x_rms_ms2": 5.2,
    }

    _, m_dict = tracker.update(assessment, telemetry)
    assert 0.0 <= m_dict["composite_degradation_score"] <= 1.0
    assert 0.0 <= m_dict["pressure_efficiency_ratio"] <= 1.0
    assert m_dict["thermal_dissipation_factor"] > 0.0
    assert m_dict["mechanical_vibration_severity"] > 1.0


def test_profiler_save_and_load(synthetic_training_data: pd.DataFrame):
    """
    Verify profiler serialization via joblib preserves centroids and inverted covariance matrices.
    """
    profiler = StateBehavioralProfiler()
    profiler.fit(synthetic_training_data, state_column="state_label")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "test_profiler.joblib"
        profiler.save_profiles(file_path)
        assert file_path.exists()

        loaded_profiler = StateBehavioralProfiler.load_profiles(file_path)
        assert loaded_profiler.is_fitted
        assert len(loaded_profiler.profiles) == len(profiler.profiles)

        # Check identical Mahalanobis distance calculation
        sample_obs = {"pressure_psi": 85.0, "delta_pressure_dt": 1.2, "motor_current_rms_a": 18.0, "vibration_x_rms_ms2": 4.5, "vibration_kurtosis": 3.1, "acoustic_energy_mv": 250.0}
        orig_res = profiler.assess_deviation(sample_obs, OperatingStateLabel.PRESSURIZING_LOADED.value)
        loaded_res = loaded_profiler.assess_deviation(sample_obs, OperatingStateLabel.PRESSURIZING_LOADED.value)

        assert orig_res.deviation_index == pytest.approx(loaded_res.deviation_index, rel=1e-5)
        assert orig_res.dominant_deviating_feature == loaded_res.dominant_deviating_feature
