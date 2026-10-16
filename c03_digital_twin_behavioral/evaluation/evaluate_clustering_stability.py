"""
evaluation/evaluate_clustering_stability.py
Component 03: Industrial Digital Twin & Behavioral Intelligence
Quantitative clustering benchmarking, state transition stability,
and temporal degradation drift evaluation.
Researcher: Kahakotuwa K.N (IT23432284) - J26-SE-326
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

# Ensure c03_digital_twin_behavioral root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.clustering.state_discovery import (
    OperatingStateLabel,
    ClusteringAlgorithm,
    CompressorStateClusterer,
    DEFAULT_FEATURE_COLUMNS,
)
from src.clustering.cluster_evaluator import (
    ClusterEvaluationMetrics,
    ClusterEvaluator,
)
from src.behavioral.state_behavioral_profiler import StateBehavioralProfiler
from src.behavioral.degradation_tracker import (
    DeviationClassification,
    DegradationTracker,
)


def run_clustering_metric_sweep(
    data: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    """
    Executes quantitative metric sweep across K in [2, 3, 4, 5, 6] for both K-Means and GMM.
    Saves results to CSV and returns summary DataFrame.
    """
    k_range = [2, 3, 4, 5, 6]
    records: List[Dict[str, Any]] = []

    for algo in [ClusteringAlgorithm.K_MEANS, ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL]:
        metrics = ClusterEvaluator.evaluate_k_range(
            X=data,
            algorithm=algo,
            k_values=k_range,
            random_state=42,
        )
        for m in metrics:
            records.append(m.to_dict())

    df_results = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    return df_results


def compute_state_transition_matrix(
    state_labels: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculates empirical first-order Markovian transition probability matrix
    P_ij = Count(i -> j) / Total Transitions from i.
    """
    unique_states = sorted(list(set(state_labels)))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    n_states = len(unique_states)

    count_matrix = np.zeros((n_states, n_states), dtype=float)

    for t in range(len(state_labels) - 1):
        src = state_to_idx[state_labels[t]]
        dst = state_to_idx[state_labels[t + 1]]
        count_matrix[src, dst] += 1.0

    # Normalize rows to form valid stochastic transition probabilities
    prob_matrix = np.zeros_like(count_matrix)
    for i in range(n_states):
        row_sum = np.sum(count_matrix[i, :])
        if row_sum > 0:
            prob_matrix[i, :] = count_matrix[i, :] / row_sum
        else:
            prob_matrix[i, i] = 1.0

    return prob_matrix, unique_states


def simulate_temporal_drift_run(
    clusterer: CompressorStateClusterer,
    profiler: StateBehavioralProfiler,
    tracker: DegradationTracker,
    n_nominal_cycles: int = 150,
    n_leak_cycles: int = 100,
) -> pd.DataFrame:
    """
    Simulates a sequential operational run:
    1. Nominal loaded compression with periodic motor start transients.
    2. Subtle developing pneumatic leak causing persistent pressure droop and acoustic hiss.
    Records Mahalanobis distance, persistence counter, and classification over time.
    """
    records: List[Dict[str, Any]] = []
    rng = np.random.RandomState(42)

    # Baseline nominal profile for loaded compression
    p_mean, dp_mean, i_mean, v_mean, k_mean, a_mean = 75.0, 1.4, 18.2, 4.8, 3.2, 260.0

    # Phase 1: Nominal operation (with transient electrical shock spikes at t=40 and t=80)
    for t in range(n_nominal_cycles):
        if t == 40:
            # Moderate transient perturbation (lasting 1 frame only, D_M in [3.0, 5.0])
            obs = {
                "pressure_psi": p_mean,
                "delta_pressure_dt": dp_mean,
                "motor_current_rms_a": i_mean + 3.2,
                "vibration_x_rms_ms2": v_mean,
                "vibration_kurtosis": k_mean,
                "acoustic_energy_mv": a_mean,
                "temperature_celsius": 60.0,
            }
        elif t == 80:
            # Severe transient shock (lasting 1 frame only, D_M >= 6.0)
            obs = {
                "pressure_psi": p_mean,
                "delta_pressure_dt": dp_mean,
                "motor_current_rms_a": i_mean + 12.0,
                "vibration_x_rms_ms2": v_mean + 5.0,
                "vibration_kurtosis": 6.5,
                "acoustic_energy_mv": a_mean + 200.0,
                "temperature_celsius": 60.0,
            }
        else:
            obs = {
                "pressure_psi": float(rng.normal(p_mean, 2.0)),
                "delta_pressure_dt": float(rng.normal(dp_mean, 0.1)),
                "motor_current_rms_a": float(rng.normal(i_mean, 0.4)),
                "vibration_x_rms_ms2": float(rng.normal(v_mean, 0.2)),
                "vibration_kurtosis": float(rng.normal(k_mean, 0.1)),
                "acoustic_energy_mv": float(rng.normal(a_mean, 15.0)),
                "temperature_celsius": float(rng.normal(60.0, 1.0)),
            }

        assignment = clusterer.predict_single(obs, delta_t_sec=0.04)
        assessment = profiler.assess_deviation(obs, assignment.state_label)
        b_dict, m_dict = tracker.update(assessment, obs, delta_t_sec=0.04)

        records.append({
            "step": t,
            "phase": "NOMINAL",
            "state_label": assignment.state_label,
            "deviation_index": assessment.deviation_index,
            "persistence_counter": tracker.persistence_counter,
            "drift_detected": tracker.drift_detected,
            "classification": b_dict["deviation_classification"],
            "composite_degradation_score": m_dict["composite_degradation_score"],
        })

    # Phase 2: Developing pneumatic leak (gradual irreversible drift)
    for t in range(n_leak_cycles):
        leak_severity = min(1.0, (t + 1) / 40.0)  # Gradual progressive leak
        obs = {
            "pressure_psi": float(rng.normal(p_mean - 15.0 * leak_severity, 3.0)),
            "delta_pressure_dt": float(rng.normal(dp_mean - 1.8 * leak_severity, 0.1)),  # Drooping pressure
            "motor_current_rms_a": float(rng.normal(i_mean + 2.5 * leak_severity, 0.5)), # Motor strains
            "vibration_x_rms_ms2": float(rng.normal(v_mean + 1.2 * leak_severity, 0.3)),
            "vibration_kurtosis": float(rng.normal(k_mean + 1.5 * leak_severity, 0.2)),
            "acoustic_energy_mv": float(rng.normal(a_mean + 400.0 * leak_severity, 30.0)), # Air hiss
            "temperature_celsius": float(rng.normal(65.0, 1.5)),
        }

        assignment = clusterer.predict_single(obs, delta_t_sec=0.04)
        assessment = profiler.assess_deviation(obs, assignment.state_label)
        b_dict, m_dict = tracker.update(assessment, obs, delta_t_sec=0.04)

        records.append({
            "step": n_nominal_cycles + t,
            "phase": "DEVELOPING_LEAK",
            "state_label": assignment.state_label,
            "deviation_index": assessment.deviation_index,
            "persistence_counter": tracker.persistence_counter,
            "drift_detected": tracker.drift_detected,
            "classification": b_dict["deviation_classification"],
            "composite_degradation_score": m_dict["composite_degradation_score"],
        })

    return pd.DataFrame(records)


def generate_evaluation_visualizations(
    df_metrics: pd.DataFrame,
    df_train: pd.DataFrame,
    clusterer: CompressorStateClusterer,
    df_drift: pd.DataFrame,
    prob_matrix: np.ndarray,
    state_names: List[str],
    plots_dir: Path,
) -> None:
    """
    Renders 4 publication-grade figures for the PP1 presentation slide deck and notebook:
    1. Clustering metrics sweep curves across K (Silhouette, DB, CH).
    2. 2D PCA projection of the 6D feature space showing cluster boundaries.
    3. Temporal Mahalanobis drift trajectory showing noise filtering vs leak detection.
    4. State transition probability matrix heatmap.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Plot 1: Clustering Metrics Sweep across K
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    for algo in [ClusteringAlgorithm.K_MEANS.value, ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL.value]:
        sub = df_metrics[df_metrics["algorithm"] == algo]
        label_name = "K-Means" if algo == ClusteringAlgorithm.K_MEANS.value else "Gaussian Mixture (GMM)"
        marker = "o" if algo == ClusteringAlgorithm.K_MEANS.value else "s"

        axes[0].plot(sub["k"], sub["silhouette_score"], marker=marker, linewidth=2, label=label_name)
        axes[1].plot(sub["k"], sub["davies_bouldin_index"], marker=marker, linewidth=2, label=label_name)
        axes[2].plot(sub["k"], sub["calinski_harabasz_index"], marker=marker, linewidth=2, label=label_name)

    axes[0].set_title("Silhouette Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].axvline(5, color="red", linestyle="--", alpha=0.7, label="Optimal K=5")
    axes[0].legend()

    axes[1].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].axvline(5, color="red", linestyle="--", alpha=0.7, label="Optimal K=5")
    axes[1].legend()

    axes[2].set_title("Calinski-Harabasz Index (Higher is Better)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Variance Ratio (CH)")
    axes[2].axvline(5, color="red", linestyle="--", alpha=0.7, label="Optimal K=5")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "clustering_metrics_sweep.png")
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: 2D PCA Latent Space Projection
    # -------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[DEFAULT_FEATURE_COLUMNS].values)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    _, state_labels, _ = clusterer.predict(df_train)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    unique_labels = sorted(list(set(state_labels)))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

    for i, lbl in enumerate(unique_labels):
        mask = np.array(state_labels) == lbl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[i % len(colors)],
            label=lbl,
            alpha=0.7,
            edgecolors="none",
            s=40,
        )

    var_ratio = pca.explained_variance_ratio_
    ax.set_title(
        f"2D PCA Projection of Discovered Compressor States (EV: {var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}%)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_cluster_projection.png")
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: Temporal Mahalanobis Drift Trajectory
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, dpi=300)

    steps = df_drift["step"].values
    d_m = df_drift["deviation_index"].values
    counter = df_drift["persistence_counter"].values

    ax1.plot(steps, d_m, color="#1f77b4", linewidth=1.5, label="Mahalanobis Distance D_M")
    ax1.axhline(3.0, color="orange", linestyle="--", linewidth=1.5, label="Normal Envelope Threshold (3.0 sigma)")
    ax1.axhline(6.0, color="red", linestyle=":", linewidth=1.5, label="Critical Drift Threshold (6.0 sigma)")
    ax1.axvline(150, color="gray", linestyle="-.", label="Developing Leak Injected")

    # Annotate transient spikes
    ax1.annotate("Transient Start Surge\n(Filtered Noise)", xy=(40, d_m[40]), xytext=(45, 9.0),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6))
    ax1.annotate("Transient Shock\n(Filtered Noise)", xy=(80, d_m[80]), xytext=(85, 9.0),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6))

    ax1.set_title("Real-Time Behavioral Deviation & Temporal Persistence Filtering", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mahalanobis Distance ($D_M$)")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 14)

    # Persistence Counter subplot
    ax2.plot(steps, counter, color="#2ca02c", linewidth=2.0, label="Persistence Counter (Consecutive Windows)")
    ax2.axhline(5, color="red", linestyle="--", linewidth=1.5, label="Persistence Threshold (tau=5 frames)")
    ax2.fill_between(steps, 0, counter, color="#2ca02c", alpha=0.2)
    ax2.set_xlabel("Time (25 Hz Telemetry Sample Steps)")
    ax2.set_ylabel("Consecutive Frames")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(plots_dir / "temporal_drift_trajectory.png")
    plt.close()

    # -------------------------------------------------------------
    # Plot 4: State Transition Matrix Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(prob_matrix, cmap="Blues", alpha=0.85)

    for i in range(len(state_names)):
        for j in range(len(state_names)):
            val = prob_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontweight="bold")

    fig.colorbar(cax)
    clean_labels = [s.replace("_", "\n") for s in state_names]
    ax.set_xticks(range(len(state_names)))
    ax.set_yticks(range(len(state_names)))
    ax.set_xticklabels(clean_labels, fontsize=8)
    ax.set_yticklabels(clean_labels, fontsize=8)
    ax.set_title("Empirical State Transition Probability Matrix ($P_{ij}$)", fontsize=12, fontweight="bold", pad=20)
    ax.set_xlabel("Next Operational State (t + 1)")
    ax.set_ylabel("Current Operational State (t)")

    plt.tight_layout()
    plt.savefig(plots_dir / "state_transition_heatmap.png")
    plt.close()


def main():
    """
    Main evaluation pipeline execution.
    """
    print("=" * 75)
    print("COMPONENT 03: DIGITAL TWIN BEHAVIORAL INTELLIGENCE - DAY 05 EVALUATION")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    output_csv = base_dir / "clustering_benchmark_metrics.csv"
    plots_dir = base_dir / "plots"

    print("\n[Step 1/5] Generating multi-modal compressor telemetry dataset...")
    df_data = ClusterEvaluator.generate_synthetic_compressor_data(
        n_samples_per_regime=150,
        noise_level=0.04,
        random_state=42,
    )
    print(f"Generated {len(df_data)} observations across 5 canonical operational regimes.")

    print("\n[Step 2/5] Running quantitative clustering metric sweep across K in [2, 3, 4, 5, 6]...")
    df_metrics = run_clustering_metric_sweep(df_data, output_csv)
    print(f"Successfully computed metric matrix. Saved to: {output_csv}")

    print("\n" + "-" * 75)
    print("QUANTITATIVE BENCHMARK MATRIX (K-Means vs GMM):")
    print("-" * 75)
    print(df_metrics[["algorithm", "k", "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index"]].to_string(index=False))

    print("\n[Step 3/5] Fitting production GMM clusterer and state behavioral profiler...")
    clusterer = CompressorStateClusterer(
        algorithm=ClusteringAlgorithm.GAUSSIAN_MIXTURE_MODEL,
        n_clusters=5,
        random_state=42,
    )
    clusterer.fit(df_data)

    _, pred_labels, _ = clusterer.predict(df_data)
    df_data["predicted_state"] = pred_labels

    profiler = StateBehavioralProfiler()
    profiler.fit(df_data, state_column="predicted_state")

    tracker = DegradationTracker(deviation_threshold=3.0, persistence_threshold=5)

    print("\n[Step 4/5] Simulating temporal operational sequence & persistence drift tracking...")
    df_drift = simulate_temporal_drift_run(clusterer, profiler, tracker)

    # Compute transition matrix
    prob_mat, state_names = compute_state_transition_matrix(pred_labels)
    print(f"Computed Markov transition matrix for {len(state_names)} operational states.")

    print(f"\n[Step 5/5] Generating publication-grade visualization figures in {plots_dir}...")
    generate_evaluation_visualizations(
        df_metrics=df_metrics,
        df_train=df_data,
        clusterer=clusterer,
        df_drift=df_drift,
        prob_matrix=prob_mat,
        state_names=state_names,
        plots_dir=plots_dir,
    )
    print("Rendered 4 publication figures:")
    print("  1. clustering_metrics_sweep.png")
    print("  2. pca_cluster_projection.png")
    print("  3. temporal_drift_trajectory.png")
    print("  4. state_transition_heatmap.png")

    print("\n" + "=" * 75)
    print("DAY 05 BENCHMARKING COMPLETE: ALL METRICS AND PLOTS EXPORTED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    main()
