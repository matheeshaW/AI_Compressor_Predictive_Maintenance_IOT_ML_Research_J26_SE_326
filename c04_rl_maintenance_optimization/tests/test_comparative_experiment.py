"""
test_comparative_experiment.py
==============================
Unit tests for comparative policy evaluation script and metric extraction.
Component 04: Reinforcement Learning-Based Maintenance Optimization
"""

import os
import tempfile

from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import (
    CompressorMaintenanceEnv,
)
from c04_rl_maintenance_optimization.src.baselines.heuristic_policies import (
    RunToFailurePolicy,
)
from c04_rl_maintenance_optimization.experiments.run_comparative_experiment import (
    evaluate_policy,
    save_metrics_to_csv,
    run_comparative_experiment,
)


def test_evaluate_policy_metrics_structure():
    """Verify evaluate_policy extracts all required quantitative metrics."""
    env = CompressorMaintenanceEnv(max_episode_steps=10, seed=42)
    policy = RunToFailurePolicy()

    results = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=5,
        base_seed=42,
        policy_name="TestRTF",
        verbose=False,
    )

    # Core required performance metrics
    assert "total_maintenance_cost" in results
    assert "cumulative_downtime_hours" in results
    assert "catastrophic_failure_prevention_rate" in results

    assert results["num_episodes"] == 5
    assert results["total_maintenance_cost"] >= 0.0
    assert results["cumulative_downtime_hours"] >= 0.0
    assert 0.0 <= results["catastrophic_failure_prevention_rate"] <= 100.0
    assert len(results["episode_records"]) == 5


def test_save_metrics_to_csv_output():
    """Verify metrics are properly formatted and written to CSV files."""
    dummy_results = [
        {
            "policy_name": "RunToFailurePolicy",
            "num_episodes": 10,
            "total_maintenance_cost": 15000.0,
            "mean_maintenance_cost": 1500.0,
            "std_maintenance_cost": 200.0,
            "cumulative_downtime_hours": 240.0,
            "mean_downtime_hours": 24.0,
            "std_downtime_hours": 4.0,
            "catastrophic_failure_prevention_rate": 20.0,
            "total_failure_events": 8,
            "episodes_zero_failure": 2,
            "mean_reward": 500.0,
            "evaluation_time_seconds": 1.5,
            "episode_records": [
                {
                    "policy": "RunToFailurePolicy",
                    "episode": 1,
                    "seed": 42,
                    "total_maintenance_cost": 1500.0,
                    "cumulative_downtime_hours": 24.0,
                    "catastrophic_failures": 1,
                    "failure_prevented": 0,
                    "cumulative_reward": 50.0,
                    "steps": 10,
                }
            ],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        summary_csv = os.path.join(tmp_dir, "summary.csv")
        detailed_csv = os.path.join(tmp_dir, "detailed.csv")

        save_metrics_to_csv(dummy_results, summary_csv, detailed_csv)

        assert os.path.isfile(summary_csv)
        assert os.path.isfile(detailed_csv)

        with open(summary_csv, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Total Maintenance Cost ($)" in content
            assert "Cumulative Downtime Hours (h)" in content
            assert "Catastrophic Failure Prevention Rate (%)" in content
            assert "RunToFailurePolicy" in content


def test_run_comparative_experiment_pipeline():
    """Verify full comparative experiment pipeline runs over small episode count."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_comparative_experiment(
            num_episodes=3,
            max_steps_per_episode=10,
            periodic_interval=5,
            seed=123,
            checkpoint_path=os.path.join(tmp_dir, "ckpt.pt"),
            results_dir=tmp_dir,
            train_if_missing=True,
            train_episodes=2,
            verbose=False,
        )

        assert len(results["summary"]) == 3
        assert os.path.isfile(results["summary_csv"])
        assert os.path.isfile(results["detailed_csv"])
