"""
run_comparative_experiment.py
=============================
Comparative Empirical Policy Evaluation: Deep Q-Network (DQN) vs Conventional Baselines.

Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
Milestone: Progress Presentation 1 (PP1) — Day 5 Execution

This experiment script instantiates the custom Gymnasium simulation environment
(CompressorMaintenanceEnv) and conducts a rigorous 500-episode Monte Carlo evaluation
comparing three distinct maintenance decision policies under identical degradation dynamics:
  1. RunToFailurePolicy (RTF): Purely reactive baseline (operate until breakdown).
  2. PeriodicMaintenancePolicy (PM): Conventional time-based scheduled preventive maintenance.
  3. DQNAgent: Reinforcement learning prescriptive policy optimizing multi-objective trade-offs.

Extracts and exports primary quantitative research metrics:
  - Total Maintenance Cost ($)
  - Cumulative Downtime Hours (h)
  - Catastrophic Failure Prevention Rate (%)
"""

from typing import Dict, Any, List, Optional, Tuple
import argparse
import csv
import json
import os
import sys
import time
import numpy as np

# Ensure project root is available on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = SCRIPT_DIR
C04_ROOT = os.path.dirname(EXPERIMENT_ROOT)
REPO_ROOT = os.path.dirname(C04_ROOT)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import (
    CompressorMaintenanceEnv,
)
from c04_rl_maintenance_optimization.src.baselines.heuristic_policies import (
    RunToFailurePolicy,
    PeriodicMaintenancePolicy,
    BasePolicy,
)
from c04_rl_maintenance_optimization.src.agents.dqn_agent import DQNAgent
from c04_rl_maintenance_optimization.src.train_dqn import train_dqn


def evaluate_policy(
    env: CompressorMaintenanceEnv,
    policy: Any,
    num_episodes: int = 500,
    base_seed: int = 42,
    policy_name: str = "Policy",
    verbose: bool = True,
    log_interval: int = 100,
) -> Dict[str, Any]:
    """
    Evaluates a maintenance policy over a fixed number of independent simulation episodes.

    Args:
        env: Gymnasium CompressorMaintenanceEnv instance.
        policy: Instance of BasePolicy or DQNAgent.
        num_episodes: Exact number of simulation episodes to run (e.g. 500).
        base_seed: Base random seed for reproducible episode initialization.
        policy_name: Descriptive name string for logging.
        verbose: Whether to display progress logs.
        log_interval: Interval of episodes to print intermediate progress.

    Returns:
        Dictionary containing extracted quantitative metrics:
            - 'policy_name': Policy name string.
            - 'num_episodes': Total evaluated episodes.
            - 'total_maintenance_cost': Aggregate maintenance & failure costs across all episodes ($).
            - 'mean_maintenance_cost': Average cost per episode ($).
            - 'std_maintenance_cost': Standard deviation of cost per episode.
            - 'cumulative_downtime_hours': Total downtime hours across all episodes (h).
            - 'mean_downtime_hours': Average downtime hours per episode (h).
            - 'std_downtime_hours': Standard deviation of downtime per episode.
            - 'catastrophic_failure_prevention_rate': Percentage of episodes with 0 catastrophic failures (%).
            - 'total_failure_events': Total catastrophic breakdown events across all episodes.
            - 'mean_reward': Average cumulative reward per episode.
            - 'episode_records': Granular list of per-episode performance dictionaries.
            - 'evaluation_time_seconds': Elapsed wall-clock time in seconds.
    """
    t_start = time.time()
    is_dqn = isinstance(policy, DQNAgent)

    episode_costs: List[float] = []
    episode_downtimes: List[float] = []
    episode_failures: List[int] = []
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    episode_records: List[Dict[str, Any]] = []

    if verbose:
        print(f"--> Evaluating {policy_name} over {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        ep_seed = base_seed + ep
        obs, info = env.reset(seed=ep_seed)

        # Reset internal state/counters for stateful heuristic policies
        if hasattr(policy, "reset"):
            policy.reset()

        ep_reward = 0.0
        step_count = 0
        done = False

        while not done:
            step_count += 1
            if is_dqn:
                # Pure greedy evaluation (no exploration noise)
                action = policy.select_action(obs, evaluate=True)
            else:
                action = policy.select_action(obs, info=info)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            obs = next_obs
            info = step_info
            ep_reward += reward

        cost = float(env.cumulative_cost)
        downtime = float(env.cumulative_downtime)
        failures = int(env.failure_events)
        prevented = 1 if failures == 0 else 0

        episode_costs.append(cost)
        episode_downtimes.append(downtime)
        episode_failures.append(failures)
        episode_rewards.append(ep_reward)
        episode_steps.append(step_count)

        episode_records.append({
            "policy": policy_name,
            "episode": ep,
            "seed": ep_seed,
            "total_maintenance_cost": cost,
            "cumulative_downtime_hours": downtime,
            "catastrophic_failures": failures,
            "failure_prevented": prevented,
            "cumulative_reward": ep_reward,
            "steps": step_count,
        })

        if verbose and (ep % log_interval == 0 or ep == num_episodes):
            avg_c = np.mean(episode_costs[-log_interval:])
            avg_dt = np.mean(episode_downtimes[-log_interval:])
            tot_f = sum(episode_failures[-log_interval:])
            print(
                f"    [{policy_name}] Ep {ep:3d}/{num_episodes:3d} | "
                f"Mean Cost: ${avg_c:8.1f} | Mean DT: {avg_dt:4.1f}h | "
                f"Failures: {tot_f:2d}"
            )

    elapsed_time = time.time() - t_start

    # Compute aggregate statistical metrics
    total_cost = float(np.sum(episode_costs))
    mean_cost = float(np.mean(episode_costs))
    std_cost = float(np.std(episode_costs))

    cumulative_dt = float(np.sum(episode_downtimes))
    mean_dt = float(np.mean(episode_downtimes))
    std_dt = float(np.std(episode_downtimes))

    total_failures = int(np.sum(episode_failures))
    episodes_zero_failure = int(sum(1 for f in episode_failures if f == 0))
    prevention_rate = float((episodes_zero_failure / num_episodes) * 100.0)

    mean_reward = float(np.mean(episode_rewards))

    return {
        "policy_name": policy_name,
        "num_episodes": num_episodes,
        "total_maintenance_cost": total_cost,
        "mean_maintenance_cost": mean_cost,
        "std_maintenance_cost": std_cost,
        "cumulative_downtime_hours": cumulative_dt,
        "mean_downtime_hours": mean_dt,
        "std_downtime_hours": std_dt,
        "catastrophic_failure_prevention_rate": prevention_rate,
        "total_failure_events": total_failures,
        "episodes_zero_failure": episodes_zero_failure,
        "mean_reward": mean_reward,
        "episode_records": episode_records,
        "evaluation_time_seconds": elapsed_time,
    }


def save_metrics_to_csv(
    summary_results: List[Dict[str, Any]],
    output_summary_csv: str,
    output_detailed_csv: Optional[str] = None,
) -> None:
    """
    Saves quantitative evaluation metrics to CSV file(s) in experiments/results/.

    Args:
        summary_results: List of policy evaluation result dictionaries.
        output_summary_csv: Filepath for the comparative summary CSV.
        output_detailed_csv: Optional filepath for full per-episode trajectory logs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_summary_csv)), exist_ok=True)

    # 1. Summary CSV table containing key quantitative metrics
    fieldnames = [
        "Policy",
        "Episodes Evaluated",
        "Total Maintenance Cost ($)",
        "Mean Maintenance Cost ($)",
        "Std Maintenance Cost ($)",
        "Cumulative Downtime Hours (h)",
        "Mean Downtime Hours (h)",
        "Std Downtime Hours (h)",
        "Catastrophic Failure Prevention Rate (%)",
        "Total Catastrophic Failures",
        "Mean Cumulative Reward",
        "Evaluation Time (s)",
    ]

    with open(output_summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for res in summary_results:
            writer.writerow([
                res["policy_name"],
                res["num_episodes"],
                f"{res['total_maintenance_cost']:.2f}",
                f"{res['mean_maintenance_cost']:.2f}",
                f"{res['std_maintenance_cost']:.2f}",
                f"{res['cumulative_downtime_hours']:.2f}",
                f"{res['mean_downtime_hours']:.2f}",
                f"{res['std_downtime_hours']:.2f}",
                f"{res['catastrophic_failure_prevention_rate']:.2f}",
                res["total_failure_events"],
                f"{res['mean_reward']:.2f}",
                f"{res['evaluation_time_seconds']:.2f}",
            ])

    print(f"\n[+] Successfully saved comparative summary metrics to:\n    {output_summary_csv}")

    # 2. Detailed per-episode CSV table if requested
    if output_detailed_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_detailed_csv)), exist_ok=True)
        all_records = []
        for res in summary_results:
            all_records.extend(res.get("episode_records", []))

        if all_records:
            detailed_fields = [
                "policy",
                "episode",
                "seed",
                "total_maintenance_cost",
                "cumulative_downtime_hours",
                "catastrophic_failures",
                "failure_prevented",
                "cumulative_reward",
                "steps",
            ]
            with open(output_detailed_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=detailed_fields)
                writer.writeheader()
                writer.writerows(all_records)

            print(f"[+] Successfully saved detailed episode metrics ({len(all_records)} rows) to:\n    {output_detailed_csv}")


def print_comparative_table(results: List[Dict[str, Any]]) -> None:
    """Prints a formatted ASCII comparison table to stdout."""
    col_w = {
        "policy": 26,
        "cost": 24,
        "downtime": 24,
        "prevention": 25,
        "failures": 16,
    }

    border = "+" + "+".join(["-" * (w + 2) for w in col_w.values()]) + "+"
    header = (
        f"| {'Policy':<{col_w['policy']}} "
        f"| {'Total Maintenance Cost':<{col_w['cost']}} "
        f"| {'Cumulative Downtime':<{col_w['downtime']}} "
        f"| {'Failure Prevention Rate':<{col_w['prevention']}} "
        f"| {'Total Failures':<{col_w['failures']}} |"
    )

    print("\n" + "=" * len(border))
    print(" " * 15 + "COMPARATIVE POLICY EVALUATION (500 EPISODES)")
    print("=" * len(border))
    print(border)
    print(header)
    print(border)

    for r in results:
        cost_str = f"${r['total_maintenance_cost']:,.0f} (avg ${r['mean_maintenance_cost']:.1f})"
        dt_str = f"{r['cumulative_downtime_hours']:,.1f}h (avg {r['mean_downtime_hours']:.1f}h)"
        prev_str = f"{r['catastrophic_failure_prevention_rate']:.1f}% ({r['episodes_zero_failure']}/{r['num_episodes']} eps)"
        fail_str = f"{r['total_failure_events']:,d}"

        row = (
            f"| {r['policy_name']:<{col_w['policy']}} "
            f"| {cost_str:<{col_w['cost']}} "
            f"| {dt_str:<{col_w['downtime']}} "
            f"| {prev_str:<{col_w['prevention']}} "
            f"| {fail_str:<{col_w['failures']}} |"
        )
        print(row)

    print(border)

    # Relative performance highlights
    rtf_res = next((r for r in results if "RunToFailure" in r["policy_name"]), None)
    dqn_res = next((r for r in results if "DQN" in r["policy_name"]), None)
    pm_res = next((r for r in results if "Periodic" in r["policy_name"]), None)

    if rtf_res and dqn_res:
        print("\nResearch Key Findings:")
        if rtf_res["total_maintenance_cost"] > 0:
            cost_diff = ((rtf_res["total_maintenance_cost"] - dqn_res["total_maintenance_cost"]) / rtf_res["total_maintenance_cost"]) * 100.0
            print(f"  * DQN vs Run-to-Failure Cost Reduction:     {cost_diff:+.1f}%")
        if rtf_res["cumulative_downtime_hours"] > 0:
            dt_diff = ((rtf_res["cumulative_downtime_hours"] - dqn_res["cumulative_downtime_hours"]) / rtf_res["cumulative_downtime_hours"]) * 100.0
            print(f"  * DQN vs Run-to-Failure Downtime Reduction: {dt_diff:+.1f}%")
        if pm_res and pm_res["total_maintenance_cost"] > 0:
            pm_cost_diff = ((pm_res["total_maintenance_cost"] - dqn_res["total_maintenance_cost"]) / pm_res["total_maintenance_cost"]) * 100.0
            print(f"  * DQN vs Periodic Maintenance Cost Delta:   {pm_cost_diff:+.1f}%")
    print("=" * len(border) + "\n")


def run_comparative_experiment(
    num_episodes: int = 500,
    max_steps_per_episode: int = 300,
    periodic_interval: int = 25,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    train_if_missing: bool = True,
    train_episodes: int = 1500,
    retrain: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main orchestration routine executing the comparative experiment across 500 episodes.

    Args:
        num_episodes: Number of test episodes per policy (default: 500).
        max_steps_per_episode: Maximum steps per episode (default: 300).
        periodic_interval: Interval steps for PeriodicMaintenancePolicy (default: 25).
        seed: Base random seed for evaluation reproducibility.
        checkpoint_path: Optional path to DQN weights checkpoint.
        results_dir: Directory where results CSV files are written.
        train_if_missing: Whether to train a checkpoint if missing.
        train_episodes: Number of training episodes if training is triggered (default: 1500).
        retrain: Force re-training even if checkpoint already exists.
        verbose: Verbose progress output flag.

    Returns:
        Dictionary containing evaluation summary results for all three policies.
    """
    experiment_dir = SCRIPT_DIR
    default_results_dir = os.path.join(experiment_dir, "results")
    default_checkpoint_path = os.path.join(experiment_dir, "checkpoints", "dqn_agent.pt")

    target_results_dir = results_dir or default_results_dir
    target_checkpoint_path = checkpoint_path or default_checkpoint_path

    os.makedirs(target_results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(target_checkpoint_path)), exist_ok=True)

    print("=" * 80)
    print("Component 04: Comparative Policy Experiment Execution (PP1 Day 5)")
    print(f"Episodes: {num_episodes} | Steps/Episode: {max_steps_per_episode} | Seed: {seed}")
    print(f"Target Results Directory: {target_results_dir}")
    print("=" * 80)

    # 1. Instantiate Evaluation Environment
    env = CompressorMaintenanceEnv(
        max_episode_steps=max_steps_per_episode,
        seed=seed,
    )

    # 2. Instantiate Conventional Baselines
    rtf_policy = RunToFailurePolicy()
    pm_policy = PeriodicMaintenancePolicy(
        maintenance_interval=periodic_interval,
        unit="steps",
    )

    # 3. Instantiate and Prepare DQN Agent
    dqn_agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=(128, 128),
        learning_rate=1e-3,
        gamma=0.99,
        seed=seed,
    )

    should_train = retrain or (not os.path.exists(target_checkpoint_path) and train_if_missing)

    if should_train:
        print(f"\n[!] Initiating DQN training run ({train_episodes} episodes)...")
        train_output = train_dqn(
            num_episodes=train_episodes,
            max_steps_per_episode=max_steps_per_episode,
            learning_rate=1e-3,
            epsilon_decay=0.997,
            batch_size=64,
            warm_up_steps=128,
            target_update_freq=100,
            seed=seed,
            checkpoint_path=target_checkpoint_path,
            verbose=verbose,
        )
        dqn_agent = train_output["agent"]
        print(f"[+] Training completed in {train_output['training_time_seconds']:.1f}s.")
        print(f"[+] Saved trained checkpoint to: {target_checkpoint_path}")
    elif os.path.exists(target_checkpoint_path):
        print(f"\n[+] Loading pre-trained DQN checkpoint from: {target_checkpoint_path}")
        dqn_agent.load(target_checkpoint_path)
    else:
        print("\n[!] Pre-trained checkpoint not found and training was skipped. Evaluating initialized DQN agent.")

    # 4. Evaluate the Three Policies over exactly 500 episodes
    policies_to_evaluate = [
        ("RunToFailurePolicy", rtf_policy),
        ("PeriodicMaintenancePolicy", pm_policy),
        ("DQNAgent", dqn_agent),
    ]

    summary_results: List[Dict[str, Any]] = []

    for name, policy_instance in policies_to_evaluate:
        res = evaluate_policy(
            env=env,
            policy=policy_instance,
            num_episodes=num_episodes,
            base_seed=seed,
            policy_name=name,
            verbose=verbose,
            log_interval=100,
        )
        summary_results.append(res)

    # 5. Save Extracted Metrics to CSV in experiments/results/
    summary_csv_path = os.path.join(target_results_dir, "comparative_policy_metrics.csv")
    detailed_csv_path = os.path.join(target_results_dir, "comparative_episodes_detailed.csv")
    save_metrics_to_csv(summary_results, summary_csv_path, detailed_csv_path)

    # 6. Save JSON Summary for programmatic access
    json_summary_path = os.path.join(target_results_dir, "comparative_metrics_summary.json")
    clean_summary = []
    for r in summary_results:
        c = dict(r)
        c.pop("episode_records", None)
        clean_summary.append(c)
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    # 7. Print Comparative Performance Table
    print_comparative_table(summary_results)

    return {
        "summary": clean_summary,
        "results": summary_results,
        "summary_csv": summary_csv_path,
        "detailed_csv": detailed_csv_path,
    }


def main() -> None:
    """CLI entry point for executing comparative policy experiments."""
    parser = argparse.ArgumentParser(
        description="Run comparative policy evaluation on CompressorMaintenanceEnv over 500 episodes."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes per policy (default: 500)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum decision steps per episode (default: 300)",
    )
    parser.add_argument(
        "--periodic-interval",
        type=int,
        default=25,
        help="Maintenance interval steps for PeriodicMaintenancePolicy (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic evaluation (default: 42)",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=1500,
        help="Number of training episodes if checkpoint is missing (default: 1500)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training for specified episodes before evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained DQN checkpoint (default: experiments/checkpoints/dqn_agent.pt)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Destination directory for output CSVs (default: experiments/results/)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training if checkpoint is not found",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate evaluation logging",
    )

    args = parser.parse_args()

    run_comparative_experiment(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        periodic_interval=args.periodic_interval,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        train_if_missing=not args.skip_train,
        train_episodes=args.train_episodes,
        retrain=args.retrain,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
