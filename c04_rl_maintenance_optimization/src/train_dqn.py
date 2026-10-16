"""
train_dqn.py
============
Training loop script for Deep Q-Network (DQN) on CompressorMaintenanceEnv.
Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
"""

from typing import Dict, Any, List, Optional
import argparse
import os
import sys
import time
import numpy as np

from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import (
    CompressorMaintenanceEnv,
)
from c04_rl_maintenance_optimization.src.agents.dqn_agent import DQNAgent


def train_dqn(
    num_episodes: int = 50,
    max_steps_per_episode: int = 300,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99,
    batch_size: int = 64,
    warm_up_steps: int = 128,
    train_every_steps: int = 1,
    target_update_freq: int = 100,
    buffer_capacity: int = 50_000,
    tau: float = 1.0,
    seed: Optional[int] = 42,
    log_interval: int = 5,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the DQN training loop inside the CompressorMaintenanceEnv.

    Args:
        num_episodes: Number of simulation episodes to train.
        max_steps_per_episode: Maximum decision steps per episode.
        learning_rate: Optimizer learning rate.
        gamma: Temporal discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Epsilon multiplicative decay per episode.
        batch_size: Mini-batch size for gradient updates.
        warm_up_steps: Steps to collect experience before training commences.
        train_every_steps: Train frequency in environment steps.
        target_update_freq: Frequency for hard target network synchronization.
        buffer_capacity: Replay buffer capacity.
        tau: Soft update factor (1.0 = hard update).
        seed: Random seed for deterministic execution.
        log_interval: Episode interval for logging summary metrics.
        checkpoint_path: Optional path to save final model checkpoint.
        verbose: Whether to print progress logs.

    Returns:
        Dictionary containing training history and convergence metrics:
            - 'episode_rewards': List of cumulative rewards per episode.
            - 'episode_costs': List of cumulative financial costs per episode.
            - 'episode_downtimes': List of cumulative downtime hours per episode.
            - 'episode_failures': List of catastrophic breakdown counts.
            - 'episode_losses': List of mean MSE losses per episode.
            - 'total_steps': Total simulation steps across all episodes.
            - 'final_epsilon': Final exploration rate.
            - 'training_time_seconds': Elapsed wall-clock time.
    """
    start_time = time.time()

    # 1. Instantiate Gymnasium environment and DQN agent
    env = CompressorMaintenanceEnv(
        max_episode_steps=max_steps_per_episode,
        seed=seed,
    )

    agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=(128, 128),
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        tau=tau,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        seed=seed,
    )

    # Metrics history
    history: Dict[str, List[Any]] = {
        "episode_rewards": [],
        "episode_costs": [],
        "episode_downtimes": [],
        "episode_failures": [],
        "episode_losses": [],
    }

    total_env_steps = 0

    if verbose:
        print("=" * 78)
        print("Starting DQN Training on CompressorMaintenanceEnv")
        print(f"Episodes: {num_episodes} | Max Steps/Ep: {max_steps_per_episode} | Batch Size: {batch_size}")
        print(f"Warm-up Steps: {warm_up_steps} | Target Update Freq: {target_update_freq}")
        print("=" * 78)

    # 2. Main Episode Training Loop
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + ep if seed is not None else None)
        ep_reward = 0.0
        ep_losses: List[float] = []

        for step in range(max_steps_per_episode):
            total_env_steps += 1

            # Select action with epsilon-greedy policy
            action = agent.select_action(obs, evaluate=False)

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            # Store transition in replay buffer
            agent.store_transition(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward

            # Perform training step once buffer warm-up is fulfilled
            if total_env_steps >= warm_up_steps and total_env_steps % train_every_steps == 0:
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

            if done:
                break

        # Decay exploration rate at the end of each episode
        current_eps = agent.decay_epsilon()

        # Record episode metrics
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["episode_rewards"].append(ep_reward)
        history["episode_costs"].append(env.cumulative_cost)
        history["episode_downtimes"].append(env.cumulative_downtime)
        history["episode_failures"].append(env.failure_events)
        history["episode_losses"].append(mean_loss)

        if verbose and (ep % log_interval == 0 or ep == 1 or ep == num_episodes):
            recent_rewards = history["episode_rewards"][-log_interval:]
            recent_losses = [l for l in history["episode_losses"][-log_interval:] if l > 0]
            avg_loss_str = f"{np.mean(recent_losses):.4f}" if recent_losses else "N/A (warm-up)"
            print(
                f"Episode {ep:3d}/{num_episodes:3d} | "
                f"Steps: {step + 1:3d} | "
                f"Reward: {ep_reward:9.2f} (Avg: {np.mean(recent_rewards):9.2f}) | "
                f"Loss: {avg_loss_str:>14s} | "
                f"Cost: ${env.cumulative_cost:8.2f} | "
                f"Downtime: {env.cumulative_downtime:4.1f}h | "
                f"Failures: {env.failure_events:1d} | "
                f"Eps: {current_eps:.3f}"
            )

    elapsed_time = time.time() - start_time

    # 3. Save model checkpoint if requested
    if checkpoint_path is not None:
        agent.save(checkpoint_path)
        if verbose:
            print(f"\nModel checkpoint successfully saved to: {checkpoint_path}")

    if verbose:
        print("=" * 78)
        print(f"Training Complete in {elapsed_time:.2f}s | Total Steps: {total_env_steps}")
        print(f"Final 10-Ep Mean Reward: {np.mean(history['episode_rewards'][-10:]):.2f}")
        print("=" * 78)

    return {
        "episode_rewards": history["episode_rewards"],
        "episode_costs": history["episode_costs"],
        "episode_downtimes": history["episode_downtimes"],
        "episode_failures": history["episode_failures"],
        "episode_losses": history["episode_losses"],
        "total_steps": total_env_steps,
        "final_epsilon": agent.epsilon,
        "training_time_seconds": elapsed_time,
        "agent": agent,
    }


def main() -> None:
    """CLI entry point for running DQN training."""
    parser = argparse.ArgumentParser(description="Train DQN Agent on CompressorMaintenanceEnv")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save checkpoint")
    args = parser.parse_args()

    train_dqn(
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        verbose=True,
    )


if __name__ == "__main__":
    main()
