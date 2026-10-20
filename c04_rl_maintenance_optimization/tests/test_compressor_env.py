"""
test_compressor_env.py
======================
Unit tests for the C04 Gymnasium compressor simulation environment.
"""

import numpy as np
from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import CompressorMaintenanceEnv
from c04_rl_maintenance_optimization.src.env.degradation_simulator import CompressorDegradationSimulator


def test_degradation_simulator_lifecycle():
    sim = CompressorDegradationSimulator(seed=42)
    telemetry = sim.reset()
    assert 0.0 <= telemetry["composite_degradation"] <= 0.10
    assert 0.0 <= telemetry["normalized_rul"] <= 1.0

    # Test step wear dynamics
    failed, info = sim.step_dynamics(hours_elapsed=4.0)
    assert not failed
    assert info["composite_degradation"] >= telemetry["composite_degradation"]

    # Test maintenance wear reset
    maint_info = sim.apply_maintenance(action=1)
    assert maint_info["action_name"] == "MINOR_PNEUMATIC_SERVICE"
    assert maint_info["new_composite_degradation"] <= info["composite_degradation"]


def test_gym_environment_spaces_and_reset():
    env = CompressorMaintenanceEnv(seed=123)
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 4

    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (8,)
    assert env.observation_space.contains(obs)
    assert info["current_step"] == 0
    assert info["cumulative_cost"] == 0.0


def test_gym_environment_step_mechanics():
    env = CompressorMaintenanceEnv(max_episode_steps=10, seed=456)
    obs, info = env.reset()

    # Step 0: Operate Normally
    obs, reward, terminated, truncated, step_info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not terminated
    assert step_info["action_taken"] == 0
    assert step_info["step_direct_cost"] == 0.0

    # Step 1: Minor Service
    obs, reward, terminated, truncated, step_info = env.step(1)
    assert step_info["step_direct_cost"] == 200.0
    assert step_info["step_downtime_hours"] == 2.0

    # Step 2: Medium Overhaul
    obs, reward, terminated, truncated, step_info = env.step(2)
    assert step_info["step_direct_cost"] == 600.0

    # Step 3: Major Overhaul
    obs, reward, terminated, truncated, step_info = env.step(3)
    assert step_info["step_direct_cost"] == 1500.0
    assert step_info["next_degradation"] <= 0.05


def test_episode_termination():
    env = CompressorMaintenanceEnv(max_episode_steps=5, seed=789)
    env.reset()
    terminated = False
    for _ in range(5):
        _, _, terminated, _, _ = env.step(0)
    assert terminated
