"""
test_baselines.py
=================
Unit tests for the heuristic baseline maintenance policies (RunToFailurePolicy,
PeriodicMaintenancePolicy, ConditionBasedMaintenancePolicy) in Component 04.
"""

import pytest
import numpy as np

from c04_rl_maintenance_optimization.src.baselines.heuristic_policies import (
    RunToFailurePolicy,
    PeriodicMaintenancePolicy,
    ConditionBasedMaintenancePolicy,
    ACTION_OPERATE_NORMALLY,
    ACTION_SCHEDULE_MEDIUM_MAINTENANCE,
    ACTION_EMERGENCY_MAJOR_MAINTENANCE,
)
from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import (
    CompressorMaintenanceEnv,
)


# ---------------------------------------------------------------------------
# 1. Run-to-Failure (RTF) Policy Unit Tests
# ---------------------------------------------------------------------------

def test_rtf_policy_initialization():
    policy = RunToFailurePolicy()
    assert policy.rul_threshold == 0.0
    assert policy.degradation_threshold == 1.0
    assert policy.total_emergency_repairs == 0
    assert policy.total_steps == 0


def test_rtf_policy_normal_operation():
    """Verify policy returns 0 (OPERATE_NORMALLY) while equipment has remaining life."""
    policy = RunToFailurePolicy()

    # Normal 8-dim observation vector [D_t=0.15, RUL_t=0.85, delta=1.2, eta=0.95, ...]
    normal_obs = np.array([0.15, 0.85, 1.2, 0.95, 1.1, 1.0, 0.0, 0.1], dtype=np.float32)
    action = policy.select_action(normal_obs)
    assert action == ACTION_OPERATE_NORMALLY

    # Normal dictionary state
    dict_state = {"composite_degradation": 0.20, "normalized_rul": 0.80}
    assert policy.select_action(dict_state) == ACTION_OPERATE_NORMALLY

    # Normal scalar state (RUL = 0.5)
    assert policy.select_action(0.5) == ACTION_OPERATE_NORMALLY

    # Calling via act() and __call__()
    assert policy.act(normal_obs) == ACTION_OPERATE_NORMALLY
    assert policy(normal_obs) == ACTION_OPERATE_NORMALLY


def test_rtf_triggers_emergency_on_rul_zero():
    """Verify policy triggers action 3 (EMERGENCY_MAJOR_MAINTENANCE) exactly when RUL <= 0.0."""
    policy = RunToFailurePolicy()

    # RUL depleted to 0.0 in observation vector
    exhausted_obs = np.array([0.95, 0.0, 5.0, 0.4, 2.5, 4.0, 1.0, 0.9], dtype=np.float32)
    action = policy.select_action(exhausted_obs)
    assert action == ACTION_EMERGENCY_MAJOR_MAINTENANCE
    assert policy.total_emergency_repairs == 1

    # Negative RUL due to numerical variance
    neg_rul_obs = np.array([0.98, -0.005, 5.0, 0.4, 2.5, 4.0, 1.0, 0.9], dtype=np.float32)
    assert policy.select_action(neg_rul_obs) == ACTION_EMERGENCY_MAJOR_MAINTENANCE
    assert policy.total_emergency_repairs == 2

    # RUL 0.0 in dict state
    assert policy.select_action({"normalized_rul": 0.0}) == ACTION_EMERGENCY_MAJOR_MAINTENANCE

    # Scalar RUL 0.0
    assert policy.select_action(0.0) == ACTION_EMERGENCY_MAJOR_MAINTENANCE


def test_rtf_triggers_emergency_on_catastrophic_failure():
    """Verify policy triggers action 3 when catastrophic breakdown or degradation limit is hit."""
    policy = RunToFailurePolicy()

    # Degradation reaches 1.0
    broken_obs = np.array([1.0, 0.05, 8.0, 0.2, 3.0, 4.0, 1.0, 1.0], dtype=np.float32)
    assert policy.select_action(broken_obs) == ACTION_EMERGENCY_MAJOR_MAINTENANCE

    # Info dictionary indicates catastrophic breakdown
    safe_obs = np.array([0.4, 0.6, 2.0, 0.8, 1.2, 1.0, 0.0, 0.2], dtype=np.float32)
    fail_info = {"catastrophic_failure": True, "failure_cause": "VALVE_SHATTER_BREAKDOWN"}
    assert policy.select_action(safe_obs, info=fail_info) == ACTION_EMERGENCY_MAJOR_MAINTENANCE

    # Info with non-normal failure cause
    fail_cause_info = {"failure_cause": "BEARING_SEIZURE_MOTOR_BURNOUT"}
    assert policy.select_action(safe_obs, info=fail_cause_info) == ACTION_EMERGENCY_MAJOR_MAINTENANCE


def test_rtf_policy_reset():
    policy = RunToFailurePolicy()
    policy.select_action(np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0]))
    assert policy.total_emergency_repairs == 1
    assert policy.total_steps == 1

    policy.reset()
    assert policy.total_emergency_repairs == 0
    assert policy.total_steps == 0


# ---------------------------------------------------------------------------
# 2. Periodic Maintenance (PM) Policy Unit Tests
# ---------------------------------------------------------------------------

def test_pm_policy_initialization():
    policy = PeriodicMaintenancePolicy(maintenance_interval=25)
    assert policy.maintenance_interval == 25.0
    assert policy.unit == "steps"
    assert policy.counter == 0
    assert policy.total_scheduled_services == 0

    with pytest.raises(ValueError):
        PeriodicMaintenancePolicy(maintenance_interval=0)

    with pytest.raises(ValueError):
        PeriodicMaintenancePolicy(maintenance_interval=-10)

    with pytest.raises(ValueError):
        PeriodicMaintenancePolicy(maintenance_interval=10, unit="invalid_unit")


def test_pm_policy_steps_timeline():
    """
    Simulate a 12-step timeline with maintenance_interval = 4 steps.
    Expect action 0 for steps 1..3, action 2 at step 4 (counter reset),
    action 0 for steps 5..7, action 2 at step 8 (counter reset), etc.
    """
    interval = 4
    policy = PeriodicMaintenancePolicy(maintenance_interval=interval)

    actions_taken = []
    for step in range(1, 13):
        action = policy.select_action()
        actions_taken.append(action)

        if step % interval == 0:
            assert action == ACTION_SCHEDULE_MEDIUM_MAINTENANCE, f"Step {step} should trigger maintenance"
            assert policy.counter == 0, f"Counter should reset to 0 after step {step}"
        else:
            assert action == ACTION_OPERATE_NORMALLY, f"Step {step} should be normal operation"
            assert policy.counter == (step % interval)

    assert actions_taken == [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]
    assert policy.total_scheduled_services == 3
    assert policy.total_steps == 12


def test_pm_policy_hours_timeline():
    """
    Simulate a timeline measured in operating hours.
    maintenance_interval = 16.0 hours, step_hours = 4.0 hours (4 steps per cycle).
    """
    policy = PeriodicMaintenancePolicy(
        maintenance_interval=16.0,
        step_hours=4.0,
        unit="hours",
    )

    actions = []
    for _ in range(8):  # 8 steps * 4h = 32 hours = 2 maintenance cycles
        act = policy.select_action()
        actions.append(act)

    assert actions == [0, 0, 0, 2, 0, 0, 0, 2]
    assert policy.total_scheduled_services == 2
    assert policy.counter == 0.0


def test_pm_policy_reset():
    policy = PeriodicMaintenancePolicy(maintenance_interval=5)
    for _ in range(3):
        policy.select_action()
    assert policy.counter == 3

    policy.reset()
    assert policy.counter == 0
    assert policy.total_scheduled_services == 0
    assert policy.total_steps == 0


# ---------------------------------------------------------------------------
# 3. Condition-Based Maintenance (CBM) Policy Unit Tests
# ---------------------------------------------------------------------------

def test_cbm_policy_thresholds():
    policy = ConditionBasedMaintenancePolicy(degradation_threshold=0.70, rul_threshold=0.20)

    # Below threshold -> action 0
    safe_obs = np.array([0.50, 0.50, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert policy.select_action(safe_obs) == ACTION_OPERATE_NORMALLY

    # High degradation -> action 2
    high_deg_obs = np.array([0.72, 0.40, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert policy.select_action(high_deg_obs) == ACTION_SCHEDULE_MEDIUM_MAINTENANCE

    # Low RUL -> action 2
    low_rul_obs = np.array([0.60, 0.15, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert policy.select_action(low_rul_obs) == ACTION_SCHEDULE_MEDIUM_MAINTENANCE


# ---------------------------------------------------------------------------
# 4. Simulation Environment Integration Tests
# ---------------------------------------------------------------------------

def test_rtf_policy_simulation_in_gym_env():
    """
    Run the RunToFailurePolicy in CompressorMaintenanceEnv over a multi-step episode.
    Verify that policy acts continuously and triggers emergency overhaul if machine fails.
    """
    env = CompressorMaintenanceEnv(max_episode_steps=40, seed=100)
    policy = RunToFailurePolicy()

    obs, info = env.reset(seed=100)
    terminated = False
    step_count = 0

    while not terminated and step_count < 40:
        action = policy.select_action(obs, info=info)
        assert action in (ACTION_OPERATE_NORMALLY, ACTION_EMERGENCY_MAJOR_MAINTENANCE)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    assert step_count == 40
    assert info["current_step"] == 40
    assert policy.total_steps == 40


def test_pm_policy_simulation_in_gym_env():
    """
    Run the PeriodicMaintenancePolicy with maintenance_interval=10 in CompressorMaintenanceEnv
    over 50 steps. Verify that Action 2 is executed exactly at steps 10, 20, 30, 40, 50.
    """
    env = CompressorMaintenanceEnv(max_episode_steps=50, seed=200)
    policy = PeriodicMaintenancePolicy(maintenance_interval=10)

    obs, info = env.reset(seed=200)
    actions_recorded = []

    for _ in range(50):
        action = policy.select_action(obs, info=info)
        actions_recorded.append(action)
        obs, reward, terminated, truncated, info = env.step(action)

    assert len(actions_recorded) == 50
    # Scheduled services triggered at indices 9, 19, 29, 39, 49 (steps 10, 20, 30, 40, 50)
    for idx, act in enumerate(actions_recorded):
        step_number = idx + 1
        if step_number % 10 == 0:
            assert act == ACTION_SCHEDULE_MEDIUM_MAINTENANCE, f"Step {step_number} must be Action 2"
        else:
            assert act == ACTION_OPERATE_NORMALLY, f"Step {step_number} must be Action 0"

    assert policy.total_scheduled_services == 5
    assert info["current_step"] == 50
