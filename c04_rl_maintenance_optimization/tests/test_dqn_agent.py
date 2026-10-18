"""
test_dqn_agent.py
=================
Comprehensive unit and integration test suite for:
- Experience Replay Buffer (ReplayBuffer)
- Deep Q-Network (QNetwork)
- Deep Q-Network Agent (DQNAgent)
- Gym Environment Training Loop (train_dqn)

Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
"""

import os
import tempfile
import pytest
import numpy as np
import torch

from c04_rl_maintenance_optimization.src.agents.replay_buffer import (
    ReplayBuffer,
)
from c04_rl_maintenance_optimization.src.agents.dqn_agent import (
    QNetwork,
    DQNAgent,
)
from c04_rl_maintenance_optimization.src.env.compressor_maintenance_env import (
    CompressorMaintenanceEnv,
)
from c04_rl_maintenance_optimization.src.train_dqn import train_dqn


# ---------------------------------------------------------------------------
# 1. Experience Replay Buffer Tests
# ---------------------------------------------------------------------------

def test_replay_buffer_initialization():
    """Verify buffer initializes with specified capacity and empty state."""
    buf = ReplayBuffer(capacity=1000, seed=42)
    assert len(buf) == 0
    assert buf.capacity == 1000
    assert not buf.can_sample(10)

    with pytest.raises(ValueError, match="capacity must be positive"):
        ReplayBuffer(capacity=0)

    with pytest.raises(ValueError, match="capacity must be positive"):
        ReplayBuffer(capacity=-5)


def test_replay_buffer_push_and_sample():
    """Verify storing and sampling transitions returns correct shapes and types."""
    buf = ReplayBuffer(capacity=50, seed=123)

    state = np.array([0.1, 0.9, 1.0, 0.95, 1.1, 1.0, 0.0, 0.1], dtype=np.float32)
    action = 0
    reward = 50.0
    next_state = np.array([0.12, 0.88, 1.05, 0.94, 1.12, 1.0, 0.0, 0.15], dtype=np.float32)
    done = False

    # Push 20 transitions
    for _ in range(20):
        buf.push(state, action, reward, next_state, done)

    assert len(buf) == 20
    assert buf.can_sample(16)
    assert not buf.can_sample(25)

    # Sample PyTorch tensors
    states_t, actions_t, rewards_t, next_states_t, dones_t = buf.sample(batch_size=16)
    assert isinstance(states_t, torch.Tensor)
    assert states_t.shape == (16, 8)
    assert actions_t.shape == (16, 1)
    assert rewards_t.shape == (16, 1)
    assert next_states_t.shape == (16, 8)
    assert dones_t.shape == (16, 1)
    assert states_t.dtype == torch.float32
    assert actions_t.dtype == torch.long
    assert rewards_t.dtype == torch.float32
    assert dones_t.dtype == torch.float32

    # Sample numpy arrays
    states_np, actions_np, rewards_np, next_states_np, dones_np = buf.sample_numpy(batch_size=8)
    assert isinstance(states_np, np.ndarray)
    assert states_np.shape == (8, 8)
    assert actions_np.shape == (8, 1)
    assert rewards_np.shape == (8, 1)
    assert next_states_np.shape == (8, 8)
    assert dones_np.shape == (8, 1)


def test_replay_buffer_fifo_eviction():
    """Verify FIFO eviction behavior when buffer reaches max capacity."""
    buf = ReplayBuffer(capacity=5, seed=42)

    for i in range(10):
        dummy_state = np.full(8, fill_value=float(i), dtype=np.float32)
        buf.push(dummy_state, action=i % 4, reward=float(i * 10), next_state=dummy_state, done=False)

    assert len(buf) == 5
    # The buffer should hold the last 5 inserted transitions (i = 5, 6, 7, 8, 9)
    stored_states = [buf[idx].state[0] for idx in range(len(buf))]
    assert stored_states == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_replay_buffer_error_on_insufficient_samples():
    """Verify ValueError is raised if sample size exceeds buffer occupancy."""
    buf = ReplayBuffer(capacity=100)
    for _ in range(5):
        buf.push(np.zeros(8), 0, 1.0, np.zeros(8), False)

    with pytest.raises(ValueError, match="Cannot sample 10 items"):
        buf.sample(10)


def test_replay_buffer_clear():
    """Verify clear() empties all stored items."""
    buf = ReplayBuffer(capacity=100)
    buf.push(np.zeros(8), 0, 1.0, np.zeros(8), False)
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# 2. Q-Network Architecture Tests
# ---------------------------------------------------------------------------

def test_q_network_initialization_and_forward():
    """Verify QNetwork forward pass handles single and batched inputs."""
    net = QNetwork(
        state_dim=8,
        action_dim=4,
        hidden_dims=(64, 64),
        activation="leaky_relu",
        use_layer_norm=True,
    )

    # 1D single-state tensor input
    single_state = torch.randn(8)
    q_single = net(single_state)
    assert q_single.shape == (4,)
    assert not torch.isnan(q_single).any()

    # 2D batched tensor input
    batch_states = torch.randn(32, 8)
    q_batch = net(batch_states)
    assert q_batch.shape == (32, 4)
    assert not torch.isnan(q_batch).any()


def test_q_network_relu_activation():
    """Verify QNetwork supports standard ReLU activation."""
    net = QNetwork(state_dim=8, action_dim=4, hidden_dims=(32,), activation="relu")
    out = net(torch.randn(10, 8))
    assert out.shape == (10, 4)


# ---------------------------------------------------------------------------
# 3. DQN Agent Mechanics Tests
# ---------------------------------------------------------------------------

def test_dqn_agent_initialization():
    """Verify DQNAgent initializes policy and target networks identically."""
    agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        hidden_dims=(64, 64),
        epsilon_start=1.0,
        epsilon_end=0.05,
        target_update_freq=50,
        batch_size=32,
        seed=42,
    )

    assert agent.state_dim == 8
    assert agent.action_dim == 4
    assert agent.epsilon == 1.0
    assert agent.train_step_count == 0

    # Policy and target net parameters should be numerically identical initially
    for p_pol, p_tgt in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_pol, p_tgt)
        assert not p_tgt.requires_grad  # Target net should be frozen


def test_dqn_agent_epsilon_greedy_selection():
    """Verify epsilon-greedy action selection during exploration and exploitation."""
    agent = DQNAgent(state_dim=8, action_dim=4, epsilon_start=1.0, seed=42)
    state = np.array([0.2, 0.8, 1.1, 0.9, 1.0, 1.0, 0.0, 0.2], dtype=np.float32)

    # Exploration (epsilon = 1.0): actions should cover all options
    actions_explored = [agent.select_action(state, evaluate=False) for _ in range(100)]
    for a in range(4):
        assert a in actions_explored
        assert 0 <= actions_explored[0] <= 3

    # Pure Exploitation (evaluate = True): action should be purely deterministic
    greedy_a1 = agent.select_action(state, evaluate=True)
    greedy_a2 = agent.select_action(state, evaluate=True)
    assert greedy_a1 == greedy_a2

    # Calling via act() and __call__()
    assert agent.act(state, evaluate=True) == greedy_a1
    assert agent(state, evaluate=True) == greedy_a1


def test_dqn_agent_epsilon_decay():
    """Verify epsilon decays correctly towards epsilon_end."""
    agent = DQNAgent(
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay=0.5,
    )
    assert agent.epsilon == 1.0
    new_eps = agent.decay_epsilon()
    assert new_eps == 0.5
    agent.decay_epsilon()  # 0.25
    agent.decay_epsilon()  # 0.125
    agent.decay_epsilon()  # clamped to 0.10
    assert agent.epsilon == 0.10
    agent.decay_epsilon()
    assert agent.epsilon == 0.10


def test_dqn_agent_target_network_updates():
    """Verify hard and soft target network update mechanisms."""
    agent = DQNAgent(state_dim=8, action_dim=4, seed=42)

    # Artificially alter policy network weights
    with torch.no_grad():
        for param in agent.policy_net.parameters():
            param.add_(10.0)

    # Prior to update, target network should NOT match modified policy net
    targets_match_before = any(
        torch.allclose(p_pol, p_tgt)
        for p_pol, p_tgt in zip(agent.policy_net.parameters(), agent.target_net.parameters())
    )
    assert not targets_match_before

    # 1. Test Hard Update: target network should now match policy network exactly
    agent.hard_update_target()
    for p_pol, p_tgt in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_pol, p_tgt)

    # 2. Test Soft Update: target network should interpolate smoothly
    with torch.no_grad():
        for param in agent.policy_net.parameters():
            param.add_(2.0)

    prev_target_val = list(agent.target_net.parameters())[0][0, 0].item()
    policy_val = list(agent.policy_net.parameters())[0][0, 0].item()

    tau = 0.5
    agent.soft_update_target(tau=tau)
    new_target_val = list(agent.target_net.parameters())[0][0, 0].item()

    expected_val = tau * policy_val + (1.0 - tau) * prev_target_val
    assert np.isclose(new_target_val, expected_val, atol=1e-5)


def test_dqn_agent_get_q_values():
    """Verify get_q_values returns 4 Q-values for any valid state."""
    agent = DQNAgent(state_dim=8, action_dim=4)
    state = np.array([0.5, 0.5, 2.0, 0.8, 1.5, 2.0, 0.1, 0.3], dtype=np.float32)

    q_vals = agent.get_q_values(state)
    assert isinstance(q_vals, np.ndarray)
    assert q_vals.shape == (4,)
    assert not np.isnan(q_vals).any()


def test_dqn_agent_train_step_loss():
    """Verify train_step() computes MSE loss and updates policy network weights."""
    agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        batch_size=16,
        learning_rate=1e-2,
        seed=42,
    )

    # When buffer is below batch size, train_step() returns None
    assert agent.train_step() is None

    # Populate buffer with 32 transitions
    for i in range(32):
        s = np.random.uniform(0.0, 1.0, size=8).astype(np.float32)
        a = i % 4
        r = float(np.random.uniform(-100.0, 100.0))
        s_next = np.random.uniform(0.0, 1.0, size=8).astype(np.float32)
        d = False
        agent.store_transition(s, a, r, s_next, d)

    # Record initial weights of first policy layer
    initial_weight = list(agent.policy_net.parameters())[0].clone().detach()

    # Execute train_step()
    loss = agent.train_step()
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert agent.train_step_count == 1

    # Verify weights actually changed due to backpropagation
    updated_weight = list(agent.policy_net.parameters())[0].clone().detach()
    assert not torch.allclose(initial_weight, updated_weight)


def test_dqn_agent_save_and_load():
    """Verify agent checkpoints can be saved and restored identically."""
    agent = DQNAgent(state_dim=8, action_dim=4, seed=42)
    state = np.array([0.3, 0.7, 1.5, 0.9, 1.2, 1.0, 0.0, 0.4], dtype=np.float32)

    q_before = agent.get_q_values(state)
    agent.epsilon = 0.42
    agent.train_step_count = 137

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dqn_test.pt")
        agent.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        new_agent = DQNAgent(state_dim=8, action_dim=4)
        new_agent.load(ckpt_path)

        assert new_agent.epsilon == 0.42
        assert new_agent.train_step_count == 137
        q_after = new_agent.get_q_values(state)
        assert np.allclose(q_before, q_after, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Environment Integration & Training Loop Tests
# ---------------------------------------------------------------------------

def test_dqn_agent_training_loop_with_env():
    """
    Integration test running the DQN agent inside CompressorMaintenanceEnv.
    Verifies that state observations, actions, rewards, and next states flow correctly
    into the replay buffer, train_step() executes successfully, and no crashes occur.
    """
    env = CompressorMaintenanceEnv(max_episode_steps=50, seed=100)
    agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        batch_size=16,
        buffer_capacity=1000,
        epsilon_start=0.5,
        target_update_freq=20,
        seed=100,
    )

    obs, info = env.reset(seed=100)
    assert obs.shape == (8,)

    step_losses = []
    total_reward = 0.0

    for step in range(50):
        action = agent.select_action(obs, evaluate=False)
        assert action in [0, 1, 2, 3]

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = bool(terminated or truncated)

        agent.store_transition(obs, action, reward, next_obs, done)
        total_reward += reward
        obs = next_obs

        # Train after gathering minimum samples
        if len(agent.replay_buffer) >= agent.batch_size:
            loss = agent.train_step()
            if loss is not None:
                step_losses.append(loss)

        if done:
            break

    assert len(agent.replay_buffer) > 0
    assert len(step_losses) > 0
    assert all(not np.isnan(l) for l in step_losses)
    assert all(not np.isinf(l) for l in step_losses)
    assert agent.train_step_count > 0


def test_train_dqn_function_execution():
    """Verify train_dqn executes multi-episode training and returns valid metrics."""
    result = train_dqn(
        num_episodes=3,
        max_steps_per_episode=20,
        batch_size=16,
        warm_up_steps=16,
        train_every_steps=1,
        target_update_freq=10,
        verbose=False,
        seed=42,
    )

    assert len(result["episode_rewards"]) == 3
    assert len(result["episode_costs"]) == 3
    assert len(result["episode_downtimes"]) == 3
    assert len(result["episode_failures"]) == 3
    assert len(result["episode_losses"]) == 3
    assert result["total_steps"] > 0
    assert result["final_epsilon"] <= 1.0
    assert result["training_time_seconds"] > 0.0
    assert isinstance(result["agent"], DQNAgent)
