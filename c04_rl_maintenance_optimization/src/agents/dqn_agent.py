"""
dqn_agent.py
============
Deep Q-Network (DQN) Agent for Industrial Air Compressor Maintenance Optimization.
Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
"""

from typing import Tuple, List, Optional, Union
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """
    Deep Q-Network multi-layer perceptron approximating Q*(s, a).
    
    Takes an 8-dimensional continuous observation vector modeling compressor health:
    [D_t, RUL_t, delta_t, eta_t, theta_t, k_t, tau_t, t_maint]
    and outputs estimated Q-values for the 4 discrete maintenance/operational actions:
    [0: OPERATE_NORMALLY, 1: MINOR_SERVICE, 2: MEDIUM_OVERHAUL, 3: MAJOR_REPLACEMENT].
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128),
        activation: str = "leaky_relu",
        use_layer_norm: bool = False,
    ):
        """
        Initialize the fully-connected Q-network.

        Args:
            state_dim: Dimension of the continuous state vector (default: 8).
            action_dim: Number of discrete maintenance actions (default: 4).
            hidden_dims: Tuple specifying sizes of hidden layers.
            activation: Activation function ('leaky_relu' or 'relu').
            use_layer_norm: Whether to apply LayerNorm across hidden representations.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        layers: List[nn.Module] = []
        prev_dim = state_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            
            if activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                layers.append(nn.ReLU())
            prev_dim = h_dim

        # Final linear projection layer to action Q-values
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        # Xavier/Kaiming initialization for stable gradient propagation
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights using orthogonal/kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass estimating action-values Q(s, ·).

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Q-values tensor of shape (batch_size, action_dim) or (action_dim,).
        """
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)

        q_vals = self.network(state)

        if is_single:
            return q_vals.squeeze(0)
        return q_vals


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Industrial Compressor Prescriptive Maintenance.

    Key algorithmic components:
    1. Fully-connected Q-Network approximating Q*(s, a) over 8 state features & 4 actions.
    2. Target Q-Network with periodic hard updates or soft Polyak updates to stabilize Bellman targets.
    3. Epsilon-greedy exploration policy with exponential decay schedule.
    4. Experience Replay Buffer breaking autocorrelation in sequential telemetry.
    5. Optimization using Mean Squared Error (MSE) loss against Bellman optimality targets.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        tau: float = 1.0,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        max_grad_norm: float = 10.0,
        activation: str = "leaky_relu",
        use_layer_norm: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        """
        Initialize the Deep Q-Network Agent.

        Args:
            state_dim: Dimensionality of state space (8).
            action_dim: Dimensionality of discrete action space (4).
            hidden_dims: Layer dimensions of the Q-network MLP.
            learning_rate: Adam optimizer learning rate.
            gamma: Temporal discount factor (0.99).
            epsilon_start: Initial exploration rate (1.0).
            epsilon_end: Minimum exploration rate (0.05).
            epsilon_decay: Multiplicative decay factor per episode/step.
            target_update_freq: Frequency (train steps) for periodic target network updates.
            tau: Polyak update factor (1.0 = hard update every target_update_freq steps;
                 < 1.0 = soft update on every train step).
            buffer_capacity: Maximum capacity of the Experience Replay Buffer.
            batch_size: Mini-batch size sampled for gradient descent.
            max_grad_norm: Maximum norm threshold for gradient clipping.
            activation: Activation function for QNetwork ('leaky_relu' or 'relu').
            use_layer_norm: Whether to enable LayerNorm in QNetwork.
            device: Computing device ('cpu', 'cuda', etc.).
            seed: Random seed for deterministic reproducibility.
            replay_buffer: Optional pre-existing ReplayBuffer instance.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # Exploration schedule parameters
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        # Set seeds for reproducibility
        if seed is not None:
            self._set_seed(seed)

        # Compute device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Primary policy network and stabilized target network
        self.policy_net = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
        ).to(self.device)

        # Initialize target network with exact identical weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer and MSE loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience Replay Buffer
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        # Counters and metrics
        self.train_step_count = 0
        self.total_episodes = 0

    def _set_seed(self, seed: int) -> None:
        """Sets random seed across all libraries for deterministic execution."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def select_action(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        evaluate: bool = False,
    ) -> int:
        """
        Selects an action using the epsilon-greedy policy.

        Args:
            state: Observation vector s_t (shape: (8,) or (1, 8)).
            evaluate: If True, executes pure greedy exploitation (epsilon = 0.0).

        Returns:
            Integer action index in {0, 1, 2, 3}.
        """
        # Exploration: choose random action with probability epsilon
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploitation: select greedy action argmax_a Q(s, a)
        return self._greedy_action(state)

    def act(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        evaluate: bool = False,
    ) -> int:
        """Convenience alias for select_action()."""
        return self.select_action(state=state, evaluate=evaluate)

    def __call__(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        evaluate: bool = False,
    ) -> int:
        """Enables callable policy interface agent(state)."""
        return self.select_action(state=state, evaluate=evaluate)

    def _greedy_action(self, state: Union[np.ndarray, List[float], torch.Tensor]) -> int:
        """Computes greedy action argmax_a Q(s, a)."""
        with torch.no_grad():
            state_tensor = self._to_state_tensor(state)
            q_values = self.policy_net(state_tensor)
            action = int(q_values.argmax(dim=-1).item())
            return action

    def get_q_values(
        self, state: Union[np.ndarray, List[float], torch.Tensor]
    ) -> np.ndarray:
        """
        Evaluates state and returns raw Q-values for all 4 discrete actions.

        Args:
            state: State vector s_t (shape: (8,) or (batch_size, 8)).

        Returns:
            Numpy array of shape (4,) for single state or (batch_size, 4) for batch.
        """
        is_single = (
            (isinstance(state, np.ndarray) and state.ndim == 1)
            or (isinstance(state, torch.Tensor) and state.dim() == 1)
            or (isinstance(state, (list, tuple)) and not isinstance(state[0], (list, tuple, np.ndarray)))
        )
        with torch.no_grad():
            state_tensor = self._to_state_tensor(state)
            q_values = self.policy_net(state_tensor)
            arr = q_values.cpu().numpy()
            if is_single:
                return arr.squeeze(0)
            return arr

    def _to_state_tensor(self, state: Union[np.ndarray, List[float], torch.Tensor]) -> torch.Tensor:
        """Normalizes state input into a float32 PyTorch tensor on the agent's device."""
        if isinstance(state, torch.Tensor):
            t = state.to(device=self.device, dtype=torch.float32)
        elif isinstance(state, np.ndarray):
            t = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)
        else:
            t = torch.tensor(state, device=self.device, dtype=torch.float32)

        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def decay_epsilon(self) -> float:
        """
        Applies exponential decay to the exploration rate epsilon.

        Returns:
            Updated epsilon value.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """Manually sets the exploration rate epsilon."""
        self.epsilon = float(np.clip(epsilon, 0.0, 1.0))

    def store_transition(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        action: Union[int, np.integer, torch.Tensor],
        reward: Union[float, int, np.floating],
        next_state: Union[np.ndarray, List[float], torch.Tensor],
        done: Union[bool, int, float],
    ) -> None:
        """Stores transition in experience replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def remember(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        action: Union[int, np.integer, torch.Tensor],
        reward: Union[float, int, np.floating],
        next_state: Union[np.ndarray, List[float], torch.Tensor],
        done: Union[bool, int, float],
    ) -> None:
        """Convenience alias for store_transition()."""
        self.store_transition(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Executes a single gradient descent training step.

        1. Samples a mini-batch of (s, a, r, s', d) from the Experience Replay Buffer.
        2. Computes current Q-values Q(s, a; theta) via the policy network.
        3. Computes target Q-values via the Bellman optimality equation:
           y = r + gamma * (1 - d) * max_a' Q(s', a'; theta^-)
        4. Calculates Mean Squared Error (MSE) loss:
           L(theta) = MSE(Q(s, a; theta), y)
        5. Backpropagates gradients, clips gradient norm, and updates policy network weights.
        6. Periodically synchronizes target network weights.

        Returns:
            Computed MSE loss as a float, or None if buffer has insufficient samples.
        """
        if not self.replay_buffer.can_sample(self.batch_size):
            return None

        # 1. Sample mini-batch from experience replay
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size=self.batch_size,
            device=self.device,
            to_tensor=True,
        )

        # 2. Compute current Q-values for selected actions Q(s, a)
        # policy_net(states) -> shape (batch_size, 4)
        # gather along action dimension -> shape (batch_size, 1)
        current_q_values = self.policy_net(states).gather(1, actions)

        # 3. Compute target Q-values using the target network (no gradient computation)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * (1.0 - dones) * max_next_q_values

        # 4. Calculate Mean Squared Error (MSE) loss
        loss = self.criterion(current_q_values, target_q_values)

        # 5. Optimize policy network
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # 6. Periodic target network update
        self.train_step_count += 1
        if self.tau < 1.0:
            # Soft Polyak update on every step
            self.soft_update_target(self.tau)
        elif self.train_step_count % self.target_update_freq == 0:
            # Hard periodic update
            self.hard_update_target()

        return float(loss.item())

    def hard_update_target(self) -> None:
        """Copies policy network weights directly to target network (theta^- <- theta)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self, tau: float) -> None:
        """
        Polyak soft target network update:
        theta^- <- tau * theta + (1 - tau) * theta^-
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self, tau: Optional[float] = None) -> None:
        """
        Updates target network weights.
        
        Args:
            tau: Optional update factor. If tau is 1.0 (or None when self.tau == 1.0),
                 performs hard copy. If tau < 1.0, performs Polyak soft update.
        """
        target_tau = tau if tau is not None else self.tau
        if target_tau >= 1.0:
            self.hard_update_target()
        else:
            self.soft_update_target(target_tau)

    def save(self, filepath: str) -> None:
        """
        Saves agent weights, optimizer state, and hyperparameters to disk.

        Args:
            filepath: Destination file path (e.g. 'checkpoint.pt').
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
            "total_episodes": self.total_episodes,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "hidden_dims": self.policy_net.hidden_dims,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str, map_location: Optional[Union[str, torch.device]] = None) -> None:
        """
        Restores agent weights, optimizer state, and hyperparameters from disk.

        Args:
            filepath: Source file path.
            map_location: Optional torch map_location.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        loc = map_location if map_location is not None else self.device
        checkpoint = torch.load(filepath, map_location=loc, weights_only=False)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "train_step_count" in checkpoint:
            self.train_step_count = checkpoint["train_step_count"]
        if "total_episodes" in checkpoint:
            self.total_episodes = checkpoint.get("total_episodes", 0)
