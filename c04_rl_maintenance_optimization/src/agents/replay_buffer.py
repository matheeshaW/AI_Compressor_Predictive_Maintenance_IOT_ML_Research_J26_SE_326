"""
replay_buffer.py
================
Experience Replay Buffer implementation for Deep Q-Network (DQN) training.
Component 04: Reinforcement Learning-Based Maintenance Optimization
Researcher: Liyanage H.G.W.R (IT23163690)
Project: SLIIT Research Project J26-SE-326 (AIR-GUARD AI)
"""

from collections import deque, namedtuple
from typing import Tuple, List, Optional, Union, Any
import random
import numpy as np
import torch

# Standard transition tuple structure
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Fixed-capacity Experience Replay Buffer for storing and sampling transitions.
    
    Breaks temporal autocorrelation in sequential telemetry by uniformly sampling
    mini-batches of (state, action, reward, next_state, done) tuples for off-policy
    DQN updates according to the Bellman optimality equation.
    """

    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store. Older transitions
                      are automatically evicted (FIFO) when capacity is exceeded.
            seed: Optional random seed for reproducible sampling.
        """
        if capacity <= 0:
            raise ValueError(f"ReplayBuffer capacity must be positive, got {capacity}")

        self.capacity = int(capacity)
        self.buffer: deque = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def push(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        action: Union[int, np.integer, torch.Tensor],
        reward: Union[float, int, np.floating],
        next_state: Union[np.ndarray, List[float], torch.Tensor],
        done: Union[bool, int, float],
    ) -> None:
        """
        Stores an experience transition tuple (s, a, r, s', d) in the buffer.

        Args:
            state: Observation vector at current timestep s_t.
            action: Discrete action executed a_t in {0, 1, 2, 3}.
            reward: Scalar multi-objective reward r_t received.
            next_state: Observation vector at next timestep s_{t+1}.
            done: Termination/truncation boolean flag indicating episode boundary.
        """
        # Ensure state arrays are float32 1D numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state_arr = np.asarray(state, dtype=np.float32).ravel()

        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()
        next_state_arr = np.asarray(next_state, dtype=np.float32).ravel()

        act = int(action.item() if isinstance(action, (torch.Tensor, np.generic)) else action)
        rew = float(reward.item() if isinstance(reward, (torch.Tensor, np.generic)) else reward)
        d = bool(done.item() if isinstance(done, (torch.Tensor, np.generic)) else done)

        transition = Transition(
            state=state_arr,
            action=act,
            reward=rew,
            next_state=next_state_arr,
            done=d,
        )
        self.buffer.append(transition)

    def add(
        self,
        state: Union[np.ndarray, List[float], torch.Tensor],
        action: Union[int, np.integer, torch.Tensor],
        reward: Union[float, int, np.floating],
        next_state: Union[np.ndarray, List[float], torch.Tensor],
        done: Union[bool, int, float],
    ) -> None:
        """Convenience alias for push()."""
        self.push(state, action, reward, next_state, done)

    def sample(
        self,
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
        to_tensor: bool = True,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Uniformly samples a random mini-batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample.
            device: Optional torch device (e.g. 'cpu', 'cuda') if to_tensor is True.
            to_tensor: If True, returns PyTorch tensors; if False, returns numpy arrays.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones):
                - states: shape (batch_size, state_dim)
                - actions: shape (batch_size, 1)
                - rewards: shape (batch_size, 1)
                - next_states: shape (batch_size, state_dim)
                - dones: shape (batch_size, 1)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} items from buffer with only {len(self.buffer)} items."
            )

        transitions = self.rng.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))

        states_np = np.stack(batch.state, axis=0).astype(np.float32)
        actions_np = np.array(batch.action, dtype=np.int64).reshape(-1, 1)
        rewards_np = np.array(batch.reward, dtype=np.float32).reshape(-1, 1)
        next_states_np = np.stack(batch.next_state, axis=0).astype(np.float32)
        dones_np = np.array(batch.done, dtype=np.float32).reshape(-1, 1)

        if not to_tensor:
            return states_np, actions_np, rewards_np, next_states_np, dones_np

        dev = torch.device(device) if device is not None else None
        states_t = torch.tensor(states_np, dtype=torch.float32, device=dev)
        actions_t = torch.tensor(actions_np, dtype=torch.long, device=dev)
        rewards_t = torch.tensor(rewards_np, dtype=torch.float32, device=dev)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32, device=dev)
        dones_t = torch.tensor(dones_np, dtype=torch.float32, device=dev)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def sample_numpy(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convenience method returning sampled mini-batch as raw numpy ndarrays."""
        return self.sample(batch_size=batch_size, to_tensor=False)

    def can_sample(self, batch_size: int) -> bool:
        """Returns True if the buffer contains at least batch_size transitions."""
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Empties all stored transitions from the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Returns the current number of transitions stored."""
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Transition:
        """Allows direct indexing into buffer for testing/inspection."""
        return self.buffer[idx]
