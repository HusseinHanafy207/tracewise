"""DQN"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DQNConfig:
    hidden_dims: Tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_size: int = 2_000
    target_sync_steps: int = 500
    gradient_clip: float = 10.0
    reward_scale: float = 100.0
    double_dqn: bool = False

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive sizes")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if not self.batch_size <= self.min_replay_size <= self.replay_capacity:
            raise ValueError(
                "min_replay_size must be between batch_size and replay_capacity"
            )
        if self.target_sync_steps <= 0:
            raise ValueError("target_sync_steps must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")


class ReplayBuffer:
    """Fixed-size replay buffer backed by contiguous NumPy arrays."""

    def __init__(self, capacity: int, observation_dim: int, seed: int = 42):
        if capacity <= 0 or observation_dim <= 0:
            raise ValueError("capacity and observation_dim must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.observation_dim,) or next_state.shape != (
            self.observation_dim,
        ):
            raise ValueError("state and next_state must match observation_dim")
        i = self._position
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than are stored")
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, n_actions: int, hidden_dims: Sequence[int]):
        super().__init__()
        dims = [observation_dim, *hidden_dims, n_actions]
        layers = []
        for input_dim, output_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def linear_epsilon(
    step: int, start: float, end: float, decay_steps: int
) -> float:
    if decay_steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(decay_steps), 0.0), 1.0)
    return float(start + fraction * (end - start))


class DQNAgent:
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        config: DQNConfig,
        device: torch.device,
        seed: int = 42,
    ):
        if observation_dim <= 0 or n_actions <= 0:
            raise ValueError("observation_dim and n_actions must be positive")
        self.observation_dim = int(observation_dim)
        self.n_actions = int(n_actions)
        self.config = config
        self.device = device
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self.online = QNetwork(observation_dim, n_actions, config.hidden_dims).to(device)
        self.target = QNetwork(observation_dim, n_actions, config.hidden_dims).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(config.replay_capacity, observation_dim, seed)
        self.learn_steps = 0

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.rand() < epsilon:
            return int(self.rng.randint(0, self.n_actions))
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            action = int(self.online(tensor).argmax(dim=1).item())
        return action

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[dict]:
        if len(self.replay) < self.config.min_replay_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device).long()
        rewards = torch.as_tensor(batch["rewards"], device=self.device)
        next_states = torch.as_tensor(batch["next_states"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)

        self.online.train()
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_states).max(dim=1).values
            targets = (
                rewards * self.config.reward_scale
                + self.config.gamma * (1.0 - dones) * next_q
            )

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.config.target_sync_steps == 0:
            self.sync_target()

        return {
            "loss": float(loss.detach().cpu()),
            "q_mean": float(q_values.detach().mean().cpu()),
            "target_mean": float(targets.detach().mean().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_dim": self.observation_dim,
                "n_actions": self.n_actions,
                "config": asdict(self.config),
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device, load_optimizer: bool = False):
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        config_values = dict(payload["config"])
        config_values["hidden_dims"] = tuple(config_values["hidden_dims"])
        agent = cls(
            observation_dim=int(payload["observation_dim"]),
            n_actions=int(payload["n_actions"]),
            config=DQNConfig(**config_values),
            device=device,
            seed=0,
        )
        agent.online.load_state_dict(payload["online_state_dict"])
        agent.target.load_state_dict(payload["target_state_dict"])
        if load_optimizer:
            agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.learn_steps = int(payload.get("learn_steps", 0))
        agent.online.eval()
        agent.target.eval()
        return agent, payload.get("extra", {})
