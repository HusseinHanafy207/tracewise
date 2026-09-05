"""Episode-replay recurrent Double DQN for partially observable environments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RecurrentDQNConfig:
    recurrent_hidden_dim: int = 64
    head_hidden_dims: Tuple[int, ...] = (64,)
    gamma: float = 0.99
    learning_rate: float = 5e-4
    episode_batch_size: int = 3
    replay_capacity_episodes: int = 2_000
    min_replay_episodes: int = 40
    target_sync_updates: int = 500
    gradient_clip: float = 10.0
    reward_scale: float = 100.0
    double_dqn: bool = True

    def __post_init__(self) -> None:
        if self.recurrent_hidden_dim <= 0:
            raise ValueError("recurrent_hidden_dim must be positive")
        if any(dim <= 0 for dim in self.head_hidden_dims):
            raise ValueError("head_hidden_dims must contain positive sizes")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.episode_batch_size <= 0:
            raise ValueError("episode_batch_size must be positive")
        if self.replay_capacity_episodes < self.episode_batch_size:
            raise ValueError("replay capacity must be at least episode_batch_size")
        if not (
            self.episode_batch_size
            <= self.min_replay_episodes
            <= self.replay_capacity_episodes
        ):
            raise ValueError(
                "min_replay_episodes must be between batch size and capacity"
            )
        if self.target_sync_updates <= 0:
            raise ValueError("target_sync_updates must be positive")
        if self.gradient_clip <= 0 or self.reward_scale <= 0:
            raise ValueError("gradient_clip and reward_scale must be positive")


class EpisodeReplayBuffer:
    """Fixed-size replay of complete, fixed-horizon episodes."""

    def __init__(
        self,
        capacity_episodes: int,
        horizon: int,
        observation_dim: int,
        seed: int = 42,
    ):
        if capacity_episodes <= 0 or horizon <= 0 or observation_dim <= 0:
            raise ValueError("capacity, horizon, and observation_dim must be positive")
        self.capacity_episodes = int(capacity_episodes)
        self.horizon = int(horizon)
        self.observation_dim = int(observation_dim)
        self.observations = np.zeros(
            (capacity_episodes, horizon + 1, observation_dim), dtype=np.float32
        )
        self.actions = np.zeros((capacity_episodes, horizon), dtype=np.int64)
        self.rewards = np.zeros((capacity_episodes, horizon), dtype=np.float32)
        self.dones = np.zeros((capacity_episodes, horizon), dtype=np.float32)
        self._position = 0
        self._size = 0
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        if observations.shape != (self.horizon + 1, self.observation_dim):
            raise ValueError("observations must have shape (horizon + 1, observation_dim)")
        if any(array.shape != (self.horizon,) for array in (actions, rewards, dones)):
            raise ValueError("actions, rewards, and dones must have shape (horizon,)")
        if not bool(dones[-1]) or bool(np.any(dones[:-1])):
            raise ValueError("only the final transition may terminate an episode")

        index = self._position
        self.observations[index] = observations
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.dones[index] = dones
        self._position = (self._position + 1) % self.capacity_episodes
        self._size = min(self._size + 1, self.capacity_episodes)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size > self._size:
            raise ValueError("cannot sample more episodes than are stored")
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }


class RecurrentQNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        recurrent_hidden_dim: int,
        head_hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=observation_dim,
            hidden_size=recurrent_hidden_dim,
            batch_first=True,
        )
        dims = [recurrent_hidden_dim, *head_hidden_dims, n_actions]
        layers = []
        for input_dim, output_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 3:
            raise ValueError("observations must have shape (batch, sequence, features)")
        features, next_hidden = self.gru(observations, hidden)
        return self.head(features), next_hidden


class RecurrentDQNAgent:
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        horizon: int,
        config: RecurrentDQNConfig,
        device: torch.device,
        seed: int = 42,
    ):
        if observation_dim <= 0 or n_actions <= 0 or horizon <= 0:
            raise ValueError("observation_dim, n_actions, and horizon must be positive")
        self.observation_dim = int(observation_dim)
        self.n_actions = int(n_actions)
        self.horizon = int(horizon)
        self.config = config
        self.device = device
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        network_args = (
            observation_dim,
            n_actions,
            config.recurrent_hidden_dim,
            config.head_hidden_dims,
        )
        self.online = RecurrentQNetwork(*network_args).to(device)
        self.target = RecurrentQNetwork(*network_args).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = EpisodeReplayBuffer(
            config.replay_capacity_episodes,
            horizon,
            observation_dim,
            seed,
        )
        self.learn_steps = 0
        self._hidden: Optional[torch.Tensor] = None

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.online.parameters())

    def reset_hidden(self) -> None:
        self._hidden = None

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError("observation must match observation_dim")
        explore = self.rng.rand() < epsilon
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        self.online.eval()
        with torch.no_grad():
            q_values, self._hidden = self.online(tensor, self._hidden)
            self._hidden = self._hidden.detach()
        if explore:
            return int(self.rng.randint(0, self.n_actions))
        return int(q_values[0, 0].argmax().item())

    def store_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.replay.add(observations, actions, rewards, dones)

    def learn(self) -> Optional[dict]:
        if len(self.replay) < self.config.min_replay_episodes:
            return None
        batch = self.replay.sample(self.config.episode_batch_size)
        observations = torch.as_tensor(batch["observations"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device).long()
        rewards = torch.as_tensor(batch["rewards"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)

        self.online.train()
        online_sequence, _ = self.online(observations)
        current_q = online_sequence[:, :-1].gather(
            2, actions.unsqueeze(2)
        ).squeeze(2)
        with torch.no_grad():
            target_sequence, _ = self.target(observations)
            if self.config.double_dqn:
                next_actions = online_sequence[:, 1:].argmax(dim=2, keepdim=True)
                next_q = target_sequence[:, 1:].gather(2, next_actions).squeeze(2)
            else:
                next_q = target_sequence[:, 1:].max(dim=2).values
            targets = (
                rewards * self.config.reward_scale
                + self.config.gamma * (1.0 - dones) * next_q
            )

        loss = F.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.config.target_sync_updates == 0:
            self.sync_target()

        return {
            "loss": float(loss.detach().cpu()),
            "q_mean": float(current_q.detach().mean().cpu()),
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
                "architecture": "gru",
                "observation_dim": self.observation_dim,
                "n_actions": self.n_actions,
                "horizon": self.horizon,
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
        if payload.get("architecture") != "gru":
            raise ValueError("checkpoint is not a GRU DQN")
        config_values = dict(payload["config"])
        config_values["head_hidden_dims"] = tuple(config_values["head_hidden_dims"])
        agent = cls(
            observation_dim=int(payload["observation_dim"]),
            n_actions=int(payload["n_actions"]),
            horizon=int(payload["horizon"]),
            config=RecurrentDQNConfig(**config_values),
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
        agent.reset_hidden()
        return agent, payload.get("extra", {})
