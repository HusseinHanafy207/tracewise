"""Feed-forward Double DQN trained from complete-episode replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.rl.dqn import QNetwork
from src.rl.recurrent_dqn import EpisodeReplayBuffer


@dataclass(frozen=True)
class EpisodeReplayDQNConfig:
    hidden_dims: Tuple[int, ...] = (128, 128)
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
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive sizes")
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


class EpisodeReplayDQNAgent:
    """An MLP DQN whose loss batches contain complete sampled episodes."""

    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        horizon: int,
        config: EpisodeReplayDQNConfig,
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

        self.online = QNetwork(
            observation_dim, n_actions, config.hidden_dims
        ).to(device)
        self.target = QNetwork(
            observation_dim, n_actions, config.hidden_dims
        ).to(device)
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

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.online.parameters())

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError("observation must match observation_dim")
        if self.rng.rand() < epsilon:
            return int(self.rng.randint(0, self.n_actions))
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            return int(self.online(tensor).argmax(dim=1).item())

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

        states = observations[:, :-1].reshape(-1, self.observation_dim)
        next_states = observations[:, 1:].reshape(-1, self.observation_dim)
        flat_actions = actions.reshape(-1)
        flat_rewards = rewards.reshape(-1)
        flat_dones = dones.reshape(-1)

        self.online.train()
        q_values = self.online(states).gather(
            1, flat_actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(
                    1, next_actions
                ).squeeze(1)
            else:
                next_q = self.target(next_states).max(dim=1).values
            targets = (
                flat_rewards * self.config.reward_scale
                + self.config.gamma * (1.0 - flat_dones) * next_q
            )

        loss = F.smooth_l1_loss(q_values, targets)
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
            "q_mean": float(q_values.detach().mean().cpu()),
            "target_mean": float(targets.detach().mean().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "loss_transitions": int(q_values.numel()),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": "feedforward_episode_replay",
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
        if payload.get("architecture") != "feedforward_episode_replay":
            raise ValueError("checkpoint is not an episode-replay feed-forward DQN")
        config_values = dict(payload["config"])
        config_values["hidden_dims"] = tuple(config_values["hidden_dims"])
        agent = cls(
            observation_dim=int(payload["observation_dim"]),
            n_actions=int(payload["n_actions"]),
            horizon=int(payload["horizon"]),
            config=EpisodeReplayDQNConfig(**config_values),
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
