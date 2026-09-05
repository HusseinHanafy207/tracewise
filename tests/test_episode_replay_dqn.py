import numpy as np
import pytest
import torch

from scripts.train_dqn import environment_config
from scripts.train_episode_replay_dqn import episode_replay_agent_config
from src.rl.environment import TutoringEnv
from src.rl.episode_replay_dqn import (
    EpisodeReplayDQNAgent,
    EpisodeReplayDQNConfig,
)
from src.utils.seed import load_config


def make_episode(horizon: int, observation_dim: int, action: int = 0):
    observations = np.arange(
        (horizon + 1) * observation_dim, dtype=np.float32
    ).reshape(horizon + 1, observation_dim)
    actions = np.full(horizon, action, dtype=np.int64)
    rewards = np.linspace(0.0, 0.1, horizon, dtype=np.float32)
    dones = np.zeros(horizon, dtype=np.float32)
    dones[-1] = 1.0
    return observations, actions, rewards, dones


def test_episode_replay_config_changes_only_replay_scheme():
    reference = load_config("configs/double_dqn_train_5000.yaml")
    control = load_config("configs/episode_replay_double_dqn_train.yaml")
    replay_only = control.pop("episode_replay_dqn")

    assert control == reference
    assert replay_only == {
        "episode_batch_size": 3,
        "replay_capacity_episodes": 2000,
        "min_replay_episodes": 40,
    }


def test_episode_replay_control_preserves_original_mlp_size():
    config = load_config("configs/episode_replay_double_dqn_train.yaml")
    environment = TutoringEnv(environment_config(config, False))
    agent = EpisodeReplayDQNAgent(
        environment.observation_dim,
        environment.n_actions,
        environment.config.horizon,
        episode_replay_agent_config(config),
        torch.device("cpu"),
    )

    assert agent.parameter_count == 19206
    assert agent.config.double_dqn is True
    assert agent.config.episode_batch_size * environment.config.horizon == 150


def test_episode_replay_feedforward_loss_uses_every_sampled_transition():
    config = EpisodeReplayDQNConfig(
        hidden_dims=(8,),
        episode_batch_size=2,
        replay_capacity_episodes=4,
        min_replay_episodes=2,
        target_sync_updates=10,
    )
    agent = EpisodeReplayDQNAgent(3, 2, 4, config, torch.device("cpu"))
    agent.store_episode(*make_episode(4, 3, action=0))
    agent.store_episode(*make_episode(4, 3, action=1))

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["loss_transitions"] == 8
    assert agent.learn_steps == 1


def test_episode_replay_feedforward_uses_double_dqn_action_selection():
    config = EpisodeReplayDQNConfig(
        hidden_dims=(1,),
        gamma=1.0,
        reward_scale=1.0,
        episode_batch_size=1,
        replay_capacity_episodes=1,
        min_replay_episodes=1,
        target_sync_updates=100,
    )
    agent = EpisodeReplayDQNAgent(1, 2, 2, config, torch.device("cpu"))
    with torch.no_grad():
        online_layers = [m for m in agent.online.modules() if isinstance(m, torch.nn.Linear)]
        target_layers = [m for m in agent.target.modules() if isinstance(m, torch.nn.Linear)]
        for layer in online_layers + target_layers:
            layer.weight.zero_()
            layer.bias.zero_()
        online_layers[-1].bias.copy_(torch.tensor([3.0, 1.0]))
        target_layers[-1].bias.copy_(torch.tensor([2.0, 9.0]))
    episode = (
        np.zeros((3, 1), dtype=np.float32),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )
    agent.store_episode(*episode)

    metrics = agent.learn()

    assert metrics["target_mean"] == pytest.approx(2.0)


def test_episode_replay_checkpoint_round_trip(tmp_path):
    config = EpisodeReplayDQNConfig(
        hidden_dims=(8,),
        episode_batch_size=1,
        replay_capacity_episodes=2,
        min_replay_episodes=1,
    )
    agent = EpisodeReplayDQNAgent(3, 2, 2, config, torch.device("cpu"), seed=9)
    path = tmp_path / "episode_replay.pt"
    agent.save(path, {"episode": 12})

    loaded, extra = EpisodeReplayDQNAgent.load(path, torch.device("cpu"))

    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    assert loaded.select_action(observation) == agent.select_action(observation)
    assert loaded.parameter_count == agent.parameter_count
    assert extra == {"episode": 12}
