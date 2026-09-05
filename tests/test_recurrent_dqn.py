import numpy as np
import pytest
import torch

from src.rl.recurrent_dqn import (
    EpisodeReplayBuffer,
    RecurrentDQNAgent,
    RecurrentDQNConfig,
    RecurrentQNetwork,
)


def episode(value, horizon=3, observation_dim=2):
    observations = np.full(
        (horizon + 1, observation_dim), value, dtype=np.float32
    )
    actions = np.arange(horizon, dtype=np.int64) % 2
    rewards = np.full(horizon, value / 10.0, dtype=np.float32)
    dones = np.zeros(horizon, dtype=np.float32)
    dones[-1] = 1.0
    return observations, actions, rewards, dones


def test_episode_replay_overwrites_oldest_and_samples_full_sequences():
    replay = EpisodeReplayBuffer(2, horizon=3, observation_dim=2, seed=7)
    for value in (1.0, 2.0, 3.0):
        replay.add(*episode(value))

    batch = replay.sample(2)

    assert len(replay) == 2
    assert batch["observations"].shape == (2, 4, 2)
    assert set(batch["observations"][:, 0, 0].tolist()) == {2.0, 3.0}
    assert batch["actions"].shape == (2, 3)


def test_episode_replay_requires_one_terminal_transition_at_the_end():
    replay = EpisodeReplayBuffer(2, horizon=3, observation_dim=2)
    observations, actions, rewards, dones = episode(1.0)
    dones[:] = 0.0

    with pytest.raises(ValueError, match="final transition"):
        replay.add(observations, actions, rewards, dones)


def test_recurrent_network_is_causal_and_has_expected_shapes():
    network = RecurrentQNetwork(2, 3, recurrent_hidden_dim=4, head_hidden_dims=(5,))
    first = torch.zeros(1, 4, 2)
    second = first.clone()
    second[:, -1] = 10.0

    first_q, first_hidden = network(first)
    second_q, _ = network(second)

    assert first_q.shape == (1, 4, 3)
    assert first_hidden.shape == (1, 1, 4)
    assert torch.equal(first_q[:, :-1], second_q[:, :-1])


def test_recurrent_learning_changes_parameters_and_syncs_target():
    config = RecurrentDQNConfig(
        recurrent_hidden_dim=4,
        head_hidden_dims=(4,),
        episode_batch_size=2,
        replay_capacity_episodes=4,
        min_replay_episodes=2,
        target_sync_updates=1,
    )
    agent = RecurrentDQNAgent(2, 2, 3, config, torch.device("cpu"), seed=5)
    agent.store_episode(*episode(1.0))
    agent.store_episode(*episode(2.0))
    before = [parameter.detach().clone() for parameter in agent.online.parameters()]

    metrics = agent.learn()

    assert metrics is not None and np.isfinite(metrics["loss"])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.online.parameters())
    )
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(online, target)


@pytest.mark.parametrize(
    ("double_dqn", "expected_target"),
    [(False, 10.0), (True, 1.0)],
)
def test_recurrent_double_dqn_separates_selection_and_evaluation(
    double_dqn,
    expected_target,
):
    config = RecurrentDQNConfig(
        recurrent_hidden_dim=2,
        head_hidden_dims=(),
        gamma=1.0,
        learning_rate=1e-3,
        episode_batch_size=2,
        replay_capacity_episodes=2,
        min_replay_episodes=2,
        target_sync_updates=100,
        reward_scale=1.0,
        double_dqn=double_dqn,
    )
    agent = RecurrentDQNAgent(1, 2, 2, config, torch.device("cpu"), seed=3)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.online.head[-1].bias.copy_(torch.tensor([2.0, 0.0]))
        agent.target.head[-1].bias.copy_(torch.tensor([1.0, 10.0]))
    observations = np.zeros((3, 1), dtype=np.float32)
    actions = np.asarray([0, 0], dtype=np.int64)
    rewards = np.asarray([0.0, expected_target], dtype=np.float32)
    dones = np.asarray([0.0, 1.0], dtype=np.float32)
    for _ in range(2):
        agent.store_episode(observations, actions, rewards, dones)

    metrics = agent.learn()

    assert metrics is not None
    assert metrics["target_mean"] == pytest.approx(expected_target)


def test_recurrent_checkpoint_round_trip_preserves_greedy_sequence(tmp_path):
    config = RecurrentDQNConfig(
        recurrent_hidden_dim=4,
        head_hidden_dims=(4,),
        episode_batch_size=2,
        replay_capacity_episodes=4,
        min_replay_episodes=2,
    )
    agent = RecurrentDQNAgent(2, 2, 3, config, torch.device("cpu"), seed=11)
    sequence = [
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.asarray([0.3, 0.4], dtype=np.float32),
    ]
    agent.reset_hidden()
    expected = [agent.select_action(item) for item in sequence]
    path = tmp_path / "recurrent.pt"
    agent.save(path, extra={"marker": "round-trip"})

    loaded, extra = RecurrentDQNAgent.load(path, torch.device("cpu"))
    actual = [loaded.select_action(item) for item in sequence]

    assert actual == expected
    assert extra == {"marker": "round-trip"}
