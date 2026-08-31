import numpy as np
import pytest
import torch

from src.rl.dqn import DQNAgent, DQNConfig, QNetwork, ReplayBuffer, linear_epsilon


def test_replay_buffer_overwrites_oldest_transitions_and_samples_shapes():
    replay = ReplayBuffer(capacity=3, observation_dim=2, seed=7)
    for i in range(5):
        state = np.asarray([i, i + 0.5], dtype=np.float32)
        replay.add(state, i % 2, i / 10.0, state + 1.0, i == 4)

    assert len(replay) == 3
    batch = replay.sample(3)
    assert batch["states"].shape == (3, 2)
    assert set(batch["states"][:, 0].tolist()) == {2.0, 3.0, 4.0}
    assert batch["actions"].dtype == np.int64


def test_q_network_and_epsilon_schedule():
    network = QNetwork(observation_dim=4, n_actions=3, hidden_dims=(8, 8))
    assert network(torch.zeros(2, 4)).shape == (2, 3)
    assert linear_epsilon(0, 1.0, 0.1, 100) == pytest.approx(1.0)
    assert linear_epsilon(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert linear_epsilon(200, 1.0, 0.1, 100) == pytest.approx(0.1)


def test_learning_step_changes_online_parameters_and_syncs_target():
    config = DQNConfig(
        hidden_dims=(8,),
        batch_size=4,
        replay_capacity=20,
        min_replay_size=4,
        target_sync_steps=1,
    )
    agent = DQNAgent(3, 2, config, torch.device("cpu"), seed=5)
    for i in range(8):
        state = np.asarray([i / 10.0, 0.2, 0.3], dtype=np.float32)
        agent.store(state, i % 2, 0.01 * i, state + 0.01, i % 4 == 3)

    before = [parameter.detach().clone() for parameter in agent.online.parameters()]
    metrics = agent.learn()
    after = list(agent.online.parameters())

    assert metrics is not None
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(online, target)


def test_checkpoint_round_trip_preserves_greedy_action(tmp_path):
    config = DQNConfig(
        hidden_dims=(8,),
        batch_size=2,
        replay_capacity=10,
        min_replay_size=2,
    )
    agent = DQNAgent(3, 2, config, torch.device("cpu"), seed=11)
    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    expected = agent.select_action(observation, epsilon=0.0)
    path = tmp_path / "agent.pt"
    agent.save(path, extra={"marker": "round-trip"})

    loaded, extra = DQNAgent.load(path, torch.device("cpu"))
    assert loaded.select_action(observation, epsilon=0.0) == expected
    assert extra == {"marker": "round-trip"}


def test_config_rejects_an_unreachable_replay_warmup():
    with pytest.raises(ValueError, match="min_replay_size"):
        DQNConfig(
            batch_size=4,
            replay_capacity=10,
            min_replay_size=11,
        )
