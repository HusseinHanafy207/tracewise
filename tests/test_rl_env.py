import numpy as np
import pytest

from src.rl.environment import OBSERVATION_NAMES, TutoringEnv, TutoringEnvConfig


class FixedEstimateTracker:
    def __init__(self):
        self.value = 0.25

    def estimate_mastery(self, skill: int) -> float:
        return self.value

    def update(self, skill: int, correct: int) -> None:
        self.value = 0.75


def test_reset_observation_has_documented_shape_and_no_diagnostics_by_default():
    env = TutoringEnv(TutoringEnvConfig(horizon=3), seed=7)
    observation, info = env.reset(seed=19)

    assert observation.shape == (len(OBSERVATION_NAMES),)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert "diagnostics" not in info
    assert set(info) == {"step", "focus_skill", "observation_mode", "episode_seed"}


def test_reward_is_mastery_change_and_horizon_terminates_episode():
    env = TutoringEnv(
        TutoringEnvConfig(horizon=2, include_diagnostics=True), seed=3
    )
    _, reset_info = env.reset(seed=23)
    mastery_before = reset_info["diagnostics"]["true_mastery"]

    _, reward, terminated, truncated, info = env.step("explain")
    mastery_after = info["diagnostics"]["true_mastery"]
    assert reward == pytest.approx(mastery_after - mastery_before)
    assert not terminated
    assert not truncated

    _, _, terminated, truncated, _ = env.step("harder_problem")
    assert terminated
    assert not truncated
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(0)


def test_seed_and_action_sequence_reproduce_exact_trajectory():
    config = TutoringEnvConfig(horizon=5, delayed_effects=True)

    def rollout():
        env = TutoringEnv(config)
        observation, _ = env.reset(seed=101)
        rows = [observation.copy()]
        for action in [0, 4, 0, 4, 2]:
            observation, reward, terminated, _, info = env.step(action)
            rows.append(
                np.concatenate(
                    [observation, np.asarray([reward, info["correct"]], dtype=np.float32)]
                )
            )
        assert terminated
        return rows

    first = rollout()
    second = rollout()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_fork_preserves_state_without_mutating_source_and_can_resample_future():
    config = TutoringEnvConfig(horizon=3, delayed_effects=True)
    source = TutoringEnv(config)
    initial, _ = source.reset(seed=101)

    exact = source.fork()
    exact_step = exact.step("harder_problem")
    source_step = source.step("harder_problem")
    assert np.array_equal(exact_step[0], source_step[0])
    assert exact_step[1:] == source_step[1:]

    untouched = TutoringEnv(config)
    untouched_initial, _ = untouched.reset(seed=101)
    first = untouched.fork(response_seed=9001).step("harder_problem")
    second = untouched.fork(response_seed=9001).step("harder_problem")
    third = untouched.fork(response_seed=9002).step("harder_problem")

    assert np.array_equal(initial, untouched_initial)
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert not np.array_equal(first[0], third[0])


def test_estimated_mode_uses_tracker_signal_not_true_mastery():
    env = TutoringEnv(
        TutoringEnvConfig(
            horizon=2,
            observation_mode="estimated",
            include_diagnostics=True,
        ),
        tracker_factory=FixedEstimateTracker,
    )
    observation, info = env.reset(seed=29)
    assert observation[0] == pytest.approx(0.25)
    assert observation[0] != pytest.approx(info["diagnostics"]["true_mastery"])

    observation, _, _, _, _ = env.step("worked_example")
    assert observation[0] == pytest.approx(0.75)


def test_no_state_mode_zeros_student_features_but_keeps_task_progress():
    env = TutoringEnv(
        TutoringEnvConfig(horizon=4, observation_mode="no_state"), seed=2
    )
    observation, _ = env.reset(seed=31)
    assert np.allclose(observation[[0, 1, 4, 5, 6, 7]], 0.0)
    assert 0.15 <= observation[2] <= 0.85
    assert observation[3] == 0.0

    observation, _, _, _, _ = env.step("explain")
    assert np.allclose(observation[[0, 1, 4, 5, 6, 7]], 0.0)
    assert observation[3] == pytest.approx(0.25)
    assert np.array_equal(observation[8:], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_delayed_effects_reduce_immediate_scaffolding_gain_for_same_episode():
    off = TutoringEnv(TutoringEnvConfig(horizon=2, delayed_effects=False), seed=5)
    on = TutoringEnv(TutoringEnvConfig(horizon=2, delayed_effects=True), seed=5)
    off.reset(seed=37)
    on.reset(seed=37)

    _, reward_off, _, _, _ = off.step("explain")
    _, reward_on, _, _, _ = on.step("explain")
    assert reward_on < reward_off


def test_invalid_configuration_and_actions_fail_loudly():
    with pytest.raises(ValueError, match="tracker_factory"):
        TutoringEnv(TutoringEnvConfig(observation_mode="estimated"))
    with pytest.raises(ValueError, match="horizon"):
        TutoringEnvConfig(horizon=0)
    with pytest.raises(ValueError, match="not implemented"):
        TutoringEnvConfig(actions=("explain", "unknown"))

    env = TutoringEnv(TutoringEnvConfig(horizon=2))
    env.reset(seed=41)
    with pytest.raises(ValueError, match="action id"):
        env.step(99)
    with pytest.raises(ValueError, match="unknown action"):
        env.step("not_an_action")


def test_action_subset_has_matching_observation_metadata():
    env = TutoringEnv(
        TutoringEnvConfig(actions=("explain", "harder_problem"), horizon=2),
        seed=9,
    )
    observation, _ = env.reset(seed=9)

    assert observation.shape == (10,)
    assert env.observation_dim == 10
    assert env.observation_names[-2:] == (
        "last_action_explain",
        "last_action_harder_problem",
    )

    observation, *_ = env.step(1)
    assert observation[-2:].tolist() == [0.0, 1.0]
