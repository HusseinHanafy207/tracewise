import numpy as np
import pytest

from scripts.validate_multiskill_env import multiskill_config
from src.rl.multiskill_environment import (
    MultiSkillTutoringEnv,
    MultiSkillTutoringEnvConfig,
)
from src.utils.seed import load_config


class VectorTracker:
    def __init__(self):
        self.values = [0.11, 0.22, 0.33]
        self.updates = []

    def estimate_mastery(self, skill: int) -> float:
        return self.values[skill]

    def update(self, skill: int, correct: int) -> None:
        self.updates.append((skill, correct))
        self.values[skill] = 0.9


def two_skill_config(**overrides):
    values = {
        "horizon": 2,
        "num_skills": 2,
        "prerequisites": ((), (0,)),
        "focus_skill_sequence": (1,),
        "initial_mastery": (0.1, 0.2),
        "delayed_effects": False,
        "vary_difficulty": False,
        "include_diagnostics": True,
    }
    values.update(overrides)
    return MultiSkillTutoringEnvConfig(**values)


def test_validation_config_builds_four_skill_diamond_graph():
    config = multiskill_config(load_config("configs/multiskill_simulator.yaml"))

    assert config.num_skills == 4
    assert config.resolved_prerequisites == ((), (0,), (0,), (1, 2))
    assert config.resolved_focus_skill_sequence == (0, 1, 0, 2, 1, 2, 3, 3)


def test_default_graph_is_chain_and_transitive_closure_is_available():
    env = MultiSkillTutoringEnv(
        MultiSkillTutoringEnvConfig(num_skills=4, horizon=2)
    )
    env.reset(seed=3)

    assert env.prerequisites == ((), (0,), (1,), (2,))
    assert env.prerequisite_closure(3) == (0, 1, 2)


def test_invalid_prerequisite_graphs_fail_loudly():
    with pytest.raises(ValueError, match="one entry per skill"):
        MultiSkillTutoringEnvConfig(num_skills=3, prerequisites=((), (0,)))
    with pytest.raises(ValueError, match="acyclic"):
        MultiSkillTutoringEnvConfig(
            num_skills=2,
            prerequisites=((1,), (0,)),
        )
    with pytest.raises(ValueError, match="own prerequisite"):
        MultiSkillTutoringEnvConfig(
            num_skills=2,
            prerequisites=((0,), ()),
        )


def test_observation_contains_mastery_vector_and_current_skill_one_hot():
    config = MultiSkillTutoringEnvConfig(
        num_skills=3,
        horizon=3,
        prerequisites=((), (0,), (0, 1)),
        focus_skill_sequence=(2, 1, 0),
        initial_mastery=(0.1, 0.2, 0.3),
        include_diagnostics=True,
    )
    env = MultiSkillTutoringEnv(config)
    observation, info = env.reset(seed=11)

    assert observation.shape == (20,)
    assert observation.dtype == np.float32
    assert observation[:3].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert observation[11:14].tolist() == [0.0, 0.0, 1.0]
    assert info["current_skill"] == 2
    assert info["diagnostics"]["true_mastery_by_skill"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )


def test_prerequisite_review_changes_prerequisite_and_current_skill():
    env = MultiSkillTutoringEnv(two_skill_config())
    _, reset_info = env.reset(seed=13)
    before = np.asarray(reset_info["diagnostics"]["true_mastery_by_skill"])

    _, reward, _, _, info = env.step("prerequisite_review")
    after = np.asarray(info["diagnostics"]["true_mastery_by_skill"])
    delta = after - before

    assert info["acted_skill"] == 1
    assert info["target_skill"] == 0
    assert delta[0] > 0.0
    assert delta[1] > 0.0
    assert reward == pytest.approx(delta.mean())
    assert info["mastery_delta_by_skill"] == pytest.approx(delta.tolist())


def test_current_skill_instruction_can_spill_over_to_weak_prerequisite():
    env = MultiSkillTutoringEnv(two_skill_config())
    _, reset_info = env.reset(seed=17)
    before = np.asarray(reset_info["diagnostics"]["true_mastery_by_skill"])

    _, _, _, _, info = env.step("explain")
    after = np.asarray(info["diagnostics"]["true_mastery_by_skill"])

    assert info["target_skill"] == 1
    assert after[1] > before[1]
    assert after[0] > before[0]


def test_mastered_prerequisite_increases_learning_on_dependent_skill():
    low = MultiSkillTutoringEnv(two_skill_config(initial_mastery=(0.1, 0.2)))
    high = MultiSkillTutoringEnv(two_skill_config(initial_mastery=(1.0, 0.2)))
    low.reset(seed=19)
    high.reset(seed=19)

    _, low_reward, _, _, low_info = low.step("explain")
    _, high_reward, _, _, high_info = high.step("explain")

    assert low_info["target_skill"] == high_info["target_skill"] == 1
    assert low_info["diagnostics"]["true_mastery_by_skill"][1] < high_info[
        "diagnostics"
    ]["true_mastery_by_skill"][1]
    assert low_reward < high_reward


def test_focus_sequence_advances_and_tracker_updates_action_target():
    trackers = []

    def tracker_factory():
        tracker = VectorTracker()
        trackers.append(tracker)
        return tracker

    config = MultiSkillTutoringEnvConfig(
        num_skills=3,
        horizon=3,
        prerequisites=((), (0,), (0, 1)),
        focus_skill_sequence=(2, 1, 0),
        initial_mastery=(0.1, 0.2, 0.3),
        observation_mode="estimated",
    )
    env = MultiSkillTutoringEnv(config, tracker_factory=tracker_factory)
    observation, _ = env.reset(seed=23)
    assert observation[:3].tolist() == pytest.approx([0.11, 0.22, 0.33])

    observation, _, _, _, info = env.step("prerequisite_review")

    assert info["acted_skill"] == 2
    assert info["target_skill"] == 0
    assert info["current_skill"] == 1
    assert trackers[0].updates[0][0] == 0
    assert observation[0] == pytest.approx(0.9)
    assert observation[11:14].tolist() == [0.0, 1.0, 0.0]


def test_no_state_hides_mastery_and_response_history_but_keeps_task_context():
    config = MultiSkillTutoringEnvConfig(
        num_skills=3,
        horizon=2,
        prerequisites=((), (0,), (1,)),
        focus_skill_sequence=(2, 1),
        initial_mastery=(0.1, 0.2, 0.3),
        observation_mode="no_state",
        vary_difficulty=False,
    )
    env = MultiSkillTutoringEnv(config)
    observation, info = env.reset(seed=29)

    assert np.allclose(observation[:3], 0.0)
    assert np.allclose(observation[[3, 6, 7, 8, 9, 10]], 0.0)
    assert observation[4] == pytest.approx(0.5)
    assert observation[11:14].tolist() == [0.0, 0.0, 1.0]
    assert "diagnostics" not in info


def test_seeded_action_sequence_and_fork_are_reproducible():
    config = MultiSkillTutoringEnvConfig(
        num_skills=3,
        horizon=4,
        prerequisites=((), (0,), (0, 1)),
        focus_skill_sequence=(2, 1, 2, 0),
        include_diagnostics=True,
    )

    def rollout():
        env = MultiSkillTutoringEnv(config)
        observation, _ = env.reset(seed=31)
        rows = [observation.copy()]
        for action in ("prerequisite_review", "explain", "harder_problem", "socratic_hint"):
            observation, reward, terminated, _, info = env.step(action)
            rows.append(
                (
                    observation.copy(),
                    reward,
                    info["correct"],
                    tuple(info["diagnostics"]["true_mastery_by_skill"]),
                )
            )
        assert terminated
        return rows

    first_rollout = rollout()
    second_rollout = rollout()
    assert np.array_equal(first_rollout[0], second_rollout[0])
    assert all(
        np.array_equal(first[0], second[0]) and first[1:] == second[1:]
        for first, second in zip(first_rollout[1:], second_rollout[1:])
    )

    source = MultiSkillTutoringEnv(config)
    source.reset(seed=37)
    clone = source.fork()
    clone_step = clone.step("prerequisite_review")
    source_step = source.step("prerequisite_review")
    assert np.array_equal(clone_step[0], source_step[0])
    assert clone_step[1:] == source_step[1:]
