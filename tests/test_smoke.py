import numpy as np

from src.policy.bandit import LinUCB
from src.policy.rule_based import RuleBasedPolicy
from src.policy.simulator import (
    bandit_context,
    run_kt_aware_simulation,
    run_policy_simulation,
    state_from_simulated,
)


ACTIONS = [
    "explain",
    "worked_example",
    "socratic_hint",
    "easier_problem",
    "harder_problem",
    "prerequisite_review",
]


class ConstantTracker:
    def __init__(self):
        self.observations = []

    def estimate_mastery(self, skill: int) -> float:
        return 0.45

    def update(self, skill: int, correct: int) -> None:
        self.observations.append((skill, correct))


def _assert_valid_result(result: dict, expected_steps: int):
    for key in ("mean_reward", "mean_final_mastery", "overall_accuracy", "final_accuracy"):
        assert np.isfinite(result[key])
    assert sum(result["action_counts"].values()) == expected_steps
    assert 0.0 <= result["mean_final_mastery"] <= 1.0
    assert 0.0 <= result["overall_accuracy"] <= 1.0
    assert 0.0 <= result["final_accuracy"] <= 1.0


def test_short_oracle_policy_loop_smoke():
    n_students, horizon = 4, 6

    rule_result = run_policy_simulation(
        RuleBasedPolicy(),
        n_students=n_students,
        n_interactions=horizon,
        context_fn=lambda student, difficulty: state_from_simulated(student),
        seed=9,
    )
    _assert_valid_result(rule_result, n_students * horizon)

    linucb_result = run_policy_simulation(
        LinUCB(ACTIONS, context_dim=8, alpha=1.0),
        n_students=n_students,
        n_interactions=horizon,
        context_fn=lambda student, difficulty: bandit_context(student, difficulty),
        seed=9,
    )
    _assert_valid_result(linucb_result, n_students * horizon)


def test_short_estimated_state_policy_loop_smoke():
    n_students, horizon = 4, 6
    result = run_kt_aware_simulation(
        LinUCB(ACTIONS, context_dim=8, alpha=1.0),
        tracker_factory=ConstantTracker,
        n_students=n_students,
        n_interactions=horizon,
        num_skills=3,
        seed=11,
        state_mode="estimated",
    )

    _assert_valid_result(result, n_students * horizon)
    assert result["state_mode"] == "estimated"
    assert np.isfinite(result["mean_abs_estimation_error"])
