import numpy as np
import pytest

from src.policy.bandit import EpsilonGreedyBandit, LinUCB
from src.policy.kt_state import bandit_context_from_estimate
from src.policy.random_policy import RandomPolicy
from src.policy.simulator import (
    SimulatedStudent,
    bandit_context,
    homogeneous_prefs,
    run_policy_simulation,
)


ACTIONS = [
    "explain",
    "worked_example",
    "socratic_hint",
    "easier_problem",
    "harder_problem",
    "prerequisite_review",
]


def test_epsilon_greedy_updates_only_selected_action():
    bandit = EpsilonGreedyBandit(
        ["a", "b"], context_dim=2, epsilon=0.0, lr=0.1, seed=1
    )
    context = np.array([2.0, -1.0])
    bandit.update("b", context, reward=0.5)

    assert np.allclose(bandit.weights["a"], np.zeros(2))
    assert np.allclose(bandit.weights["b"], np.array([0.1, -0.05]))


def test_linucb_update_matches_disjoint_linear_equations():
    bandit = LinUCB(["a", "b"], context_dim=2, alpha=1.0)
    context = np.array([1.0, 2.0])
    bandit.update("a", context, reward=0.25)

    assert np.allclose(bandit.A["a"], np.eye(2) + np.outer(context, context))
    assert np.allclose(bandit.b["a"], 0.25 * context)
    assert np.allclose(bandit.A["b"], np.eye(2))
    assert bandit.select_action(context) in {"a", "b"}


def test_policy_context_excludes_hidden_simulator_variables():
    student = SimulatedStudent(
        student_id=7,
        mastery=0.3,
        difficulty_pref=0.5,
        prefs=homogeneous_prefs(),
        independence=0.91,
        fatigue=0.83,
    )

    context = bandit_context(student, difficulty=0.5, include_mastery=True)
    assert np.allclose(context, [0.3, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5])

    no_state = bandit_context_from_estimate(
        estimated_mastery=0.99,
        history_correct=[1, 0, 1],
        difficulty=0.4,
        include_mastery=False,
    )
    assert np.allclose(no_state, [0.0, 0.0, 0.4, 0.06, 0.0, 0.0, 0.0, 0.0])


def test_delayed_simulation_is_deterministic_and_bounded():
    def run_once():
        return run_policy_simulation(
            RandomPolicy(ACTIONS, seed=123),
            n_students=8,
            n_interactions=12,
            context_fn=lambda student, difficulty: None,
            seed=123,
            delayed_effects=True,
        )

    first = run_once()
    second = run_once()

    assert first["mean_reward"] == pytest.approx(second["mean_reward"])
    assert first["mean_final_mastery"] == pytest.approx(
        second["mean_final_mastery"]
    )
    assert first["accuracy_curve"] == second["accuracy_curve"]
    assert sum(first["action_counts"].values()) == 8 * 12
    assert 0.0 <= first["mean_final_mastery"] <= 1.0
    assert 0.0 <= first["mean_final_independence"] <= 1.0
    assert 0.0 <= first["mean_final_fatigue"] <= 1.0
