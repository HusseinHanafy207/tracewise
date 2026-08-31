import numpy as np
import pytest

from src.rl.environment import TutoringEnvConfig
from src.rl.evaluation import bootstrap_mean_ci, evaluate_policy, paired_difference


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    values = [0.1, 0.2, 0.3, 0.4]
    first = bootstrap_mean_ci(values, n_resamples=500, seed=7)
    second = bootstrap_mean_ci(values, n_resamples=500, seed=7)

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]
    assert bootstrap_mean_ci([0.25]) == [0.25, 0.25]


def test_evaluate_policy_keeps_episode_rows_and_action_mix():
    config = TutoringEnvConfig(
        actions=("explain", "harder_problem"),
        horizon=3,
        include_diagnostics=True,
    )

    result = evaluate_policy(
        "interleave",
        lambda episode_seed: (
            lambda observation, step: "explain" if step % 2 == 0 else "harder_problem"
        ),
        config,
        [10, 11, 12],
        bootstrap_resamples=100,
    )

    assert result["n_episodes"] == 3
    assert len(result["episodes"]) == 3
    assert len(result["mastery_curve_mean"]) == 4
    assert result["action_frequency"]["explain"] == pytest.approx(2.0 / 3.0)
    assert result["action_frequency"]["harder_problem"] == pytest.approx(1.0 / 3.0)


def test_paired_difference_uses_matching_episode_seeds():
    candidate = {
        "name": "candidate",
        "episodes": [
            {"episode_seed": 1, "final_mastery": 0.7},
            {"episode_seed": 2, "final_mastery": 0.5},
        ],
    }
    reference = {
        "name": "reference",
        "episodes": [
            {"episode_seed": 1, "final_mastery": 0.4},
            {"episode_seed": 2, "final_mastery": 0.4},
        ],
    }

    result = paired_difference(
        candidate,
        reference,
        bootstrap_resamples=100,
    )
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["candidate_win_rate"] == 1.0

    reference["episodes"][1]["episode_seed"] = 3
    with pytest.raises(ValueError, match="identical episode seeds"):
        paired_difference(candidate, reference)
