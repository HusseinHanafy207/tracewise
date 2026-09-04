import numpy as np
import pytest

from scripts.eval_q_overestimation import (
    compare_bias_across_training_seeds,
    paired_bias_difference,
    summarize_bias,
)


def test_bias_summary_uses_q_minus_monte_carlo_return():
    rows = [
        {
            "step": 0,
            "q_max": 4.0,
            "mc_return_mean": 3.0,
            "mc_return_standard_error": 0.2,
        },
        {
            "step": 10,
            "q_max": 1.0,
            "mc_return_mean": 2.0,
            "mc_return_standard_error": 0.4,
        },
    ]

    summary = summarize_bias(rows, bootstrap_resamples=100, seed=7)

    assert summary["mean_signed_bias"] == pytest.approx(0.0)
    assert summary["mean_absolute_error"] == pytest.approx(1.0)
    assert summary["root_mean_squared_error"] == pytest.approx(1.0)
    assert summary["overestimation_rate"] == pytest.approx(0.5)
    assert summary["mean_mc_standard_error"] == pytest.approx(0.3)
    assert summary["bias_by_step"] == {"0": 1.0, "10": -1.0}
    assert np.isfinite(summary["mean_signed_bias_ci95_across_states"]).all()


def test_bias_comparison_bootstraps_matched_training_seeds():
    rows = []
    for algorithm, biases, errors in (
        ("dqn", [2.0, 4.0, 6.0], [3.0, 5.0, 7.0]),
        ("double_dqn", [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
    ):
        for seed, bias, error in zip((42, 43, 44), biases, errors):
            rows.append(
                {
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "summary": {
                        "mean_signed_bias": bias,
                        "mean_absolute_error": error,
                        "root_mean_squared_error": error + 1.0,
                        "overestimation_rate": bias / 10.0,
                    },
                }
            )

    result = compare_bias_across_training_seeds(rows, 1000, 7)

    assert result["replication_unit"] == "training_seed"
    assert result["n_training_seeds"] == 3
    assert result["comparisons"][
        "double_minus_vanilla_mean_signed_bias"
    ]["mean_of_seed_level_differences"] == pytest.approx(-2.0)
    assert result["comparisons"][
        "double_minus_vanilla_mean_absolute_error"
    ]["mean_of_seed_level_differences"] == pytest.approx(-1.0)


def test_on_policy_bias_difference_does_not_pair_different_latent_states():
    def checkpoint(name, bias):
        return {
            "name": name,
            "training_seed": 42,
            "summary": {"mean_signed_bias": bias},
            "states": [
                {"state_id": 0, "signed_bias": bias - 1.0},
                {"state_id": 1, "signed_bias": bias + 1.0},
            ],
        }

    result = paired_bias_difference(
        checkpoint("double", -2.0),
        checkpoint("vanilla", -1.0),
        bootstrap_resamples=100,
        bootstrap_seed=7,
        pair_states=False,
    )

    assert result["state_comparison"] == "unpaired_on_policy"
    assert result["mean_difference"] == pytest.approx(-1.0)
    assert result["difference_ci95_across_states"] is None
    assert result["double_dqn_lower_bias_state_rate"] is None
