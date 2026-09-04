import pytest

from scripts.eval_double_dqn_budget import (
    budget_specs,
    compare_budgets_across_training_seeds,
    summarize_budget,
)
from src.utils.seed import load_config


def test_5000_episode_config_changes_only_training_budget():
    reference = load_config("configs/double_dqn_train.yaml")
    candidate = load_config("configs/double_dqn_train_5000.yaml")

    assert reference["dqn"]["train_episodes"] == 2000
    assert candidate["dqn"]["train_episodes"] == 5000
    candidate["dqn"]["train_episodes"] = 2000
    assert candidate == reference


def test_budget_manifest_pairs_two_budgets_and_checkpoint_estimands():
    config = load_config("configs/double_dqn_budget_evaluation.yaml")

    specs = budget_specs(config)

    assert len(specs) == 40
    assert {row["training_seed"] for row in specs} == set(range(42, 52))
    assert {row["training_episodes"] for row in specs} == {2000, 5000}
    assert {row["checkpoint_kind"] for row in specs} == {"best", "final"}
    assert all(row["double_dqn"] for row in specs)
    assert specs[0]["checkpoint"] == "dqn_oracle_best_double_seed42.pt"
    assert specs[10]["checkpoint"] == (
        "dqn_oracle_best_double_budget5000_seed42.pt"
    )
    assert specs[20]["checkpoint"] == "dqn_oracle_final_double_seed42.pt"


def synthetic_rows():
    rows = []
    for budget, values in ((2000, (0.50, 0.60, 0.40)), (5000, (0.55, 0.61, 0.50))):
        for seed, value in zip((42, 43, 44), values):
            rows.append(
                {
                    "training_episodes": budget,
                    "checkpoint_kind": "final",
                    "training_seed": seed,
                    "final_mastery": value,
                    "checkpoint_extra": {"episode": budget},
                }
            )
    return rows


def test_budget_summary_uses_training_seed_as_replication_unit():
    summary = summarize_budget(synthetic_rows(), 5000, "final")

    assert summary["n_training_seeds"] == 3
    assert summary["mean_final_mastery_across_agents"] == pytest.approx(
        (0.55 + 0.61 + 0.50) / 3.0
    )
    assert summary["checkpoint_episode_by_seed"] == {
        "42": 5000,
        "43": 5000,
        "44": 5000,
    }


def test_budget_comparison_bootstraps_matched_training_seeds():
    result = compare_budgets_across_training_seeds(
        synthetic_rows(),
        checkpoint_kind="final",
        reference_budget=2000,
        candidate_budget=5000,
        bootstrap_resamples=1000,
        bootstrap_seed=19,
    )

    assert result["replication_unit"] == "training_seed"
    assert result["mean_of_seed_level_differences"] == pytest.approx(
        (0.05 + 0.01 + 0.10) / 3.0
    )
    assert result["candidate_better_seed_count"] == 3
    assert result["mean_difference_ci95_training_seed_bootstrap"][0] > 0.0
