import pytest

from scripts.eval_memory_ablation import (
    compare_across_training_seeds,
    expected_update_count,
    memory_ablation_specs,
)
from src.utils.seed import load_config


def test_memory_ablation_protocol_matches_recurrent_replay():
    config = load_config("configs/memory_ablation_evaluation.yaml")

    assert config["episode_replay_dqn"] == {
        key: config["recurrent_dqn"][key]
        for key in (
            "episode_batch_size",
            "replay_capacity_episodes",
            "min_replay_episodes",
        )
    }
    assert expected_update_count(config) == 62000
    assert config["dqn"]["train_episodes"] == 5000
    assert config["dqn"]["double_dqn"] is True


def test_memory_ablation_manifest_has_two_estimands_and_matched_seeds():
    specs = memory_ablation_specs(
        load_config("configs/memory_ablation_evaluation.yaml")
    )

    assert len(specs) == 40
    assert {row["training_seed"] for row in specs} == set(range(42, 52))
    assert {row["architecture"] for row in specs} == {
        "feedforward_episode_replay",
        "gru_episode_replay",
    }
    assert {row["checkpoint_kind"] for row in specs} == {"best", "final"}


def test_memory_comparison_bootstraps_matched_training_seeds():
    rows = []
    for architecture, values in (
        ("feedforward_episode_replay", (0.50, 0.60, 0.40)),
        ("gru_episode_replay", (0.55, 0.61, 0.50)),
    ):
        for seed, value in zip((42, 43, 44), values):
            rows.append(
                {
                    "architecture": architecture,
                    "checkpoint_kind": "best",
                    "training_seed": seed,
                    "final_mastery": value,
                }
            )

    comparison = compare_across_training_seeds(
        rows, "best", bootstrap_resamples=1000, bootstrap_seed=7
    )

    assert comparison["replication_unit"] == "training_seed"
    assert comparison["mean_of_seed_level_differences"] == pytest.approx(
        (0.05 + 0.01 + 0.10) / 3.0
    )
    assert comparison["gru_better_seed_count"] == 3
    assert comparison["mean_difference_ci95_training_seed_bootstrap"][0] > 0.0
