import pytest

from scripts.eval_double_dqn import comparison_specs, summarize_stability
from src.utils.seed import load_config


def test_comparison_manifest_pairs_algorithms_on_the_same_seeds():
    config = load_config("configs/double_dqn_evaluation.yaml")

    specs = comparison_specs(config)

    assert len(specs) == 6
    assert [row["training_seed"] for row in specs[:3]] == [42, 43, 44]
    assert [row["training_seed"] for row in specs[3:]] == [42, 43, 44]
    assert all(not row["double_dqn"] for row in specs[:3])
    assert all(row["double_dqn"] for row in specs[3:])
    assert specs[0]["checkpoint"] == "dqn_oracle_best.pt"
    assert specs[3]["checkpoint"] == "dqn_oracle_best_double_seed42.pt"


def test_double_dqn_training_config_changes_only_the_algorithm_flag():
    vanilla = load_config("configs/dqn_train.yaml")
    double = load_config("configs/double_dqn_train.yaml")

    assert vanilla["dqn"]["double_dqn"] is False
    assert double["dqn"]["double_dqn"] is True
    vanilla["dqn"]["double_dqn"] = True
    assert double == vanilla


def test_stability_summary_uses_training_seed_as_the_replication_unit():
    rows = [
        {
            "algorithm": "double_dqn",
            "training_seed": seed,
            "final_mastery": value,
            "checkpoint_extra": {"episode": episode},
        }
        for seed, value, episode in (
            (44, 0.60, 800),
            (42, 0.50, 400),
            (43, 0.55, 600),
        )
    ]

    summary = summarize_stability(rows, "double_dqn")

    assert summary["training_seeds"] == [42, 43, 44]
    assert summary["mean_final_mastery_across_agents"] == pytest.approx(0.55)
    assert summary["range_final_mastery_across_agents"] == pytest.approx(0.10)
    assert summary["selected_checkpoint_episode_by_seed"] == {
        "42": 400,
        "43": 600,
        "44": 800,
    }
