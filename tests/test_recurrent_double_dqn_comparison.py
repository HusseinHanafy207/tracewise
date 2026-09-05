import pytest

from scripts.eval_recurrent_double_dqn import (
    compare_architectures_across_training_seeds,
    recurrent_comparison_specs,
    summarize_architecture,
)
from scripts.train_recurrent_dqn import (
    feedforward_parameter_count,
    recurrent_agent_config,
)
from src.rl.environment import TutoringEnv
from src.rl.recurrent_dqn import RecurrentDQNAgent
from scripts.train_dqn import environment_config
from src.utils.seed import load_config
import torch


def test_recurrent_config_keeps_shared_double_dqn_protocol():
    feedforward = load_config("configs/double_dqn_train_5000.yaml")
    recurrent = load_config("configs/recurrent_double_dqn_train.yaml")

    recurrent_only = recurrent.pop("recurrent_dqn")

    assert recurrent == feedforward
    assert recurrent_only["recurrent_hidden_dim"] == 64
    assert recurrent["dqn"]["double_dqn"] is True
    assert recurrent["dqn"]["train_episodes"] == 5000


def test_recurrent_model_is_close_to_feedforward_parameter_count():
    config = load_config("configs/recurrent_double_dqn_train.yaml")
    environment = TutoringEnv(environment_config(config, False))
    agent = RecurrentDQNAgent(
        environment.observation_dim,
        environment.n_actions,
        environment.config.horizon,
        recurrent_agent_config(config),
        torch.device("cpu"),
    )
    feedforward = feedforward_parameter_count(
        config, environment.observation_dim, environment.n_actions
    )

    assert agent.parameter_count == 19910
    assert feedforward == 19206
    assert agent.parameter_count / feedforward < 1.05


def test_recurrent_comparison_manifest_pairs_architectures_and_estimands():
    config = load_config("configs/recurrent_double_dqn_evaluation.yaml")

    specs = recurrent_comparison_specs(config)

    assert len(specs) == 40
    assert {row["training_seed"] for row in specs} == set(range(42, 52))
    assert {row["architecture"] for row in specs} == {"feedforward", "gru"}
    assert {row["checkpoint_kind"] for row in specs} == {"best", "final"}
    assert specs[0]["checkpoint"] == (
        "dqn_oracle_best_double_budget5000_seed42.pt"
    )
    assert specs[10]["checkpoint"] == "recurrent_dqn_oracle_best_gru_seed42.pt"


def synthetic_rows():
    rows = []
    for architecture, values in (
        ("feedforward", (0.50, 0.60, 0.40)),
        ("gru", (0.55, 0.61, 0.50)),
    ):
        for seed, value in zip((42, 43, 44), values):
            rows.append(
                {
                    "architecture": architecture,
                    "checkpoint_kind": "best",
                    "training_seed": seed,
                    "final_mastery": value,
                    "parameter_count": 10,
                    "checkpoint_extra": {"episode": 100},
                }
            )
    return rows


def test_recurrent_summary_and_comparison_use_training_seed_replications():
    summary = summarize_architecture(synthetic_rows(), "gru", "best")
    comparison = compare_architectures_across_training_seeds(
        synthetic_rows(), "best", bootstrap_resamples=1000, bootstrap_seed=7
    )

    assert summary["n_training_seeds"] == 3
    assert summary["mean_final_mastery_across_agents"] == pytest.approx(
        (0.55 + 0.61 + 0.50) / 3.0
    )
    assert comparison["replication_unit"] == "training_seed"
    assert comparison["mean_of_seed_level_differences"] == pytest.approx(
        (0.05 + 0.01 + 0.10) / 3.0
    )
    assert comparison["gru_better_seed_count"] == 3
    assert comparison["mean_difference_ci95_training_seed_bootstrap"][0] > 0.0
