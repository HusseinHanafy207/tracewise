"""Compare feed-forward and GRU Double DQN on matched training seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_dqn_suite import env_config, git_provenance, sha256
from src.rl.dqn import DQNAgent
from src.rl.evaluation import evaluate_policy, paired_difference
from src.rl.recurrent_dqn import RecurrentDQNAgent
from src.utils.seed import load_config, set_seed


def recurrent_comparison_specs(cfg: dict) -> List[dict]:
    evaluation = cfg["recurrent_double_dqn_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    checkpoint_kinds = list(evaluation["checkpoint_kinds"])
    if len(seeds) < 2:
        raise ValueError("At least two matched training seeds are required")
    if checkpoint_kinds != ["best", "final"]:
        raise ValueError("checkpoint_kinds must be [best, final]")
    templates = {
        "feedforward": evaluation["feedforward_tag_template"],
        "gru": evaluation["recurrent_tag_template"],
    }
    stems = {"feedforward": "dqn_oracle", "gru": "recurrent_dqn_oracle"}
    specs = []
    for checkpoint_kind in checkpoint_kinds:
        for architecture in ("feedforward", "gru"):
            for seed in seeds:
                tag = str(templates[architecture]).format(seed=seed)
                specs.append(
                    {
                        "name": f"{architecture}_{checkpoint_kind}_seed{seed}",
                        "architecture": architecture,
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint": (
                            f"{stems[architecture]}_{checkpoint_kind}_{tag}.pt"
                        ),
                        "training_seed": seed,
                    }
                )
    return specs


def load_condition(
    spec: dict,
    checkpoint_dir: Path,
    cfg: dict,
    device: torch.device,
    episode_seeds: List[int],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict:
    checkpoint_path = checkpoint_dir / spec["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing recurrent comparison checkpoint: {checkpoint_path}")
    environment = env_config(cfg, "oracle")
    architecture = spec["architecture"]
    if architecture == "feedforward":
        agent, checkpoint_extra = DQNAgent.load(checkpoint_path, device)
        if not agent.config.double_dqn:
            raise ValueError(f"Expected Double DQN checkpoint: {checkpoint_path}")

        def policy_factory(episode_seed: int):
            return lambda observation, step: agent.select_action(
                observation, epsilon=0.0
            )

        parameter_count = sum(
            parameter.numel() for parameter in agent.online.parameters()
        )
        gamma = agent.config.gamma
    elif architecture == "gru":
        agent, checkpoint_extra = RecurrentDQNAgent.load(checkpoint_path, device)
        if not agent.config.double_dqn:
            raise ValueError(f"Expected recurrent Double DQN: {checkpoint_path}")

        def policy_factory(episode_seed: int):
            agent.reset_hidden()
            return lambda observation, step: agent.select_action(
                observation, epsilon=0.0
            )

        parameter_count = agent.parameter_count
        gamma = agent.config.gamma
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    if not np.isclose(gamma, float(cfg["rl"]["gamma"])):
        raise ValueError(f"Checkpoint gamma mismatch: {checkpoint_path}")
    expected_episode = int(cfg["dqn"]["train_episodes"])
    actual_episode = int(checkpoint_extra["episode"])
    if spec["checkpoint_kind"] == "final" and actual_episode != expected_episode:
        raise ValueError(f"Final checkpoint is not episode {expected_episode}")
    if spec["checkpoint_kind"] == "best" and actual_episode > expected_episode:
        raise ValueError("Selected checkpoint exceeds the training budget")

    summary = evaluate_policy(
        spec["name"],
        policy_factory,
        environment,
        episode_seeds,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    summary.update(
        {
            "architecture": architecture,
            "checkpoint_kind": spec["checkpoint_kind"],
            "training_seed": int(spec["training_seed"]),
            "training_episodes": expected_episode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_extra": checkpoint_extra,
            "parameter_count": parameter_count,
            "gamma": float(gamma),
            "double_dqn": True,
        }
    )
    return summary


def summarize_architecture(
    rows: List[dict], architecture: str, checkpoint_kind: str
) -> dict:
    selected = sorted(
        (
            row
            for row in rows
            if row["architecture"] == architecture
            and row["checkpoint_kind"] == checkpoint_kind
        ),
        key=lambda row: row["training_seed"],
    )
    values = np.asarray(
        [row["final_mastery"] for row in selected], dtype=np.float64
    )
    if len(values) < 2:
        raise ValueError("Each architecture requires at least two trained agents")
    return {
        "architecture": architecture,
        "checkpoint_kind": checkpoint_kind,
        "n_training_seeds": len(selected),
        "training_seeds": [int(row["training_seed"]) for row in selected],
        "final_mastery_by_seed": {
            str(row["training_seed"]): float(row["final_mastery"])
            for row in selected
        },
        "mean_final_mastery_across_agents": float(values.mean()),
        "std_final_mastery_across_agents": float(values.std(ddof=1)),
        "range_final_mastery_across_agents": float(values.max() - values.min()),
        "parameter_count": int(selected[0]["parameter_count"]),
        "checkpoint_episode_by_seed": {
            str(row["training_seed"]): int(row["checkpoint_extra"]["episode"])
            for row in selected
        },
    }


def compare_architectures_across_training_seeds(
    rows: List[dict],
    checkpoint_kind: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    selected = {
        (row["architecture"], int(row["training_seed"])): float(
            row["final_mastery"]
        )
        for row in rows
        if row["checkpoint_kind"] == checkpoint_kind
    }
    feedforward_seeds = sorted(
        seed for architecture, seed in selected if architecture == "feedforward"
    )
    gru_seeds = sorted(seed for architecture, seed in selected if architecture == "gru")
    if feedforward_seeds != gru_seeds or len(feedforward_seeds) < 2:
        raise ValueError("Architectures require at least two identical training seeds")
    feedforward = np.asarray(
        [selected[("feedforward", seed)] for seed in feedforward_seeds],
        dtype=np.float64,
    )
    gru = np.asarray(
        [selected[("gru", seed)] for seed in feedforward_seeds], dtype=np.float64
    )
    differences = gru - feedforward
    rng = np.random.RandomState(bootstrap_seed)
    indices = rng.randint(
        0,
        len(feedforward_seeds),
        size=(bootstrap_resamples, len(feedforward_seeds)),
    )
    mean_differences = differences[indices].mean(axis=1)
    std_differences = (
        gru[indices].std(axis=1, ddof=1)
        - feedforward[indices].std(axis=1, ddof=1)
    )
    return {
        "replication_unit": "training_seed",
        "checkpoint_kind": checkpoint_kind,
        "training_seeds": feedforward_seeds,
        "n_training_seeds": len(feedforward_seeds),
        "gru_minus_feedforward_by_seed": {
            str(seed): float(value)
            for seed, value in zip(feedforward_seeds, differences)
        },
        "mean_of_seed_level_differences": float(differences.mean()),
        "mean_difference_ci95_training_seed_bootstrap": [
            float(np.quantile(mean_differences, 0.025)),
            float(np.quantile(mean_differences, 0.975)),
        ],
        "median_seed_level_difference": float(np.median(differences)),
        "gru_better_seed_count": int(np.sum(differences > 0.0)),
        "gru_tied_seed_count": int(np.sum(differences == 0.0)),
        "gru_minus_feedforward_std": float(
            gru.std(ddof=1) - feedforward.std(ddof=1)
        ),
        "std_difference_ci95_training_seed_bootstrap": [
            float(np.quantile(std_differences, 0.025)),
            float(np.quantile(std_differences, 0.975)),
        ],
        "std_ratio_gru_over_feedforward": float(
            gru.std(ddof=1) / feedforward.std(ddof=1)
        ),
        "note": (
            "Paired bootstrap resamples trained agents by matched training "
            "seed, not held-out episodes within a trained agent."
        ),
    }


def audit_training_artifacts(cfg: dict) -> dict:
    evaluation = cfg["recurrent_double_dqn_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    results_dir = Path(cfg["paths"]["results_dir"])
    expected_episodes = int(cfg["dqn"]["train_episodes"])
    expected_steps = expected_episodes * int(cfg["rl"]["horizon"])
    per_seed = {}
    parameter_counts = set()
    for seed in seeds:
        tag = str(evaluation["recurrent_tag_template"]).format(seed=seed)
        path = results_dir / f"recurrent_dqn_oracle_training_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing recurrent training result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(payload["seed"]) != seed
            or payload["algorithm"] != "recurrent_double_dqn"
            or payload["architecture"] != "gru"
            or int(payload["train_episodes"]) != expected_episodes
            or int(payload["global_steps"]) != expected_steps
            or len(payload["training_history"]) != expected_episodes
            or int(payload["training_history"][-1]["episode"]) != expected_episodes
            or not bool(payload["effective_agent_config"]["double_dqn"])
        ):
            raise ValueError(f"Recurrent training audit failed for seed {seed}")
        parameter_counts.add(int(payload["parameter_count"]))
        per_seed[str(seed)] = {
            "result": str(path),
            "result_sha256": sha256(path),
            "gradient_updates": int(payload["gradient_updates"]),
            "best_checkpoint_episode": int(
                payload["best_checkpoint_extra"]["episode"]
            ),
        }
    if len(parameter_counts) != 1:
        raise ValueError("Recurrent parameter count differs across training seeds")
    return {
        "all_checks_passed": True,
        "n_training_seeds": len(seeds),
        "training_episodes_per_agent": expected_episodes,
        "environment_steps_per_agent": expected_steps,
        "parameter_count": parameter_counts.pop(),
        "per_seed": per_seed,
    }


def plot_comparison(rows: List[dict], summaries: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    colors = {"feedforward": "#4c78a8", "gru": "#f58518"}
    for axis, checkpoint_kind in zip(axes, ("best", "final")):
        for architecture in ("feedforward", "gru"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["architecture"] == architecture
                    and row["checkpoint_kind"] == checkpoint_kind
                ),
                key=lambda row: row["training_seed"],
            )
            summary = summaries[checkpoint_kind][architecture]
            axis.plot(
                [row["training_seed"] for row in selected],
                [row["final_mastery"] for row in selected],
                marker="o",
                color=colors[architecture],
                label=(
                    f"{architecture.title()}: "
                    f"{summary['mean_final_mastery_across_agents']:.4f} "
                    f"± {summary['std_final_mastery_across_agents']:.4f} SD"
                ),
            )
        axis.set_xlabel("Matched training seed")
        axis.set_title(
            "Validation-selected checkpoint"
            if checkpoint_kind == "best"
            else "Exact episode-5,000 endpoint"
        )
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean held-out final mastery")
    fig.suptitle("Feed-forward versus GRU Double DQN")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(config_path: str, device_name: str) -> None:
    cfg = load_config(config_path)
    evaluation = cfg["recurrent_double_dqn_evaluation"]
    training_seeds = [int(seed) for seed in evaluation["training_seeds"]]
    checkpoint_kinds = list(evaluation["checkpoint_kinds"])
    n_episodes = int(evaluation["evaluation_episodes"])
    bootstrap_resamples = int(evaluation["bootstrap_resamples"])
    if n_episodes <= 0 or bootstrap_resamples <= 0:
        raise ValueError("evaluation_episodes and bootstrap_resamples must be positive")
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg["dqn"].get("torch_num_threads", 1)))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    heldout_start = int(cfg["dqn"]["heldout_seed_start"])
    episode_seeds = [heldout_start + index for index in range(n_episodes)]
    bootstrap_seed = int(cfg["seed"]) + 41_000
    training_audit = audit_training_artifacts(cfg)

    specs = recurrent_comparison_specs(cfg)
    print(
        "Feed-forward vs GRU Double DQN | "
        f"seeds={training_seeds} | heldout_episodes={n_episodes} | device={device}"
    )
    print(
        "Training audit passed. Oracle mastery is visible; independence, "
        "fatigue, preferences, and response noise remain hidden."
    )
    rows = []
    for offset, spec in enumerate(specs):
        print(f"  evaluating {spec['name']} ...", flush=True)
        rows.append(
            load_condition(
                spec,
                checkpoint_dir,
                cfg,
                device,
                episode_seeds,
                bootstrap_seed + 100 * offset,
                bootstrap_resamples,
            )
        )

    by_key: Dict[tuple, dict] = {
        (row["checkpoint_kind"], row["architecture"], row["training_seed"]): row
        for row in rows
    }
    summaries = {
        checkpoint_kind: {
            architecture: summarize_architecture(
                rows, architecture, checkpoint_kind
            )
            for architecture in ("feedforward", "gru")
        }
        for checkpoint_kind in checkpoint_kinds
    }
    comparisons = {}
    for kind_index, checkpoint_kind in enumerate(checkpoint_kinds):
        paired = [
            paired_difference(
                by_key[(checkpoint_kind, "gru", seed)],
                by_key[(checkpoint_kind, "feedforward", seed)],
                bootstrap_seed=bootstrap_seed + 5_000 + 100 * kind_index + index,
                bootstrap_resamples=bootstrap_resamples,
            )
            for index, seed in enumerate(training_seeds)
        ]
        across = compare_architectures_across_training_seeds(
            rows,
            checkpoint_kind,
            bootstrap_resamples,
            bootstrap_seed + 7_000 + kind_index,
        )
        comparisons[checkpoint_kind] = {
            "paired_heldout_episode_differences_by_training_seed": paired,
            "across_training_seeds": across,
        }
        feedforward = summaries[checkpoint_kind]["feedforward"]
        gru = summaries[checkpoint_kind]["gru"]
        mean_low, mean_high = across[
            "mean_difference_ci95_training_seed_bootstrap"
        ]
        std_low, std_high = across[
            "std_difference_ci95_training_seed_bootstrap"
        ]
        print(
            f"{checkpoint_kind}: feed-forward mean/std="
            f"{feedforward['mean_final_mastery_across_agents']:.4f}/"
            f"{feedforward['std_final_mastery_across_agents']:.4f} | "
            f"GRU mean/std={gru['mean_final_mastery_across_agents']:.4f}/"
            f"{gru['std_final_mastery_across_agents']:.4f}"
        )
        print(
            f"  GRU difference={across['mean_of_seed_level_differences']:+.4f} "
            f"CI [{mean_low:+.4f}, {mean_high:+.4f}] | "
            f"std difference={across['gru_minus_feedforward_std']:+.4f} "
            f"CI [{std_low:+.4f}, {std_high:+.4f}]"
        )

    result_path = results_dir / "recurrent_double_dqn_comparison.json"
    figure_path = results_dir / "figures" / "recurrent_double_dqn_comparison.png"
    plot_comparison(rows, summaries, figure_path)
    payload = {
        "disclaimer": (
            "Oracle-state simulator comparison only. No intervention policy "
            "was evaluated on real students."
        ),
        "research_question": (
            "Does a longer remembered interaction history help Double DQN "
            "handle hidden independence/fatigue and improve final mastery?"
        ),
        "observability": {
            "visible_or_derived": [
                "oracle mastery",
                "recent correctness summaries",
                "difficulty",
                "episode progress",
                "previous action",
            ],
            "hidden": [
                "independence",
                "fatigue",
                "learner preferences",
                "response randomness",
            ],
        },
        "controlled_comparison": (
            "Both architectures use Double DQN, 5,000 episodes, matched seeds, "
            "the same epsilon/reward/discount/target-sync schedule, validation "
            "episodes, and held-out episodes. GRU replay samples complete "
            "50-step episodes; its loss batch has 150 correlated transitions "
            "versus 128 independently sampled transitions for feed-forward DQN."
        ),
        "training_episodes": int(cfg["dqn"]["train_episodes"]),
        "training_seeds": training_seeds,
        "n_heldout_episodes": n_episodes,
        "heldout_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "bootstrap_resamples": bootstrap_resamples,
        "training_artifact_audit": training_audit,
        "conditions": rows,
        "summaries": summaries,
        "comparisons": comparisons,
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "environment": sha256(Path("src/rl/environment.py")),
                "feedforward_dqn": sha256(Path("src/rl/dqn.py")),
                "recurrent_dqn": sha256(Path("src/rl/recurrent_dqn.py")),
                "feedforward_training_script": sha256(Path("scripts/train_dqn.py")),
                "recurrent_training_script": sha256(
                    Path("scripts/train_recurrent_dqn.py")
                ),
                "evaluation_script": sha256(Path(__file__)),
            },
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote {result_path} and {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/recurrent_double_dqn_evaluation.yaml"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    arguments = parser.parse_args()
    main(arguments.config, arguments.device)
