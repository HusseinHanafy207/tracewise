"""Compare feed-forward and GRU Double DQN under matched episode replay."""
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
from scripts.eval_recurrent_double_dqn import summarize_architecture
from src.rl.episode_replay_dqn import EpisodeReplayDQNAgent
from src.rl.evaluation import evaluate_policy, paired_difference
from src.rl.recurrent_dqn import RecurrentDQNAgent
from src.utils.seed import load_config, set_seed


def memory_ablation_specs(cfg: dict) -> List[dict]:
    evaluation = cfg["memory_ablation_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    checkpoint_kinds = list(evaluation["checkpoint_kinds"])
    if len(seeds) < 2:
        raise ValueError("At least two matched training seeds are required")
    if checkpoint_kinds != ["best", "final"]:
        raise ValueError("checkpoint_kinds must be [best, final]")
    templates = {
        "feedforward_episode_replay": evaluation["feedforward_tag_template"],
        "gru_episode_replay": evaluation["gru_tag_template"],
    }
    stems = {
        "feedforward_episode_replay": "episode_replay_dqn_oracle",
        "gru_episode_replay": "recurrent_dqn_oracle",
    }
    specs = []
    for checkpoint_kind in checkpoint_kinds:
        for architecture in ("feedforward_episode_replay", "gru_episode_replay"):
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


def expected_update_count(cfg: dict) -> int:
    total_steps = int(cfg["dqn"]["train_episodes"]) * int(cfg["rl"]["horizon"])
    first = int(cfg["dqn"]["min_replay_size"])
    frequency = int(cfg["dqn"]["train_frequency"])
    first = ((first + frequency - 1) // frequency) * frequency
    last = ((total_steps - 1) // frequency) * frequency
    return 0 if first > last else (last - first) // frequency + 1


def audit_training_artifacts(cfg: dict) -> dict:
    evaluation = cfg["memory_ablation_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    results_dir = Path(cfg["paths"]["results_dir"])
    episodes = int(cfg["dqn"]["train_episodes"])
    steps = episodes * int(cfg["rl"]["horizon"])
    updates = expected_update_count(cfg)
    expected_replay = {
        "episode_batch_size": int(cfg["episode_replay_dqn"]["episode_batch_size"]),
        "replay_capacity_episodes": int(
            cfg["episode_replay_dqn"]["replay_capacity_episodes"]
        ),
        "min_replay_episodes": int(
            cfg["episode_replay_dqn"]["min_replay_episodes"]
        ),
    }
    recurrent_replay = {
        key: int(cfg["recurrent_dqn"][key]) for key in expected_replay
    }
    if expected_replay != recurrent_replay:
        raise ValueError("Feed-forward and GRU episode replay settings differ")

    per_seed = {}
    feedforward_parameters = set()
    gru_parameters = set()
    for seed in seeds:
        feedforward_tag = str(evaluation["feedforward_tag_template"]).format(seed=seed)
        gru_tag = str(evaluation["gru_tag_template"]).format(seed=seed)
        paths = {
            "feedforward_episode_replay": results_dir
            / f"episode_replay_dqn_oracle_training_{feedforward_tag}.json",
            "gru_episode_replay": results_dir
            / f"recurrent_dqn_oracle_training_{gru_tag}.json",
        }
        payloads = {}
        for architecture, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing memory-ablation result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected_algorithm = (
                "episode_replay_double_dqn"
                if architecture == "feedforward_episode_replay"
                else "recurrent_double_dqn"
            )
            if (
                int(payload["seed"]) != seed
                or payload["algorithm"] != expected_algorithm
                or int(payload["train_episodes"]) != episodes
                or int(payload["global_steps"]) != steps
                or int(payload["gradient_updates"]) != updates
                or len(payload["training_history"]) != episodes
                or int(payload["training_history"][-1]["episode"]) != episodes
                or int(payload["replay_loss_transitions_per_update"]) != 150
                or not bool(payload["effective_agent_config"]["double_dqn"])
            ):
                raise ValueError(f"Training audit failed for {architecture}, seed {seed}")
            payloads[architecture] = payload

        feedforward = payloads["feedforward_episode_replay"]
        gru = payloads["gru_episode_replay"]
        if (
            feedforward["dqn_shared_config"] != gru["dqn_shared_config"]
            or feedforward["rl_config"] != gru["rl_config"]
            or feedforward["train_seed_range"] != gru["train_seed_range"]
            or feedforward["validation_seed_range"] != gru["validation_seed_range"]
            or feedforward["heldout_seed_range"] != gru["heldout_seed_range"]
        ):
            raise ValueError(f"Matched protocol differs for seed {seed}")
        feedforward_parameters.add(int(feedforward["parameter_count"]))
        gru_parameters.add(int(gru["parameter_count"]))
        per_seed[str(seed)] = {
            architecture: {
                "result": str(paths[architecture]),
                "result_sha256": sha256(paths[architecture]),
                "best_checkpoint_episode": int(
                    payloads[architecture]["best_checkpoint_extra"]["episode"]
                ),
            }
            for architecture in paths
        }

    if len(feedforward_parameters) != 1 or len(gru_parameters) != 1:
        raise ValueError("Parameter count varies across training seeds")
    return {
        "all_checks_passed": True,
        "n_training_seeds": len(seeds),
        "training_episodes_per_agent": episodes,
        "environment_steps_per_agent": steps,
        "optimizer_updates_per_agent": updates,
        "episode_replay": expected_replay,
        "loss_transitions_per_update": 150,
        "updates_deferred_to_episode_boundary": True,
        "feedforward_parameter_count": feedforward_parameters.pop(),
        "gru_parameter_count": gru_parameters.pop(),
        "per_seed": per_seed,
    }


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
        raise FileNotFoundError(f"Missing memory-ablation checkpoint: {checkpoint_path}")
    architecture = spec["architecture"]
    if architecture == "feedforward_episode_replay":
        agent, checkpoint_extra = EpisodeReplayDQNAgent.load(checkpoint_path, device)

        def policy_factory(episode_seed: int):
            return lambda observation, step: agent.select_action(observation, epsilon=0.0)

    elif architecture == "gru_episode_replay":
        agent, checkpoint_extra = RecurrentDQNAgent.load(checkpoint_path, device)

        def policy_factory(episode_seed: int):
            agent.reset_hidden()
            return lambda observation, step: agent.select_action(observation, epsilon=0.0)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    expected_episode = int(cfg["dqn"]["train_episodes"])
    actual_episode = int(checkpoint_extra["episode"])
    if spec["checkpoint_kind"] == "final" and actual_episode != expected_episode:
        raise ValueError(f"Final checkpoint is not episode {expected_episode}")
    if spec["checkpoint_kind"] == "best" and actual_episode > expected_episode:
        raise ValueError("Selected checkpoint exceeds the training budget")
    if not agent.config.double_dqn or not np.isclose(
        agent.config.gamma, float(cfg["rl"]["gamma"])
    ):
        raise ValueError(f"Checkpoint algorithm mismatch: {checkpoint_path}")

    summary = evaluate_policy(
        spec["name"],
        policy_factory,
        env_config(cfg, "oracle"),
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
            "parameter_count": agent.parameter_count,
            "gamma": float(agent.config.gamma),
            "double_dqn": True,
            "replay_scheme": "complete_episode",
        }
    )
    return summary


def compare_across_training_seeds(
    rows: List[dict],
    checkpoint_kind: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    architectures = ("feedforward_episode_replay", "gru_episode_replay")
    selected = {
        (row["architecture"], int(row["training_seed"])): float(
            row["final_mastery"]
        )
        for row in rows
        if row["checkpoint_kind"] == checkpoint_kind
    }
    feedforward_seeds = sorted(
        seed for architecture, seed in selected if architecture == architectures[0]
    )
    gru_seeds = sorted(
        seed for architecture, seed in selected if architecture == architectures[1]
    )
    if feedforward_seeds != gru_seeds or len(feedforward_seeds) < 2:
        raise ValueError("Architectures require at least two identical training seeds")
    feedforward = np.asarray(
        [selected[(architectures[0], seed)] for seed in feedforward_seeds]
    )
    gru = np.asarray([selected[(architectures[1], seed)] for seed in feedforward_seeds])
    differences = gru - feedforward
    rng = np.random.RandomState(bootstrap_seed)
    indices = rng.randint(
        0, len(feedforward_seeds), size=(bootstrap_resamples, len(feedforward_seeds))
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
            "Paired bootstrap resamples trained agents by matched training seed; "
            "both architectures use the same complete-episode replay scheme."
        ),
    }


def plot_comparison(rows: List[dict], summaries: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    labels = {
        "feedforward_episode_replay": "Feed-forward",
        "gru_episode_replay": "GRU",
    }
    colors = {
        "feedforward_episode_replay": "#4c78a8",
        "gru_episode_replay": "#f58518",
    }
    for axis, checkpoint_kind in zip(axes, ("best", "final")):
        for architecture in labels:
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
                    f"{labels[architecture]}: "
                    f"{summary['mean_final_mastery_across_agents']:.4f} +/- "
                    f"{summary['std_final_mastery_across_agents']:.4f} SD"
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
    fig.suptitle("Memory ablation with matched complete-episode replay")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(config_path: str, device_name: str) -> None:
    cfg = load_config(config_path)
    evaluation = cfg["memory_ablation_evaluation"]
    training_seeds = [int(seed) for seed in evaluation["training_seeds"]]
    checkpoint_kinds = list(evaluation["checkpoint_kinds"])
    n_episodes = int(evaluation["evaluation_episodes"])
    bootstrap_resamples = int(evaluation["bootstrap_resamples"])
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
    bootstrap_seed = int(cfg["seed"]) + 53_000
    training_audit = audit_training_artifacts(cfg)

    specs = memory_ablation_specs(cfg)
    print(
        "Matched episode-replay memory ablation | "
        f"seeds={training_seeds} | heldout_episodes={n_episodes} | device={device}"
    )
    print("Training audit passed: both agents use 150 transitions and 62,000 updates.")
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

    architectures = ("feedforward_episode_replay", "gru_episode_replay")
    by_key: Dict[tuple, dict] = {
        (row["checkpoint_kind"], row["architecture"], row["training_seed"]): row
        for row in rows
    }
    summaries = {
        checkpoint_kind: {
            architecture: summarize_architecture(
                rows, architecture, checkpoint_kind
            )
            for architecture in architectures
        }
        for checkpoint_kind in checkpoint_kinds
    }
    comparisons = {}
    for kind_index, checkpoint_kind in enumerate(checkpoint_kinds):
        paired = [
            paired_difference(
                by_key[(checkpoint_kind, architectures[1], seed)],
                by_key[(checkpoint_kind, architectures[0], seed)],
                bootstrap_seed=bootstrap_seed + 5_000 + 100 * kind_index + index,
                bootstrap_resamples=bootstrap_resamples,
            )
            for index, seed in enumerate(training_seeds)
        ]
        across = compare_across_training_seeds(
            rows,
            checkpoint_kind,
            bootstrap_resamples,
            bootstrap_seed + 7_000 + kind_index,
        )
        comparisons[checkpoint_kind] = {
            "paired_heldout_episode_differences_by_training_seed": paired,
            "across_training_seeds": across,
        }
        feedforward = summaries[checkpoint_kind][architectures[0]]
        gru = summaries[checkpoint_kind][architectures[1]]
        low, high = across["mean_difference_ci95_training_seed_bootstrap"]
        print(
            f"{checkpoint_kind}: feed-forward mean/std="
            f"{feedforward['mean_final_mastery_across_agents']:.4f}/"
            f"{feedforward['std_final_mastery_across_agents']:.4f} | "
            f"GRU mean/std={gru['mean_final_mastery_across_agents']:.4f}/"
            f"{gru['std_final_mastery_across_agents']:.4f}"
        )
        print(
            f"  GRU difference={across['mean_of_seed_level_differences']:+.4f} "
            f"CI [{low:+.4f}, {high:+.4f}] | "
            f"wins={across['gru_better_seed_count']}/{len(training_seeds)}"
        )

    result_path = results_dir / "memory_ablation_comparison.json"
    figure_path = results_dir / "figures" / "memory_ablation_comparison.png"
    plot_comparison(rows, summaries, figure_path)
    payload = {
        "disclaimer": (
            "Oracle-state simulator memory ablation only. No intervention "
            "policy was evaluated on real students."
        ),
        "research_question": (
            "Does GRU history improve final mastery when replay, loss-batch "
            "size, update count, and training protocol are matched?"
        ),
        "controlled_comparison": (
            "Both architectures use complete 50-step episode replay, three "
            "episodes (150 transitions) per loss, 62,000 episode-boundary "
            "updates, Double DQN, 5,000 episodes, matched seeds, and identical "
            "environment/evaluation protocols. Only the near-parameter-matched "
            "MLP versus GRU state representation differs."
        ),
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
                "feedforward_agent": sha256(Path("src/rl/episode_replay_dqn.py")),
                "gru_agent": sha256(Path("src/rl/recurrent_dqn.py")),
                "feedforward_training_script": sha256(
                    Path("scripts/train_episode_replay_dqn.py")
                ),
                "gru_training_script": sha256(Path("scripts/train_recurrent_dqn.py")),
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
    parser.add_argument("--config", default="configs/memory_ablation_evaluation.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    arguments = parser.parse_args()
    main(arguments.config, arguments.device)
