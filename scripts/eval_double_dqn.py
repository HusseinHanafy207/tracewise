"""Compare vanilla DQN and Double DQN across matched training/evaluation seeds."""
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

from scripts.eval_dqn_suite import (
    git_provenance,
    load_dqn_condition,
    sha256,
)
from src.rl.evaluation import paired_difference
from src.utils.seed import load_config, set_seed


def comparison_specs(cfg: dict) -> List[dict]:
    evaluation = cfg["double_dqn_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    if len(seeds) < 2:
        raise ValueError("At least two training seeds are required for stability")
    if seeds[0] != int(cfg["seed"]):
        raise ValueError("training_seeds must start with the canonical seed")
    gamma = float(cfg["rl"]["gamma"])
    specs = []
    for algorithm, double_dqn in (("dqn", False), ("double_dqn", True)):
        for index, seed in enumerate(seeds):
            if double_dqn:
                suffix = f"_double_seed{seed}"
            else:
                suffix = "" if index == 0 else f"_seed{seed}"
            specs.append(
                {
                    "name": f"{algorithm}_oracle_seed{seed}",
                    "checkpoint": f"dqn_oracle_best{suffix}.pt",
                    "condition": "oracle",
                    "training_seed": seed,
                    "gamma": gamma,
                    "double_dqn": double_dqn,
                }
            )
    return specs


def summarize_stability(rows: List[dict], algorithm: str) -> dict:
    selected = sorted(
        (row for row in rows if row["algorithm"] == algorithm),
        key=lambda row: row["training_seed"],
    )
    values = np.asarray(
        [row["final_mastery"] for row in selected], dtype=np.float64
    )
    if not len(values):
        raise ValueError(f"No evaluation rows for {algorithm}")
    return {
        "algorithm": algorithm,
        "n_training_seeds": len(selected),
        "training_seeds": [row["training_seed"] for row in selected],
        "final_mastery_by_seed": {
            str(row["training_seed"]): row["final_mastery"] for row in selected
        },
        "mean_final_mastery_across_agents": float(values.mean()),
        "std_final_mastery_across_agents": float(
            values.std(ddof=1) if len(values) > 1 else 0.0
        ),
        "range_final_mastery_across_agents": float(values.max() - values.min()),
        "selected_checkpoint_episode_by_seed": {
            str(row["training_seed"]): int(row["checkpoint_extra"]["episode"])
            for row in selected
        },
    }


def compare_across_training_seeds(
    rows: List[dict],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    by_key = {
        (row["algorithm"], int(row["training_seed"])): float(row["final_mastery"])
        for row in rows
    }
    vanilla_seeds = sorted(seed for algorithm, seed in by_key if algorithm == "dqn")
    double_seeds = sorted(
        seed for algorithm, seed in by_key if algorithm == "double_dqn"
    )
    if vanilla_seeds != double_seeds or len(vanilla_seeds) < 2:
        raise ValueError("Algorithms require at least two identical training seeds")
    vanilla = np.asarray(
        [by_key[("dqn", seed)] for seed in vanilla_seeds], dtype=np.float64
    )
    double = np.asarray(
        [by_key[("double_dqn", seed)] for seed in vanilla_seeds],
        dtype=np.float64,
    )
    differences = double - vanilla
    rng = np.random.RandomState(bootstrap_seed)
    indices = rng.randint(
        0,
        len(vanilla_seeds),
        size=(bootstrap_resamples, len(vanilla_seeds)),
    )
    mean_differences = differences[indices].mean(axis=1)
    std_differences = (
        double[indices].std(axis=1, ddof=1)
        - vanilla[indices].std(axis=1, ddof=1)
    )
    return {
        "replication_unit": "training_seed",
        "training_seeds": vanilla_seeds,
        "n_training_seeds": len(vanilla_seeds),
        "mean_of_seed_level_differences": float(differences.mean()),
        "mean_difference_ci95_training_seed_bootstrap": [
            float(np.quantile(mean_differences, 0.025)),
            float(np.quantile(mean_differences, 0.975)),
        ],
        "std_of_seed_level_differences": float(differences.std(ddof=1)),
        "median_seed_level_difference": float(np.median(differences)),
        "double_dqn_better_seed_count": int(np.sum(differences > 0.0)),
        "double_dqn_tied_seed_count": int(np.sum(differences == 0.0)),
        "double_minus_vanilla_std": float(double.std(ddof=1) - vanilla.std(ddof=1)),
        "std_difference_ci95_training_seed_bootstrap": [
            float(np.quantile(std_differences, 0.025)),
            float(np.quantile(std_differences, 0.975)),
        ],
        "std_ratio_double_over_vanilla": float(
            double.std(ddof=1) / vanilla.std(ddof=1)
        ),
        "note": (
            "Paired bootstrap resamples training seeds, not the 500 episodes "
            "within a fixed trained agent."
        ),
    }


def plot_comparison(
    rows: List[dict],
    paired: List[dict],
    path: Path,
) -> None:
    seeds = sorted({int(row["training_seed"]) for row in rows})
    by_key = {
        (row["algorithm"], int(row["training_seed"])): row for row in rows
    }
    positions = np.arange(len(seeds))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    for offset, (algorithm, label, color) in enumerate(
        (("dqn", "Vanilla DQN", "#4c78a8"), ("double_dqn", "Double DQN", "#f58518"))
    ):
        values = [by_key[(algorithm, seed)]["final_mastery"] for seed in seeds]
        axes[0].bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([str(seed) for seed in seeds])
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Mean held-out final mastery")
    axes[0].set_title("Performance on 500 matched episodes")
    axes[0].legend()

    differences = [row["mean_difference"] for row in paired]
    lower = [
        row["mean_difference"] - row["difference_ci95"][0] for row in paired
    ]
    upper = [
        row["difference_ci95"][1] - row["mean_difference"] for row in paired
    ]
    axes[1].bar(positions, differences, color="#54a24b")
    axes[1].errorbar(
        positions,
        differences,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(seed) for seed in seeds])
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Double DQN - DQN final mastery")
    axes[1].set_title("Paired episode difference (95% CI)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(config_path: str, device_name: str, tag: str) -> None:
    cfg = load_config(config_path)
    evaluation = cfg["double_dqn_evaluation"]
    n_episodes = int(evaluation["evaluation_episodes"])
    bootstrap_resamples = int(evaluation["bootstrap_resamples"])
    if n_episodes <= 0 or bootstrap_resamples <= 0:
        raise ValueError("evaluation_episodes and bootstrap_resamples must be positive")

    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg["dqn"].get("torch_num_threads", 1)))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    heldout_start = int(cfg["dqn"]["heldout_seed_start"])
    episode_seeds = [heldout_start + index for index in range(n_episodes)]
    bootstrap_seed = int(cfg["seed"]) + 17_000

    print(
        "Vanilla DQN vs Double DQN | "
        f"training_seeds={evaluation['training_seeds']} | "
        f"heldout_episodes={n_episodes} | device={device}"
    )
    print(
        "DISCLAIMER: oracle-state simulator evaluation only; no real-student "
        "learning claim."
    )

    rows = []
    for offset, spec in enumerate(comparison_specs(cfg)):
        print(f"  evaluating {spec['name']} ...", flush=True)
        rows.append(
            load_dqn_condition(
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
        (row["algorithm"], int(row["training_seed"])): row for row in rows
    }
    training_seeds = [int(seed) for seed in evaluation["training_seeds"]]
    paired = [
        paired_difference(
            by_key[("double_dqn", seed)],
            by_key[("dqn", seed)],
            bootstrap_seed=bootstrap_seed + 2_000 + index,
            bootstrap_resamples=bootstrap_resamples,
        )
        for index, seed in enumerate(training_seeds)
    ]
    stability = {
        "dqn": summarize_stability(rows, "dqn"),
        "double_dqn": summarize_stability(rows, "double_dqn"),
        "double_minus_vanilla_across_training_seeds": compare_across_training_seeds(
            rows,
            bootstrap_resamples,
            bootstrap_seed + 3_000,
        ),
    }

    print("\nseed   vanilla    double     paired difference [95% CI]")
    for seed, effect in zip(training_seeds, paired):
        vanilla = by_key[("dqn", seed)]["final_mastery"]
        double = by_key[("double_dqn", seed)]["final_mastery"]
        low, high = effect["difference_ci95"]
        print(
            f"{seed:<6} {vanilla:.4f}     {double:.4f}     "
            f"{effect['mean_difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
        )
    for algorithm in ("dqn", "double_dqn"):
        summary = stability[algorithm]
        print(
            f"{algorithm:<11} across-seed mean="
            f"{summary['mean_final_mastery_across_agents']:.4f} | std="
            f"{summary['std_final_mastery_across_agents']:.4f}"
        )
    across = stability["double_minus_vanilla_across_training_seeds"]
    low, high = across["mean_difference_ci95_training_seed_bootstrap"]
    std_low, std_high = across["std_difference_ci95_training_seed_bootstrap"]
    print(
        "training-seed bootstrap: mean difference="
        f"{across['mean_of_seed_level_differences']:+.4f} "
        f"CI [{low:+.4f}, {high:+.4f}] | std difference="
        f"{across['double_minus_vanilla_std']:+.4f} "
        f"CI [{std_low:+.4f}, {std_high:+.4f}]"
    )

    suffix = f"_{tag}" if tag else ""
    result_path = results_dir / f"double_dqn_comparison{suffix}.json"
    figure_path = results_dir / "figures" / f"double_dqn_comparison{suffix}.png"
    plot_comparison(rows, paired, figure_path)
    payload = {
        "disclaimer": (
            "Simulated oracle-state comparison only. No intervention policy "
            "was evaluated on real students."
        ),
        "controlled_change": (
            "Double DQN uses the online network to select the bootstrap action "
            "and the target network to evaluate it; all other configured "
            "training and evaluation settings match vanilla DQN."
        ),
        "training_episodes_per_agent": int(cfg["dqn"]["train_episodes"]),
        "training_seeds": training_seeds,
        "n_heldout_episodes": n_episodes,
        "heldout_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "bootstrap_resamples": bootstrap_resamples,
        "conditions": rows,
        "double_dqn_minus_vanilla_by_training_seed": paired,
        "stability": stability,
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "dqn": sha256(Path("src/rl/dqn.py")),
                "training_script": sha256(Path("scripts/train_dqn.py")),
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
    parser.add_argument("--config", default="configs/double_dqn_evaluation.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()
    main(arguments.config, arguments.device, arguments.tag)
