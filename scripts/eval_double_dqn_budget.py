"""Compare Double DQN trained for 2,000 versus 5,000 episodes."""
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

from scripts.eval_dqn_suite import git_provenance, load_dqn_condition, sha256
from src.rl.evaluation import paired_difference
from src.utils.seed import load_config, set_seed


def budget_specs(cfg: dict) -> List[dict]:
    evaluation = cfg["training_budget_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    budgets = [int(budget) for budget in evaluation["budgets"]]
    checkpoint_kinds = list(evaluation["checkpoint_kinds"])
    if len(seeds) < 2:
        raise ValueError("At least two matched training seeds are required")
    if len(budgets) != 2 or budgets != sorted(budgets):
        raise ValueError("budgets must contain two increasing values")
    if checkpoint_kinds != ["best", "final"]:
        raise ValueError("checkpoint_kinds must be [best, final]")

    tag_templates = {
        budgets[0]: evaluation["reference_tag_template"],
        budgets[1]: evaluation["candidate_tag_template"],
    }
    specs = []
    for checkpoint_kind in checkpoint_kinds:
        for budget in budgets:
            for seed in seeds:
                tag = str(tag_templates[budget]).format(seed=seed)
                specs.append(
                    {
                        "name": (
                            f"double_dqn_{budget}_{checkpoint_kind}_seed{seed}"
                        ),
                        "checkpoint": f"dqn_oracle_{checkpoint_kind}_{tag}.pt",
                        "condition": "oracle",
                        "training_seed": seed,
                        "training_episodes": budget,
                        "checkpoint_kind": checkpoint_kind,
                        "gamma": float(cfg["rl"]["gamma"]),
                        "double_dqn": True,
                    }
                )
    return specs


def audit_training_artifacts(cfg: dict) -> dict:
    """Verify the larger budget extends each matched deterministic run."""
    evaluation = cfg["training_budget_evaluation"]
    seeds = [int(seed) for seed in evaluation["training_seeds"]]
    reference_budget, candidate_budget = [
        int(budget) for budget in evaluation["budgets"]
    ]
    results_dir = Path(cfg["paths"]["results_dir"])
    per_seed = {}
    for seed in seeds:
        reference_tag = str(evaluation["reference_tag_template"]).format(seed=seed)
        candidate_tag = str(evaluation["candidate_tag_template"]).format(seed=seed)
        reference_path = results_dir / f"dqn_oracle_training_{reference_tag}.json"
        candidate_path = results_dir / f"dqn_oracle_training_{candidate_tag}.json"
        if not reference_path.exists() or not candidate_path.exists():
            raise FileNotFoundError(
                f"Missing matched training artifacts for seed {seed}: "
                f"{reference_path}, {candidate_path}"
            )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        expected = (
            (reference, reference_budget, reference_budget * int(cfg["rl"]["horizon"])),
            (candidate, candidate_budget, candidate_budget * int(cfg["rl"]["horizon"])),
        )
        for payload, episodes, steps in expected:
            if (
                int(payload["seed"]) != seed
                or payload["algorithm"] != "double_dqn"
                or not bool(payload["effective_agent_config"]["double_dqn"])
                or int(payload["train_episodes"]) != episodes
                or int(payload["global_steps"]) != steps
                or len(payload["training_history"]) != episodes
                or int(payload["training_history"][-1]["episode"]) != episodes
            ):
                raise ValueError(f"Training metadata audit failed for seed {seed}")

        candidate_prefix = candidate["training_history"][:reference_budget]
        candidate_validation_prefix = [
            row
            for row in candidate["validation_history"]
            if int(row["episode"]) <= reference_budget
        ]
        if reference["training_history"] != candidate_prefix:
            raise ValueError(f"Training trajectory prefix differs for seed {seed}")
        if reference["validation_history"] != candidate_validation_prefix:
            raise ValueError(f"Validation trajectory prefix differs for seed {seed}")

        reference_config = dict(reference["dqn_config"])
        candidate_config = dict(candidate["dqn_config"])
        reference_config["train_episodes"] = candidate_budget
        if reference_config != candidate_config:
            raise ValueError(f"Training config differs beyond budget for seed {seed}")
        per_seed[str(seed)] = {
            "reference_result": str(reference_path),
            "reference_result_sha256": sha256(reference_path),
            "candidate_result": str(candidate_path),
            "candidate_result_sha256": sha256(candidate_path),
            "training_prefix_exact": True,
            "validation_prefix_exact": True,
            "reference_best_checkpoint_episode": int(
                reference["best_checkpoint_extra"]["episode"]
            ),
            "candidate_best_checkpoint_episode": int(
                candidate["best_checkpoint_extra"]["episode"]
            ),
        }
    return {
        "all_checks_passed": True,
        "n_matched_training_seeds": len(seeds),
        "reference_training_episodes": reference_budget,
        "candidate_training_episodes": candidate_budget,
        "first_2000_training_and_validation_records_exact": True,
        "per_seed": per_seed,
    }


def summarize_budget(rows: List[dict], budget: int, checkpoint_kind: str) -> dict:
    selected = sorted(
        (
            row
            for row in rows
            if int(row["training_episodes"]) == budget
            and row["checkpoint_kind"] == checkpoint_kind
        ),
        key=lambda row: row["training_seed"],
    )
    values = np.asarray(
        [row["final_mastery"] for row in selected], dtype=np.float64
    )
    if len(values) < 2:
        raise ValueError("Each budget requires at least two trained agents")
    return {
        "training_episodes": budget,
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
        "checkpoint_episode_by_seed": {
            str(row["training_seed"]): int(row["checkpoint_extra"]["episode"])
            for row in selected
        },
    }


def compare_budgets_across_training_seeds(
    rows: List[dict],
    checkpoint_kind: str,
    reference_budget: int,
    candidate_budget: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    selected = {
        (int(row["training_episodes"]), int(row["training_seed"])): float(
            row["final_mastery"]
        )
        for row in rows
        if row["checkpoint_kind"] == checkpoint_kind
    }
    reference_seeds = sorted(
        seed for budget, seed in selected if budget == reference_budget
    )
    candidate_seeds = sorted(
        seed for budget, seed in selected if budget == candidate_budget
    )
    if reference_seeds != candidate_seeds or len(reference_seeds) < 2:
        raise ValueError("Budgets require at least two identical training seeds")

    reference = np.asarray(
        [selected[(reference_budget, seed)] for seed in reference_seeds],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [selected[(candidate_budget, seed)] for seed in reference_seeds],
        dtype=np.float64,
    )
    differences = candidate - reference
    rng = np.random.RandomState(bootstrap_seed)
    indices = rng.randint(
        0,
        len(reference_seeds),
        size=(bootstrap_resamples, len(reference_seeds)),
    )
    mean_differences = differences[indices].mean(axis=1)
    std_differences = (
        candidate[indices].std(axis=1, ddof=1)
        - reference[indices].std(axis=1, ddof=1)
    )
    return {
        "replication_unit": "training_seed",
        "checkpoint_kind": checkpoint_kind,
        "reference_training_episodes": reference_budget,
        "candidate_training_episodes": candidate_budget,
        "training_seeds": reference_seeds,
        "n_training_seeds": len(reference_seeds),
        "candidate_minus_reference_by_seed": {
            str(seed): float(value)
            for seed, value in zip(reference_seeds, differences)
        },
        "mean_of_seed_level_differences": float(differences.mean()),
        "mean_difference_ci95_training_seed_bootstrap": [
            float(np.quantile(mean_differences, 0.025)),
            float(np.quantile(mean_differences, 0.975)),
        ],
        "median_seed_level_difference": float(np.median(differences)),
        "candidate_better_seed_count": int(np.sum(differences > 0.0)),
        "candidate_tied_seed_count": int(np.sum(differences == 0.0)),
        "candidate_minus_reference_std": float(
            candidate.std(ddof=1) - reference.std(ddof=1)
        ),
        "std_difference_ci95_training_seed_bootstrap": [
            float(np.quantile(std_differences, 0.025)),
            float(np.quantile(std_differences, 0.975)),
        ],
        "std_ratio_candidate_over_reference": float(
            candidate.std(ddof=1) / reference.std(ddof=1)
        ),
        "note": (
            "Paired bootstrap resamples trained agents by training seed; the "
            "held-out episodes within one checkpoint are not algorithm "
            "replications."
        ),
    }


def plot_budget_comparison(
    rows: List[dict],
    summaries: dict,
    path: Path,
) -> None:
    checkpoint_kinds = ("best", "final")
    budgets = sorted({int(row["training_episodes"]) for row in rows})
    colors = {budgets[0]: "#4c78a8", budgets[1]: "#f58518"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for axis, checkpoint_kind in zip(axes, checkpoint_kinds):
        for budget in budgets:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["checkpoint_kind"] == checkpoint_kind
                    and int(row["training_episodes"]) == budget
                ),
                key=lambda row: row["training_seed"],
            )
            summary = summaries[checkpoint_kind][str(budget)]
            axis.plot(
                [row["training_seed"] for row in selected],
                [row["final_mastery"] for row in selected],
                marker="o",
                color=colors[budget],
                label=(
                    f"{budget:,} episodes: "
                    f"{summary['mean_final_mastery_across_agents']:.4f} "
                    f"± {summary['std_final_mastery_across_agents']:.4f} SD"
                ),
            )
        axis.set_xlabel("Matched training seed")
        axis.set_title(
            "Validation-selected checkpoint"
            if checkpoint_kind == "best"
            else "Exact budget endpoint"
        )
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean held-out final mastery")
    fig.suptitle("Double DQN training-budget comparison")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(config_path: str, device_name: str) -> None:
    cfg = load_config(config_path)
    evaluation = cfg["training_budget_evaluation"]
    budgets = [int(budget) for budget in evaluation["budgets"]]
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
    bootstrap_seed = int(cfg["seed"]) + 31_000

    training_audit = audit_training_artifacts(cfg)
    print(
        "Training audit passed: all candidate runs exactly reproduce their "
        "matched first 2,000 training and validation records."
    )
    specs = budget_specs(cfg)
    print(
        "Double DQN training-budget evaluation | "
        f"budgets={budgets} | seeds={training_seeds} | "
        f"heldout_episodes={n_episodes} | device={device}"
    )
    print(
        "DISCLAIMER: oracle-state simulator evaluation only; no real-student "
        "learning claim."
    )

    rows = []
    for offset, spec in enumerate(specs):
        print(f"  evaluating {spec['name']} ...", flush=True)
        row = load_dqn_condition(
            spec,
            checkpoint_dir,
            cfg,
            device,
            episode_seeds,
            bootstrap_seed + 100 * offset,
            bootstrap_resamples,
        )
        row["training_episodes"] = int(spec["training_episodes"])
        row["checkpoint_kind"] = spec["checkpoint_kind"]
        rows.append(row)

    by_key: Dict[tuple, dict] = {
        (
            row["checkpoint_kind"],
            int(row["training_episodes"]),
            int(row["training_seed"]),
        ): row
        for row in rows
    }
    summaries = {
        checkpoint_kind: {
            str(budget): summarize_budget(rows, budget, checkpoint_kind)
            for budget in budgets
        }
        for checkpoint_kind in checkpoint_kinds
    }
    comparisons = {}
    for kind_index, checkpoint_kind in enumerate(checkpoint_kinds):
        paired = [
            paired_difference(
                by_key[(checkpoint_kind, budgets[1], seed)],
                by_key[(checkpoint_kind, budgets[0], seed)],
                bootstrap_seed=bootstrap_seed + 5_000 + 100 * kind_index + index,
                bootstrap_resamples=bootstrap_resamples,
            )
            for index, seed in enumerate(training_seeds)
        ]
        across = compare_budgets_across_training_seeds(
            rows,
            checkpoint_kind,
            budgets[0],
            budgets[1],
            bootstrap_resamples,
            bootstrap_seed + 7_000 + kind_index,
        )
        comparisons[checkpoint_kind] = {
            "paired_heldout_episode_differences_by_training_seed": paired,
            "across_training_seeds": across,
        }

        reference = summaries[checkpoint_kind][str(budgets[0])]
        candidate = summaries[checkpoint_kind][str(budgets[1])]
        mean_low, mean_high = across[
            "mean_difference_ci95_training_seed_bootstrap"
        ]
        std_low, std_high = across[
            "std_difference_ci95_training_seed_bootstrap"
        ]
        print(
            f"{checkpoint_kind}: {budgets[0]} mean/std="
            f"{reference['mean_final_mastery_across_agents']:.4f}/"
            f"{reference['std_final_mastery_across_agents']:.4f} | "
            f"{budgets[1]} mean/std="
            f"{candidate['mean_final_mastery_across_agents']:.4f}/"
            f"{candidate['std_final_mastery_across_agents']:.4f}"
        )
        print(
            f"  mean difference={across['mean_of_seed_level_differences']:+.4f} "
            f"CI [{mean_low:+.4f}, {mean_high:+.4f}] | "
            f"std difference={across['candidate_minus_reference_std']:+.4f} "
            f"CI [{std_low:+.4f}, {std_high:+.4f}]"
        )

    result_path = results_dir / "double_dqn_budget_comparison.json"
    figure_path = results_dir / "figures" / "double_dqn_budget_comparison.png"
    plot_budget_comparison(rows, summaries, figure_path)
    payload = {
        "disclaimer": (
            "Simulated oracle-state Double DQN comparison only. No intervention "
            "policy was evaluated on real students."
        ),
        "research_question": (
            "Does increasing Double DQN training from 2,000 to 5,000 episodes "
            "improve mean held-out final mastery or reduce training-seed "
            "variability?"
        ),
        "controlled_change": (
            "Only dqn.train_episodes changes from 2000 to 5000. Seeds are "
            "matched, and each 5000-episode run repeats the same first 2000 "
            "training episodes before receiving 3000 additional episodes."
        ),
        "checkpoint_estimands": {
            "best": (
                "Best validation checkpoint available within each budget; the "
                "5000-episode condition has more checkpoint-selection chances."
            ),
            "final": (
                "Checkpoint at the exact budget endpoint; avoids the unequal "
                "number of validation-selection opportunities."
            ),
        },
        "training_budgets": budgets,
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
                "dqn": sha256(Path("src/rl/dqn.py")),
                "environment": sha256(Path("src/rl/environment.py")),
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
    parser.add_argument(
        "--config", default="configs/double_dqn_budget_evaluation.yaml"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    arguments = parser.parse_args()
    main(arguments.config, arguments.device)
