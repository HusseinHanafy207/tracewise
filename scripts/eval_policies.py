"""Offline / simulated comparison of adaptive tutoring policies.

Policies:
  - random
  - rule-based (mastery thresholds)
  - epsilon-greedy contextual bandit (with student state)
  - LinUCB contextual bandit (with student state)
  - ablation: epsilon-greedy WITHOUT mastery features (no student state)

IMPORTANT: results are simulated under the documented INTERVENTION_EFFECT
assumptions in src/policy/simulator.py. They are NOT evidence of real
student learning gains.

Usage:
    python scripts/eval_policies.py --config configs/config.yaml
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.bandit import EpsilonGreedyBandit, LinUCB
from src.policy.random_policy import RandomPolicy
from src.policy.rule_based import RuleBasedPolicy
from src.policy.simulator import (
    bandit_context,
    run_policy_simulation,
    state_from_simulated,
)
from src.utils.seed import load_config, set_seed


def _mean_std(rows: List[dict], key: str):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def _aggregate_action_counts(rows: List[dict], actions: List[str]) -> Dict[str, float]:
    totals = {a: 0.0 for a in actions}
    for r in rows:
        for a, c in r.get("action_counts", {}).items():
            totals[a] = totals.get(a, 0.0) + c
    grand = sum(totals.values()) or 1.0
    return {a: totals[a] / grand for a in actions}


def run_multi_seed(
    name: str,
    policy_factory: Callable,
    context_fn: Callable,
    n_students: int,
    n_interactions: int,
    seeds: List[int],
    actions: List[str],
) -> dict:
    rows = []
    for seed in seeds:
        policy = policy_factory(seed)
        result = run_policy_simulation(
            policy,
            n_students=n_students,
            n_interactions=n_interactions,
            context_fn=context_fn,
            seed=seed,
        )
        rows.append(result)

    summary = {"name": name, "n_seeds": len(seeds)}
    for key in ("mean_reward", "mean_final_mastery", "overall_accuracy", "final_accuracy"):
        mu, sd = _mean_std(rows, key)
        summary[key] = mu
        summary[f"{key}_std"] = sd

    curves = np.stack([np.asarray(r["accuracy_curve"], dtype=np.float64) for r in rows])
    summary["accuracy_curve_mean"] = curves.mean(axis=0).tolist()
    summary["action_freq"] = _aggregate_action_counts(rows, actions)
    return summary


def print_table(summaries: List[dict]) -> str:
    header = (
        "| Policy | mean mastery gain | final mastery | overall acc | final-window acc |\n"
        "|---|---|---|---|---|\n"
    )
    rows = ""
    for s in summaries:
        rows += (
            f"| {s['name']} | "
            f"{s['mean_reward']:.4f} ± {s['mean_reward_std']:.4f} | "
            f"{s['mean_final_mastery']:.3f} ± {s['mean_final_mastery_std']:.3f} | "
            f"{s['overall_accuracy']:.3f} ± {s['overall_accuracy_std']:.3f} | "
            f"{s['final_accuracy']:.3f} ± {s['final_accuracy_std']:.3f} |\n"
        )
    table = header + rows
    print(table)
    return table


def plot_learning_curves(summaries: List[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for s in summaries:
        curve = s.get("accuracy_curve_mean") or []
        if not curve:
            continue
        ax.plot(curve, label=s["name"], linewidth=2)
    ax.set_xlabel("Interaction t")
    ax.set_ylabel("Mean accuracy across students")
    ax.set_title("Simulated learning curves (offline)")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_action_freq(summaries: List[dict], actions: List[str], out_path: Path) -> None:
    # one grouped bar chart for policies that actually choose
    names = [s["name"] for s in summaries]
    x = np.arange(len(actions))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, s in enumerate(names):
        freqs = [summaries[i]["action_freq"].get(a, 0.0) for a in actions]
        ax.bar(x + i * width, freqs, width, label=s)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(actions, rotation=20, ha="right")
    ax.set_ylabel("Selection frequency")
    ax.set_title("Intervention mix (simulated)")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    actions = list(cfg["policy"]["actions"])
    low = float(cfg["policy"]["rule_thresholds"]["low_mastery"])
    high = float(cfg["policy"]["rule_thresholds"]["high_mastery"])
    bandit_cfg = cfg["policy"]["bandit"]
    context_dim = int(bandit_cfg["context_dim"])
    epsilon = float(bandit_cfg["epsilon"])
    alpha = float(bandit_cfg.get("alpha", 1.0))

    n_students = int(cfg["simulator"]["n_students"])
    n_interactions = int(cfg["simulator"]["n_interactions_per_student"])
    n_seeds = int(cfg["simulator"].get("n_seeds", 5))
    seeds = [cfg["seed"] + i for i in range(n_seeds)]

    print(
        f"Simulated policy eval | students={n_students} | "
        f"T={n_interactions} | seeds={n_seeds}"
    )
    print(
        "DISCLAIMER: offline simulation under stated intervention-effect "
        "assumptions — not real student learning."
    )

    def ctx_full(student, difficulty):
        return bandit_context(student, difficulty, include_mastery=True)

    def ctx_nostate(student, difficulty):
        return bandit_context(student, difficulty, include_mastery=False)

    def ctx_rule(student, difficulty):
        return state_from_simulated(student)

    def ctx_random(student, difficulty):
        return None

    summaries = []

    summaries.append(
        run_multi_seed(
            "random",
            lambda seed: RandomPolicy(actions, seed=seed),
            ctx_random,
            n_students,
            n_interactions,
            seeds,
            actions,
        )
    )
    summaries.append(
        run_multi_seed(
            "rule_based",
            lambda seed: RuleBasedPolicy(low_thresh=low, high_thresh=high),
            ctx_rule,
            n_students,
            n_interactions,
            seeds,
            actions,
        )
    )
    summaries.append(
        run_multi_seed(
            "eps_greedy",
            lambda seed: EpsilonGreedyBandit(
                actions, context_dim, epsilon=epsilon, seed=seed
            ),
            ctx_full,
            n_students,
            n_interactions,
            seeds,
            actions,
        )
    )
    summaries.append(
        run_multi_seed(
            "linucb",
            lambda seed: LinUCB(actions, context_dim, alpha=alpha),
            ctx_full,
            n_students,
            n_interactions,
            seeds,
            actions,
        )
    )
    # Ablation: same bandit, context without mastery features
    summaries.append(
        run_multi_seed(
            "eps_greedy_no_state",
            lambda seed: EpsilonGreedyBandit(
                actions, context_dim, epsilon=epsilon, seed=seed + 1000
            ),
            ctx_nostate,
            n_students,
            n_interactions,
            seeds,
            actions,
        )
    )

    table = print_table(summaries)

    print("\n### Action mix (fraction of steps)")
    print("| Policy | " + " | ".join(actions) + " |")
    print("|---|" + "|".join(["---"] * len(actions)) + "|")
    for s in summaries:
        cells = " | ".join(f"{s['action_freq'].get(a, 0.0):.2f}" for a in actions)
        print(f"| {s['name']} | {cells} |")

    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(summaries, fig_dir / "policy_learning_curves.png")
    plot_action_freq(summaries, actions, fig_dir / "policy_action_mix.png")
    print(f"\nWrote figures to {fig_dir}")

    # drop long curves from the JSON payload? keep them — useful
    payload = {
        "disclaimer": (
            "Simulated learning outcomes under state-dependent "
            "INTERVENTION_EFFECT assumptions in src/policy/simulator.py. "
            "Not evidence of real student learning."
        ),
        "n_students": n_students,
        "n_interactions": n_interactions,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "actions": actions,
        "policies": summaries,
        "markdown_table": table,
    }
    out = results_dir / "policy_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
