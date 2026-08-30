"""M6 Step 1 — calibrate delayed Independence/Fatigue dynamics.

Fixed open-loop / myopic policies (no DQN). Checks whether myopic high
immediate Δm can lose on final mastery m_T once delayed effects are on.

Usage:
    python scripts/calibrate_delayed_effects.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.simulator import (
    intervention_effect,
    run_policy_simulation,
)
from src.utils.seed import load_config, set_seed

ACTIONS = [
    "explain",
    "worked_example",
    "socratic_hint",
    "easier_problem",
    "harder_problem",
    "prerequisite_review",
]


class FixedActionPolicy:
    def __init__(self, action: str):
        self.action = action

    def select_action(self, context) -> str:
        return self.action


class MyopicOraclePolicy:
    """Argmax_a Effect(a, m) using true mastery — ignores I, F."""

    def select_action(self, student) -> str:
        m = float(student.mastery)
        best, best_e = ACTIONS[0], -1e9
        for a in ACTIONS:
            e = intervention_effect(a, m, student.prefs)
            if e > best_e:
                best, best_e = a, e
        return best


class ScaffoldThenChallengePolicy:
    """Support early, then challenge with periodic recovery (manages fatigue)."""

    def __init__(self, switch_t: int = 12):
        self.switch_t = switch_t
        self._t = 0

    def select_action(self, student) -> str:
        t = self._t
        self._t += 1
        m = float(student.mastery)
        if t < self.switch_t:
            if m < 0.35:
                return "prerequisite_review"
            return "explain"
        # long horizon: productive struggle + recovery every 3rd step
        if m < 0.30:
            return "explain"
        if t % 3 == 0:
            return "easier_problem"  # recover fatigue without much dependence
        if m < 0.50:
            return "socratic_hint" if t % 2 else "harder_problem"
        return "harder_problem"


class BalancedChallengePolicy:
    """Alternate challenge and light recovery after a short warm-up."""

    def __init__(self, warm_t: int = 8):
        self.warm_t = warm_t
        self._t = 0

    def select_action(self, student) -> str:
        t = self._t
        self._t += 1
        m = float(student.mastery)
        if t < self.warm_t:
            return "explain" if m < 0.45 else "worked_example"
        if t % 2 == 0:
            return "harder_problem" if m >= 0.35 else "explain"
        return "easier_problem" if t % 4 == 1 else "socratic_hint"


class InterleaveExplainHarderPolicy:
    def __init__(self):
        self._t = 0

    def select_action(self, student) -> str:
        t = self._t
        self._t += 1
        return "explain" if t % 2 == 0 else "harder_problem"


def _ctx_student(student, difficulty):
    """Pass the student object so oracle/hand policies can read mastery only."""
    return student


def run_named(
    name: str,
    factory: Callable,
    n_students: int,
    n_interactions: int,
    seeds: List[int],
    delayed: bool,
) -> dict:
    rows = []
    for seed in seeds:
        policy = factory()
        # reset step counters for stateful policies
        if hasattr(policy, "_t"):
            policy._t = 0
        result = run_policy_simulation(
            policy,
            n_students=n_students,
            n_interactions=n_interactions,
            context_fn=_ctx_student,
            seed=seed,
            delayed_effects=delayed,
        )
        # per-student policies that count steps need fresh instance per student;
        # FixedAction/Myopic are fine; Scaffold resets _t only once per seed.
        # Re-run with per-student reset via wrapper below if needed.
        rows.append(result)

    def mean_std(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    summary = {"name": name, "delayed_effects": delayed, "n_seeds": len(seeds)}
    for key in (
        "mean_reward",
        "mean_final_mastery",
        "early_mean_reward",
        "late_mean_reward",
        "overall_accuracy",
        "mean_final_independence",
        "mean_final_fatigue",
    ):
        mu, sd = mean_std(key)
        summary[key] = mu
        summary[f"{key}_std"] = sd
    curves = np.stack([r["reward_curve"] for r in rows])
    summary["reward_curve_mean"] = curves.mean(axis=0).tolist()
    return summary


def run_named_per_student(
    name: str,
    factory: Callable,
    n_students: int,
    n_interactions: int,
    seeds: List[int],
    delayed: bool,
) -> dict:
    """Like run_named but rebuilds policy each student (for step counters)."""
    from src.policy.simulator import make_population, difficulty_for_action

    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        students = make_population(
            n_students, seed, delayed_effects=delayed
        )
        total_reward = 0.0
        mastery_final = []
        indep_final = []
        fat_final = []
        reward_by_t = np.zeros(n_interactions, dtype=np.float64)
        correct_by_t = np.zeros(n_interactions, dtype=np.float64)

        for student in students:
            policy = factory()
            for t in range(n_interactions):
                base_diff = float(
                    np.clip(student.difficulty_pref + rng.normal(0, 0.05), 0.15, 0.85)
                )
                action = policy.select_action(student)
                difficulty = difficulty_for_action(action, base_diff)
                pre = student.mastery
                correct = student.respond(action, difficulty, rng)
                reward = student.mastery - pre
                total_reward += reward
                reward_by_t[t] += reward
                correct_by_t[t] += correct
            mastery_final.append(student.mastery)
            indep_final.append(student.independence)
            fat_final.append(student.fatigue)

        n_steps = n_students * n_interactions
        reward_by_t /= n_students
        correct_by_t /= n_students
        tail = min(5, n_interactions)
        early = min(10, n_interactions)
        rows.append(
            {
                "mean_reward": total_reward / n_steps,
                "mean_final_mastery": float(np.mean(mastery_final)),
                "early_mean_reward": float(np.mean(reward_by_t[:early])),
                "late_mean_reward": float(np.mean(reward_by_t[-tail:])),
                "overall_accuracy": float(np.mean(correct_by_t)),
                "mean_final_independence": float(np.mean(indep_final)),
                "mean_final_fatigue": float(np.mean(fat_final)),
                "reward_curve": reward_by_t.tolist(),
            }
        )

    def mean_std(key):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    summary = {"name": name, "delayed_effects": delayed, "n_seeds": len(seeds)}
    for key in (
        "mean_reward",
        "mean_final_mastery",
        "early_mean_reward",
        "late_mean_reward",
        "overall_accuracy",
        "mean_final_independence",
        "mean_final_fatigue",
    ):
        mu, sd = mean_std(key)
        summary[key] = mu
        summary[f"{key}_std"] = sd
    curves = np.stack([r["reward_curve"] for r in rows])
    summary["reward_curve_mean"] = curves.mean(axis=0).tolist()
    return summary


def print_table(summaries: List[dict]) -> str:
    header = (
        "| Policy | early Δm | late Δm | mean Δm | final m | final I | final F |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for s in summaries:
        rows += (
            f"| {s['name']} | "
            f"{s['early_mean_reward']:.4f} | "
            f"{s['late_mean_reward']:.4f} | "
            f"{s['mean_reward']:.4f} | "
            f"{s['mean_final_mastery']:.3f} | "
            f"{s['mean_final_independence']:.3f} | "
            f"{s['mean_final_fatigue']:.3f} |\n"
        )
    table = header + rows
    print(table)
    return table


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    n_students = int(cfg["simulator"]["n_students"])
    n_interactions = int(cfg["simulator"]["n_interactions_per_student"])
    n_seeds = int(cfg["simulator"].get("n_seeds", 5))
    seeds = [cfg["seed"] + i for i in range(n_seeds)]

    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        ("always_explain", lambda: FixedActionPolicy("explain")),
        ("always_harder", lambda: FixedActionPolicy("harder_problem")),
        ("myopic_oracle", lambda: MyopicOraclePolicy()),
        ("scaffold_then_challenge", lambda: ScaffoldThenChallengePolicy(12)),
        ("balanced_challenge", lambda: BalancedChallengePolicy(8)),
        ("interleave_explain_harder", lambda: InterleaveExplainHarderPolicy()),
    ]

    payload: Dict = {
        "disclaimer": (
            "M6 Step 1 calibration only. Delayed I/F are latent. "
            "No DQN. Simulated outcomes only."
        ),
        "n_students": n_students,
        "n_interactions": n_interactions,
        "n_seeds": n_seeds,
    }

    for delayed in (False, True):
        label = "ON" if delayed else "OFF"
        print(f"\n=== delayed_effects={label} ===")
        summaries = []
        for name, factory in policies:
            s = run_named_per_student(
                name, factory, n_students, n_interactions, seeds, delayed
            )
            summaries.append(s)
        table = print_table(summaries)
        key = "delayed_on" if delayed else "delayed_off"
        payload[key] = {"markdown_table": table, "conditions": summaries}

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for s in summaries:
            ax.plot(s["reward_curve_mean"], label=s["name"], linewidth=2)
        ax.set_xlabel("Interaction t")
        ax.set_ylabel("Mean mastery gain (reward)")
        ax.set_title(f"Delayed effects {label}: reward over time (simulated)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_fig = fig_dir / f"delayed_calibration_{key}.png"
        fig.savefig(out_fig, dpi=140)
        plt.close(fig)
        print(f"Wrote {out_fig}")

    out = results_dir / "delayed_effects_calibration.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
