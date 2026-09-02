"""Phase 4: paired DQN comparison, state ablations, and sensitivity checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.rule_based import RuleBasedPolicy, StudentState
from src.policy.simulator import intervention_effect
from src.rl.dqn import DQNAgent
from src.rl.environment import TutoringEnvConfig
from src.rl.evaluation import evaluate_policy, paired_difference
from src.rl.state_tracking import load_state_tracking
from src.utils.seed import load_config, set_seed


def full_model_specs(cfg: dict) -> List[dict]:
    phase4 = cfg["phase4"]
    seeds = [int(seed) for seed in phase4["training_seeds"]]
    if not seeds or seeds[0] != int(cfg["seed"]):
        raise ValueError("phase4.training_seeds must start with the canonical seed")
    default_gamma = float(cfg["rl"]["gamma"])
    gamma_ablation = float(phase4["gamma_ablation"])
    specs = []
    for index, seed in enumerate(seeds):
        suffix = "" if index == 0 else f"_seed{seed}"
        specs.append(
            {
                "name": f"dqn_oracle_seed{seed}",
                "checkpoint": f"dqn_oracle_best{suffix}.pt",
                "condition": "oracle",
                "training_seed": seed,
                "gamma": default_gamma,
            }
        )
    gamma_label = (
        str(int(gamma_ablation))
        if gamma_ablation.is_integer()
        else str(gamma_ablation).replace(".", "p")
    )
    specs.append(
        {
            "name": f"dqn_oracle_gamma{gamma_label}",
            "checkpoint": f"dqn_oracle_best_gamma{gamma_label}.pt",
            "condition": "oracle",
            "training_seed": seeds[0],
            "gamma": gamma_ablation,
        }
    )
    for condition in phase4["state_conditions"]:
        condition = str(condition)
        if condition == "oracle":
            continue
        specs.append(
            {
                "name": f"dqn_{condition}_seed{seeds[0]}",
                "checkpoint": f"dqn_{condition}_best.pt",
                "condition": condition,
                "training_seed": seeds[0],
                "gamma": default_gamma,
            }
        )
    return specs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"head_commit": commit, "working_tree_dirty": dirty}


def env_config(cfg: dict, observation_mode: str) -> TutoringEnvConfig:
    rl_cfg = cfg["rl"]
    return TutoringEnvConfig(
        actions=tuple(cfg["policy"]["actions"]),
        horizon=int(rl_cfg["horizon"]),
        num_skills=int(rl_cfg["num_skills"]),
        observation_mode=observation_mode,
        delayed_effects=bool(rl_cfg["delayed_effects"]),
        heterogeneous=bool(rl_cfg["heterogeneous"]),
        include_diagnostics=True,
    )


def baseline_factories(cfg: dict) -> Dict[str, Callable]:
    actions = list(cfg["policy"]["actions"])
    low = float(cfg["policy"]["rule_thresholds"]["low_mastery"])
    high = float(cfg["policy"]["rule_thresholds"]["high_mastery"])

    def random_factory(episode_seed: int):
        rng = np.random.RandomState(episode_seed + 3_000_003)
        return lambda observation, step: str(rng.choice(actions))

    def rule_factory(episode_seed: int):
        policy = RuleBasedPolicy(low_thresh=low, high_thresh=high)

        def choose(observation: np.ndarray, step: int) -> str:
            state = StudentState(
                mastery=float(observation[0]),
                recent_accuracy=float(observation[1]),
                attempts=step,
                consecutive_failures=int(round(float(observation[6]) * 5.0)),
            )
            return policy.select_action(state)

        return choose

    def myopic_factory(episode_seed: int):
        def choose(observation: np.ndarray, step: int) -> str:
            mastery = float(observation[0])
            return max(
                actions,
                key=lambda action: intervention_effect(action, mastery),
            )

        return choose

    def interleave_factory(episode_seed: int):
        return lambda observation, step: (
            "explain" if step % 2 == 0 else "harder_problem"
        )

    def explain_factory(episode_seed: int):
        return lambda observation, step: "explain"

    return {
        "random": random_factory,
        "rule_oracle": rule_factory,
        "myopic_oracle": myopic_factory,
        "interleave_explain_harder": interleave_factory,
        "always_explain": explain_factory,
    }


def load_dqn_condition(
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
        raise FileNotFoundError(f"Missing Phase 4 checkpoint: {checkpoint_path}")
    tracking = load_state_tracking(spec["condition"], checkpoint_dir, device)
    agent, checkpoint_extra = DQNAgent.load(checkpoint_path, device)
    environment = env_config(cfg, tracking.environment_mode)
    expected_observation_dim = 8 + len(environment.actions)
    if agent.observation_dim != expected_observation_dim:
        raise ValueError(
            f"{checkpoint_path} expects {agent.observation_dim} observations, "
            f"but environment provides {expected_observation_dim}"
        )
    if agent.n_actions != len(environment.actions):
        raise ValueError("DQN action count does not match configured environment")
    if not np.isclose(agent.config.gamma, float(spec["gamma"])):
        raise ValueError(
            f"{checkpoint_path} gamma={agent.config.gamma} does not match "
            f"manifest gamma={spec['gamma']}"
        )
    expected_double_dqn = bool(spec.get("double_dqn", False))
    if agent.config.double_dqn != expected_double_dqn:
        raise ValueError(
            f"{checkpoint_path} double_dqn={agent.config.double_dqn} does not "
            f"match manifest double_dqn={expected_double_dqn}"
        )

    def policy_factory(episode_seed: int):
        return lambda observation, step: agent.select_action(observation, epsilon=0.0)

    summary = evaluate_policy(
        spec["name"],
        policy_factory,
        environment,
        episode_seeds,
        tracker_factory=tracking.tracker_factory,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    summary["checkpoint"] = str(checkpoint_path)
    summary["checkpoint_sha256"] = sha256(checkpoint_path)
    summary["checkpoint_extra"] = checkpoint_extra
    summary["observation_condition"] = spec["condition"]
    summary["training_seed"] = int(spec["training_seed"])
    summary["gamma"] = float(spec["gamma"])
    summary["algorithm"] = "double_dqn" if agent.config.double_dqn else "dqn"
    return summary


def aggregate_training_seeds(conditions: List[dict], gamma: float) -> dict:
    rows = [
        condition
        for condition in conditions
        if condition["observation_condition"] == "oracle"
        and np.isclose(condition["gamma"], gamma)
    ]
    values = np.asarray([row["final_mastery"] for row in rows], dtype=np.float64)
    return {
        "condition": f"oracle_gamma_{gamma:g}",
        "n_training_seeds": len(rows),
        "training_seeds": [row["training_seed"] for row in rows],
        "heldout_final_mastery_mean_across_agents": float(values.mean()),
        "heldout_final_mastery_std_across_agents": float(
            values.std(ddof=1) if len(values) > 1 else 0.0
        ),
        "per_agent": {row["name"]: row["final_mastery"] for row in rows},
    }


def plot_results(baselines: List[dict], learned: List[dict], path: Path) -> None:
    canonical = learned[0]
    comparison = [*baselines, canonical]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for row in comparison:
        axes[0].plot(row["mastery_curve_mean"], label=row["name"], linewidth=2)
    axes[0].set_xlabel("Interaction")
    axes[0].set_ylabel("Mean true mastery (evaluation only)")
    axes[0].set_title("Paired held-out mastery trajectories")
    axes[0].legend(fontsize=7)

    labels = [row["name"].replace("dqn_", "") for row in learned]
    means = [row["final_mastery"] for row in learned]
    lower = [
        row["final_mastery"] - row["final_mastery_ci95"][0] for row in learned
    ]
    upper = [
        row["final_mastery_ci95"][1] - row["final_mastery"] for row in learned
    ]
    positions = np.arange(len(learned))
    axes[1].bar(positions, means, color="#4c78a8")
    axes[1].errorbar(
        positions,
        means,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean final mastery (95% episode bootstrap CI)")
    axes[1].set_title("DQN stability and ablations")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(
    config_path: str,
    n_episodes: Optional[int],
    device_name: str,
    suite: str,
    tag: str,
) -> None:
    cfg = load_config(config_path)
    phase4_cfg = cfg["phase4"]
    n_episodes = int(
        n_episodes
        if n_episodes is not None
        else phase4_cfg["evaluation_episodes"]
    )
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    bootstrap_resamples = int(phase4_cfg["bootstrap_resamples"])
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg["dqn"].get("torch_num_threads", 1)))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    heldout_start = int(cfg["dqn"]["heldout_seed_start"])
    episode_seeds = [heldout_start + i for i in range(n_episodes)]
    bootstrap_seed = int(cfg["seed"]) + 9_000

    print(
        f"Phase 4 DQN evaluation | suite={suite} | episodes={n_episodes} | "
        f"device={device}"
    )
    print(
        "DISCLAIMER: simulated outcomes only. Oracle policies are upper bounds; "
        "BKT/DKT policies receive estimated state only."
    )

    oracle_environment = env_config(cfg, "oracle")
    baselines = []
    for offset, (name, factory) in enumerate(baseline_factories(cfg).items()):
        print(f"  evaluating {name} ...", flush=True)
        baselines.append(
            evaluate_policy(
                name,
                factory,
                oracle_environment,
                episode_seeds,
                bootstrap_seed=bootstrap_seed + 100 * offset,
                bootstrap_resamples=bootstrap_resamples,
            )
        )

    all_specs = full_model_specs(cfg)
    specs = all_specs if suite == "full" else all_specs[:1]
    learned = []
    for offset, spec in enumerate(specs):
        print(f"  evaluating {spec['name']} ...", flush=True)
        learned.append(
            load_dqn_condition(
                spec,
                checkpoint_dir,
                cfg,
                device,
                episode_seeds,
                bootstrap_seed + 1_000 + 100 * offset,
                bootstrap_resamples,
            )
        )

    canonical = learned[0]
    paired_vs_baselines = [
        paired_difference(
            canonical,
            baseline,
            bootstrap_seed=bootstrap_seed + 2_000 + i,
            bootstrap_resamples=bootstrap_resamples,
        )
        for i, baseline in enumerate(baselines)
    ]

    sensitivity = {}
    if suite == "full":
        default_gamma = float(cfg["rl"]["gamma"])
        oracle_replicates = [
            row
            for row in learned
            if row["observation_condition"] == "oracle"
            and np.isclose(row["gamma"], default_gamma)
        ]
        gamma_ablation = next(
            row
            for row in learned
            if row["observation_condition"] == "oracle"
            and not np.isclose(row["gamma"], default_gamma)
        )
        state_conditions = [
            row for row in learned if row["observation_condition"] != "oracle"
        ]
        no_state = next(
            row for row in state_conditions if row["observation_condition"] == "no_state"
        )
        estimated_conditions = [
            row for row in state_conditions if row["observation_condition"] in {"bkt", "dkt"}
        ]
        interleave = next(
            row for row in baselines if row["name"] == "interleave_explain_harder"
        )
        sensitivity = {
            "training_seed_stability": aggregate_training_seeds(
                learned, default_gamma
            ),
            "oracle_training_seeds_minus_interleave": [
                paired_difference(
                    condition,
                    interleave,
                    bootstrap_seed=bootstrap_seed + 2_500 + i,
                    bootstrap_resamples=bootstrap_resamples,
                )
                for i, condition in enumerate(oracle_replicates)
            ],
            "gamma_ablation_minus_default": paired_difference(
                gamma_ablation,
                canonical,
                bootstrap_seed=bootstrap_seed + 3_000,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "state_conditions_minus_oracle": [
                paired_difference(
                    condition,
                    canonical,
                    bootstrap_seed=bootstrap_seed + 3_100 + i,
                    bootstrap_resamples=bootstrap_resamples,
                )
                for i, condition in enumerate(state_conditions)
            ],
            "estimated_state_minus_no_state": [
                paired_difference(
                    condition,
                    no_state,
                    bootstrap_seed=bootstrap_seed + 3_200 + i,
                    bootstrap_resamples=bootstrap_resamples,
                )
                for i, condition in enumerate(estimated_conditions)
            ],
        }

    print("\npolicy                         final_m    CI95 low   CI95 high")
    for row in [*baselines, *learned]:
        low, high = row["final_mastery_ci95"]
        print(f"{row['name']:<30} {row['final_mastery']:.4f}     {low:.4f}      {high:.4f}")
    print("\ncanonical DQN paired final-mastery differences:")
    for row in paired_vs_baselines:
        low, high = row["difference_ci95"]
        print(
            f"  vs {row['reference']:<26} {row['mean_difference']:+.4f} "
            f"CI [{low:+.4f}, {high:+.4f}]"
        )

    suffix = f"_{tag}" if tag else ""
    result_path = results_dir / f"dqn_phase4_evaluation{suffix}.json"
    figure_path = results_dir / "figures" / f"dqn_phase4_evaluation{suffix}.png"
    plot_results(baselines, learned, figure_path)
    payload = {
        "disclaimer": (
            "Simulated paired policy evaluation. Oracle state is an upper bound; "
            "no result demonstrates real-student learning."
        ),
        "suite": suite,
        "n_episodes": n_episodes,
        "episode_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "bootstrap_resamples": bootstrap_resamples,
        "device": str(device),
        "baselines": baselines,
        "learned_conditions": learned,
        "canonical_dqn_paired_vs_baselines": paired_vs_baselines,
        "sensitivity": sensitivity,
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "environment": sha256(Path("src/rl/environment.py")),
                "dqn": sha256(Path("src/rl/dqn.py")),
                "evaluation": sha256(Path("src/rl/evaluation.py")),
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
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--suite", default="full", choices=["canonical", "full"])
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.n_episodes,
        arguments.device,
        arguments.suite,
        arguments.tag,
    )
