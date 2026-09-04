"""Diagnose DQN value overestimation with Monte Carlo policy returns.

The learned online-network max Q value is compared with the empirical scaled,
discounted return from the same state after taking its greedy action and then
following the checkpoint's greedy policy. The diagnostic supports either a
shared algorithm-independent state bank or each checkpoint's on-policy states.
"""
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

from scripts.eval_double_dqn import comparison_specs
from scripts.eval_dqn_suite import env_config, git_provenance, sha256
from src.rl.dqn import DQNAgent
from src.rl.environment import TutoringEnv
from src.rl.evaluation import bootstrap_mean_ci
from src.utils.seed import load_config, set_seed


def collect_state_bank(
    cfg: dict,
    behavior_agent: DQNAgent | None = None,
) -> List[dict]:
    diagnostic = cfg["q_overestimation"]
    episode_count = int(diagnostic["state_episode_count"])
    state_steps = sorted({int(step) for step in diagnostic["state_steps"]})
    horizon = int(cfg["rl"]["horizon"])
    if episode_count <= 0:
        raise ValueError("state_episode_count must be positive")
    if not state_steps or state_steps[0] < 0 or state_steps[-1] >= horizon:
        raise ValueError("state_steps must be within the pre-action episode horizon")
    if behavior_agent is None and diagnostic["behavior_policy"] != "interleave_explain_harder":
        raise ValueError("Only the fixed interleave state-bank policy is supported")

    heldout_start = int(cfg["dqn"]["heldout_seed_start"])
    environment = TutoringEnv(env_config(cfg, "oracle"), seed=heldout_start)
    state_bank = []
    state_id = 0
    for episode_seed in range(heldout_start, heldout_start + episode_count):
        observation, _ = environment.reset(seed=episode_seed)
        for step in range(horizon):
            if step in state_steps:
                state_bank.append(
                    {
                        "state_id": state_id,
                        "episode_seed": episode_seed,
                        "step": step,
                        "observation": observation.copy(),
                        "environment": environment.fork(),
                    }
                )
                state_id += 1
            behavior_action = (
                behavior_agent.select_action(observation, epsilon=0.0)
                if behavior_agent is not None
                else ("explain" if step % 2 == 0 else "harder_problem")
            )
            observation, _, terminated, truncated, _ = environment.step(
                behavior_action
            )
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
        if not terminated:
            raise RuntimeError("state-bank episode did not reach its horizon")
    return state_bank


def discounted_policy_return(
    snapshot: TutoringEnv,
    agent: DQNAgent,
    first_action: int,
    response_seed: int,
) -> float:
    """Sample one scaled, discounted greedy-policy return from a fixed state."""
    environment = snapshot.fork(response_seed=response_seed)
    action = int(first_action)
    discount = 1.0
    total = 0.0
    while True:
        observation, reward, terminated, truncated, _ = environment.step(action)
        if truncated:
            raise RuntimeError("fixed-horizon environment must not truncate")
        total += discount * float(reward) * agent.config.reward_scale
        if terminated:
            return float(total)
        discount *= agent.config.gamma
        action = agent.select_action(observation, epsilon=0.0)


def summarize_bias(rows: List[dict], bootstrap_resamples: int, seed: int) -> dict:
    if not rows:
        raise ValueError("bias rows must not be empty")
    q_values = np.asarray([row["q_max"] for row in rows], dtype=np.float64)
    mc_values = np.asarray([row["mc_return_mean"] for row in rows], dtype=np.float64)
    bias = q_values - mc_values
    steps = sorted({int(row["step"]) for row in rows})
    return {
        "n_states": len(rows),
        "q_max_mean": float(q_values.mean()),
        "mc_return_mean": float(mc_values.mean()),
        "mean_signed_bias": float(bias.mean()),
        "mean_signed_bias_ci95_across_states": bootstrap_mean_ci(
            bias,
            n_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "median_signed_bias": float(np.median(bias)),
        "mean_absolute_error": float(np.abs(bias).mean()),
        "root_mean_squared_error": float(np.sqrt(np.mean(np.square(bias)))),
        "overestimation_rate": float(np.mean(bias > 0.0)),
        "mean_mc_standard_error": float(
            np.mean([row["mc_return_standard_error"] for row in rows])
        ),
        "bias_by_step": {
            str(step): float(
                np.mean(
                    [
                        float(row["q_max"]) - float(row["mc_return_mean"])
                        for row in rows
                        if int(row["step"]) == step
                    ]
                )
            )
            for step in steps
        },
    }


def evaluate_checkpoint_bias(
    spec: dict,
    checkpoint_dir: Path,
    state_bank: List[dict] | None,
    cfg: dict,
    monte_carlo_rollouts: int,
    rollout_seed_start: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    device: torch.device,
) -> dict:
    checkpoint_path = checkpoint_dir / spec["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing diagnostic checkpoint: {checkpoint_path}")
    agent, checkpoint_extra = DQNAgent.load(checkpoint_path, device)
    if agent.config.double_dqn != bool(spec["double_dqn"]):
        raise ValueError(f"Algorithm flag mismatch for {checkpoint_path}")
    if not np.isclose(agent.config.gamma, float(spec["gamma"])):
        raise ValueError(f"Discount mismatch for {checkpoint_path}")
    if monte_carlo_rollouts <= 1:
        raise ValueError("monte_carlo_rollouts must be greater than one")
    effective_state_bank = (
        state_bank if state_bank is not None else collect_state_bank(cfg, agent)
    )

    observations = np.stack(
        [row["observation"] for row in effective_state_bank], axis=0
    ).astype(np.float32)
    with torch.no_grad():
        q_matrix = agent.online(
            torch.as_tensor(observations, dtype=torch.float32, device=device)
        ).cpu().numpy()
    greedy_actions = q_matrix.argmax(axis=1)
    q_max = q_matrix.max(axis=1)

    rows = []
    for index, state in enumerate(effective_state_bank):
        returns = np.asarray(
            [
                discounted_policy_return(
                    state["environment"],
                    agent,
                    int(greedy_actions[index]),
                    rollout_seed_start
                    + int(state["state_id"]) * monte_carlo_rollouts
                    + replicate,
                )
                for replicate in range(monte_carlo_rollouts)
            ],
            dtype=np.float64,
        )
        mc_mean = float(returns.mean())
        mc_std = float(returns.std(ddof=1))
        estimate = float(q_max[index])
        rows.append(
            {
                "state_id": int(state["state_id"]),
                "episode_seed": int(state["episode_seed"]),
                "step": int(state["step"]),
                "greedy_action": int(greedy_actions[index]),
                "q_max": estimate,
                "mc_return_mean": mc_mean,
                "mc_return_std": mc_std,
                "mc_return_standard_error": mc_std
                / float(np.sqrt(monte_carlo_rollouts)),
                "signed_bias": estimate - mc_mean,
            }
        )

    summary = summarize_bias(rows, bootstrap_resamples, bootstrap_seed)
    return {
        "name": spec["name"],
        "algorithm": "double_dqn" if agent.config.double_dqn else "dqn",
        "training_seed": int(spec["training_seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_checkpoint_episode": int(checkpoint_extra["episode"]),
        "gamma": float(agent.config.gamma),
        "reward_scale": float(agent.config.reward_scale),
        "summary": summary,
        "states": rows,
    }


def algorithm_summary(rows: List[dict], algorithm: str) -> dict:
    selected = sorted(
        (row for row in rows if row["algorithm"] == algorithm),
        key=lambda row: row["training_seed"],
    )
    biases = np.asarray(
        [row["summary"]["mean_signed_bias"] for row in selected],
        dtype=np.float64,
    )
    errors = np.asarray(
        [row["summary"]["mean_absolute_error"] for row in selected],
        dtype=np.float64,
    )
    rates = np.asarray(
        [row["summary"]["overestimation_rate"] for row in selected],
        dtype=np.float64,
    )
    return {
        "algorithm": algorithm,
        "n_training_seeds": len(selected),
        "training_seeds": [row["training_seed"] for row in selected],
        "mean_signed_bias_by_seed": {
            str(row["training_seed"]): row["summary"]["mean_signed_bias"]
            for row in selected
        },
        "mean_signed_bias_across_agents": float(biases.mean()),
        "std_signed_bias_across_agents": float(
            biases.std(ddof=1) if len(biases) > 1 else 0.0
        ),
        "mean_absolute_error_across_agents": float(errors.mean()),
        "mean_overestimation_rate_across_agents": float(rates.mean()),
    }


def paired_bias_difference(
    double: dict,
    vanilla: dict,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    pair_states: bool = True,
) -> dict:
    double_by_state = {
        int(row["state_id"]): float(row["signed_bias"])
        for row in double["states"]
    }
    vanilla_by_state = {
        int(row["state_id"]): float(row["signed_bias"])
        for row in vanilla["states"]
    }
    if double_by_state.keys() != vanilla_by_state.keys():
        raise ValueError("Bias comparison requires identical state ids")
    differences = np.asarray(
        [double_by_state[key] - vanilla_by_state[key] for key in sorted(double_by_state)],
        dtype=np.float64,
    )
    return {
        "training_seed": int(double["training_seed"]),
        "candidate": double["name"],
        "reference": vanilla["name"],
        "metric": "double_dqn_signed_bias_minus_vanilla",
        "state_comparison": "paired" if pair_states else "unpaired_on_policy",
        "n_states_per_checkpoint": len(differences),
        "mean_difference": float(
            double["summary"]["mean_signed_bias"]
            - vanilla["summary"]["mean_signed_bias"]
        ),
        "difference_ci95_across_states": (
            bootstrap_mean_ci(
                differences,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            if pair_states
            else None
        ),
        "double_dqn_lower_bias_state_rate": (
            float(np.mean(differences < 0.0)) if pair_states else None
        ),
    }


def compare_bias_across_training_seeds(
    rows: List[dict],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict:
    """Compare checkpoint-level calibration metrics across matched seeds."""
    by_key = {
        (row["algorithm"], int(row["training_seed"])): row["summary"]
        for row in rows
    }
    vanilla_seeds = sorted(seed for algorithm, seed in by_key if algorithm == "dqn")
    double_seeds = sorted(
        seed for algorithm, seed in by_key if algorithm == "double_dqn"
    )
    if vanilla_seeds != double_seeds or len(vanilla_seeds) < 2:
        raise ValueError("Algorithms require at least two identical training seeds")

    metric_names = (
        "mean_signed_bias",
        "mean_absolute_error",
        "root_mean_squared_error",
        "overestimation_rate",
    )
    differences = {
        metric: np.asarray(
            [
                float(by_key[("double_dqn", seed)][metric])
                - float(by_key[("dqn", seed)][metric])
                for seed in vanilla_seeds
            ],
            dtype=np.float64,
        )
        for metric in metric_names
    }
    rng = np.random.RandomState(bootstrap_seed)
    indices = rng.randint(
        0,
        len(vanilla_seeds),
        size=(bootstrap_resamples, len(vanilla_seeds)),
    )
    comparisons = {}
    for metric, values in differences.items():
        bootstrap_means = values[indices].mean(axis=1)
        comparisons[f"double_minus_vanilla_{metric}"] = {
            "mean_of_seed_level_differences": float(values.mean()),
            "ci95_training_seed_bootstrap": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "seed_level_differences": {
                str(seed): float(value)
                for seed, value in zip(vanilla_seeds, values)
            },
        }
    return {
        "replication_unit": "training_seed",
        "training_seeds": vanilla_seeds,
        "n_training_seeds": len(vanilla_seeds),
        "comparisons": comparisons,
        "note": (
            "Paired bootstrap resamples trained agents by matched training seed; "
            "it does not treat states or Monte Carlo rollouts as independent "
            "algorithm replications."
        ),
    }
def performance_association(diagnostics: List[dict], performance: dict) -> dict:
    performance_by_name = {
        row["name"]: float(row["final_mastery"])
        for row in performance["conditions"]
    }

    def correlation(rows: List[dict]) -> float:
        bias = np.asarray(
            [row["summary"]["mean_signed_bias"] for row in rows],
            dtype=np.float64,
        )
        mastery = np.asarray(
            [performance_by_name[row["name"]] for row in rows],
            dtype=np.float64,
        )
        if len(rows) < 2 or np.isclose(bias.std(), 0.0) or np.isclose(mastery.std(), 0.0):
            return float("nan")
        return float(np.corrcoef(bias, mastery)[0, 1])

    missing = [row["name"] for row in diagnostics if row["name"] not in performance_by_name]
    if missing:
        raise ValueError(f"Performance result is missing diagnostic models: {missing}")
    return {
        "pearson_bias_vs_final_mastery_all_agents": correlation(diagnostics),
        "pearson_bias_vs_final_mastery_dqn": correlation(
            [row for row in diagnostics if row["algorithm"] == "dqn"]
        ),
        "pearson_bias_vs_final_mastery_double_dqn": correlation(
            [row for row in diagnostics if row["algorithm"] == "double_dqn"]
        ),
        "note": "Descriptive association across trained agents; not a causal estimate.",
    }


def plot_diagnostic(rows: List[dict], path: Path, state_bank_mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = {"dqn": "#4c78a8", "double_dqn": "#f58518"}
    for algorithm in ("dqn", "double_dqn"):
        selected = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: row["training_seed"],
        )
        axes[0].plot(
            [row["training_seed"] for row in selected],
            [row["summary"]["mean_signed_bias"] for row in selected],
            marker="o",
            label=algorithm.replace("_", " ").title(),
            color=colors[algorithm],
        )
        axes[1].scatter(
            [state["mc_return_mean"] for row in selected for state in row["states"]],
            [state["q_max"] for row in selected for state in row["states"]],
            s=8,
            alpha=0.18,
            label=algorithm.replace("_", " ").title(),
            color=colors[algorithm],
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Mean Q max - Monte Carlo return")
    state_label = "shared states" if state_bank_mode == "shared_interleave" else "on-policy states"
    axes[0].set_title(f"Signed value bias on {state_label}")
    axes[0].legend()

    all_values = [
        value
        for row in rows
        for state in row["states"]
        for value in (state["mc_return_mean"], state["q_max"])
    ]
    low, high = float(min(all_values)), float(max(all_values))
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set_xlabel("Monte Carlo greedy-policy return")
    axes[1].set_ylabel("Learned max Q")
    axes[1].set_title("Value calibration (scaled discounted units)")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(
    config_path: str,
    performance_path: Path,
    device_name: str,
    state_bank_mode: str,
) -> None:
    cfg = load_config(config_path)
    diagnostic = cfg["q_overestimation"]
    monte_carlo_rollouts = int(diagnostic["monte_carlo_rollouts"])
    bootstrap_resamples = int(diagnostic["bootstrap_resamples"])
    rollout_seed_start = int(diagnostic["rollout_seed_start"])
    if diagnostic["continuation_policy"] != "checkpoint_greedy":
        raise ValueError("Only checkpoint_greedy continuation is supported")

    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg["dqn"].get("torch_num_threads", 1)))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if state_bank_mode not in {"shared_interleave", "on_policy"}:
        raise ValueError("state_bank_mode must be shared_interleave or on_policy")
    state_bank = collect_state_bank(cfg) if state_bank_mode == "shared_interleave" else None
    expected_states = int(diagnostic["state_episode_count"]) * len(
        diagnostic["state_steps"]
    )
    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])
    print(
        f"Q-overestimation diagnostic | models={len(comparison_specs(cfg))} | "
        f"state_bank={state_bank_mode} | states/model={expected_states} | "
        f"MC/state={monte_carlo_rollouts}"
    )
    print(
        "Q and Monte Carlo returns use the checkpoint reward scale and discount. "
        "Simulator/oracle-state evidence only."
    )

    rows = []
    for offset, spec in enumerate(comparison_specs(cfg)):
        print(f"  diagnosing {spec['name']} ...", flush=True)
        rows.append(
            evaluate_checkpoint_bias(
                spec,
                checkpoint_dir,
                state_bank,
                cfg,
                monte_carlo_rollouts,
                rollout_seed_start,
                bootstrap_resamples,
                int(cfg["seed"]) + 25_000 + offset,
                device,
            )
        )

    by_key: Dict[tuple, dict] = {
        (row["algorithm"], int(row["training_seed"])): row for row in rows
    }
    seeds = [int(seed) for seed in cfg["double_dqn_evaluation"]["training_seeds"]]
    paired = [
        paired_bias_difference(
            by_key[("double_dqn", seed)],
            by_key[("dqn", seed)],
            bootstrap_resamples,
            int(cfg["seed"]) + 27_000 + index,
            pair_states=state_bank_mode == "shared_interleave",
        )
        for index, seed in enumerate(seeds)
    ]
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    algorithm_summaries = {
        algorithm: algorithm_summary(rows, algorithm)
        for algorithm in ("dqn", "double_dqn")
    }
    association = performance_association(rows, performance)
    across_training_seeds = compare_bias_across_training_seeds(
        rows,
        bootstrap_resamples,
        int(cfg["seed"]) + 29_000,
    )

    print("\nalgorithm    mean bias   seed SD    mean MAE   over-rate")
    for algorithm, summary in algorithm_summaries.items():
        print(
            f"{algorithm:<12} {summary['mean_signed_bias_across_agents']:+.4f}    "
            f"{summary['std_signed_bias_across_agents']:.4f}    "
            f"{summary['mean_absolute_error_across_agents']:.4f}    "
            f"{summary['mean_overestimation_rate_across_agents']:.3f}"
        )
    interval_label = (
        " [95% state-bootstrap CI]" if state_bank_mode == "shared_interleave" else ""
    )
    print(f"\nseed   Double DQN bias - vanilla bias{interval_label}")
    for effect in paired:
        interval = effect["difference_ci95_across_states"]
        interval_text = (
            f" [{interval[0]:+.4f}, {interval[1]:+.4f}]"
            if interval is not None
            else ""
        )
        print(f"{effect['training_seed']:<6} {effect['mean_difference']:+.4f}{interval_text}")
    print("\ntraining-seed bootstrap (Double DQN - vanilla)")
    for metric in ("mean_signed_bias", "mean_absolute_error"):
        effect = across_training_seeds["comparisons"][
            f"double_minus_vanilla_{metric}"
        ]
        low, high = effect["ci95_training_seed_bootstrap"]
        print(
            f"  {metric}: {effect['mean_of_seed_level_differences']:+.4f} "
            f"[{low:+.4f}, {high:+.4f}]"
        )

    results_dir = Path(cfg["paths"]["results_dir"])
    suffix = "" if state_bank_mode == "shared_interleave" else "_on_policy"
    result_path = results_dir / f"q_overestimation_diagnostic{suffix}.json"
    figure_path = results_dir / "figures" / f"q_overestimation_diagnostic{suffix}.png"
    plot_diagnostic(rows, figure_path, state_bank_mode)
    payload = {
        "disclaimer": (
            "Simulated oracle-state value diagnostic only. No intervention "
            "policy was evaluated on real students."
        ),
        "estimand": (
            "online max_a Q(s,a) minus the empirical scaled discounted return "
            "after its greedy action and greedy-policy continuation"
        ),
        "state_bank": {
            "mode": state_bank_mode,
            "behavior_policy": (
                diagnostic["behavior_policy"]
                if state_bank_mode == "shared_interleave"
                else "each_checkpoint_greedy_policy"
            ),
            "episode_seed_range": [
                int(cfg["dqn"]["heldout_seed_start"]),
                int(cfg["dqn"]["heldout_seed_start"])
                + int(diagnostic["state_episode_count"])
                - 1,
            ],
            "steps": [int(step) for step in diagnostic["state_steps"]],
            "n_states_per_checkpoint": expected_states,
            "comparability_note": (
                "All checkpoints share the same latent states."
                if state_bank_mode == "shared_interleave"
                else (
                    "Episode seeds and steps match, but each checkpoint reaches "
                    "its own on-policy latent states."
                )
            ),
        },
        "monte_carlo_rollouts_per_state": monte_carlo_rollouts,
        "continuation_policy": diagnostic["continuation_policy"],
        "training_seeds": seeds,
        "checkpoints": rows,
        "algorithm_summary": algorithm_summaries,
        "double_dqn_minus_vanilla_bias_by_seed": paired,
        "double_dqn_minus_vanilla_across_training_seeds": across_training_seeds,
        "bias_performance_association": association,
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "environment": sha256(Path("src/rl/environment.py")),
                "dqn": sha256(Path("src/rl/dqn.py")),
                "diagnostic_script": sha256(Path(__file__)),
                "performance_result": sha256(performance_path),
            },
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote {result_path} and {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q_overestimation.yaml")
    parser.add_argument(
        "--performance-result",
        type=Path,
        default=Path("results/double_dqn_comparison.json"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--state-bank-mode",
        default="shared_interleave",
        choices=["shared_interleave", "on_policy"],
    )
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.performance_result,
        arguments.device,
        arguments.state_bank_mode,
    )
