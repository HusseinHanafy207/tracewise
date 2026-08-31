"""Train DQN under oracle, BKT, DKT, or no-state observations.

Training, checkpoint-selection, and held-out episode seeds are disjoint. This
script trains/evaluates one DQN; `eval_dqn_suite.py` performs the Phase 4
multi-policy comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.dqn import DQNAgent, DQNConfig, linear_epsilon
from src.rl.environment import OnlineTracker, TutoringEnv, TutoringEnvConfig
from src.rl.state_tracking import load_state_tracking
from src.utils.seed import load_config, set_seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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


def environment_config(
    cfg: dict,
    include_diagnostics: bool,
    observation_mode: Optional[str] = None,
) -> TutoringEnvConfig:
    rl_cfg = cfg["rl"]
    return TutoringEnvConfig(
        actions=tuple(cfg["policy"]["actions"]),
        horizon=int(rl_cfg["horizon"]),
        num_skills=int(rl_cfg["num_skills"]),
        observation_mode=str(observation_mode or rl_cfg["observation_mode"]),
        delayed_effects=bool(rl_cfg["delayed_effects"]),
        heterogeneous=bool(rl_cfg["heterogeneous"]),
        include_diagnostics=include_diagnostics,
    )


def agent_config(cfg: dict, gamma_override: Optional[float] = None) -> DQNConfig:
    dqn_cfg = cfg["dqn"]
    return DQNConfig(
        hidden_dims=tuple(int(x) for x in dqn_cfg["hidden_dims"]),
        gamma=float(
            gamma_override if gamma_override is not None else cfg["rl"]["gamma"]
        ),
        learning_rate=float(dqn_cfg["learning_rate"]),
        batch_size=int(dqn_cfg["batch_size"]),
        replay_capacity=int(dqn_cfg["replay_capacity"]),
        min_replay_size=int(dqn_cfg["min_replay_size"]),
        target_sync_steps=int(dqn_cfg["target_sync_steps"]),
        gradient_clip=float(dqn_cfg["gradient_clip"]),
        reward_scale=float(dqn_cfg["reward_scale"]),
        double_dqn=bool(dqn_cfg["double_dqn"]),
    )


def evaluate_agent(
    agent: DQNAgent,
    env_config: TutoringEnvConfig,
    episode_seeds: List[int],
    tracker_factory: Optional[Callable[[], OnlineTracker]] = None,
) -> dict:
    env = TutoringEnv(
        env_config,
        tracker_factory=tracker_factory,
        seed=episode_seeds[0],
    )
    rows = []
    action_counts = {name: 0 for name in env.action_names}
    for seed in episode_seeds:
        observation, _ = env.reset(seed=seed)
        terminated = False
        episode_return = 0.0
        correct = []
        info = {}
        while not terminated:
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
            episode_return += reward
            correct.append(int(info["correct"]))
            action_counts[info["action_name"]] += 1
        diagnostics = info["diagnostics"]
        rows.append(
            {
                "return": episode_return,
                "final_mastery": diagnostics["true_mastery"],
                "overall_accuracy": float(np.mean(correct)),
                "final_accuracy": float(np.mean(correct[-min(5, len(correct)) :])),
                "final_independence": diagnostics["independence"],
                "final_fatigue": diagnostics["fatigue"],
            }
        )

    summary = {"n_episodes": len(rows)}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
    total_actions = sum(action_counts.values()) or 1
    summary["action_frequency"] = {
        action: count / total_actions for action, count in action_counts.items()
    }
    return summary


def plot_training(train_rows: List[dict], validation_rows: List[dict], path: Path) -> None:
    episodes = np.asarray([row["episode"] for row in train_rows])
    returns = np.asarray([row["return"] for row in train_rows])
    window = min(100, max(len(returns), 1))
    if len(returns) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(returns, kernel, mode="valid")
        smooth_x = episodes[window - 1 :]
    else:
        smooth, smooth_x = returns, episodes

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(episodes, returns, alpha=0.15, color="#3b6ea5")
    axes[0].plot(smooth_x, smooth, color="#3b6ea5", linewidth=2)
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Undiscounted return")
    axes[0].set_title("DQN training return")

    axes[1].plot(
        [row["episode"] for row in validation_rows],
        [row["final_mastery"] for row in validation_rows],
        marker="o",
        color="#b35c5c",
    )
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Mean final mastery")
    axes[1].set_title("Frozen greedy validation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(
    config_path: str,
    episodes_override: Optional[int],
    device_name: str,
    tag: Optional[str],
    seed_override: Optional[int] = None,
    gamma_override: Optional[float] = None,
    observation_condition: str = "oracle",
) -> None:
    cfg = load_config(config_path)
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    set_seed(seed)
    dqn_cfg = cfg["dqn"]
    torch.set_num_threads(int(dqn_cfg.get("torch_num_threads", 1)))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    train_episodes = int(
        episodes_override
        if episodes_override is not None
        else dqn_cfg["train_episodes"]
    )
    if train_episodes <= 0:
        raise ValueError("train_episodes must be positive")

    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])
    state_tracking = load_state_tracking(
        observation_condition,
        checkpoint_dir,
        device,
    )
    train_env_cfg = environment_config(
        cfg,
        include_diagnostics=False,
        observation_mode=state_tracking.environment_mode,
    )
    eval_env_cfg = environment_config(
        cfg,
        include_diagnostics=True,
        observation_mode=state_tracking.environment_mode,
    )
    probe_env = TutoringEnv(
        train_env_cfg,
        tracker_factory=state_tracking.tracker_factory,
        seed=seed,
    )
    agent = DQNAgent(
        probe_env.observation_dim,
        probe_env.n_actions,
        agent_config(cfg, gamma_override),
        device,
        seed=seed,
    )

    train_seed_start = int(dqn_cfg["train_seed_start"])
    train_seed_end = train_seed_start + train_episodes - 1
    validation_count = int(dqn_cfg["validation_episodes"])
    heldout_count = int(dqn_cfg["heldout_eval_episodes"])
    if validation_count <= 0 or heldout_count <= 0:
        raise ValueError("validation and held-out episode counts must be positive")
    validation_seeds = [
        int(dqn_cfg["validation_seed_start"]) + i
        for i in range(validation_count)
    ]
    heldout_seeds = [
        int(dqn_cfg["heldout_seed_start"]) + i
        for i in range(heldout_count)
    ]
    train_seed_range = set(range(train_seed_start, train_seed_end + 1))
    if (
        train_seed_range & set(validation_seeds)
        or train_seed_range & set(heldout_seeds)
        or set(validation_seeds) & set(heldout_seeds)
    ):
        raise ValueError("training, validation, and held-out seeds must be disjoint")

    results_dir = Path(cfg["paths"]["results_dir"])
    suffix = f"_{tag}" if tag else ""
    stem = f"dqn_{state_tracking.condition}"
    best_path = checkpoint_dir / f"{stem}_best{suffix}.pt"
    final_path = checkpoint_dir / f"{stem}_final{suffix}.pt"
    figure_path = results_dir / "figures" / f"{stem}_training{suffix}.png"
    result_path = results_dir / f"{stem}_training{suffix}.json"

    print(
        f"DQN {state_tracking.condition} training | device={device} | "
        f"seed={seed} | gamma={agent.config.gamma:g} | episodes={train_episodes} | "
        f"obs={probe_env.observation_dim} | actions={probe_env.n_actions}"
    )
    print(
        "DISCLAIMER: simulated delayed-effect environment; oracle mastery is an "
        "upper bound and estimated-state results are not real-student evidence."
    )

    validation_history = []
    initial_validation = evaluate_agent(
        agent,
        eval_env_cfg,
        validation_seeds,
        state_tracking.tracker_factory,
    )
    validation_history.append({"episode": 0, **initial_validation})
    best_mastery = initial_validation["final_mastery"]
    agent.save(
        best_path,
        extra={
            "episode": 0,
            "global_step": 0,
            "seed": seed,
            "observation_condition": state_tracking.condition,
            "validation": initial_validation,
            "actions": list(probe_env.action_names),
            "observation_names": list(probe_env.observation_names),
        },
    )

    train_rows = []
    global_step = 0
    env = TutoringEnv(
        train_env_cfg,
        tracker_factory=state_tracking.tracker_factory,
        seed=train_seed_start,
    )
    t0 = time.time()
    validation_interval = int(dqn_cfg["validation_interval"])
    train_frequency = int(dqn_cfg["train_frequency"])

    for episode in range(1, train_episodes + 1):
        observation, _ = env.reset(seed=train_seed_start + episode - 1)
        terminated = False
        episode_return = 0.0
        episode_losses = []
        while not terminated:
            epsilon = linear_epsilon(
                global_step,
                float(dqn_cfg["epsilon_start"]),
                float(dqn_cfg["epsilon_end"]),
                int(dqn_cfg["epsilon_decay_steps"]),
            )
            action = agent.select_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
            agent.store(observation, action, reward, next_observation, terminated)
            if global_step % train_frequency == 0:
                metrics = agent.learn()
                if metrics is not None:
                    episode_losses.append(metrics["loss"])
            observation = next_observation
            episode_return += reward
            global_step += 1

        train_rows.append(
            {
                "episode": episode,
                "return": episode_return,
                "epsilon": epsilon,
                "mean_loss": (
                    float(np.mean(episode_losses)) if episode_losses else None
                ),
            }
        )

        should_validate = episode % validation_interval == 0 or episode == train_episodes
        if should_validate:
            validation = evaluate_agent(
                agent,
                eval_env_cfg,
                validation_seeds,
                state_tracking.tracker_factory,
            )
            validation_history.append({"episode": episode, **validation})
            print(
                f"episode={episode:4d} step={global_step:7d} "
                f"eps={epsilon:.3f} train_return={episode_return:.4f} "
                f"val_final_m={validation['final_mastery']:.4f}"
            )
            if validation["final_mastery"] > best_mastery:
                best_mastery = validation["final_mastery"]
                agent.save(
                    best_path,
                    extra={
                        "episode": episode,
                        "global_step": global_step,
                        "seed": seed,
                        "observation_condition": state_tracking.condition,
                        "validation": validation,
                        "actions": list(probe_env.action_names),
                        "observation_names": list(probe_env.observation_names),
                    },
                )

    elapsed = time.time() - t0
    agent.save(
        final_path,
        extra={
            "episode": train_episodes,
            "global_step": global_step,
            "seed": seed,
            "observation_condition": state_tracking.condition,
            "actions": list(probe_env.action_names),
            "observation_names": list(probe_env.observation_names),
        },
    )
    best_agent, best_extra = DQNAgent.load(best_path, device)
    heldout = evaluate_agent(
        best_agent,
        eval_env_cfg,
        heldout_seeds,
        state_tracking.tracker_factory,
    )
    plot_training(train_rows, validation_history, figure_path)

    payload = {
        "disclaimer": (
            "Simulated DQN training. Oracle mastery is an upper bound; estimated "
            "states remain simulator evaluations. No real-student learning claim."
        ),
        "device": str(device),
        "seed": seed,
        "observation_condition": state_tracking.condition,
        "train_episodes": train_episodes,
        "global_steps": global_step,
        "training_seconds": elapsed,
        "observation_names": list(probe_env.observation_names),
        "actions": list(probe_env.action_names),
        "rl_config": {
            **cfg["rl"],
            "observation_mode": state_tracking.environment_mode,
            "gamma": agent.config.gamma,
        },
        "dqn_config": cfg["dqn"],
        "effective_agent_config": asdict(agent.config),
        "train_seed_range": [train_seed_start, train_seed_end],
        "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
        "heldout_seed_range": [heldout_seeds[0], heldout_seeds[-1]],
        "initial_validation": initial_validation,
        "best_checkpoint": str(best_path),
        "best_checkpoint_extra": best_extra,
        "heldout": heldout,
        "training_history": train_rows,
        "validation_history": validation_history,
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "environment": sha256(Path("src/rl/environment.py")),
                "dqn": sha256(Path("src/rl/dqn.py")),
                "training_script": sha256(Path(__file__)),
                **(
                    {"state_checkpoint": sha256(state_tracking.checkpoint_path)}
                    if state_tracking.checkpoint_path is not None
                    else {}
                ),
            },
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"best validation final mastery={best_mastery:.4f} @ "
        f"episode={best_extra.get('episode')}"
    )
    print(
        f"held-out final mastery={heldout['final_mastery']:.4f} +/- "
        f"{heldout['final_mastery_std']:.4f} | return={heldout['return']:.4f}"
    )
    print(f"Saved {best_path}, {final_path}, {result_path}, {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--observation-mode",
        default="oracle",
        choices=["oracle", "bkt", "dkt", "no_state"],
        help="State signal used for both training and evaluation",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.episodes,
        args.device,
        args.tag,
        seed_override=args.seed,
        gamma_override=args.gamma,
        observation_condition=args.observation_mode,
    )
