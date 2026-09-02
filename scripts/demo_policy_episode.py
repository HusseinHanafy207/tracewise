from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.simulator import intervention_effect
from src.rl.dqn import DQNAgent
from src.rl.environment import TutoringEnv, TutoringEnvConfig
from src.utils.seed import load_config


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
        return {"head_commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head_commit": None, "working_tree_dirty": None}


def environment_config(config: dict) -> TutoringEnvConfig:
    rl = config["rl"]
    return TutoringEnvConfig(
        actions=tuple(config["policy"]["actions"]),
        horizon=int(rl["horizon"]),
        num_skills=int(rl["num_skills"]),
        observation_mode="oracle",
        delayed_effects=bool(rl["delayed_effects"]),
        heterogeneous=bool(rl["heterogeneous"]),
        include_diagnostics=True,
    )


def make_policy(
    policy_name: str,
    actions: Tuple[str, ...],
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Callable[[np.ndarray, int], int | str], Optional[DQNAgent]]:
    if policy_name == "interleave":
        return (
            lambda observation, step: (
                "explain" if step % 2 == 0 else "harder_problem"
            ),
            None,
        )
    if policy_name == "myopic":
        return (
            lambda observation, step: max(
                actions,
                key=lambda action: intervention_effect(action, float(observation[0])),
            ),
            None,
        )
    if policy_name != "dqn":
        raise ValueError("policy_name must be dqn, interleave, or myopic")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Use --replay docs/examples/dqn_episode_seed200000.json "
            "for the committed no-checkpoint demo."
        )
    agent, _ = DQNAgent.load(checkpoint_path, device)
    if agent.n_actions != len(actions):
        raise ValueError("DQN checkpoint action count does not match config")
    return (
        lambda observation, step: agent.select_action(observation, epsilon=0.0),
        agent,
    )


def run_episode(
    config_path: str,
    policy_name: str,
    checkpoint_path: Path,
    episode_seed: int,
    device: torch.device,
) -> dict:
    config = load_config(config_path)
    env = TutoringEnv(environment_config(config), seed=episode_seed)
    policy, agent = make_policy(
        policy_name,
        env.action_names,
        checkpoint_path,
        device,
    )
    observation, info = env.reset(seed=episode_seed)
    initial = info["diagnostics"]
    steps = []
    terminated = False
    step = 0
    total_return = 0.0
    while not terminated:
        mastery_before = float(info["diagnostics"]["true_mastery"])
        action = policy(observation, step)
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("fixed-horizon environment must not truncate")
        diagnostics = info["diagnostics"]
        total_return += float(reward)
        steps.append(
            {
                "step": step + 1,
                "action": info["action_name"],
                "correct": int(info["correct"]),
                "reward": float(reward),
                "mastery_before": mastery_before,
                "mastery_after": float(diagnostics["true_mastery"]),
                "independence": float(diagnostics["independence"]),
                "fatigue": float(diagnostics["fatigue"]),
                "base_difficulty": float(info["base_difficulty"]),
            }
        )
        step += 1

    action_counts = {
        action: sum(row["action"] == action for row in steps)
        for action in env.action_names
    }
    final = info["diagnostics"]
    payload = {
        "schema_version": 1,
        "disclaimer": (
            "Simulated oracle-state evaluation only. No real student was evaluated; "
            "true mastery is shown only to explain the saved research demo."
        ),
        "policy": policy_name,
        "episode_seed": int(episode_seed),
        "horizon": len(steps),
        "provenance": {
            **git_provenance(),
            "generator": "scripts/demo_policy_episode.py",
            "generator_sha256": sha256(Path(__file__)),
        },
        "config": str(Path(config_path).as_posix()),
        "checkpoint": (
            {
                "name": checkpoint_path.name,
                "sha256": sha256(checkpoint_path),
            }
            if agent is not None
            else None
        ),
        "initial_mastery": float(initial["initial_mastery"]),
        "final_mastery": float(final["true_mastery"]),
        "return": total_return,
        "overall_accuracy": float(np.mean([row["correct"] for row in steps])),
        "final_independence": float(final["independence"]),
        "final_fatigue": float(final["fatigue"]),
        "action_counts": action_counts,
        "steps": steps,
    }
    validate_trace(payload)
    return payload


def validate_trace(payload: dict) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported demo trace schema")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Demo trace must contain at least one step")
    if int(payload.get("horizon", -1)) != len(steps):
        raise ValueError("Demo trace horizon does not match its step count")
    required = {
        "step",
        "action",
        "correct",
        "reward",
        "mastery_before",
        "mastery_after",
        "independence",
        "fatigue",
        "base_difficulty",
    }
    if any(not required.issubset(row) for row in steps):
        raise ValueError("Demo trace step is missing required fields")


def load_trace(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_trace(payload)
    return payload


def save_trace(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_trace(payload: dict) -> None:
    print(payload["disclaimer"])
    print(
        f"policy={payload['policy']} | seed={payload['episode_seed']} | "
        f"mastery={payload['initial_mastery']:.3f}->{payload['final_mastery']:.3f} | "
        f"return={payload['return']:.4f} | accuracy={payload['overall_accuracy']:.3f}"
    )
    print("step action                 correct reward   mastery  independence fatigue")
    for row in payload["steps"]:
        print(
            f"{row['step']:>4} {row['action']:<22} {row['correct']:>7} "
            f"{row['reward']:>+7.4f} {row['mastery_after']:>8.3f} "
            f"{row['independence']:>12.3f} {row['fatigue']:>7.3f}"
        )


def render_gif(payload: dict, path: Path, max_frames: int = 16) -> None:
    if max_frames <= 1:
        raise ValueError("max_frames must be greater than one")
    steps = payload["steps"]
    frame_indices = np.unique(
        np.linspace(0, len(steps) - 1, min(max_frames, len(steps)), dtype=int)
    )
    actions = sorted({row["action"] for row in steps})
    color_map = {
        action: plt.get_cmap("tab10")(index % 10)
        for index, action in enumerate(actions)
    }
    x = np.arange(0, len(steps) + 1)
    mastery = np.asarray(
        [payload["initial_mastery"]]
        + [row["mastery_after"] for row in steps],
        dtype=np.float64,
    )
    rewards = np.asarray([row["reward"] for row in steps], dtype=np.float64)

    fig, (mastery_ax, reward_ax) = plt.subplots(
        2,
        1,
        figsize=(8.4, 5.2),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    def draw(frame_position: int):
        index = int(frame_indices[frame_position])
        row = steps[index]
        mastery_ax.clear()
        reward_ax.clear()
        mastery_ax.plot(x[: index + 2], mastery[: index + 2], color="#2f6f9f", linewidth=3)
        mastery_ax.scatter(index + 1, mastery[index + 1], color="#c44e52", s=55, zorder=3)
        mastery_ax.set_xlim(0, len(steps))
        mastery_ax.set_ylim(0.0, 1.0)
        mastery_ax.set_ylabel("Oracle mastery\n(evaluation only)")
        mastery_ax.grid(alpha=0.2)
        mastery_ax.set_title(
            f"Tracewise DQN episode - step {index + 1}/{len(steps)}\n"
            f"action: {row['action']} | correct: {row['correct']} | "
            f"reward: {row['reward']:+.4f}"
        )

        positions = np.arange(index + 1)
        reward_ax.bar(
            positions,
            rewards[: index + 1],
            color=[color_map[item["action"]] for item in steps[: index + 1]],
        )
        reward_ax.axhline(0.0, color="black", linewidth=0.8)
        reward_ax.set_xlim(-0.5, len(steps) - 0.5)
        reward_ax.set_ylabel("Mastery gain")
        reward_ax.set_xlabel("Interaction")
        fig.tight_layout()
        return []

    animation = FuncAnimation(
        fig,
        draw,
        frames=len(frame_indices),
        interval=650,
        repeat=True,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn_train.yaml")
    parser.add_argument("--policy", choices=["dqn", "interleave", "myopic"], default="dqn")
    parser.add_argument("--checkpoint", default="results/checkpoints/dqn_oracle_best.pt")
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--save-trace", type=Path, default=None)
    parser.add_argument("--save-gif", type=Path, default=None)
    parser.add_argument("--gif-frames", type=int, default=16)
    args = parser.parse_args()

    if args.replay is not None:
        payload = load_trace(args.replay)
    else:
        payload = run_episode(
            args.config,
            args.policy,
            Path(args.checkpoint),
            args.seed,
            torch.device(args.device),
        )
    print_trace(payload)
    if args.save_trace is not None:
        save_trace(payload, args.save_trace)
        print(f"Wrote {args.save_trace}")
    if args.save_gif is not None:
        render_gif(payload, args.save_gif, max_frames=args.gif_frames)
        print(f"Wrote {args.save_gif}")


if __name__ == "__main__":
    main()
