"""Validate that the episodic RL environment has a delayed-reward challenge.

This is environment calibration, not DQN training and not evidence about real
students. It compares fixed/oracle policies with delayed effects off and on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.simulator import intervention_effect
from src.rl.environment import TutoringEnv, TutoringEnvConfig
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


def always(action: str) -> Callable[[np.ndarray, int], str]:
    return lambda observation, step: action


def myopic_oracle(actions: List[str]) -> Callable[[np.ndarray, int], str]:
    def choose(observation: np.ndarray, step: int) -> str:
        mastery = float(observation[0])
        return max(actions, key=lambda action: intervention_effect(action, mastery))

    return choose


def interleave(observation: np.ndarray, step: int) -> str:
    return "explain" if step % 2 == 0 else "harder_problem"


def evaluate(
    name: str,
    policy: Callable[[np.ndarray, int], str],
    env_config: TutoringEnvConfig,
    episode_seeds: List[int],
) -> dict:
    rows = []
    for seed in episode_seeds:
        env = TutoringEnv(env_config, seed=seed)
        observation, _ = env.reset(seed=seed)
        rewards = []
        terminated = False
        info = {}
        while not terminated:
            action = policy(observation, len(rewards))
            observation, reward, terminated, truncated, info = env.step(action)
            assert not truncated
            rewards.append(reward)

        diagnostics = info["diagnostics"]
        rows.append(
            {
                "return": float(sum(rewards)),
                "early_reward": float(np.mean(rewards[: min(10, len(rewards))])),
                "late_reward": float(np.mean(rewards[-min(5, len(rewards)) :])),
                "final_mastery": diagnostics["true_mastery"],
                "final_independence": diagnostics["independence"],
                "final_fatigue": diagnostics["fatigue"],
            }
        )

    summary: Dict[str, float] = {"name": name, "n_episodes": len(rows)}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
    return summary


def print_results(label: str, summaries: List[dict]) -> None:
    print(f"\n=== delayed_effects={label} ===")
    print("policy                  early_r   late_r    return   final_m   final_I")
    for row in summaries:
        print(
            f"{row['name']:<23} "
            f"{row['early_reward']:>8.4f} "
            f"{row['late_reward']:>8.4f} "
            f"{row['return']:>9.4f} "
            f"{row['final_mastery']:>9.3f} "
            f"{row['final_independence']:>9.3f}"
        )


def main(config_path: str, n_episodes: int) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    actions = list(cfg["policy"]["actions"])
    rl_cfg = cfg["rl"]
    episode_seeds = [seed + i for i in range(n_episodes)]

    policies = {
        "always_explain": always("explain"),
        "always_harder": always("harder_problem"),
        "myopic_oracle": myopic_oracle(actions),
        "interleave_explain_harder": interleave,
    }
    payload = {
        "disclaimer": "Simulated environment validation only; no DQN and no real students.",
        "n_episodes": n_episodes,
        "episode_seeds": episode_seeds,
        "horizon": int(rl_cfg["horizon"]),
        "observation_mode": "oracle",
        "reward": rl_cfg["reward"],
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "environment": sha256(Path("src/rl/environment.py")),
                "simulator": sha256(Path("src/policy/simulator.py")),
                "validation_script": sha256(Path(__file__)),
            },
        },
    }

    all_results = {}
    for delayed in (False, True):
        env_config = TutoringEnvConfig(
            actions=tuple(actions),
            horizon=int(rl_cfg["horizon"]),
            num_skills=int(rl_cfg["num_skills"]),
            observation_mode="oracle",
            delayed_effects=delayed,
            heterogeneous=False,
            include_diagnostics=True,
        )
        summaries = [
            evaluate(name, policy, env_config, episode_seeds)
            for name, policy in policies.items()
        ]
        label = "on" if delayed else "off"
        all_results[label] = summaries
        print_results(label.upper(), summaries)

    def by_name(label: str, name: str) -> dict:
        return next(row for row in all_results[label] if row["name"] == name)

    checks = {
        "off_myopic_beats_interleave_final": (
            by_name("off", "myopic_oracle")["final_mastery"]
            > by_name("off", "interleave_explain_harder")["final_mastery"]
        ),
        "on_interleave_beats_myopic_final": (
            by_name("on", "interleave_explain_harder")["final_mastery"]
            > by_name("on", "myopic_oracle")["final_mastery"]
        ),
        "on_myopic_beats_interleave_early": (
            by_name("on", "myopic_oracle")["early_reward"]
            > by_name("on", "interleave_explain_harder")["early_reward"]
        ),
    }
    payload["delayed_off"] = all_results["off"]
    payload["delayed_on"] = all_results["on"]
    payload["checks"] = checks
    payload["calibration_passed"] = all(checks.values())

    print("\nchecks:")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "rl_env_validation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")

    if not payload["calibration_passed"]:
        raise SystemExit("RL environment calibration failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-episodes", type=int, default=500)
    args = parser.parse_args()
    if args.n_episodes <= 0:
        parser.error("--n-episodes must be positive")
    main(args.config, args.n_episodes)
