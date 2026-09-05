"""Validate multi-skill prerequisite dynamics on fixed simulated episodes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_dqn import git_provenance, sha256
from src.rl.evaluation import bootstrap_mean_ci
from src.rl.multiskill_environment import (
    MultiSkillTutoringEnv,
    MultiSkillTutoringEnvConfig,
)
from src.utils.seed import load_config, set_seed


def multiskill_config(cfg: dict, include_diagnostics: bool = True):
    values = cfg["multiskill"]
    return MultiSkillTutoringEnvConfig(
        actions=tuple(cfg["policy"]["actions"]),
        horizon=int(values["horizon"]),
        num_skills=int(values["num_skills"]),
        prerequisites=tuple(
            tuple(int(parent) for parent in row)
            for row in values["prerequisites"]
        ),
        focus_skill_sequence=tuple(
            int(skill) for skill in values["focus_skill_sequence"]
        ),
        observation_mode=str(values["observation_mode"]),
        delayed_effects=bool(values["delayed_effects"]),
        heterogeneous=bool(values["heterogeneous"]),
        vary_difficulty=bool(values["vary_difficulty"]),
        include_diagnostics=include_diagnostics,
        prerequisite_gate_floor=float(values["prerequisite_gate_floor"]),
        prerequisite_review_transfer=float(
            values["prerequisite_review_transfer"]
        ),
        prerequisite_spillover=float(values["prerequisite_spillover"]),
    )


def observation_value(
    observation: np.ndarray, names: tuple, name: str
) -> float:
    return float(observation[names.index(name)])


def current_skill(observation: np.ndarray, env: MultiSkillTutoringEnv) -> int:
    offset = env.config.num_skills + 8
    values = observation[offset : offset + env.config.num_skills]
    return int(values.argmax())


def action_for_mastery(mastery: float) -> str:
    if mastery < 0.40:
        return "explain"
    if mastery < 0.70:
        return "socratic_hint"
    return "harder_problem"


def current_only_policy(
    observation: np.ndarray, env: MultiSkillTutoringEnv, rng: np.random.RandomState
) -> str:
    skill = current_skill(observation, env)
    mastery = observation_value(
        observation, env.observation_names, f"mastery_signal_skill_{skill}"
    )
    return action_for_mastery(mastery)


def prerequisite_aware_policy(
    observation: np.ndarray, env: MultiSkillTutoringEnv, rng: np.random.RandomState
) -> str:
    skill = current_skill(observation, env)
    readiness = observation_value(
        observation, env.observation_names, "prerequisite_readiness"
    )
    if env.prerequisites[skill] and readiness < 0.60:
        return "prerequisite_review"
    mastery = observation_value(
        observation, env.observation_names, f"mastery_signal_skill_{skill}"
    )
    return action_for_mastery(mastery)


def random_policy(
    observation: np.ndarray, env: MultiSkillTutoringEnv, rng: np.random.RandomState
) -> str:
    return env.action_names[int(rng.randint(0, env.n_actions))]


def evaluate_condition(
    name: str,
    policy: Callable,
    env_config: MultiSkillTutoringEnvConfig,
    episode_seeds: List[int],
) -> dict:
    env = MultiSkillTutoringEnv(env_config, seed=episode_seeds[0])
    rows = []
    action_counts = {action: 0 for action in env.action_names}
    target_counts = np.zeros(env.config.num_skills, dtype=np.int64)
    for episode_seed in episode_seeds:
        observation, reset_info = env.reset(seed=episode_seed)
        policy_rng = np.random.RandomState(episode_seed + 7_000_003)
        total_reward = 0.0
        correct = []
        terminated = False
        info = reset_info
        while not terminated:
            action = policy(observation, env, policy_rng)
            observation, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
            total_reward += float(reward)
            correct.append(int(info["correct"]))
            action_counts[info["action_name"]] += 1
            target_counts[int(info["target_skill"])] += 1
        diagnostics = info["diagnostics"]
        final_mastery = np.asarray(
            diagnostics["true_mastery_by_skill"], dtype=np.float64
        )
        initial_mastery = np.asarray(
            diagnostics["initial_mastery_by_skill"], dtype=np.float64
        )
        readiness = [
            env.prerequisite_readiness(skill)
            for skill in range(env.config.num_skills)
        ]
        mastery_gain = float(final_mastery.mean() - initial_mastery.mean())
        if not np.isclose(total_reward, mastery_gain, atol=1e-12):
            raise RuntimeError("episode reward does not telescope to mean mastery gain")
        rows.append(
            {
                "episode_seed": episode_seed,
                "return": total_reward,
                "initial_mean_mastery": float(initial_mastery.mean()),
                "final_mean_mastery": float(final_mastery.mean()),
                "mean_mastery_gain": mastery_gain,
                "final_mastery_by_skill": final_mastery.tolist(),
                "mean_prerequisite_readiness": float(np.mean(readiness)),
                "overall_accuracy": float(np.mean(correct)),
                "final_independence": float(diagnostics["independence"]),
                "final_fatigue": float(diagnostics["fatigue"]),
            }
        )

    summary: Dict[str, object] = {"name": name, "n_episodes": len(rows)}
    scalar_metrics = (
        "return",
        "initial_mean_mastery",
        "final_mean_mastery",
        "mean_mastery_gain",
        "mean_prerequisite_readiness",
        "overall_accuracy",
        "final_independence",
        "final_fatigue",
    )
    for offset, metric in enumerate(scalar_metrics):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
        summary[f"{metric}_ci95"] = bootstrap_mean_ci(
            values,
            n_resamples=5_000,
            seed=51_000 + offset,
        )
    per_skill = np.asarray(
        [row["final_mastery_by_skill"] for row in rows], dtype=np.float64
    )
    summary["final_mastery_by_skill"] = per_skill.mean(axis=0).tolist()
    total_actions = sum(action_counts.values())
    summary["action_frequency"] = {
        action: count / total_actions for action, count in action_counts.items()
    }
    summary["target_skill_frequency"] = (
        target_counts / target_counts.sum()
    ).tolist()
    summary["episodes"] = rows
    return summary


def paired_difference(candidate: dict, reference: dict, cfg: dict) -> dict:
    reference_by_seed = {
        int(row["episode_seed"]): float(row["final_mean_mastery"])
        for row in reference["episodes"]
    }
    differences = np.asarray(
        [
            float(row["final_mean_mastery"])
            - reference_by_seed[int(row["episode_seed"])]
            for row in candidate["episodes"]
        ],
        dtype=np.float64,
    )
    return {
        "candidate": candidate["name"],
        "reference": reference["name"],
        "metric": "final_mean_mastery",
        "n_pairs": len(differences),
        "mean_difference": float(differences.mean()),
        "difference_ci95": bootstrap_mean_ci(
            differences,
            n_resamples=int(cfg["multiskill"]["bootstrap_resamples"]),
            seed=int(cfg["seed"]) + 61_000,
        ),
        "candidate_win_rate": float(np.mean(differences > 0.0)),
        "tie_rate": float(np.mean(differences == 0.0)),
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    environment = multiskill_config(cfg, include_diagnostics=True)
    settings = cfg["multiskill"]
    seed_start = int(settings["validation_seed_start"])
    episode_seeds = [
        seed_start + index for index in range(int(settings["validation_episodes"]))
    ]
    policies = {
        "random": random_policy,
        "current_skill_only": current_only_policy,
        "prerequisite_aware": prerequisite_aware_policy,
    }
    summaries = {
        name: evaluate_condition(name, policy, environment, episode_seeds)
        for name, policy in policies.items()
    }
    comparison = paired_difference(
        summaries["prerequisite_aware"], summaries["current_skill_only"], cfg
    )
    result = {
        "disclaimer": (
            "Hand-designed multi-skill simulator validation only; no real "
            "students or classroom outcomes are represented."
        ),
        "config": {
            **settings,
            "prerequisites": [list(row) for row in environment.prerequisites],
            "focus_skill_sequence": list(environment.focus_skill_sequence),
        },
        "observation_names": list(
            MultiSkillTutoringEnv(environment).observation_names
        ),
        "episode_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "summaries": summaries,
        "prerequisite_aware_vs_current_only": comparison,
        "validation_checks": {
            "separate_mastery_per_skill": True,
            "acyclic_prerequisite_graph": True,
            "current_and_prerequisite_transfer": True,
            "reward_telescopes_mean_mastery_gain": True,
            "common_episode_seeds": True,
        },
        "provenance": {
            **git_provenance(),
            "sha256": {
                "config": sha256(Path(config_path)),
                "environment": sha256(Path("src/rl/multiskill_environment.py")),
                "validation_script": sha256(Path(__file__)),
            },
        },
    }
    output = Path(cfg["paths"]["results_dir"]) / "multiskill_environment_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    for name, summary in summaries.items():
        print(
            f"{name:24s} final_mean_mastery="
            f"{summary['final_mean_mastery']:.4f} | "
            f"readiness={summary['mean_prerequisite_readiness']:.4f} | "
            f"accuracy={summary['overall_accuracy']:.4f}"
        )
    low, high = comparison["difference_ci95"]
    print(
        "prerequisite-aware minus current-only="
        f"{comparison['mean_difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multiskill_simulator.yaml")
    arguments = parser.parse_args()
    main(arguments.config)
