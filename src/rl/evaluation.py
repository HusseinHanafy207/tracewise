"""Paired episode evaluation utilities for sequential tutoring policies."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from src.rl.environment import OnlineTracker, TutoringEnv, TutoringEnvConfig

Action = Union[int, str]
EpisodePolicy = Callable[[np.ndarray, int], Action]
PolicyFactory = Callable[[int], EpisodePolicy]


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 5_000,
    seed: int = 42,
) -> List[float]:
    """Return a deterministic percentile-bootstrap interval for a mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(array), size=(n_resamples, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    ]


def evaluate_policy(
    name: str,
    policy_factory: PolicyFactory,
    env_config: TutoringEnvConfig,
    episode_seeds: Sequence[int],
    tracker_factory: Optional[Callable[[], OnlineTracker]] = None,
    bootstrap_seed: int = 42,
    bootstrap_resamples: int = 5_000,
) -> dict:
    """Evaluate one policy on fixed episode seeds and retain paired rows."""
    if not episode_seeds:
        raise ValueError("episode_seeds must not be empty")
    env = TutoringEnv(
        env_config,
        tracker_factory=tracker_factory,
        seed=int(episode_seeds[0]),
    )
    rows = []
    mastery_curves = []
    action_counts = {action: 0 for action in env.action_names}

    for episode_seed in episode_seeds:
        policy = policy_factory(int(episode_seed))
        observation, info = env.reset(seed=int(episode_seed))
        diagnostics = info["diagnostics"]
        mastery_curve = [float(diagnostics["true_mastery"])]
        correct = []
        episode_return = 0.0
        terminated = False
        step = 0
        while not terminated:
            action = policy(observation, step)
            observation, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
            episode_return += float(reward)
            correct.append(int(info["correct"]))
            action_counts[info["action_name"]] += 1
            mastery_curve.append(float(info["diagnostics"]["true_mastery"]))
            step += 1

        final = info["diagnostics"]
        rows.append(
            {
                "episode_seed": int(episode_seed),
                "return": episode_return,
                "final_mastery": float(final["true_mastery"]),
                "overall_accuracy": float(np.mean(correct)),
                "final_accuracy": float(np.mean(correct[-min(5, len(correct)) :])),
                "final_independence": float(final["independence"]),
                "final_fatigue": float(final["fatigue"]),
            }
        )
        mastery_curves.append(mastery_curve)

    summary: Dict[str, object] = {"name": name, "n_episodes": len(rows)}
    metric_names = [key for key in rows[0] if key != "episode_seed"]
    for offset, metric in enumerate(metric_names):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = float(values.mean())
        summary[f"{metric}_std"] = float(
            values.std(ddof=1) if len(values) > 1 else 0.0
        )
        summary[f"{metric}_ci95"] = bootstrap_mean_ci(
            values,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + offset,
        )
    total_actions = sum(action_counts.values()) or 1
    summary["action_frequency"] = {
        action: count / total_actions for action, count in action_counts.items()
    }
    summary["mastery_curve_mean"] = np.mean(
        np.asarray(mastery_curves, dtype=np.float64), axis=0
    ).tolist()
    summary["episodes"] = rows
    return summary


def paired_difference(
    candidate: dict,
    reference: dict,
    metric: str = "final_mastery",
    bootstrap_seed: int = 42,
    bootstrap_resamples: int = 5_000,
) -> dict:
    """Compute candidate-minus-reference effects on matching episode seeds."""
    candidate_by_seed = {
        int(row["episode_seed"]): float(row[metric]) for row in candidate["episodes"]
    }
    reference_by_seed = {
        int(row["episode_seed"]): float(row[metric]) for row in reference["episodes"]
    }
    if candidate_by_seed.keys() != reference_by_seed.keys():
        raise ValueError("paired comparisons require identical episode seeds")
    seeds = sorted(candidate_by_seed)
    differences = np.asarray(
        [candidate_by_seed[seed] - reference_by_seed[seed] for seed in seeds],
        dtype=np.float64,
    )
    return {
        "candidate": candidate["name"],
        "reference": reference["name"],
        "metric": metric,
        "n_pairs": len(differences),
        "mean_difference": float(differences.mean()),
        "difference_std": float(
            differences.std(ddof=1) if len(differences) > 1 else 0.0
        ),
        "difference_ci95": bootstrap_mean_ci(
            differences,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "candidate_win_rate": float(np.mean(differences > 0.0)),
        "tie_rate": float(np.mean(differences == 0.0)),
    }
