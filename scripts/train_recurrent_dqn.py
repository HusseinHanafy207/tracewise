"""Train GRU Double DQN with complete-episode recurrent replay."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_dqn import (
    environment_config,
    git_provenance,
    plot_training,
    sha256,
)
from src.rl.dqn import QNetwork, linear_epsilon
from src.rl.environment import OnlineTracker, TutoringEnv, TutoringEnvConfig
from src.rl.recurrent_dqn import RecurrentDQNAgent, RecurrentDQNConfig
from src.rl.state_tracking import load_state_tracking
from src.utils.seed import load_config, set_seed


def recurrent_agent_config(cfg: dict) -> RecurrentDQNConfig:
    dqn = cfg["dqn"]
    recurrent = cfg["recurrent_dqn"]
    return RecurrentDQNConfig(
        recurrent_hidden_dim=int(recurrent["recurrent_hidden_dim"]),
        head_hidden_dims=tuple(int(value) for value in recurrent["head_hidden_dims"]),
        gamma=float(cfg["rl"]["gamma"]),
        learning_rate=float(dqn["learning_rate"]),
        episode_batch_size=int(recurrent["episode_batch_size"]),
        replay_capacity_episodes=int(recurrent["replay_capacity_episodes"]),
        min_replay_episodes=int(recurrent["min_replay_episodes"]),
        target_sync_updates=int(dqn["target_sync_steps"]),
        gradient_clip=float(dqn["gradient_clip"]),
        reward_scale=float(dqn["reward_scale"]),
        double_dqn=bool(dqn["double_dqn"]),
    )


def feedforward_parameter_count(cfg: dict, observation_dim: int, n_actions: int) -> int:
    network = QNetwork(
        observation_dim,
        n_actions,
        tuple(int(value) for value in cfg["dqn"]["hidden_dims"]),
    )
    return sum(parameter.numel() for parameter in network.parameters())


def evaluate_recurrent_agent(
    agent: RecurrentDQNAgent,
    env_config: TutoringEnvConfig,
    episode_seeds: List[int],
    tracker_factory: Optional[Callable[[], OnlineTracker]] = None,
) -> dict:
    environment = TutoringEnv(
        env_config,
        tracker_factory=tracker_factory,
        seed=episode_seeds[0],
    )
    rows = []
    action_counts = {name: 0 for name in environment.action_names}
    for episode_seed in episode_seeds:
        observation, _ = environment.reset(seed=episode_seed)
        agent.reset_hidden()
        terminated = False
        episode_return = 0.0
        correct = []
        info = {}
        while not terminated:
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = environment.step(action)
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
            episode_return += float(reward)
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
        summary[f"{key}_std"] = float(
            values.std(ddof=1) if len(values) > 1 else 0.0
        )
    total_actions = sum(action_counts.values()) or 1
    summary["action_frequency"] = {
        action: count / total_actions for action, count in action_counts.items()
    }
    return summary


def main(
    config_path: str,
    device_name: str,
    tag: Optional[str],
    seed_override: Optional[int],
) -> None:
    cfg = load_config(config_path)
    seed = int(seed_override if seed_override is not None else cfg["seed"])
    set_seed(seed)
    dqn_cfg = cfg["dqn"]
    train_episodes = int(dqn_cfg["train_episodes"])
    torch.set_num_threads(int(dqn_cfg.get("torch_num_threads", 1)))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    checkpoint_dir = Path(cfg["paths"]["checkpoints_dir"])
    tracking = load_state_tracking("oracle", checkpoint_dir, device)
    train_env_config = environment_config(
        cfg,
        include_diagnostics=False,
        observation_mode=tracking.environment_mode,
    )
    eval_env_config = environment_config(
        cfg,
        include_diagnostics=True,
        observation_mode=tracking.environment_mode,
    )
    probe = TutoringEnv(
        train_env_config,
        tracker_factory=tracking.tracker_factory,
        seed=seed,
    )
    agent = RecurrentDQNAgent(
        probe.observation_dim,
        probe.n_actions,
        train_env_config.horizon,
        recurrent_agent_config(cfg),
        device,
        seed=seed,
    )
    feedforward_parameters = feedforward_parameter_count(
        cfg, probe.observation_dim, probe.n_actions
    )

    train_seed_start = int(dqn_cfg["train_seed_start"])
    validation_seeds = [
        int(dqn_cfg["validation_seed_start"]) + index
        for index in range(int(dqn_cfg["validation_episodes"]))
    ]
    heldout_seeds = [
        int(dqn_cfg["heldout_seed_start"]) + index
        for index in range(int(dqn_cfg["heldout_eval_episodes"]))
    ]
    train_seed_range = set(range(train_seed_start, train_seed_start + train_episodes))
    if (
        train_seed_range & set(validation_seeds)
        or train_seed_range & set(heldout_seeds)
        or set(validation_seeds) & set(heldout_seeds)
    ):
        raise ValueError("training, validation, and held-out seeds must be disjoint")

    results_dir = Path(cfg["paths"]["results_dir"])
    suffix = f"_{tag}" if tag else ""
    stem = "recurrent_dqn_oracle"
    best_path = checkpoint_dir / f"{stem}_best{suffix}.pt"
    final_path = checkpoint_dir / f"{stem}_final{suffix}.pt"
    figure_path = results_dir / "figures" / f"{stem}_training{suffix}.png"
    result_path = results_dir / f"{stem}_training{suffix}.json"

    print(
        "GRU Double DQN oracle training | "
        f"device={device} | seed={seed} | episodes={train_episodes} | "
        f"parameters={agent.parameter_count} "
        f"(feed-forward reference={feedforward_parameters})"
    )
    print(
        "DISCLAIMER: simulated delayed-effect environment; independence and "
        "fatigue are hidden, and no real students were evaluated."
    )

    validation_history = []
    initial_validation = evaluate_recurrent_agent(
        agent,
        eval_env_config,
        validation_seeds,
        tracking.tracker_factory,
    )
    validation_history.append({"episode": 0, **initial_validation})
    best_mastery = initial_validation["final_mastery"]
    common_extra = {
        "seed": seed,
        "observation_condition": tracking.condition,
        "actions": list(probe.action_names),
        "observation_names": list(probe.observation_names),
        "architecture": "gru",
    }
    agent.save(
        best_path,
        extra={
            **common_extra,
            "episode": 0,
            "global_step": 0,
            "validation": initial_validation,
        },
    )

    train_rows = []
    global_step = 0
    environment = TutoringEnv(
        train_env_config,
        tracker_factory=tracking.tracker_factory,
        seed=train_seed_start,
    )
    validation_interval = int(dqn_cfg["validation_interval"])
    train_frequency = int(dqn_cfg["train_frequency"])
    min_replay_transitions = int(dqn_cfg["min_replay_size"])
    started = time.time()

    for episode in range(1, train_episodes + 1):
        observation, _ = environment.reset(seed=train_seed_start + episode - 1)
        agent.reset_hidden()
        observations = [observation.copy()]
        actions = []
        rewards = []
        dones = []
        episode_return = 0.0
        scheduled_updates = 0
        terminated = False
        while not terminated:
            epsilon = linear_epsilon(
                global_step,
                float(dqn_cfg["epsilon_start"]),
                float(dqn_cfg["epsilon_end"]),
                int(dqn_cfg["epsilon_decay_steps"]),
            )
            action = agent.select_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            if truncated:
                raise RuntimeError("fixed-horizon environment must not truncate")
            observations.append(next_observation.copy())
            actions.append(action)
            rewards.append(float(reward))
            dones.append(terminated)
            if (
                global_step >= min_replay_transitions
                and global_step % train_frequency == 0
            ):
                scheduled_updates += 1
            observation = next_observation
            episode_return += float(reward)
            global_step += 1

        agent.store_episode(
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )
        episode_losses = []
        for _ in range(scheduled_updates):
            metrics = agent.learn()
            if metrics is not None:
                episode_losses.append(metrics["loss"])
        train_rows.append(
            {
                "episode": episode,
                "return": episode_return,
                "epsilon": epsilon,
                "scheduled_updates": scheduled_updates,
                "completed_updates": len(episode_losses),
                "mean_loss": (
                    float(np.mean(episode_losses)) if episode_losses else None
                ),
            }
        )

        should_validate = episode % validation_interval == 0 or episode == train_episodes
        if should_validate:
            validation = evaluate_recurrent_agent(
                agent,
                eval_env_config,
                validation_seeds,
                tracking.tracker_factory,
            )
            validation_history.append({"episode": episode, **validation})
            print(
                f"episode={episode:4d} step={global_step:7d} "
                f"updates={agent.learn_steps:6d} eps={epsilon:.3f} "
                f"train_return={episode_return:.4f} "
                f"val_final_m={validation['final_mastery']:.4f}"
            )
            if validation["final_mastery"] > best_mastery:
                best_mastery = validation["final_mastery"]
                agent.save(
                    best_path,
                    extra={
                        **common_extra,
                        "episode": episode,
                        "global_step": global_step,
                        "validation": validation,
                    },
                )

    elapsed = time.time() - started
    agent.save(
        final_path,
        extra={
            **common_extra,
            "episode": train_episodes,
            "global_step": global_step,
        },
    )
    best_agent, best_extra = RecurrentDQNAgent.load(best_path, device)
    heldout = evaluate_recurrent_agent(
        best_agent,
        eval_env_config,
        heldout_seeds,
        tracking.tracker_factory,
    )
    plot_training(train_rows, validation_history, figure_path)

    payload = {
        "disclaimer": (
            "Simulated recurrent Double DQN training only. Hidden-state and "
            "oracle-mastery results are not real-student evidence."
        ),
        "device": str(device),
        "algorithm": "recurrent_double_dqn",
        "architecture": "gru",
        "seed": seed,
        "observation_condition": tracking.condition,
        "train_episodes": train_episodes,
        "global_steps": global_step,
        "gradient_updates": agent.learn_steps,
        "training_seconds": elapsed,
        "parameter_count": agent.parameter_count,
        "feedforward_reference_parameter_count": feedforward_parameters,
        "parameter_count_ratio": agent.parameter_count / feedforward_parameters,
        "replay_loss_transitions_per_update": (
            agent.config.episode_batch_size * train_env_config.horizon
        ),
        "observation_names": list(probe.observation_names),
        "hidden_simulator_state": [
            "independence",
            "fatigue",
            "learner_preferences",
            "response_randomness",
        ],
        "actions": list(probe.action_names),
        "rl_config": cfg["rl"],
        "dqn_shared_config": cfg["dqn"],
        "recurrent_config": cfg["recurrent_dqn"],
        "effective_agent_config": asdict(agent.config),
        "train_seed_range": [
            train_seed_start,
            train_seed_start + train_episodes - 1,
        ],
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
                "recurrent_dqn": sha256(Path("src/rl/recurrent_dqn.py")),
                "training_script": sha256(Path(__file__)),
            },
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(
        f"best validation final mastery={best_mastery:.4f} @ "
        f"episode={best_extra['episode']}"
    )
    print(
        f"held-out final mastery={heldout['final_mastery']:.4f} +/- "
        f"{heldout['final_mastery_std']:.4f} | return={heldout['return']:.4f}"
    )
    print(f"Saved {best_path}, {final_path}, {result_path}, {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recurrent_double_dqn_train.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=None)
    arguments = parser.parse_args()
    main(arguments.config, arguments.device, arguments.tag, arguments.seed)
