"""Episodic tutoring environment for sequential RL.

The API follows Gymnasium's reset/step shape without requiring Gymnasium:

    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(action_id)

The environment is a POMDP in deployable modes. True mastery, independence,
fatigue, and learner preferences belong to the simulator's hidden state and
never appear in the observation. Oracle mastery is available only as an
explicit diagnostic upper-bound mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple, Union

import numpy as np

from src.policy.simulator import (
    SimulatedStudent,
    difficulty_for_action,
    make_population_with_skills,
)


DEFAULT_ACTIONS: Tuple[str, ...] = (
    "explain",
    "worked_example",
    "socratic_hint",
    "easier_problem",
    "harder_problem",
    "prerequisite_review",
)

BASE_OBSERVATION_NAMES: Tuple[str, ...] = (
    "mastery_signal",
    "recent_accuracy",
    "base_difficulty",
    "progress",
    "recent_improvement",
    "worked_example_success",
    "consecutive_failures",
    "socratic_hint_success",
)
OBSERVATION_NAMES: Tuple[str, ...] = BASE_OBSERVATION_NAMES + tuple(
    f"last_action_{action}" for action in DEFAULT_ACTIONS
)


class OnlineTracker(Protocol):
    """Minimal BKT/DKT online-tracker interface used by estimated mode."""

    def estimate_mastery(self, skill: int) -> float:
        ...

    def update(self, skill: int, correct: int) -> None:
        ...


@dataclass(frozen=True)
class TutoringEnvConfig:
    actions: Tuple[str, ...] = DEFAULT_ACTIONS
    horizon: int = 50
    num_skills: int = 123
    observation_mode: str = "oracle"  # oracle | estimated | no_state
    delayed_effects: bool = True
    heterogeneous: bool = False
    vary_difficulty: bool = True
    include_diagnostics: bool = False

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        unknown_actions = set(self.actions) - set(DEFAULT_ACTIONS)
        if unknown_actions:
            raise ValueError(f"actions are not implemented by the simulator: {unknown_actions}")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.num_skills <= 0:
            raise ValueError("num_skills must be positive")
        if self.observation_mode not in {"oracle", "estimated", "no_state"}:
            raise ValueError(
                "observation_mode must be 'oracle', 'estimated', or 'no_state'"
            )


class TutoringEnv:
    """One simulated student per fixed-horizon episode."""

    def __init__(
        self,
        config: TutoringEnvConfig,
        tracker_factory: Optional[Callable[[], OnlineTracker]] = None,
        seed: int = 42,
    ):
        if config.observation_mode == "estimated" and tracker_factory is None:
            raise ValueError("estimated observation mode requires tracker_factory")
        self.config = config
        self.tracker_factory = tracker_factory
        self._master_rng = np.random.RandomState(seed)
        self._episode_rng: Optional[np.random.RandomState] = None
        self._student: Optional[SimulatedStudent] = None
        self._tracker: Optional[OnlineTracker] = None
        self._base_difficulty: float = 0.5
        self._history: List[Tuple[str, int]] = []
        self._step_count: int = 0
        self._terminated: bool = False
        self._initial_mastery: float = 0.0

    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    @property
    def observation_names(self) -> Tuple[str, ...]:
        return BASE_OBSERVATION_NAMES + tuple(
            f"last_action_{action}" for action in self.config.actions
        )

    @property
    def n_actions(self) -> int:
        return len(self.config.actions)

    @property
    def action_names(self) -> Tuple[str, ...]:
        return self.config.actions

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Start a fresh student episode and return `(observation, info)`."""
        if seed is None:
            episode_seed = int(self._master_rng.randint(0, 2**31 - 1))
        else:
            episode_seed = int(seed)
            self._master_rng = np.random.RandomState(episode_seed)

        self._student = make_population_with_skills(
            n_students=1,
            num_skills=self.config.num_skills,
            seed=episode_seed,
            heterogeneous=self.config.heterogeneous,
            delayed_effects=self.config.delayed_effects,
        )[0]
        response_seed = (episode_seed + 1_000_003) % (2**31 - 1)
        self._episode_rng = np.random.RandomState(response_seed)
        self._tracker = self.tracker_factory() if self.tracker_factory else None
        self._history = []
        self._step_count = 0
        self._terminated = False
        self._initial_mastery = float(self._student.mastery)
        self._base_difficulty = self._sample_base_difficulty()

        observation = self._observation()
        info = self._info()
        info["episode_seed"] = episode_seed
        return observation, info

    def step(
        self, action: Union[int, np.integer, str]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one intervention and return the Gymnasium-style step tuple."""
        student = self._require_student()
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before step()")
        if self._episode_rng is None:
            raise RuntimeError("environment RNG is unavailable; call reset()")

        action_id, action_name = self._decode_action(action)
        base_difficulty = self._base_difficulty
        effective_difficulty = difficulty_for_action(action_name, base_difficulty)
        mastery_before = float(student.mastery)
        correct = student.respond(action_name, effective_difficulty, self._episode_rng)
        mastery_after = float(student.mastery)
        reward = mastery_after - mastery_before

        self._history.append((action_name, int(correct)))
        if self._tracker is not None:
            self._tracker.update(int(student.focus_skill), int(correct))

        self._step_count += 1
        self._terminated = self._step_count >= self.config.horizon
        if not self._terminated:
            self._base_difficulty = self._sample_base_difficulty()

        observation = self._observation()
        info = self._info()
        info.update(
            {
                "action_id": action_id,
                "action_name": action_name,
                "correct": int(correct),
                "base_difficulty": float(base_difficulty),
                "effective_difficulty": float(effective_difficulty),
            }
        )
        return observation, float(reward), self._terminated, False, info

    def _decode_action(self, action: Union[int, np.integer, str]) -> Tuple[int, str]:
        if isinstance(action, str):
            if action not in self.config.actions:
                raise ValueError(f"unknown action {action!r}")
            return self.config.actions.index(action), action
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer id or configured action name")
        action_id = int(action)
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError(f"action id must be in [0, {self.n_actions - 1}]")
        return action_id, self.config.actions[action_id]

    def _sample_base_difficulty(self) -> float:
        student = self._require_student()
        if not self.config.vary_difficulty:
            return 0.5
        if self._episode_rng is None:
            raise RuntimeError("environment RNG is unavailable")
        return float(
            np.clip(
                student.difficulty_pref + self._episode_rng.normal(0.0, 0.05),
                0.15,
                0.85,
            )
        )

    def _observation(self) -> np.ndarray:
        student = self._require_student()
        correct_history = [correct for _, correct in self._history]
        recent = correct_history[-5:]
        recent_accuracy = float(np.mean(recent)) if recent else 0.0
        improvement = 0.0
        if len(correct_history) >= 10:
            improvement = recent_accuracy - float(np.mean(correct_history[-10:-5]))

        consecutive_failures = 0
        for correct in reversed(correct_history):
            if correct:
                break
            consecutive_failures += 1

        worked_rate = self._action_success_rate("worked_example")
        socratic_rate = self._action_success_rate("socratic_hint")

        if self.config.observation_mode == "oracle":
            mastery_signal = float(student.mastery)
        elif self.config.observation_mode == "estimated":
            if self._tracker is None:
                raise RuntimeError("estimated mode tracker is unavailable")
            mastery_signal = float(
                self._tracker.estimate_mastery(int(student.focus_skill))
            )
        else:
            mastery_signal = 0.0

        if self.config.observation_mode == "no_state":
            recent_accuracy = 0.0
            improvement = 0.0
            worked_rate = 0.0
            socratic_rate = 0.0
            failure_feature = 0.0
        else:
            failure_feature = float(min(consecutive_failures, 5)) / 5.0

        last_action = np.zeros(len(self.config.actions), dtype=np.float32)
        if self._history:
            previous_action = self._history[-1][0]
            last_action[self.config.actions.index(previous_action)] = 1.0

        base_observation = np.asarray(
            [
                mastery_signal,
                recent_accuracy,
                self._base_difficulty,
                float(self._step_count) / float(self.config.horizon),
                improvement,
                worked_rate,
                failure_feature,
                socratic_rate,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base_observation, last_action])

    def _action_success_rate(self, action: str, window: int = 10) -> float:
        matched = [correct for name, correct in self._history if name == action][-window:]
        return float(np.mean(matched)) if matched else 0.5

    def _info(self) -> dict:
        student = self._require_student()
        info = {
            "step": self._step_count,
            "focus_skill": int(student.focus_skill),
            "observation_mode": self.config.observation_mode,
        }
        if self.config.include_diagnostics:
            info["diagnostics"] = {
                "initial_mastery": self._initial_mastery,
                "true_mastery": float(student.mastery),
                "independence": float(student.independence),
                "fatigue": float(student.fatigue),
                "learner_type": student.prefs.type_name,
            }
        return info

    def _require_student(self) -> SimulatedStudent:
        if self._student is None:
            raise RuntimeError("environment is not initialized; call reset()")
        return self._student
