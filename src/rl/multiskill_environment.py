"""Multi-skill tutoring POMDP with prerequisite-aware learning dynamics."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from src.policy.simulator import (
    LearnerPrefs,
    difficulty_for_action,
    homogeneous_prefs,
    intervention_effect,
    learning_scale,
    sample_learner_prefs,
    update_independence_fatigue,
)
from src.rl.environment import DEFAULT_ACTIONS, OnlineTracker


@dataclass(frozen=True)
class MultiSkillTutoringEnvConfig:
    actions: Tuple[str, ...] = DEFAULT_ACTIONS
    horizon: int = 60
    num_skills: int = 4
    prerequisites: Optional[Tuple[Tuple[int, ...], ...]] = None
    focus_skill_sequence: Optional[Tuple[int, ...]] = None
    initial_mastery: Optional[Tuple[float, ...]] = None
    observation_mode: str = "oracle"  # oracle | estimated | no_state
    delayed_effects: bool = True
    heterogeneous: bool = False
    vary_difficulty: bool = True
    include_diagnostics: bool = False
    prerequisite_gate_floor: float = 0.35
    prerequisite_review_transfer: float = 0.25
    prerequisite_spillover: float = 0.12

    def __post_init__(self) -> None:
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")
        unknown = set(self.actions) - set(DEFAULT_ACTIONS)
        if unknown:
            raise ValueError(f"actions are not implemented: {unknown}")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.num_skills < 2:
            raise ValueError("multi-skill environment requires at least two skills")
        if self.observation_mode not in {"oracle", "estimated", "no_state"}:
            raise ValueError(
                "observation_mode must be 'oracle', 'estimated', or 'no_state'"
            )
        for name, value in (
            ("prerequisite_gate_floor", self.prerequisite_gate_floor),
            ("prerequisite_review_transfer", self.prerequisite_review_transfer),
            ("prerequisite_spillover", self.prerequisite_spillover),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        graph = self.resolved_prerequisites
        self._validate_graph(graph)
        sequence = self.resolved_focus_skill_sequence
        if not sequence:
            raise ValueError("focus_skill_sequence must not be empty")
        if any(skill < 0 or skill >= self.num_skills for skill in sequence):
            raise ValueError("focus_skill_sequence contains an invalid skill")
        if self.initial_mastery is not None:
            if len(self.initial_mastery) != self.num_skills:
                raise ValueError("initial_mastery must contain one value per skill")
            if any(value < 0.0 or value > 1.0 for value in self.initial_mastery):
                raise ValueError("initial_mastery values must be in [0, 1]")

    @property
    def resolved_prerequisites(self) -> Tuple[Tuple[int, ...], ...]:
        if self.prerequisites is not None:
            return tuple(tuple(int(parent) for parent in row) for row in self.prerequisites)
        return ((),) + tuple((skill - 1,) for skill in range(1, self.num_skills))

    @property
    def resolved_focus_skill_sequence(self) -> Tuple[int, ...]:
        if self.focus_skill_sequence is not None:
            return tuple(int(skill) for skill in self.focus_skill_sequence)
        return tuple(range(self.num_skills))

    def _validate_graph(self, graph: Tuple[Tuple[int, ...], ...]) -> None:
        if len(graph) != self.num_skills:
            raise ValueError("prerequisites must contain one entry per skill")
        for skill, parents in enumerate(graph):
            if len(set(parents)) != len(parents):
                raise ValueError(f"skill {skill} contains duplicate prerequisites")
            if any(parent < 0 or parent >= self.num_skills for parent in parents):
                raise ValueError(f"skill {skill} contains an invalid prerequisite")
            if skill in parents:
                raise ValueError("a skill cannot be its own prerequisite")

        visiting = set()
        visited = set()

        def visit(skill: int) -> None:
            if skill in visiting:
                raise ValueError("prerequisite graph must be acyclic")
            if skill in visited:
                return
            visiting.add(skill)
            for parent in graph[skill]:
                visit(parent)
            visiting.remove(skill)
            visited.add(skill)

        for skill in range(self.num_skills):
            visit(skill)


@dataclass
class MultiSkillStudent:
    mastery: np.ndarray
    difficulty_preferences: np.ndarray
    prefs: LearnerPrefs = field(default_factory=homogeneous_prefs)
    independence: float = 0.70
    fatigue: float = 0.15


class MultiSkillTutoringEnv:
    """Fixed-horizon simulator with separate skill states and a known DAG."""

    def __init__(
        self,
        config: MultiSkillTutoringEnvConfig,
        tracker_factory: Optional[Callable[[], OnlineTracker]] = None,
        seed: int = 42,
    ):
        if config.observation_mode == "estimated" and tracker_factory is None:
            raise ValueError("estimated observation mode requires tracker_factory")
        self.config = config
        self.tracker_factory = tracker_factory
        self.prerequisites = config.resolved_prerequisites
        self.focus_skill_sequence = config.resolved_focus_skill_sequence
        self._master_rng = np.random.RandomState(seed)
        self._episode_rng: Optional[np.random.RandomState] = None
        self._student: Optional[MultiSkillStudent] = None
        self._tracker: Optional[OnlineTracker] = None
        self._history: List[dict] = []
        self._step_count = 0
        self._terminated = False
        self._current_skill = self.focus_skill_sequence[0]
        self._base_difficulty = 0.5
        self._initial_mastery = np.zeros(config.num_skills, dtype=np.float64)

    @property
    def observation_names(self) -> Tuple[str, ...]:
        mastery = tuple(f"mastery_signal_skill_{skill}" for skill in range(self.config.num_skills))
        history = (
            "recent_accuracy",
            "base_difficulty",
            "progress",
            "recent_improvement",
            "worked_example_success",
            "consecutive_failures",
            "socratic_hint_success",
            "prerequisite_readiness",
        )
        current = tuple(f"current_skill_{skill}" for skill in range(self.config.num_skills))
        actions = tuple(f"last_action_{action}" for action in self.config.actions)
        return mastery + history + current + actions

    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    @property
    def n_actions(self) -> int:
        return len(self.config.actions)

    @property
    def action_names(self) -> Tuple[str, ...]:
        return self.config.actions

    @property
    def current_skill(self) -> int:
        self._require_student()
        return self._current_skill

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is None:
            episode_seed = int(self._master_rng.randint(0, 2**31 - 1))
        else:
            episode_seed = int(seed)
            self._master_rng = np.random.RandomState(episode_seed)
        population_rng = np.random.RandomState(episode_seed)
        if self.config.initial_mastery is None:
            depths = self._skill_depths()
            mastery = np.asarray(
                [
                    np.clip(
                        population_rng.uniform(0.20, 0.50) - 0.06 * depths[skill],
                        0.05,
                        0.65,
                    )
                    for skill in range(self.config.num_skills)
                ],
                dtype=np.float64,
            )
        else:
            mastery = np.asarray(self.config.initial_mastery, dtype=np.float64).copy()
        difficulty_preferences = population_rng.uniform(
            0.35, 0.65, size=self.config.num_skills
        )
        prefs = (
            sample_learner_prefs(population_rng)
            if self.config.heterogeneous
            else homogeneous_prefs()
        )
        self._student = MultiSkillStudent(
            mastery=mastery,
            difficulty_preferences=difficulty_preferences,
            prefs=prefs,
            independence=float(population_rng.uniform(0.60, 0.80)),
            fatigue=float(population_rng.uniform(0.08, 0.22)),
        )
        response_seed = (episode_seed + 1_000_003) % (2**31 - 1)
        self._episode_rng = np.random.RandomState(response_seed)
        self._tracker = self.tracker_factory() if self.tracker_factory else None
        self._history = []
        self._step_count = 0
        self._terminated = False
        self._current_skill = self.focus_skill_sequence[0]
        self._initial_mastery = mastery.copy()
        self._base_difficulty = self._sample_base_difficulty()

        observation = self._observation()
        info = self._info()
        info["episode_seed"] = episode_seed
        return observation, info

    def step(
        self, action: Union[int, np.integer, str]
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        student = self._require_student()
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before step()")
        if self._episode_rng is None:
            raise RuntimeError("environment RNG is unavailable; call reset()")

        action_id, action_name = self._decode_action(action)
        acted_skill = self._current_skill
        target_skill = self._target_skill(action_name, acted_skill)
        base_difficulty = self._base_difficulty
        effective_difficulty = difficulty_for_action(action_name, base_difficulty)
        mastery_before = student.mastery.copy()
        independence_before = student.independence
        fatigue_before = student.fatigue

        base_effect = intervention_effect(
            action_name,
            float(student.mastery[target_skill]),
            student.prefs,
        )
        delayed_scale = (
            learning_scale(student.independence, student.fatigue)
            if self.config.delayed_effects
            else 1.0
        )
        prerequisite_readiness = self._prerequisite_readiness(
            acted_skill, student.mastery
        )
        target_prerequisite_readiness = self._prerequisite_readiness(
            target_skill, student.mastery
        )
        gate = 1.0
        if base_effect > 0.0 and self.prerequisites[target_skill]:
            gate = self.config.prerequisite_gate_floor + (
                1.0 - self.config.prerequisite_gate_floor
            ) * target_prerequisite_readiness
        primary_effect = float(base_effect * delayed_scale * gate)
        primary_gain = 0.35 * primary_effect
        student.mastery[target_skill] = np.clip(
            student.mastery[target_skill] + primary_gain, 0.0, 1.0
        )

        transfers = {}
        if target_skill != acted_skill and primary_gain > 0.0:
            transfer = (
                self.config.prerequisite_review_transfer
                * primary_gain
                * (0.5 + 0.5 * prerequisite_readiness)
            )
            before = float(student.mastery[acted_skill])
            student.mastery[acted_skill] = np.clip(before + transfer, 0.0, 1.0)
            transfers[acted_skill] = float(student.mastery[acted_skill] - before)
        elif target_skill == acted_skill and primary_gain > 0.0:
            parents = self.prerequisites[acted_skill]
            if parents:
                weakest = min(parents, key=lambda skill: student.mastery[skill])
                transfer = self.config.prerequisite_spillover * primary_gain
                before = float(student.mastery[weakest])
                student.mastery[weakest] = np.clip(before + transfer, 0.0, 1.0)
                transfers[weakest] = float(student.mastery[weakest] - before)

        effective_target_mastery = float(
            np.clip(mastery_before[target_skill] + primary_effect, 0.0, 1.0)
        )
        logit = 4.0 * (effective_target_mastery - effective_difficulty)
        probability_correct = 1.0 / (1.0 + np.exp(-logit))
        correct = int(self._episode_rng.rand() < probability_correct)
        if self.config.delayed_effects:
            student.independence, student.fatigue = update_independence_fatigue(
                action_name,
                correct,
                student.independence,
                student.fatigue,
            )

        mastery_after = student.mastery.copy()
        reward = float(mastery_after.mean() - mastery_before.mean())
        self._history.append(
            {
                "action": action_name,
                "acted_skill": acted_skill,
                "target_skill": target_skill,
                "correct": correct,
                "base_difficulty": float(base_difficulty),
                "effective_difficulty": float(effective_difficulty),
                "probability_correct": float(probability_correct),
                "prerequisite_readiness": float(prerequisite_readiness),
                "target_prerequisite_readiness": float(
                    target_prerequisite_readiness
                ),
                "gate": float(gate),
                "base_effect": float(base_effect),
                "primary_effect": primary_effect,
                "mastery_before": mastery_before.tolist(),
                "mastery_after": mastery_after.tolist(),
                "mastery_delta": (mastery_after - mastery_before).tolist(),
                "transfers": transfers,
                "independence_before": float(independence_before),
                "independence_after": float(student.independence),
                "fatigue_before": float(fatigue_before),
                "fatigue_after": float(student.fatigue),
            }
        )
        if self._tracker is not None:
            self._tracker.update(target_skill, correct)

        self._step_count += 1
        self._terminated = self._step_count >= self.config.horizon
        if not self._terminated:
            self._current_skill = self.focus_skill_sequence[
                self._step_count % len(self.focus_skill_sequence)
            ]
            self._base_difficulty = self._sample_base_difficulty()

        observation = self._observation()
        info = self._info()
        info.update(
            {
                "action_id": action_id,
                "action_name": action_name,
                "acted_skill": acted_skill,
                "target_skill": target_skill,
                "correct": correct,
                "base_difficulty": float(base_difficulty),
                "effective_difficulty": float(effective_difficulty),
                "mastery_delta_by_skill": (mastery_after - mastery_before).tolist(),
            }
        )
        return observation, reward, self._terminated, False, info

    def fork(self, response_seed: Optional[int] = None) -> "MultiSkillTutoringEnv":
        self._require_student()
        clone = copy.deepcopy(self)
        if response_seed is not None:
            clone._episode_rng = np.random.RandomState(int(response_seed))
        return clone

    def prerequisite_closure(self, skill: int) -> Tuple[int, ...]:
        if skill < 0 or skill >= self.config.num_skills:
            raise ValueError("skill index is out of range")
        found = set()

        def collect(node: int) -> None:
            for parent in self.prerequisites[node]:
                if parent not in found:
                    found.add(parent)
                    collect(parent)

        collect(skill)
        return tuple(sorted(found))

    def prerequisite_readiness(self, skill: int) -> float:
        """Return mean true mastery over all transitive prerequisites."""
        student = self._require_student()
        return self._prerequisite_readiness(skill, student.mastery)

    def _target_skill(self, action: str, acted_skill: int) -> int:
        parents = self.prerequisites[acted_skill]
        if action == "prerequisite_review" and parents:
            student = self._require_student()
            return int(min(parents, key=lambda skill: student.mastery[skill]))
        return acted_skill

    def _prerequisite_readiness(self, skill: int, mastery: np.ndarray) -> float:
        closure = self.prerequisite_closure(skill)
        if not closure:
            return 1.0
        return float(np.mean(mastery[list(closure)]))

    def _skill_depths(self) -> Tuple[int, ...]:
        memo = {}

        def depth(skill: int) -> int:
            if skill not in memo:
                parents = self.prerequisites[skill]
                memo[skill] = 0 if not parents else 1 + max(depth(p) for p in parents)
            return memo[skill]

        return tuple(depth(skill) for skill in range(self.config.num_skills))

    def _mastery_signals(self) -> np.ndarray:
        student = self._require_student()
        if self.config.observation_mode == "oracle":
            return student.mastery.astype(np.float32, copy=True)
        if self.config.observation_mode == "estimated":
            if self._tracker is None:
                raise RuntimeError("estimated mode tracker is unavailable")
            return np.asarray(
                [
                    self._tracker.estimate_mastery(skill)
                    for skill in range(self.config.num_skills)
                ],
                dtype=np.float32,
            )
        return np.zeros(self.config.num_skills, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        mastery_signals = self._mastery_signals()
        correct_history = [int(row["correct"]) for row in self._history]
        recent = correct_history[-5:]
        recent_accuracy = float(np.mean(recent)) if recent else 0.0
        improvement = 0.0
        if len(correct_history) >= 10:
            improvement = recent_accuracy - float(np.mean(correct_history[-10:-5]))
        failures = 0
        for correct in reversed(correct_history):
            if correct:
                break
            failures += 1
        worked_rate = self._action_success_rate("worked_example")
        socratic_rate = self._action_success_rate("socratic_hint")
        readiness = self._prerequisite_readiness(
            self._current_skill, mastery_signals
        )
        if self.config.observation_mode == "no_state":
            recent_accuracy = 0.0
            improvement = 0.0
            worked_rate = 0.0
            socratic_rate = 0.0
            failure_feature = 0.0
            readiness = 0.0
        else:
            failure_feature = float(min(failures, 5)) / 5.0

        history_features = np.asarray(
            [
                recent_accuracy,
                self._base_difficulty,
                float(self._step_count) / float(self.config.horizon),
                improvement,
                worked_rate,
                failure_feature,
                socratic_rate,
                readiness,
            ],
            dtype=np.float32,
        )
        current_skill = np.zeros(self.config.num_skills, dtype=np.float32)
        current_skill[self._current_skill] = 1.0
        last_action = np.zeros(len(self.config.actions), dtype=np.float32)
        if self._history:
            last_action[
                self.config.actions.index(str(self._history[-1]["action"]))
            ] = 1.0
        return np.concatenate(
            [mastery_signals, history_features, current_skill, last_action]
        )

    def _action_success_rate(self, action: str, window: int = 10) -> float:
        matched = [
            int(row["correct"])
            for row in self._history
            if row["action"] == action
        ][-window:]
        return float(np.mean(matched)) if matched else 0.5

    def _sample_base_difficulty(self) -> float:
        student = self._require_student()
        if not self.config.vary_difficulty:
            return 0.5
        if self._episode_rng is None:
            raise RuntimeError("environment RNG is unavailable")
        return float(
            np.clip(
                student.difficulty_preferences[self._current_skill]
                + self._episode_rng.normal(0.0, 0.05),
                0.15,
                0.85,
            )
        )

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

    def _info(self) -> dict:
        student = self._require_student()
        info = {
            "step": self._step_count,
            "current_skill": int(self._current_skill),
            "observation_mode": self.config.observation_mode,
        }
        if self.config.include_diagnostics:
            info["diagnostics"] = {
                "initial_mastery_by_skill": self._initial_mastery.tolist(),
                "true_mastery_by_skill": student.mastery.tolist(),
                "initial_mean_mastery": float(self._initial_mastery.mean()),
                "mean_true_mastery": float(student.mastery.mean()),
                "independence": float(student.independence),
                "fatigue": float(student.fatigue),
                "learner_type": student.prefs.type_name,
                "prerequisites": [list(row) for row in self.prerequisites],
            }
        return info

    def _require_student(self) -> MultiSkillStudent:
        if self._student is None:
            raise RuntimeError("environment is not initialized; call reset()")
        return self._student
