"""Student simulator for OFFLINE policy comparison only.


Response model
--------------
P(correct | mastery, difficulty, intervention) is logistic in
(effective_mastery - difficulty). Intervention effects are
*state-dependent* by design: the best action depends on current mastery.
That makes this a real sequential decision problem for a contextual
bandit, rather than a trivial "always pick the action with the largest
fixed bump."

Heterogeneous learners
---------------------------------
When enabled, each student draws a latent preference vector z_i that
multiplies base Effect(a, m). The simulator knows z_i; the policy must
never observe it — only mastery estimates and observed response history.

Delayed effects (M6, optional)
------------------------------
When delayed_effects=True, each student also has latent Independence I_t
and Fatigue F_t. Actions update (m, I, F); learning rate depends on I and F.
I and F are NEVER exposed to the policy — only used inside the generative
process so myopic Δm ≠ long-horizon optimal.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.policy.rule_based import StudentState

# Named latent types. Multipliers scale the base Effect(a, m).
# Challenge tolerance mainly scales easier/harder problem effects.
LEARNER_TYPES: Dict[str, Dict[str, float]] = {
    "example_preferring": {
        "explain": 1.10,
        "worked_example": 1.40,
        "socratic_hint": 0.70,
        "easier_problem": 1.15,
        "harder_problem": 0.55,
        "prerequisite_review": 1.05,
    },
    "socratic_preferring": {
        "explain": 0.85,
        "worked_example": 0.70,
        "socratic_hint": 1.45,
        "easier_problem": 1.05,
        "harder_problem": 0.85,
        "prerequisite_review": 0.95,
    },
    "challenge_seeking": {
        "explain": 0.90,
        "worked_example": 0.85,
        "socratic_hint": 1.05,
        "easier_problem": 0.45,
        "harder_problem": 1.50,
        "prerequisite_review": 0.80,
    },
}

# Actions that tend to create dependence (lower independence).
_SUPPORT_ACTIONS = frozenset(
    {
        "explain",
        "worked_example",
        "socratic_hint",
        "prerequisite_review",
        "easier_problem",
    }
)


@dataclass
class LearnerPrefs:
    """Latent student preferences z_i. Never passed to the policy."""

    type_name: str
    multipliers: Dict[str, float]

    def scale(self, action: str) -> float:
        return float(self.multipliers.get(action, 1.0))


def sample_learner_prefs(rng: np.random.RandomState) -> LearnerPrefs:
    names = list(LEARNER_TYPES.keys())
    name = names[int(rng.randint(0, len(names)))]
    return LearnerPrefs(type_name=name, multipliers=dict(LEARNER_TYPES[name]))


def homogeneous_prefs() -> LearnerPrefs:
    return LearnerPrefs(
        type_name="homogeneous",
        multipliers={a: 1.0 for a in LEARNER_TYPES["example_preferring"]},
    )


def intervention_effect(
    action: str,
    mastery: float,
    prefs: Optional[LearnerPrefs] = None,
) -> float:
    """State- and (optionally) type-dependent mastery nudge.

    Base pedagogy (stated assumption):
      - low mastery  → prerequisite / explain help most
      - mid mastery  → worked example / Socratic hint help most
      - high mastery → harder problem helps; easier problem hurts

    If prefs is set: Effect(a, m, z_i) = z_i[a] * Effect(a, m).
    """
    m = float(np.clip(mastery, 0.0, 1.0))
    if action == "prerequisite_review":
        base = 0.14 * (1.0 - m)
    elif action == "explain":
        base = 0.10 * (1.0 - m)
    elif action == "worked_example":
        base = 0.10 * max(0.0, 1.0 - abs(m - 0.45) * 2.2)
    elif action == "socratic_hint":
        base = 0.08 * max(0.0, 1.0 - abs(m - 0.55) * 2.2)
    elif action == "easier_problem":
        base = 0.05 if m < 0.4 else -0.03
    elif action == "harder_problem":
        base = 0.10 * m - 0.04  # negative when low, positive when high
    else:
        base = 0.0

    if prefs is None:
        return float(base)
    return float(base * prefs.scale(action))


def learning_scale(independence: float, fatigue: float) -> float:
    """How much of the base Effect(a,m) actually applies given I and F."""
    i = float(np.clip(independence, 0.0, 1.0))
    f = float(np.clip(fatigue, 0.0, 1.0))
    # Collapsing independence nearly freezes further mastery gains — the
    # long-horizon cost of myopic scaffolding.
    return float(np.clip((0.08 + 0.92 * i) * (1.0 - 0.35 * f), 0.05, 1.20))


def update_independence_fatigue(
    action: str,
    correct: int,
    independence: float,
    fatigue: float,
) -> Tuple[float, float]:
    """Latent (I, F) transition. Never shown to the policy."""
    i = float(independence)
    f = float(fatigue)

    # mild natural recovery from fatigue each step
    f = max(0.0, f - 0.030)

    if action in _SUPPORT_ACTIONS:
        # repeated scaffolding creates dependence quickly
        i -= 0.085 if action != "easier_problem" else 0.035
        f -= 0.045 if action != "easier_problem" else 0.080
    elif action == "harder_problem":
        f += 0.040
        if correct:
            i += 0.125
        else:
            i += 0.040
            f += 0.020

    return float(np.clip(i, 0.0, 1.0)), float(np.clip(f, 0.0, 1.0))


def difficulty_for_action(action: str, base_difficulty: float) -> float:
    """Easier/harder interventions also shift the problem difficulty."""
    if action == "easier_problem":
        return float(np.clip(base_difficulty - 0.15, 0.05, 0.95))
    if action == "harder_problem":
        return float(np.clip(base_difficulty + 0.15, 0.05, 0.95))
    return float(base_difficulty)


@dataclass
class SimulatedStudent:
    student_id: int
    mastery: float = 0.3
    difficulty_pref: float = 0.5
    focus_skill: int = 0
    prefs: LearnerPrefs = field(default_factory=homogeneous_prefs)
    # Latent long-horizon variables (M6). Hidden from every policy.
    independence: float = 0.70
    fatigue: float = 0.15
    delayed_effects: bool = False
    history: List[dict] = field(default_factory=list)

    def respond(
        self,
        intervention: str,
        difficulty: float,
        rng: np.random.RandomState,
    ) -> int:
        base_effect = intervention_effect(intervention, self.mastery, self.prefs)
        if self.delayed_effects:
            scale = learning_scale(self.independence, self.fatigue)
            effect = float(base_effect * scale)
        else:
            scale = 1.0
            effect = float(base_effect)

        effective_mastery = float(np.clip(self.mastery + effect, 0.0, 1.0))
        logit = 4.0 * (effective_mastery - difficulty)
        p_correct = 1.0 / (1.0 + np.exp(-logit))
        correct = int(rng.rand() < p_correct)

        # mastery drifts toward the post-intervention effective mastery
        self.mastery = float(
            np.clip(self.mastery + 0.35 * (effective_mastery - self.mastery), 0.0, 1.0)
        )

        if self.delayed_effects:
            self.independence, self.fatigue = update_independence_fatigue(
                intervention, correct, self.independence, self.fatigue
            )

        self.history.append(
            {
                "intervention": intervention,
                "difficulty": difficulty,
                "correct": correct,
                "effect": effect,
                "base_effect": base_effect,
                "learning_scale": scale,
                "mastery_after": self.mastery,
                "independence_after": self.independence,
                "fatigue_after": self.fatigue,
            }
        )
        return correct


def state_from_simulated(student: SimulatedStudent) -> StudentState:
    """Build the rule-based StudentState from a simulated student's history."""
    recent = student.history[-5:]
    recent_acc = float(np.mean([h["correct"] for h in recent])) if recent else 0.0
    consecutive_failures = 0
    for h in reversed(student.history):
        if h["correct"] == 0:
            consecutive_failures += 1
        else:
            break
    return StudentState(
        mastery=student.mastery,
        recent_accuracy=recent_acc,
        attempts=len(student.history),
        consecutive_failures=consecutive_failures,
    )


def _action_success_rate(
    history: List[Tuple[str, int]],
    action: str,
    window: int = 10,
) -> float:
    """Mean correctness for a given action over recent matching steps (0.5 if none)."""
    matched = [c for a, c in history if a == action][-window:]
    if not matched:
        return 0.5
    return float(np.mean(matched))


def bandit_context(
    student: SimulatedStudent,
    difficulty: float,
    include_mastery: bool = True,
) -> np.ndarray:
    """8-dim context matching configs/config.yaml -> policy.bandit.context_dim.

    Dims 5 and 7 carry recent worked_example / socratic_hint success rates so
    the bandit can infer latent type from experience — never from prefs.

    Set include_mastery=False for the ablation: policy without student state
    (zeros out mastery / recent accuracy / improvement / failures / type cues).
    """
    state = state_from_simulated(student)
    improvement = 0.0
    if len(student.history) >= 10:
        prev = float(np.mean([h["correct"] for h in student.history[-10:-5]]))
        improvement = state.recent_accuracy - prev

    hist = [(h["intervention"], int(h["correct"])) for h in student.history]
    we_rate = _action_success_rate(hist, "worked_example")
    sh_rate = _action_success_rate(hist, "socratic_hint")

    if include_mastery:
        return np.array(
            [
                state.mastery,
                state.recent_accuracy,
                difficulty,
                float(state.attempts) / 50.0,
                improvement,
                we_rate,
                float(state.consecutive_failures) / 5.0,
                sh_rate,
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            0.0,
            0.0,
            difficulty,
            float(state.attempts) / 50.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def make_population(
    n_students: int,
    seed: int = 42,
    heterogeneous: bool = False,
    delayed_effects: bool = False,
) -> List[SimulatedStudent]:
    rng = np.random.RandomState(seed)
    students = []
    for i in range(n_students):
        prefs = sample_learner_prefs(rng) if heterogeneous else homogeneous_prefs()
        students.append(
            SimulatedStudent(
                student_id=i,
                mastery=float(rng.uniform(0.1, 0.5)),
                difficulty_pref=float(rng.uniform(0.35, 0.65)),
                prefs=prefs,
                independence=float(rng.uniform(0.55, 0.85)),
                fatigue=float(rng.uniform(0.05, 0.25)),
                delayed_effects=delayed_effects,
            )
        )
    return students


def make_population_with_skills(
    n_students: int,
    num_skills: int,
    seed: int = 42,
    heterogeneous: bool = False,
    delayed_effects: bool = False,
) -> List[SimulatedStudent]:
    """Population where each student is assigned one ASSISTments skill id."""
    rng = np.random.RandomState(seed)
    students = []
    for i in range(n_students):
        prefs = sample_learner_prefs(rng) if heterogeneous else homogeneous_prefs()
        students.append(
            SimulatedStudent(
                student_id=i,
                mastery=float(rng.uniform(0.1, 0.5)),
                difficulty_pref=float(rng.uniform(0.35, 0.65)),
                focus_skill=int(rng.randint(0, max(num_skills, 1))),
                prefs=prefs,
                independence=float(rng.uniform(0.55, 0.85)),
                fatigue=float(rng.uniform(0.05, 0.25)),
                delayed_effects=delayed_effects,
            )
        )
    return students


def run_policy_simulation(
    policy,
    n_students: int,
    n_interactions: int,
    context_fn: Callable,
    seed: int = 42,
    vary_difficulty: bool = True,
    heterogeneous: bool = False,
    delayed_effects: bool = False,
) -> Dict[str, float]:
    """Run one policy over a fresh simulated population.

    `context_fn(student, difficulty)` returns whatever `policy.select_action`
    expects: StudentState for the rule-based policy, a vector for bandits,
    or unused state for the random policy.

    Returns aggregate metrics plus action counts and a learning curve.
    """
    rng = np.random.RandomState(seed)
    students = make_population(
        n_students, seed, heterogeneous=heterogeneous, delayed_effects=delayed_effects
    )
    total_reward = 0.0
    mastery_final: List[float] = []
    action_counts: Dict[str, int] = {}
    # accuracy at each timestep, averaged later over students
    correct_by_t = np.zeros(n_interactions, dtype=np.float64)
    reward_by_t = np.zeros(n_interactions, dtype=np.float64)
    independence_final: List[float] = []
    fatigue_final: List[float] = []

    for student in students:
        for t in range(n_interactions):
            if vary_difficulty:
                base_diff = float(
                    np.clip(student.difficulty_pref + rng.normal(0, 0.05), 0.15, 0.85)
                )
            else:
                base_diff = 0.5

            context = context_fn(student, base_diff)
            action = policy.select_action(context)
            difficulty = difficulty_for_action(action, base_diff)

            pre_mastery = student.mastery
            correct = student.respond(action, difficulty, rng)
            reward = student.mastery - pre_mastery

            if hasattr(policy, "update"):
                # bandits see the pre-action context; reward is mastery gain
                policy.update(action, context, reward)

            total_reward += reward
            correct_by_t[t] += correct
            reward_by_t[t] += reward
            action_counts[action] = action_counts.get(action, 0) + 1

        mastery_final.append(student.mastery)
        independence_final.append(student.independence)
        fatigue_final.append(student.fatigue)

    n_steps = n_students * n_interactions
    correct_by_t /= max(n_students, 1)
    reward_by_t /= max(n_students, 1)
    # Final-window accuracy should mean the last 5 timesteps (e.g., t=46..50)
    # averaged across students, not the tail of student-major ordering.
    tail = min(5, n_interactions)
    early = min(10, n_interactions)
    final_window_accuracy = float(np.mean(correct_by_t[-tail:]))
    overall_accuracy = float(np.mean(correct_by_t))

    return {
        "mean_reward": total_reward / max(n_steps, 1),
        "mean_final_mastery": float(np.mean(mastery_final)),
        "final_accuracy": final_window_accuracy,
        "overall_accuracy": overall_accuracy,
        "early_mean_reward": float(np.mean(reward_by_t[:early])),
        "late_mean_reward": float(np.mean(reward_by_t[-tail:])),
        "mean_final_independence": float(np.mean(independence_final)),
        "mean_final_fatigue": float(np.mean(fatigue_final)),
        "action_counts": action_counts,
        "accuracy_curve": correct_by_t.tolist(),
        "reward_curve": reward_by_t.tolist(),
        "delayed_effects": delayed_effects,
    }


def run_kt_aware_simulation(
    policy,
    tracker_factory: Optional[Callable],
    n_students: int,
    n_interactions: int,
    num_skills: int,
    seed: int = 42,
    state_mode: str = "estimated",
    vary_difficulty: bool = True,
    for_rule: bool = False,
    heterogeneous: bool = False,
    feature_mode: str = "linear",
) -> Dict[str, float]:
    """Policy loop where the bandit/rule sees KT-estimated state, not true mastery.

    Parameters
    ----------
    tracker_factory :
        zero-arg callable returning a fresh online tracker per student, or None.
    state_mode :
        "oracle"     — context uses true simulator mastery (upper bound)
        "estimated"  — context uses tracker.estimate_mastery(focus_skill)
        "no_state"   — mastery features zeroed (tracker still updated if given)
    for_rule :
        if True, pass a StudentState to policy.select_action; else a context vector.
    heterogeneous :
        if True, each student draws a latent preference type z_i. Policy never
        observes z_i.
    feature_mode :
        "linear" or "basis" — how mastery enters the bandit context (rules ignore).

    Reward always uses true mastery change. Under "estimated" / "no_state" the
    policy never observes true mastery in its context.
    """
    from src.policy.kt_state import (
        bandit_context_from_estimate,
        observed_student_state,
    )

    rng = np.random.RandomState(seed)
    students = make_population_with_skills(
        n_students,
        num_skills,
        seed,
        heterogeneous=heterogeneous,
        delayed_effects=False,
    )
    total_reward = 0.0
    mastery_final: List[float] = []
    action_counts: Dict[str, int] = {}
    correct_by_t = np.zeros(n_interactions, dtype=np.float64)
    est_abs_err: List[float] = []

    for student in students:
        tracker = tracker_factory() if tracker_factory is not None else None
        history_obs: List[Tuple[str, int]] = []
        skill = int(student.focus_skill)

        for t in range(n_interactions):
            if vary_difficulty:
                base_diff = float(
                    np.clip(student.difficulty_pref + rng.normal(0, 0.05), 0.15, 0.85)
                )
            else:
                base_diff = 0.5

            if state_mode == "oracle":
                est = float(student.mastery)
            elif tracker is None:
                est = 0.5
            else:
                est = float(tracker.estimate_mastery(skill))

            if tracker is not None and state_mode != "oracle":
                est_abs_err.append(abs(est - student.mastery))

            include = state_mode != "no_state"
            history_correct = [c for _, c in history_obs]
            if for_rule:
                context = observed_student_state(
                    est if include else 0.5, history_correct
                )
            else:
                context = bandit_context_from_estimate(
                    est,
                    history_correct,
                    base_diff,
                    include_mastery=include,
                    history_actions=history_obs if include else None,
                    feature_mode=feature_mode,
                )

            action = policy.select_action(context)
            difficulty = difficulty_for_action(action, base_diff)
            pre_mastery = student.mastery
            correct = student.respond(action, difficulty, rng)
            reward = student.mastery - pre_mastery

            history_obs.append((action, correct))
            if tracker is not None:
                tracker.update(skill, correct)

            if hasattr(policy, "update") and not for_rule:
                policy.update(action, context, reward)

            total_reward += reward
            correct_by_t[t] += correct
            action_counts[action] = action_counts.get(action, 0) + 1

        mastery_final.append(student.mastery)

    n_steps = n_students * n_interactions
    correct_by_t /= max(n_students, 1)
    tail = min(5, n_interactions)

    out = {
        "mean_reward": total_reward / max(n_steps, 1),
        "mean_final_mastery": float(np.mean(mastery_final)),
        "final_accuracy": float(np.mean(correct_by_t[-tail:])),
        "overall_accuracy": float(np.mean(correct_by_t)),
        "action_counts": action_counts,
        "accuracy_curve": correct_by_t.tolist(),
        "state_mode": state_mode,
        "heterogeneous": heterogeneous,
        "feature_mode": feature_mode,
    }
    if est_abs_err:
        out["mean_abs_estimation_error"] = float(np.mean(est_abs_err))
    return out
