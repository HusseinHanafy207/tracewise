"""Student simulator for OFFLINE policy comparison only.

IMPORTANT: this is a simulation, not real student data. Any results from
this module must be reported as "simulated learning outcomes," not as
evidence about real students. See docs/roadmap.md Day 10.

Response model: P(correct | mastery, difficulty, intervention) is a
logistic function of mastery vs difficulty, modulated by an
intervention-specific "effectiveness" bump that is loosely informed by
skill-level accuracy deltas observed in the real ASSISTments data (e.g.
students who get more attempts/hints on a skill before answering
correctly again) — but the exact bump values are a modeling assumption
you should state explicitly in the writeup, not a fitted causal effect.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.policy.rule_based import StudentState

# Rough, documented assumption: how much each intervention nudges the
# student's effective mastery for the NEXT attempt. Tune / replace with
# data-informed estimates if time permits.
INTERVENTION_EFFECT: Dict[str, float] = {
    "explain": 0.05,
    "worked_example": 0.10,
    "socratic_hint": 0.07,
    "easier_problem": 0.02,
    "harder_problem": -0.02,
    "prerequisite_review": 0.12,
}


@dataclass
class SimulatedStudent:
    student_id: int
    mastery: float = 0.3
    difficulty_pref: float = 0.5
    history: List[dict] = field(default_factory=list)

    def respond(self, intervention: str, difficulty: float, rng: np.random.RandomState) -> int:
        effect = INTERVENTION_EFFECT.get(intervention, 0.0)
        effective_mastery = np.clip(self.mastery + effect, 0.0, 1.0)
        logit = 4 * (effective_mastery - difficulty)
        p_correct = 1 / (1 + np.exp(-logit))
        correct = int(rng.rand() < p_correct)

        # mastery drifts slightly toward effective_mastery after each interaction
        self.mastery = np.clip(self.mastery + 0.3 * (effective_mastery - self.mastery), 0, 1)
        self.history.append(
            {"intervention": intervention, "difficulty": difficulty, "correct": correct}
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


def bandit_context(student: SimulatedStudent, difficulty: float) -> np.ndarray:
    """8-dim context matching configs/config.yaml -> policy.bandit.context_dim."""
    state = state_from_simulated(student)
    improvement = 0.0
    if len(student.history) >= 10:
        prev = float(np.mean([h["correct"] for h in student.history[-10:-5]]))
        improvement = state.recent_accuracy - prev
    return np.array(
        [
            state.mastery,
            state.recent_accuracy,
            difficulty,
            float(state.attempts),
            improvement,
            0.0,  # time_on_task placeholder (not in ASSISTments-derived sim)
            float(state.consecutive_failures),
            0.0,  # skill_priority placeholder
        ],
        dtype=np.float64,
    )


def make_population(n_students: int, seed: int = 42) -> List[SimulatedStudent]:
    rng = np.random.RandomState(seed)
    return [
        SimulatedStudent(student_id=i, mastery=rng.uniform(0.1, 0.5))
        for i in range(n_students)
    ]


def run_policy_simulation(
    policy,
    n_students: int,
    n_interactions: int,
    context_fn,
    seed: int = 42,
) -> Dict[str, float]:
    """Run one policy over a fresh simulated population.

    `context_fn(student, difficulty)` returns whatever `policy.select_action`
    expects: `StudentState` for the rule-based policy, a vector for bandits,
    or unused state for the random policy.
    """
    rng = np.random.RandomState(seed)
    students = make_population(n_students, seed)
    total_reward = 0.0
    correct_over_time = []

    for student in students:
        for t in range(n_interactions):
            difficulty = 0.5  # could be randomized/curriculum-based
            context = context_fn(student, difficulty)
            action = policy.select_action(context)

            pre_mastery = student.mastery
            correct = student.respond(action, difficulty, rng)
            reward = student.mastery - pre_mastery

            if hasattr(policy, "update"):
                policy.update(action, context, reward)

            total_reward += reward
            correct_over_time.append(correct)

    return {
        "mean_reward": total_reward / (n_students * n_interactions),
        "final_accuracy": float(np.mean(correct_over_time[-n_students * 5 :])),
        "overall_accuracy": float(np.mean(correct_over_time)),
    }
