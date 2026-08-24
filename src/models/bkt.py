"""Bayesian Knowledge Tracing (BKT), implemented from scratch (no pyBKT).

Standard 4-parameter HMM per skill:
    p_init  = P(known at t=0)
    p_learn = P(transition unknown -> known between attempts)
    p_guess = P(correct | unknown)
    p_slip  = P(incorrect | known)

Forward update given observation o_t (correct=1/0):

    P(known_t | o_t) = P(o_t | known_t) P(known_t) / P(o_t)

    P(o_t=1 | known_t=1) = 1 - p_slip
    P(o_t=1 | known_t=0) = p_guess

Then apply learning transition before the next observation:

    P(known_{t+1}) = P(known_t | o_t) + (1 - P(known_t | o_t)) * p_learn

Parameters are fit per-skill via a simple EM/grid-search hybrid: since
BKT has only 4 parameters per skill, coordinate-wise grid search over a
coarse grid + refinement is stable and easy to reason about (full EM is
implementable too, but the grid approach is simpler to debug and still a
legitimate baseline for a hackathon/portfolio project).
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class BKTParams:
    p_init: float = 0.4
    p_learn: float = 0.2
    p_guess: float = 0.25
    p_slip: float = 0.1


def bkt_forward(observations: List[int], params: BKTParams) -> List[float]:
    """Run the BKT forward pass for one student on one skill.

    Returns P(known) *before* each observation is seen (i.e. the
    prediction used for evaluating P(correct) at that step), length
    == len(observations).
    """
    p_known = params.p_init
    predictions = []

    for o in observations:
        predictions.append(p_known)

        p_correct_given_known = 1 - params.p_slip
        p_correct_given_unknown = params.p_guess

        if o == 1:
            numerator = p_known * p_correct_given_known
            denom = numerator + (1 - p_known) * p_correct_given_unknown
        else:
            numerator = p_known * params.p_slip
            denom = numerator + (1 - p_known) * (1 - params.p_guess)

        p_known_given_o = numerator / denom if denom > 0 else p_known
        p_known = p_known_given_o + (1 - p_known_given_o) * params.p_learn

    return predictions


def predicted_correct_prob(observations: List[int], params: BKTParams) -> List[float]:
    """P(correct_t) = P(known_t) * (1-slip) + P(unknown_t) * guess."""
    p_knowns = bkt_forward(observations, params)
    return [
        pk * (1 - params.p_slip) + (1 - pk) * params.p_guess for pk in p_knowns
    ]


def _neg_log_likelihood(observations: List[int], params: BKTParams) -> float:
    probs = predicted_correct_prob(observations, params)
    eps = 1e-6
    nll = 0.0
    for o, p in zip(observations, probs):
        p = min(max(p, eps), 1 - eps)
        nll += -(o * np.log(p) + (1 - o) * np.log(1 - p))
    return nll


def fit_skill_bkt(
    student_observations: List[List[int]],
    grid: Dict[str, np.ndarray] = None,
) -> BKTParams:
    """Fit BKT params for a single skill via coarse grid search over all
    students' observation sequences for that skill, minimizing total NLL.

    `student_observations` is a list of sequences (one per student) of
    0/1 correctness for this skill only.
    """
    if grid is None:
        grid = {
            "p_init": np.linspace(0.1, 0.9, 5),
            "p_learn": np.linspace(0.05, 0.5, 5),
            "p_guess": np.linspace(0.1, 0.4, 4),
            "p_slip": np.linspace(0.05, 0.3, 4),
        }

    best_params = BKTParams()
    best_nll = float("inf")

    for p_init in grid["p_init"]:
        for p_learn in grid["p_learn"]:
            for p_guess in grid["p_guess"]:
                for p_slip in grid["p_slip"]:
                    # Standard BKT identifiability constraint.
                    if p_guess + p_slip >= 1.0:
                        continue
                    params = BKTParams(p_init, p_learn, p_guess, p_slip)
                    total_nll = sum(
                        _neg_log_likelihood(obs, params) for obs in student_observations
                    )
                    if total_nll < best_nll:
                        best_nll = total_nll
                        best_params = params

    return best_params


class BKTModel:
    """Wraps per-skill BKTParams, trained across a set of StudentSequence."""

    def __init__(self):
        self.skill_params: Dict[int, BKTParams] = {}

    def fit(self, sequences, num_skills: int) -> None:
        by_skill: Dict[int, List[List[int]]] = {k: [] for k in range(num_skills)}
        for seq in sequences:
            per_skill_obs: Dict[int, List[int]] = {}
            for skill, correct in zip(seq.skill_ids, seq.correct):
                per_skill_obs.setdefault(skill, []).append(correct)
            for skill, obs in per_skill_obs.items():
                by_skill[skill].append(obs)

        for skill, seqs in by_skill.items():
            if len(seqs) == 0:
                self.skill_params[skill] = BKTParams()
                continue
            self.skill_params[skill] = fit_skill_bkt(seqs)

    def predict_sequence(self, skill_ids: List[int], correct: List[int]) -> List[float]:
        """Predict P(correct_t) at each step, per-skill state tracked independently."""
        per_skill_state: Dict[int, float] = {}
        preds = []
        for skill, o in zip(skill_ids, correct):
            params = self.skill_params.get(skill, BKTParams())
            p_known = per_skill_state.get(skill, params.p_init)

            p_correct = p_known * (1 - params.p_slip) + (1 - p_known) * params.p_guess
            preds.append(p_correct)

            p_correct_given_known = 1 - params.p_slip
            p_correct_given_unknown = params.p_guess
            if o == 1:
                num = p_known * p_correct_given_known
                den = num + (1 - p_known) * p_correct_given_unknown
            else:
                num = p_known * params.p_slip
                den = num + (1 - p_known) * (1 - params.p_guess)
            p_known_given_o = num / den if den > 0 else p_known
            per_skill_state[skill] = p_known_given_o + (1 - p_known_given_o) * params.p_learn

        return preds
