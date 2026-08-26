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

Parameters are fit per-skill by coarse grid search (4 parameters, so a
small grid is enough). Likelihood is evaluated with a NumPy batch over
students x grid points — same math as the per-sequence Python loop, but
tractable on ASSISTments (~100 skills).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BKTParams:
    p_init: float = 0.4
    p_learn: float = 0.2
    p_guess: float = 0.25
    p_slip: float = 0.1


DEFAULT_GRID = {
    "p_init": np.linspace(0.1, 0.9, 5),
    "p_learn": np.linspace(0.05, 0.5, 5),
    "p_guess": np.linspace(0.1, 0.4, 4),
    "p_slip": np.linspace(0.05, 0.3, 4),
}


def bkt_forward(observations: Sequence[int], params: BKTParams) -> List[float]:
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


def predicted_correct_prob(observations: Sequence[int], params: BKTParams) -> List[float]:
    """P(correct_t) = P(known_t) * (1-slip) + P(unknown_t) * guess."""
    p_knowns = bkt_forward(observations, params)
    return [
        pk * (1 - params.p_slip) + (1 - pk) * params.p_guess for pk in p_knowns
    ]


def _neg_log_likelihood(observations: Sequence[int], params: BKTParams) -> float:
    probs = predicted_correct_prob(observations, params)
    eps = 1e-6
    nll = 0.0
    for o, p in zip(observations, probs):
        p = min(max(p, eps), 1 - eps)
        nll += -(o * np.log(p) + (1 - o) * np.log(1 - p))
    return nll


def _pad_observations(sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad per-student 0/1 sequences to (n_students, max_len)."""
    n = len(sequences)
    lengths = [len(s) for s in sequences]
    t_max = max(lengths) if lengths else 0
    obs = np.zeros((n, t_max), dtype=np.int8)
    mask = np.zeros((n, t_max), dtype=bool)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        obs[i, : len(seq)] = np.asarray(seq, dtype=np.int8)
        mask[i, : len(seq)] = True
    return obs, mask


def _valid_combos(grid: Dict[str, np.ndarray]) -> np.ndarray:
    """(C, 4) array of (p_init, p_learn, p_guess, p_slip) with guess+slip < 1."""
    rows = []
    for p_init in grid["p_init"]:
        for p_learn in grid["p_learn"]:
            for p_guess in grid["p_guess"]:
                for p_slip in grid["p_slip"]:
                    if p_guess + p_slip >= 1.0:
                        continue
                    rows.append((float(p_init), float(p_learn), float(p_guess), float(p_slip)))
    return np.asarray(rows, dtype=np.float64)


def _batch_nll(obs: np.ndarray, mask: np.ndarray, combos: np.ndarray) -> np.ndarray:
    """NLL for every grid point. obs/mask: (n, T); combos: (C, 4). Returns (C,)."""
    n_students, t_max = obs.shape
    n_combos = combos.shape[0]
    p_init, p_learn, p_guess, p_slip = (combos[:, 0], combos[:, 1], combos[:, 2], combos[:, 3])

    p_known = np.repeat(p_init[:, None], n_students, axis=1)  # (C, n)
    nll = np.zeros(n_combos, dtype=np.float64)
    eps = 1e-6
    p_slip_c = p_slip[:, None]
    p_guess_c = p_guess[:, None]
    p_learn_c = p_learn[:, None]

    for t in range(t_max):
        valid = mask[:, t]
        if not np.any(valid):
            continue
        o_t = obs[:, t].astype(np.float64)
        p_emit_correct = p_known * (1.0 - p_slip_c) + (1.0 - p_known) * p_guess_c
        p_obs = np.where(o_t[None, :] == 1.0, p_emit_correct, 1.0 - p_emit_correct)
        p_obs_clip = np.clip(p_obs, eps, 1.0 - eps)
        nll -= np.sum(np.log(p_obs_clip) * valid[None, :], axis=1)

        num = np.where(
            o_t[None, :] == 1.0,
            p_known * (1.0 - p_slip_c),
            p_known * p_slip_c,
        )
        p_post = np.divide(num, p_obs, out=np.copy(p_known), where=p_obs > 0)
        p_known = p_post + (1.0 - p_post) * p_learn_c

    return nll


def fit_skill_bkt(
    student_observations: List[List[int]],
    grid: Optional[Dict[str, np.ndarray]] = None,
) -> BKTParams:
    """Fit BKT params for a single skill via coarse grid search over all
    students' observation sequences for that skill, minimizing total NLL.

    `student_observations` is a list of sequences (one per student) of
    0/1 correctness for this skill only.
    """
    seqs = [s for s in student_observations if len(s) > 0]
    if not seqs:
        return BKTParams()

    if grid is None:
        grid = DEFAULT_GRID

    combos = _valid_combos(grid)
    obs, mask = _pad_observations(seqs)
    nll = _batch_nll(obs, mask, combos)
    best = combos[int(np.argmin(nll))]
    return BKTParams(p_init=best[0], p_learn=best[1], p_guess=best[2], p_slip=best[3])


class BKTModel:
    """Wraps per-skill BKTParams, trained across a set of StudentSequence."""

    def __init__(self):
        self.skill_params: Dict[int, BKTParams] = {}
        self.n_fitted: int = 0
        self.n_defaulted: int = 0

    def fit(
        self,
        sequences,
        num_skills: int,
        min_students: int = 5,
        min_obs: int = 20,
    ) -> None:
        by_skill: Dict[int, List[List[int]]] = {k: [] for k in range(num_skills)}
        for seq in sequences:
            per_skill_obs: Dict[int, List[int]] = {}
            for skill, correct in zip(seq.skill_ids, seq.correct):
                per_skill_obs.setdefault(skill, []).append(correct)
            for skill, obs in per_skill_obs.items():
                by_skill[skill].append(obs)

        self.n_fitted = 0
        self.n_defaulted = 0
        for skill in range(num_skills):
            seqs = by_skill[skill]
            n_obs = sum(len(s) for s in seqs)
            if len(seqs) < min_students or n_obs < min_obs:
                self.skill_params[skill] = BKTParams()
                self.n_defaulted += 1
                continue
            self.skill_params[skill] = fit_skill_bkt(seqs)
            self.n_fitted += 1

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
