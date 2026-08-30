"""Online student-state trackers for connecting KT models to the bandit.

The policy must NOT see true simulator mastery. It sees only estimates
updated from observed (skill, correct) responses — the same partial
observability a real tutor has.

BKTOnlineTracker  — classical per-skill P(known) / P(correct)
DKTOnlineTracker  — LSTM hidden state → P(correct | skill) vector
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.dataset import encode_interaction
from src.models.bkt import BKTModel, BKTParams
from src.models.dkt import DKT
from src.policy.rule_based import StudentState


class BKTOnlineTracker:
    """Per-student online BKT belief state."""

    def __init__(self, bkt: BKTModel):
        self.bkt = bkt
        self.p_known: Dict[int, float] = {}
        self.history_correct: List[int] = []

    def estimate_mastery(self, skill: int) -> float:
        """P(correct) for the active skill under current belief."""
        params = self.bkt.skill_params.get(skill, BKTParams())
        pk = self.p_known.get(skill, params.p_init)
        return float(pk * (1.0 - params.p_slip) + (1.0 - pk) * params.p_guess)

    def mean_mastery(self) -> float:
        """Average P(correct) over skills seen so far (or prior if none)."""
        if not self.p_known:
            # average prior P(correct) over all fitted skills
            vals = []
            for skill, params in self.bkt.skill_params.items():
                vals.append(
                    params.p_init * (1.0 - params.p_slip)
                    + (1.0 - params.p_init) * params.p_guess
                )
            return float(np.mean(vals)) if vals else 0.5
        return float(np.mean([self.estimate_mastery(s) for s in self.p_known]))

    def update(self, skill: int, correct: int) -> None:
        params = self.bkt.skill_params.get(skill, BKTParams())
        p_known = self.p_known.get(skill, params.p_init)
        if correct == 1:
            num = p_known * (1.0 - params.p_slip)
            den = num + (1.0 - p_known) * params.p_guess
        else:
            num = p_known * params.p_slip
            den = num + (1.0 - p_known) * (1.0 - params.p_guess)
        p_post = num / den if den > 0 else p_known
        self.p_known[skill] = p_post + (1.0 - p_post) * params.p_learn
        self.history_correct.append(int(correct))


class DKTOnlineTracker:
    """Per-student online DKT state via incremental LSTM steps."""

    def __init__(self, model: DKT, device: torch.device):
        self.model = model
        self.device = device
        self.num_skills = model.num_skills
        self.hidden: Optional[tuple] = None
        self.skill_probs: Optional[np.ndarray] = None  # P(correct) per skill
        self.history_correct: List[int] = []
        self.model.eval()

    def estimate_mastery(self, skill: int) -> float:
        if self.skill_probs is None:
            return 0.5
        return float(self.skill_probs[int(skill)])

    def mean_mastery(self) -> float:
        if self.skill_probs is None:
            return 0.5
        return float(self.skill_probs.mean())

    @torch.no_grad()
    def update(self, skill: int, correct: int) -> None:
        """Feed (skill, correct) and refresh next-step P(correct) for all skills."""
        input_id = encode_interaction(skill, correct, self.num_skills)
        x = torch.tensor([[input_id]], dtype=torch.long, device=self.device)
        emb = self.model.embedding(x)
        out, self.hidden = self.model.lstm(emb, self.hidden)
        logits = self.model.output(out)  # (1, 1, num_skills)
        self.skill_probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        self.history_correct.append(int(correct))


def observed_student_state(
    estimated_mastery: float,
    history_correct: List[int],
) -> StudentState:
    """Build a StudentState from *estimated* mastery + observed responses only."""
    recent = history_correct[-5:]
    recent_acc = float(np.mean(recent)) if recent else 0.0
    consecutive_failures = 0
    for c in reversed(history_correct):
        if c == 0:
            consecutive_failures += 1
        else:
            break
    return StudentState(
        mastery=float(estimated_mastery),
        recent_accuracy=recent_acc,
        attempts=len(history_correct),
        consecutive_failures=consecutive_failures,
    )


# Linear context: 8-d. Basis: 12-d (m, m², 3 mastery bins + same non-mastery dims).
CONTEXT_DIM_LINEAR = 8
CONTEXT_DIM_BASIS = 12


def mastery_basis(mastery: float) -> np.ndarray:
    """Nonlinear mastery features for LinUCB: [m, m², I_low, I_mid, I_high].

    Bins match the rule-policy thresholds / pedagogical regimes (0.4, 0.7).
    A raw linear term w·m cannot represent mid-mastery peaks in Effect(a,m).
    """
    m = float(np.clip(mastery, 0.0, 1.0))
    return np.array(
        [
            m,
            m * m,
            1.0 if m < 0.4 else 0.0,
            1.0 if 0.4 <= m < 0.7 else 0.0,
            1.0 if m >= 0.7 else 0.0,
        ],
        dtype=np.float64,
    )


def bandit_context_from_estimate(
    estimated_mastery: float,
    history_correct: List[int],
    difficulty: float,
    include_mastery: bool = True,
    history_actions: Optional[List[tuple]] = None,
    feature_mode: str = "linear",
) -> np.ndarray:
    """Bandit context using estimated mastery, never true latent mastery / z_i.

    feature_mode:
      "linear" — 8-d with raw mastery (legacy)
      "basis"  — 12-d with mastery basis [m, m², bins] so LinUCB can express
                 regime-dependent Effect(a,m)

    If history_actions is provided as (action, correct) pairs, the WE/SH
    success-rate dims carry type cues from experience only.
    """
    if feature_mode not in ("linear", "basis"):
        raise ValueError(f"Unknown feature_mode={feature_mode!r}")

    state = observed_student_state(estimated_mastery, history_correct)
    improvement = 0.0
    if len(history_correct) >= 10:
        prev = float(np.mean(history_correct[-10:-5]))
        improvement = state.recent_accuracy - prev

    we_rate = 0.5
    sh_rate = 0.5
    if history_actions:
        we = [c for a, c in history_actions if a == "worked_example"][-10:]
        sh = [c for a, c in history_actions if a == "socratic_hint"][-10:]
        if we:
            we_rate = float(np.mean(we))
        if sh:
            sh_rate = float(np.mean(sh))

    non_mastery = np.array(
        [
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

    if feature_mode == "linear":
        if include_mastery:
            return np.concatenate(
                [np.array([state.mastery], dtype=np.float64), non_mastery]
            )
        # no_state: keep difficulty + attempts only
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

    # basis
    if include_mastery:
        return np.concatenate([mastery_basis(state.mastery), non_mastery])
    # no_state under basis dim: zero mastery basis + accuracy/type cues;
    # keep difficulty and attempts
    zeros5 = np.zeros(5, dtype=np.float64)
    open_loop = np.array(
        [
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
    return np.concatenate([zeros5, open_loop])

