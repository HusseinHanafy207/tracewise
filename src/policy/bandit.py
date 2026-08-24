"""Contextual bandit for intervention selection.

State (context) vector, e.g. 8-dim:
    [mastery, recent_accuracy, difficulty, attempts, improvement,
     time_on_task_norm, consecutive_failures, skill_priority]

Two implementations:
    - EpsilonGreedyBandit: simple, per-action running average reward
      conditioned on discretized context (or a linear model per action).
    - LinUCB: linear contextual bandit with upper-confidence-bound
      exploration — the more principled option, still cheap to run.

Reward design (see docs/roadmap.md Day 8-10):
    r_t = accuracy_{t+1} - accuracy_t   (simple)
    or a shaped version subtracting a small penalty for excessive hints.
"""
from typing import Dict, List

import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, actions: List[str], context_dim: int, epsilon: float = 0.1, lr: float = 0.05):
        self.actions = actions
        self.epsilon = epsilon
        self.lr = lr
        # one linear reward-estimator weight vector per action
        self.weights: Dict[str, np.ndarray] = {
            a: np.zeros(context_dim) for a in actions
        }

    def _estimate(self, action: str, context: np.ndarray) -> float:
        return float(self.weights[action] @ context)

    def select_action(self, context: np.ndarray) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        estimates = {a: self._estimate(a, context) for a in self.actions}
        return max(estimates, key=estimates.get)

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        pred = self._estimate(action, context)
        error = reward - pred
        self.weights[action] += self.lr * error * context


class LinUCB:
    """LinUCB (Li et al. 2010) — disjoint linear model per action."""

    def __init__(self, actions: List[str], context_dim: int, alpha: float = 1.0):
        self.actions = actions
        self.alpha = alpha
        self.A: Dict[str, np.ndarray] = {a: np.eye(context_dim) for a in actions}
        self.b: Dict[str, np.ndarray] = {a: np.zeros(context_dim) for a in actions}

    def select_action(self, context: np.ndarray) -> str:
        best_action, best_score = None, -np.inf
        for a in self.actions:
            # solve() is more stable than forming A^{-1} explicitly
            theta = np.linalg.solve(self.A[a], self.b[a])
            mean = float(theta @ context)
            bound = self.alpha * np.sqrt(float(context @ np.linalg.solve(self.A[a], context)))
            score = mean + bound
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
