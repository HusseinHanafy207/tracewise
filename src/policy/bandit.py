"""Contextual bandit for intervention selection.

State (context) vector, e.g. 8-dim:
    [mastery, recent_accuracy, difficulty, attempts_norm, improvement,
     time_on_task_norm, consecutive_failures_norm, skill_priority]

Two implementations:
    - EpsilonGreedyBandit: linear reward model per action + ε-greedy explore
    - LinUCB: linear contextual bandit with UCB exploration (Li et al. 2010)

Reward (in the simulator): r_t = mastery_{t+1} - mastery_t
"""
from typing import Dict, List, Optional

import numpy as np


class EpsilonGreedyBandit:
    def __init__(
        self,
        actions: List[str],
        context_dim: int,
        epsilon: float = 0.1,
        lr: float = 0.05,
        seed: int = 42,
    ):
        self.actions = list(actions)
        self.epsilon = epsilon
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self.weights: Dict[str, np.ndarray] = {
            a: np.zeros(context_dim, dtype=np.float64) for a in self.actions
        }

    def _estimate(self, action: str, context: np.ndarray) -> float:
        return float(self.weights[action] @ context)

    def select_action(self, context: np.ndarray) -> str:
        if self.rng.rand() < self.epsilon:
            return str(self.rng.choice(self.actions))
        estimates = {a: self._estimate(a, context) for a in self.actions}
        return max(estimates, key=estimates.get)

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        pred = self._estimate(action, context)
        error = reward - pred
        self.weights[action] += self.lr * error * context


class LinUCB:
    """LinUCB (Li et al. 2010) — disjoint linear model per action."""

    def __init__(self, actions: List[str], context_dim: int, alpha: float = 1.0):
        self.actions = list(actions)
        self.alpha = alpha
        self.A: Dict[str, np.ndarray] = {
            a: np.eye(context_dim, dtype=np.float64) for a in self.actions
        }
        self.b: Dict[str, np.ndarray] = {
            a: np.zeros(context_dim, dtype=np.float64) for a in self.actions
        }

    def select_action(self, context: np.ndarray) -> str:
        best_action, best_score = self.actions[0], -np.inf
        for a in self.actions:
            theta = np.linalg.solve(self.A[a], self.b[a])
            mean = float(theta @ context)
            # quadratic form x^T A^{-1} x via solve
            bound = self.alpha * np.sqrt(
                float(context @ np.linalg.solve(self.A[a], context))
            )
            score = mean + bound
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
