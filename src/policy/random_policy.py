"""Random intervention policy — Baseline 1 before rule-based / bandit."""
from typing import Any, List

import numpy as np


class RandomPolicy:
    def __init__(self, actions: List[str], seed: int = 42):
        self.actions = list(actions)
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: Any = None) -> str:
        return str(self.rng.choice(self.actions))
