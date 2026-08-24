"""Rule-based adaptive policy — the simplest baseline before the bandit.

State is a dict (or dataclass) with at least `mastery: float in [0,1]`.
Action space is the fixed list from configs/config.yaml -> policy.actions.
"""
from dataclasses import dataclass


@dataclass
class StudentState:
    mastery: float          # current estimated mastery of the active skill
    recent_accuracy: float  # accuracy over last k attempts
    attempts: int           # attempts on current skill
    consecutive_failures: int


class RuleBasedPolicy:
    """Uses a subset of the full action list on purpose: this is a simple
    mastery-threshold baseline, not a full 6-way decision tree.
    """

    def __init__(self, low_thresh: float = 0.4, high_thresh: float = 0.7):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def select_action(self, state: StudentState) -> str:
        if state.consecutive_failures >= 2:
            return "prerequisite_review"
        if state.mastery < self.low_thresh:
            return "explain"
        if state.mastery < self.high_thresh:
            return "worked_example"
        return "harder_problem"
