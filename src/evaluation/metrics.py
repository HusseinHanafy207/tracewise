"""Shared evaluation utilities for comparing student models (BKT vs DKT).

Both models ultimately produce, per (student, step), a predicted
P(correct) and the actual observed correctness — collect these into flat
arrays and score with standard metrics.
"""
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def flatten_predictions(
    y_true: List[List[int]], y_pred: List[List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    flat_true, flat_pred = [], []
    for t_seq, p_seq in zip(y_true, y_pred):
        flat_true.extend(t_seq)
        flat_pred.extend(p_seq)
    return np.array(flat_true), np.array(flat_pred)


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    binary_pred = (y_pred >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, binary_pred)
    return {
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "n_predictions": int(len(y_true)),
    }


def evaluate(y_true: List[List[int]], y_pred: List[List[float]]) -> dict:
    flat_true, flat_pred = flatten_predictions(y_true, y_pred)
    return evaluate_arrays(flat_true, flat_pred)


def compare_models(results: dict) -> str:
    """results: {"BKT": {...}, "DKT": {...}} -> markdown table string."""
    header = "| Model | ROC-AUC | Accuracy | N predictions |\n|---|---|---|---|\n"
    rows = ""
    for name, metrics in results.items():
        rows += (
            f"| {name} | {metrics['roc_auc']:.4f} | {metrics['accuracy']:.4f} "
            f"| {metrics['n_predictions']} |\n"
        )
    return header + rows
