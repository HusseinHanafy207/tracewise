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


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[dict]]:
    """ECE with equal-width bins on [0, 1]. Also returns per-bin reliability stats."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.clip(np.asarray(y_pred).astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {"lo": float(lo), "hi": float(hi), "n": 0, "acc": None, "conf": None}
            )
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_pred[mask].mean())
        ece += (count / n) * abs(acc - conf)
        bins.append(
            {"lo": float(lo), "hi": float(hi), "n": count, "acc": acc, "conf": conf}
        )
    return float(ece), bins


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    return float(np.mean((y_pred - y_true) ** 2))


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
