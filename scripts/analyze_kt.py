"""M4 analysis: length/skill slices + calibration for BKT vs DKT on test.

Does not retrain. Needs data/processed + results/checkpoints/{bkt.pkl,dkt_best.pt}.

Usage:
    python scripts/analyze_kt.py --config configs/config.yaml
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.kt_eval import collect_aligned_records
from src.evaluation.metrics import (
    brier_score,
    evaluate_arrays,
    expected_calibration_error,
)
from src.models.dkt import DKT
from src.utils.seed import load_config, set_seed


LENGTH_BINS = [
    ("2-20", 2, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
]
POSITION_BINS = [
    ("t=1-10", 1, 10),
    ("t=11-50", 11, 50),
    ("t=51-200", 51, 200),
]


def load_split(processed_dir: Path, name: str):
    with open(processed_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_dkt(ckpt_path: Path, device: torch.device):
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    dkt_cfg = blob.get("dkt_cfg") or {}
    model = DKT(
        num_skills=blob["num_skills"],
        embedding_dim=dkt_cfg.get("embedding_dim", 128),
        hidden_dim=dkt_cfg.get("hidden_dim", 128),
        num_layers=dkt_cfg.get("num_layers", 1),
        dropout=0.0,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, dkt_cfg, blob["num_skills"]


def train_skill_counts(train_seqs) -> np.ndarray:
    counts = {}
    for seq in train_seqs:
        for sk in seq.skill_ids:
            counts[sk] = counts.get(sk, 0) + 1
    if not counts:
        return np.array([])
    max_id = max(counts)
    arr = np.zeros(max_id + 1, dtype=np.int64)
    for sk, n in counts.items():
        arr[sk] = n
    return arr


def slice_auc(y, p_bkt, p_dkt, mask, label):
    n = int(mask.sum())
    if n < 50 or len(np.unique(y[mask])) < 2:
        return {
            "slice": label,
            "n": n,
            "bkt_auc": None,
            "dkt_auc": None,
            "delta": None,
        }
    bkt = evaluate_arrays(y[mask], p_bkt[mask])
    dkt = evaluate_arrays(y[mask], p_dkt[mask])
    return {
        "slice": label,
        "n": n,
        "bkt_auc": bkt["roc_auc"],
        "dkt_auc": dkt["roc_auc"],
        "delta": dkt["roc_auc"] - bkt["roc_auc"],
    }


def print_slice_table(title, rows):
    print(f"\n### {title}")
    print("| slice | n | BKT AUC | DKT AUC | DKT-BKT |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["bkt_auc"] is None:
            print(f"| {r['slice']} | {r['n']} | n/a | n/a | n/a |")
            continue
        print(
            f"| {r['slice']} | {r['n']} | {r['bkt_auc']:.4f} | "
            f"{r['dkt_auc']:.4f} | {r['delta']:+.4f} |"
        )


def plot_calibration(y, p_bkt, p_dkt, bkt_bins, dkt_bins, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", label="perfect")

    def xy(bins):
        xs, ys = [], []
        for b in bins:
            if b["n"] and b["conf"] is not None:
                xs.append(b["conf"])
                ys.append(b["acc"])
        return xs, ys

    bx, by = xy(bkt_bins)
    dx, dy = xy(dkt_bins)
    ax.plot(bx, by, "o-", label="BKT aligned", color="#b35c5c")
    ax.plot(dx, dy, "s-", label="DKT", color="#3b6ea5")
    ax.set_xlabel("Mean predicted P(correct)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability (test, aligned next-step)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_slice_bars(rows, title, out_path: Path):
    labels = [r["slice"] for r in rows if r["bkt_auc"] is not None]
    bkt = [r["bkt_auc"] for r in rows if r["bkt_auc"] is not None]
    dkt = [r["dkt_auc"] for r in rows if r["dkt_auc"] is not None]
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, bkt, w, label="BKT aligned", color="#b35c5c")
    ax.bar(x + w / 2, dkt, w, label="DKT", color="#3b6ea5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg["data"]["processed_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_seqs = load_split(processed_dir, "train")
    test_seqs = load_split(processed_dir, "test")
    with open(ckpt_dir / "bkt.pkl", "rb") as f:
        bkt = pickle.load(f)
    dkt, dkt_cfg, num_skills = load_dkt(ckpt_dir / "dkt_best.pt", device)
    max_seq_len = int(dkt_cfg.get("max_seq_len", cfg["dkt"]["max_seq_len"]))

    rec = collect_aligned_records(bkt, dkt, test_seqs, num_skills, max_seq_len, device)
    y, pb, pd = rec["y_true"], rec["p_bkt"], rec["p_dkt"]

    overall = {
        "BKT_aligned": evaluate_arrays(y, pb),
        "DKT": evaluate_arrays(y, pd),
    }
    overall["BKT_aligned"]["ece"] = expected_calibration_error(y, pb)[0]
    overall["DKT"]["ece"] = expected_calibration_error(y, pd)[0]
    overall["BKT_aligned"]["brier"] = brier_score(y, pb)
    overall["DKT"]["brier"] = brier_score(y, pd)

    n_bkt_params = 4 * len(getattr(bkt, "skill_params", {}))
    n_dkt_params = sum(p.numel() for p in dkt.parameters())

    print("### Overall (test, aligned)")
    print("| Model | params | ROC-AUC | Acc | Brier | ECE | n |")
    print("|---|---|---|---|---|---|---|")
    print(
        f"| BKT aligned | {n_bkt_params} | {overall['BKT_aligned']['roc_auc']:.4f} | "
        f"{overall['BKT_aligned']['accuracy']:.4f} | {overall['BKT_aligned']['brier']:.4f} | "
        f"{overall['BKT_aligned']['ece']:.4f} | {overall['BKT_aligned']['n_predictions']} |"
    )
    print(
        f"| DKT | {n_dkt_params} | {overall['DKT']['roc_auc']:.4f} | "
        f"{overall['DKT']['accuracy']:.4f} | {overall['DKT']['brier']:.4f} | "
        f"{overall['DKT']['ece']:.4f} | {overall['DKT']['n_predictions']} |"
    )

    length_rows = []
    for name, lo, hi in LENGTH_BINS:
        mask = (rec["seq_len"] >= lo) & (rec["seq_len"] <= hi)
        length_rows.append(slice_auc(y, pb, pd, mask, name))
    print_slice_table("AUC by student sequence length (truncated at 200)", length_rows)

    pos_rows = []
    for name, lo, hi in POSITION_BINS:
        mask = (rec["position"] >= lo) & (rec["position"] <= hi)
        pos_rows.append(slice_auc(y, pb, pd, mask, name))
    print_slice_table("AUC by position in the sequence", pos_rows)

    counts = train_skill_counts(train_seqs)
    skill_freq = np.array([counts[s] if s < len(counts) else 0 for s in rec["skill"]])
    median = np.median(counts[counts > 0]) if np.any(counts > 0) else 0
    skill_rows = [
        slice_auc(y, pb, pd, skill_freq >= median, "head skills (train count >= median)"),
        slice_auc(y, pb, pd, skill_freq < median, "tail skills (train count < median)"),
    ]
    print_slice_table("AUC by skill frequency (train)", skill_rows)

    bkt_ece, bkt_bins = expected_calibration_error(y, pb)
    dkt_ece, dkt_bins = expected_calibration_error(y, pd)
    print(f"\nECE BKT={bkt_ece:.4f} | DKT={dkt_ece:.4f} (lower is better calibrated)")

    plot_calibration(y, pb, pd, bkt_bins, dkt_bins, fig_dir / "calibration_test.png")
    plot_slice_bars(length_rows, "Test AUC by sequence length", fig_dir / "auc_by_length.png")
    plot_slice_bars(pos_rows, "Test AUC by position t", fig_dir / "auc_by_position.png")
    print(f"Wrote figures to {fig_dir}")

    payload = {
        "split": "test",
        "max_seq_len": max_seq_len,
        "n_bkt_params": n_bkt_params,
        "n_dkt_params": n_dkt_params,
        "overall": overall,
        "by_length": length_rows,
        "by_position": pos_rows,
        "by_skill_freq": skill_rows,
        "calibration_bins": {"BKT": bkt_bins, "DKT": dkt_bins},
    }
    out = results_dir / "kt_slices.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
