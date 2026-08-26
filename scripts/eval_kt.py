"""Score BKT and DKT on val/test. DKT is next-step prediction; BKT is reported
both on all steps (M2 protocol) and aligned to DKT's next-step positions.

Usage:
    python scripts/eval_kt.py --config configs/config.yaml
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.kt_eval import bkt_predict_aligned, collect_dkt_predictions, make_loader
from src.evaluation.metrics import compare_models, evaluate
from src.models.dkt import DKT
from src.utils.seed import load_config, set_seed


def load_split(processed_dir: Path, name: str):
    with open(processed_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_dkt(ckpt_path: Path, device: torch.device):
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    if isinstance(blob, dict) and "state_dict" in blob:
        num_skills = blob["num_skills"]
        dkt_cfg = blob.get("dkt_cfg") or {}
        state = blob["state_dict"]
    else:
        raise ValueError(f"{ckpt_path} is not a DKT checkpoint dict with state_dict")
    model = DKT(
        num_skills=num_skills,
        embedding_dim=dkt_cfg.get("embedding_dim", 128),
        hidden_dim=dkt_cfg.get("hidden_dim", 128),
        num_layers=dkt_cfg.get("num_layers", 1),
        dropout=0.0,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, dkt_cfg, num_skills


def score_bkt_full(model, sequences) -> dict:
    y_true, y_pred = [], []
    for seq in sequences:
        if len(seq.correct) == 0:
            continue
        y_true.append(seq.correct)
        y_pred.append(model.predict_sequence(seq.skill_ids, seq.correct))
    return evaluate(y_true, y_pred)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg["data"]["processed_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "skill_to_idx.pkl", "rb") as f:
        skill_to_idx = pickle.load(f)
    num_skills_data = len(skill_to_idx)

    with open(ckpt_dir / "bkt.pkl", "rb") as f:
        bkt = pickle.load(f)
    dkt, dkt_cfg, num_skills = load_dkt(ckpt_dir / "dkt_best.pt", device)
    if num_skills != num_skills_data:
        raise ValueError(f"DKT num_skills={num_skills} != data {num_skills_data}")

    max_seq_len = int(dkt_cfg.get("max_seq_len", cfg["dkt"]["max_seq_len"]))
    batch_size = int(dkt_cfg.get("batch_size", cfg["dkt"]["batch_size"]))

    payload = {"seed": cfg["seed"], "max_seq_len": max_seq_len, "splits": {}}
    table = {}

    for split in ("val", "test"):
        seqs = load_split(processed_dir, split)
        bkt_full = score_bkt_full(bkt, seqs)
        bkt_aligned = bkt_predict_aligned(bkt, seqs, max_seq_len)
        loader = make_loader(seqs, num_skills, max_seq_len, batch_size)
        dkt_m = collect_dkt_predictions(dkt, loader, device)
        dkt_m.pop("loss", None)

        payload["splits"][split] = {
            "BKT_full": bkt_full,
            "BKT_aligned": bkt_aligned,
            "DKT": dkt_m,
        }
        table[f"BKT_full/{split}"] = bkt_full
        table[f"BKT_aligned/{split}"] = bkt_aligned
        table[f"DKT/{split}"] = dkt_m
        print(
            f"{split:4} | BKT_full auc={bkt_full['roc_auc']:.4f} n={bkt_full['n_predictions']} | "
            f"BKT_aligned auc={bkt_aligned['roc_auc']:.4f} n={bkt_aligned['n_predictions']} | "
            f"DKT auc={dkt_m['roc_auc']:.4f} n={dkt_m['n_predictions']}"
        )

    print()
    print(compare_models(table))

    out = results_dir / "kt_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
