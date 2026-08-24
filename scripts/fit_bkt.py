"""Fit per-skill BKT on the train split and score the val split.

Usage:
    python scripts/fit_bkt.py --config configs/config.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate
from src.models.bkt import BKTModel
from src.utils.seed import load_config, set_seed


def load_split(processed_dir: Path, name: str):
    with open(processed_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_seqs = load_split(processed_dir, "train")
    val_seqs = load_split(processed_dir, "val")
    with open(processed_dir / "skill_to_idx.pkl", "rb") as f:
        skill_to_idx = pickle.load(f)
    num_skills = len(skill_to_idx)

    model = BKTModel()
    model.fit(train_seqs, num_skills=num_skills)

    y_true, y_pred = [], []
    for seq in val_seqs:
        if len(seq.correct) == 0:
            continue
        y_true.append(seq.correct)
        y_pred.append(model.predict_sequence(seq.skill_ids, seq.correct))

    metrics = evaluate(y_true, y_pred)
    print(
        f"BKT val | roc_auc={metrics['roc_auc']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | n={metrics['n_predictions']}"
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "bkt.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {ckpt_dir / 'bkt.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
