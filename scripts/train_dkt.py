"""Train DKT on preprocessed ASSISTments sequences.

Usage (Kaggle/Colab or local with GPU):
    python scripts/train_dkt.py --config configs/config.yaml
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import KTDataset, collate_fn
from src.evaluation.kt_eval import collect_dkt_predictions
from src.models.dkt import DKT, masked_bce_loss
from src.utils.seed import load_config, set_seed


def load_split(processed_dir: Path, name: str):
    with open(processed_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_seqs = load_split(processed_dir, "train")
    val_seqs = load_split(processed_dir, "val")
    with open(processed_dir / "skill_to_idx.pkl", "rb") as f:
        skill_to_idx = pickle.load(f)
    num_skills = len(skill_to_idx)

    dkt_cfg = cfg["dkt"]
    train_ds = KTDataset(train_seqs, num_skills, dkt_cfg["max_seq_len"])
    val_ds = KTDataset(val_seqs, num_skills, dkt_cfg["max_seq_len"])
    train_loader = DataLoader(
        train_ds, batch_size=dkt_cfg["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=dkt_cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    model = DKT(
        num_skills=num_skills,
        embedding_dim=dkt_cfg["embedding_dim"],
        hidden_dim=dkt_cfg["hidden_dim"],
        num_layers=dkt_cfg["num_layers"],
        dropout=dkt_cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"DKT params={n_params:,} | train_seq={len(train_ds)} | val_seq={len(val_ds)} | "
        f"skills={num_skills}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=dkt_cfg["lr"])

    best_val_auc = -1.0
    best_epoch = 0
    patience = dkt_cfg["early_stopping_patience"]
    bad_epochs = 0
    history = []

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "dkt_best.pt"

    t0 = time.time()
    for epoch in range(dkt_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["lengths"])
            probs = model.predict_next_skill_prob(logits, batch["target_skill"])
            loss = masked_bce_loss(probs, batch["target_correct"], batch["mask"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_metrics = collect_dkt_predictions(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["roc_auc"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch+1}/{dkt_cfg['epochs']} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_skills": num_skills,
                    "dkt_cfg": dkt_cfg,
                    "val_auc": best_val_auc,
                    "epoch": best_epoch,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val_auc={best_val_auc:.4f} @ {best_epoch})")
                break

    elapsed = time.time() - t0
    print(f"Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}. Saved {ckpt_path}")

    payload = {
        "split": "val",
        "roc_auc": best_val_auc,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "n_train_sequences": len(train_ds),
        "n_val_sequences": len(val_ds),
        "num_skills": num_skills,
        "max_seq_len": dkt_cfg["max_seq_len"],
        "early_stopping": "val_auc",
        "train_seconds": round(elapsed, 2),
        "device": str(device),
        "seed": cfg["seed"],
        "history": history,
    }
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "dkt_val.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
