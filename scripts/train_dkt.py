"""Train DKT on preprocessed ASSISTments sequences.

Usage (Kaggle/Colab or local with GPU):
    python scripts/train_dkt.py --config configs/config.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import KTDataset, collate_fn
from src.models.dkt import DKT, masked_bce_loss
from src.utils.seed import load_config, set_seed


def load_split(processed_dir: Path, name: str):
    with open(processed_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    optimizer = torch.optim.Adam(model.parameters(), lr=dkt_cfg["lr"])

    best_val_loss = float("inf")
    patience = dkt_cfg["early_stopping_patience"]
    bad_epochs = 0

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"], batch["lengths"])
                probs = model.predict_next_skill_prob(logits, batch["target_skill"])
                loss = masked_bce_loss(probs, batch["target_correct"], batch["mask"])
                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"epoch {epoch+1}/{dkt_cfg['epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_dir / "dkt_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best val loss: {best_val_loss:.4f}. Checkpoint saved to {ckpt_dir / 'dkt_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
