"""Collect next-step predictions for DKT (and BKT aligned to the same steps)."""
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KTDataset, collate_fn
from src.evaluation.metrics import evaluate_arrays
from src.models.dkt import masked_bce_loss


def collect_dkt_predictions(model, loader: DataLoader, device: torch.device) -> dict:
    """Flatten masked next-step P(correct) from a KT DataLoader."""
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["lengths"])
            probs = model.predict_next_skill_prob(logits, batch["target_skill"])
            total_loss += masked_bce_loss(
                probs, batch["target_correct"], batch["mask"]
            ).item()
            mask = batch["mask"].bool()
            ys.append(batch["target_correct"][mask].detach().cpu().numpy())
            ps.append(probs[mask].detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(int) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=float)
    metrics = evaluate_arrays(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def make_loader(sequences, num_skills: int, max_seq_len: int, batch_size: int) -> DataLoader:
    ds = KTDataset(sequences, num_skills, max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def bkt_predict_aligned(model, sequences, max_seq_len: int) -> dict:
    """BKT P(correct) on the same next-step, truncated positions DKT uses.

    DKT cannot score t=0 (it predicts t+1 from (skill_t, correct_t)). Sequences
    are truncated to max_seq_len like KTDataset.
    """
    y_true: List[int] = []
    y_pred: List[float] = []
    for seq in sequences:
        skills = seq.skill_ids[:max_seq_len]
        correct = seq.correct[:max_seq_len]
        if len(correct) < 2:
            continue
        preds = model.predict_sequence(skills, correct)
        y_true.extend(correct[1:])
        y_pred.extend(preds[1:])
    return evaluate_arrays(np.asarray(y_true), np.asarray(y_pred))
