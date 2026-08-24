"""Turn raw ASSISTments CSV into per-student (skill, correct) sequences,
split by student into train/val/test, and persist to data/processed/.

This is intentionally dependency-light (pandas + numpy only) so it runs
identically in a Kaggle notebook or locally.
"""
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.download import check_raw_exists


@dataclass
class StudentSequence:
    user_id: int
    skill_ids: List[int]   # contiguous integer skill indices
    correct: List[int]     # 0/1


def load_raw(raw_path: str) -> pd.DataFrame:
    """Load the raw ASSISTments CSV.

    Note: the file is not UTF-8 in some mirrors — try latin-1 fallback.
    Mixed-type columns are common; disable the pandas dtype warning.
    """
    path = Path(raw_path)
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    return df


def _parse_skill_id(value):
    """Coerce skill_id to int; drop unparseable / multi-skill leftovers as NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return np.nan if np.isnan(value) else int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    if "," in text:
        text = text.split(",")[0].strip()
    try:
        return int(float(text))
    except ValueError:
        return np.nan


def clean(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Drop null skill/correct rows, filter short student histories,
    sort by (user_id, order_id).
    """
    required = ["user_id", "order_id", "skill_id", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {required} in raw data, missing: {missing}. "
            "Check that you downloaded the skill_builder_data.csv variant."
        )

    df = df.copy()
    df["skill_id"] = df["skill_id"].map(_parse_skill_id)
    df = df.dropna(subset=["skill_id", "correct"])
    df["correct"] = (pd.to_numeric(df["correct"], errors="coerce") > 0).astype(int)
    df["skill_id"] = df["skill_id"].astype(int)

    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= min_interactions].index
    df = df[df["user_id"].isin(keep_users)]

    df = df.sort_values(["user_id", "order_id"])
    return df.reset_index(drop=True)


def build_skill_index(df: pd.DataFrame) -> Dict[int, int]:
    """Map raw skill_id values to contiguous 0..K-1 indices."""
    unique_skills = sorted(df["skill_id"].unique())
    return {sid: i for i, sid in enumerate(unique_skills)}


def build_sequences(df: pd.DataFrame, skill_to_idx: Dict[int, int]) -> List[StudentSequence]:
    sequences = []
    for user_id, group in df.groupby("user_id"):
        skill_ids = [skill_to_idx[s] for s in group["skill_id"].tolist()]
        correct = group["correct"].tolist()
        sequences.append(StudentSequence(user_id=user_id, skill_ids=skill_ids, correct=correct))
    return sequences


def split_by_student(
    sequences: List[StudentSequence],
    train_split: float,
    val_split: float,
    seed: int = 42,
) -> Tuple[List[StudentSequence], List[StudentSequence], List[StudentSequence]]:
    """Split at the student level to avoid leakage across train/val/test."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(sequences))
    rng.shuffle(idx)

    n = len(sequences)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train = [sequences[i] for i in train_idx]
    val = [sequences[i] for i in val_idx]
    test = [sequences[i] for i in test_idx]
    return train, val, test


def run(cfg: dict) -> None:
    """Full pipeline entry point, driven by config.yaml's `data` section."""
    raw_path = check_raw_exists(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(str(raw_path))
    df = clean(df, cfg["data"]["min_interactions_per_student"])
    skill_to_idx = build_skill_index(df)
    sequences = build_sequences(df, skill_to_idx)

    train, val, test = split_by_student(
        sequences,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        seed=cfg["seed"],
    )

    with open(processed_dir / "train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(processed_dir / "val.pkl", "wb") as f:
        pickle.dump(val, f)
    with open(processed_dir / "test.pkl", "wb") as f:
        pickle.dump(test, f)
    with open(processed_dir / "skill_to_idx.pkl", "wb") as f:
        pickle.dump(skill_to_idx, f)

    idx_to_skill = {i: sid for sid, i in skill_to_idx.items()}
    with open(processed_dir / "idx_to_skill.pkl", "wb") as f:
        pickle.dump(idx_to_skill, f)

    meta = {
        "num_skills": len(skill_to_idx),
        "n_students_total": len(sequences),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_interactions": int(sum(len(s.skill_ids) for s in sequences)),
        "min_interactions_per_student": cfg["data"]["min_interactions_per_student"],
        "seed": cfg["seed"],
    }
    with open(processed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Students: train={len(train)}, val={len(val)}, test={len(test)} | "
        f"skills={len(skill_to_idx)} | interactions={meta['n_interactions']}"
    )
