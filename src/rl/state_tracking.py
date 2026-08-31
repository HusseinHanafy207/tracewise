"""Load online KT trackers for DQN observation conditions."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from src.models.dkt import DKT
from src.policy.kt_state import BKTOnlineTracker, DKTOnlineTracker
from src.rl.environment import OnlineTracker


@dataclass(frozen=True)
class StateTracking:
    """Environment mode and optional fresh-tracker factory for one condition."""

    condition: str
    environment_mode: str
    tracker_factory: Optional[Callable[[], OnlineTracker]]
    checkpoint_path: Optional[Path]


def load_state_tracking(
    condition: str,
    checkpoint_dir: Path,
    device: torch.device,
) -> StateTracking:
    """Build the tracker dependency for oracle/BKT/DKT/no-state runs."""
    condition = str(condition).lower()
    if condition == "oracle":
        return StateTracking(condition, "oracle", None, None)
    if condition == "no_state":
        return StateTracking(condition, "no_state", None, None)
    if condition == "bkt":
        path = Path(checkpoint_dir) / "bkt.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run scripts/fit_bkt.py first")
        with open(path, "rb") as file:
            model = pickle.load(file)
        return StateTracking(
            condition,
            "estimated",
            lambda: BKTOnlineTracker(model),
            path,
        )
    if condition == "dkt":
        path = Path(checkpoint_dir) / "dkt_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run scripts/train_dkt.py first")
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device)
        dkt_cfg = payload.get("dkt_cfg") or {}
        model = DKT(
            num_skills=int(payload["num_skills"]),
            embedding_dim=int(dkt_cfg.get("embedding_dim", 128)),
            hidden_dim=int(dkt_cfg.get("hidden_dim", 128)),
            num_layers=int(dkt_cfg.get("num_layers", 1)),
            dropout=0.0,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return StateTracking(
            condition,
            "estimated",
            lambda: DKTOnlineTracker(model, device),
            path,
        )
    raise ValueError("condition must be oracle, bkt, dkt, or no_state")
