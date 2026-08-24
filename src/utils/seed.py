"""Reproducibility helpers: seeding and config loading.

Usage:
    from src.utils.seed import set_seed, load_config
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["seed"])
"""
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load the project YAML config."""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
