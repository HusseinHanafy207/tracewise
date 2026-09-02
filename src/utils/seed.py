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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings while replacing scalar/list values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path, loading: set) -> Dict[str, Any]:
    resolved = path.resolve()
    if resolved in loading:
        chain = " -> ".join(str(item) for item in [*loading, resolved])
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    if not resolved.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved}")

    with open(resolved, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {resolved}")

    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"extends must be a non-empty path string: {resolved}")
    parent_path = (resolved.parent / parent).resolve()
    base = _load_config(parent_path, loading | {resolved})
    return _deep_merge(base, payload)


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config, optionally inheriting via a relative `extends` path."""
    return _load_config(Path(path), set())
