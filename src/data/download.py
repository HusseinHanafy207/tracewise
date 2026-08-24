"""Locate the ASSISTments skill-builder CSV. We do not redistribute it.

Place the file at `data/raw/skill_builder_data.csv`. Download sources and
column notes are in `data/README.md`.
"""
from pathlib import Path

EXPECTED_NAME = "skill_builder_data.csv"


def expected_raw_path(raw_dir: str = "data/raw") -> Path:
    return Path(raw_dir) / EXPECTED_NAME


def check_raw_exists(raw_dir: str = "data/raw") -> Path:
    path = expected_raw_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the ASSISTments 2009-2010 skill-builder "
            "CSV and save it under that name. See data/README.md."
        )
    return path
