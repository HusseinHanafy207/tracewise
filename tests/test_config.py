from pathlib import Path

import pytest

from src.utils.seed import load_config


def write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_config_inheritance_deep_merges_nested_values(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    write_yaml(
        base,
        "seed: 42\nmodel:\n  hidden: 64\n  dropout: 0.2\nactions: [a, b]\n",
    )
    write_yaml(
        child,
        "extends: base.yaml\nmodel:\n  dropout: 0.0\nactions: [c]\n",
    )

    config = load_config(str(child))

    assert config["seed"] == 42
    assert config["model"] == {"hidden": 64, "dropout": 0.0}
    assert config["actions"] == ["c"]
    assert "extends" not in config


def test_config_inheritance_rejects_cycles(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    write_yaml(first, "extends: second.yaml\n")
    write_yaml(second, "extends: first.yaml\n")

    with pytest.raises(ValueError, match="cycle"):
        load_config(str(first))
