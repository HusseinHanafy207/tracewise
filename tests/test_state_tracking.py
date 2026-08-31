import pytest
import torch

from src.rl.state_tracking import load_state_tracking


def test_state_tracking_modes_without_checkpoints(tmp_path):
    oracle = load_state_tracking("oracle", tmp_path, torch.device("cpu"))
    no_state = load_state_tracking("no_state", tmp_path, torch.device("cpu"))

    assert oracle.environment_mode == "oracle"
    assert oracle.tracker_factory is None
    assert no_state.environment_mode == "no_state"
    assert no_state.tracker_factory is None


def test_state_tracking_rejects_unknown_or_missing_condition(tmp_path):
    with pytest.raises(ValueError, match="oracle, bkt, dkt, or no_state"):
        load_state_tracking("unknown", tmp_path, torch.device("cpu"))
    with pytest.raises(FileNotFoundError, match="bkt.pkl"):
        load_state_tracking("bkt", tmp_path, torch.device("cpu"))
