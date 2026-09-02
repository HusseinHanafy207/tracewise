import json
from pathlib import Path

import pytest
import torch

from scripts.demo_policy_episode import (
    load_trace,
    run_episode,
    save_trace,
    validate_trace,
)


def test_interleave_demo_runs_without_a_checkpoint(tmp_path):
    payload = run_episode(
        "configs/dqn_train.yaml",
        "interleave",
        tmp_path / "not-needed.pt",
        episode_seed=17,
        device=torch.device("cpu"),
    )

    assert payload["policy"] == "interleave"
    assert payload["horizon"] == 50
    assert payload["steps"][0]["action"] == "explain"
    assert payload["steps"][1]["action"] == "harder_problem"
    assert payload["return"] == pytest.approx(
        payload["final_mastery"] - payload["initial_mastery"]
    )

    path = tmp_path / "trace.json"
    save_trace(payload, path)
    assert load_trace(path) == json.loads(path.read_text(encoding="utf-8"))


def test_trace_validation_rejects_incomplete_steps():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_trace(
            {
                "schema_version": 1,
                "horizon": 1,
                "steps": [{"step": 1, "action": "explain"}],
            }
        )
