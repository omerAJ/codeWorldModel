from __future__ import annotations

import json
from pathlib import Path

import pytest

from cwmodel.config import WrapperTokenConfig
from cwmodel.data.dataset import TransitionDataset


def test_transition_dataset_loads_and_wraps_segments() -> None:
    dataset = TransitionDataset(
        "data/humaneval_transition_1task.jsonl",
        WrapperTokenConfig(),
    )
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["code_context"].startswith("<CTX_BOS>\n")
    assert sample["action"].startswith("<ACT_BOS>\n")
    assert sample["current_state"].startswith("<STATE_BOS>\n")
    assert sample["next_action"].startswith("<NEXT_ACT_BOS>\n")
    assert sample["next_state"].startswith("<NEXT_STATE_BOS>\n")
    assert "<CTX_SUM>" in sample["code_context"]
    assert "<ACT_SUM>" in sample["action"]
    assert "<STATE_SUM>" in sample["current_state"]


def test_transition_dataset_schema_validation(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_dataset.jsonl"
    bad_record = {
        "example_id": "x",
        "task_id": "t",
        "step_index": 0,
        "code_context": "ctx",
        "action": "act",
        "current_state": "cur",
        "next_action": "nact",
        # next_state intentionally omitted
    }

    bad_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        TransitionDataset(bad_path, WrapperTokenConfig())
