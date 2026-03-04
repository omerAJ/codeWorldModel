from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

from cwmodel.config import WrapperTokenConfig
from cwmodel.data.schema import TransitionExample


class TransitionDataset(Dataset[Dict[str, str]]):
    def __init__(self, jsonl_path: str | Path, wrappers: WrapperTokenConfig) -> None:
        self.path = Path(jsonl_path)
        self.wrappers = wrappers
        self.examples = self._load_examples(self.path)

    @staticmethod
    def _load_examples(path: Path) -> List[TransitionExample]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        examples: List[TransitionExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                examples.append(TransitionExample.from_record(record, line_no=line_no))

        if not examples:
            raise ValueError(f"Dataset file {path} is empty")
        return examples

    @staticmethod
    def _wrap_segment(text: str, bos: str, sep: str) -> str:
        cleaned = text.rstrip("\n")
        return f"{bos}\n{cleaned}\n{sep}"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        ex = self.examples[index]
        sep = self.wrappers.segment_sep

        return {
            "example_id": ex.example_id,
            "task_id": ex.task_id,
            "step_index": str(ex.step_index),
            "code_context": self._wrap_segment(
                ex.code_context,
                self.wrappers.context_bos,
                sep,
            ),
            "action": self._wrap_segment(
                ex.action,
                self.wrappers.action_bos,
                sep,
            ),
            "current_state": self._wrap_segment(
                ex.current_state,
                self.wrappers.state_bos,
                sep,
            ),
            "next_action": self._wrap_segment(
                ex.next_action,
                self.wrappers.next_action_bos,
                sep,
            ),
            "next_state": self._wrap_segment(
                ex.next_state,
                self.wrappers.next_state_bos,
                sep,
            ),
        }
