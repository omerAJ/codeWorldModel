from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cwmodel.config import MaxLengthConfig, WrapperTokenConfig
from cwmodel.data.schema import TransitionExample


class TransitionDataset(Dataset[Dict[str, str]]):
    def __init__(self, jsonl_paths: str | Path | Sequence[str | Path], wrappers: WrapperTokenConfig) -> None:
        if isinstance(jsonl_paths, (str, Path)):
            paths = [Path(jsonl_paths)]
        else:
            paths = [Path(path) for path in jsonl_paths]

        if not paths:
            raise ValueError("TransitionDataset requires at least one dataset path")

        self.paths = paths
        self.path = paths[0]
        self.wrappers = wrappers
        self.examples = self._load_examples(self.paths)
        self.filter_stats: Dict[str, object] | None = None

    @staticmethod
    def _load_examples(paths: Sequence[Path]) -> List[TransitionExample]:
        examples: List[TransitionExample] = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")

            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    examples.append(TransitionExample.from_record(record, line_no=line_no))

        if not examples:
            rendered = ", ".join(str(path) for path in paths)
            raise ValueError(f"Dataset files are empty: {rendered}")
        return examples

    @staticmethod
    def _wrap_segment(text: str, bos: str, sep: str) -> str:
        cleaned = text.rstrip("\n")
        return f"{bos}\n{cleaned}\n{sep}"

    @staticmethod
    def _merge_task_text(task_text: str | None, code_context: str) -> str:
        if task_text is None or not task_text.strip():
            return code_context
        cleaned_task = task_text.strip()
        return f"# TASK_TEXT\n{cleaned_task}\n\n{code_context}"

    def __len__(self) -> int:
        return len(self.examples)

    def _build_item(self, ex: TransitionExample) -> Dict[str, str]:
        sep = self.wrappers.segment_sep
        return {
            "example_id": ex.example_id,
            "task_id": ex.task_id,
            "step_index": str(ex.step_index),
            "code_context": self._wrap_segment(
                self._merge_task_text(ex.task_text, ex.code_context),
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

    def filter_overlength_examples(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_length: MaxLengthConfig,
    ) -> Dict[str, object]:
        segment_limits = {
            "code_context": int(max_length.code_context),
            "action": int(max_length.action),
            "current_state": int(max_length.current_state),
            "next_action": int(max_length.next_action),
            "next_state": int(max_length.next_state),
        }

        before_count = len(self.examples)
        dropped_by_segment = {name: 0 for name in segment_limits}
        kept_examples: List[TransitionExample] = []

        for ex in self.examples:
            item = self._build_item(ex)
            drop_sample = False
            for segment_name, limit in segment_limits.items():
                encoded = tokenizer(
                    item[segment_name],
                    truncation=False,
                    return_attention_mask=False,
                )
                token_count = len(encoded["input_ids"])
                if token_count > limit:
                    dropped_by_segment[segment_name] += 1
                    drop_sample = True
                    break

            if not drop_sample:
                kept_examples.append(ex)

        self.examples = kept_examples
        after_count = len(self.examples)
        dropped_count = before_count - after_count

        stats: Dict[str, object] = {
            "enabled": True,
            "before": before_count,
            "after": after_count,
            "dropped": dropped_count,
            "dropped_ratio": (float(dropped_count) / float(before_count)) if before_count else 0.0,
            "dropped_by_segment": dropped_by_segment,
            "limits": segment_limits,
        }
        self.filter_stats = stats
        return stats

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self._build_item(self.examples[index])
