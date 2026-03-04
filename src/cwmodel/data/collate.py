from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from cwmodel.config import MaxLengthConfig


@dataclass
class SegmentBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    summary_token_id: int

    def to(self, device: torch.device) -> "SegmentBatch":
        return SegmentBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            summary_token_id=self.summary_token_id,
        )


@dataclass
class TransitionBatch:
    example_ids: List[str]
    task_ids: List[str]
    step_indices: List[int]
    code_context: SegmentBatch
    action: SegmentBatch
    current_state: SegmentBatch
    next_action: SegmentBatch
    next_state: SegmentBatch

    def to(self, device: torch.device) -> "TransitionBatch":
        return TransitionBatch(
            example_ids=self.example_ids,
            task_ids=self.task_ids,
            step_indices=self.step_indices,
            code_context=self.code_context.to(device),
            action=self.action.to(device),
            current_state=self.current_state.to(device),
            next_action=self.next_action.to(device),
            next_state=self.next_state.to(device),
        )


class TransitionCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: MaxLengthConfig,
        summary_token_ids: Dict[str, int],
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summary_token_ids = summary_token_ids

    def _encode_segment(
        self,
        texts: List[str],
        *,
        max_length: int,
        summary_token_id: int,
        segment_name: str,
    ) -> SegmentBatch:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Hard assertion required by training design to avoid invalid latent extraction.
        summary_present = (input_ids == summary_token_id).any(dim=1)
        if not bool(summary_present.all()):
            bad_indices = [idx for idx, ok in enumerate(summary_present.tolist()) if not ok]
            raise ValueError(
                f"Summary token missing after tokenization for segment '{segment_name}' "
                f"at batch indices {bad_indices}. Increase max_length.{segment_name}."
            )

        return SegmentBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            summary_token_id=summary_token_id,
        )

    def __call__(self, items: List[Dict[str, str]]) -> TransitionBatch:
        code_context_texts = [item["code_context"] for item in items]
        action_texts = [item["action"] for item in items]
        current_state_texts = [item["current_state"] for item in items]
        next_action_texts = [item["next_action"] for item in items]
        next_state_texts = [item["next_state"] for item in items]

        return TransitionBatch(
            example_ids=[item["example_id"] for item in items],
            task_ids=[item["task_id"] for item in items],
            step_indices=[int(item["step_index"]) for item in items],
            code_context=self._encode_segment(
                code_context_texts,
                max_length=self.max_length.code_context,
                summary_token_id=self.summary_token_ids["code_context"],
                segment_name="code_context",
            ),
            action=self._encode_segment(
                action_texts,
                max_length=self.max_length.action,
                summary_token_id=self.summary_token_ids["action"],
                segment_name="action",
            ),
            current_state=self._encode_segment(
                current_state_texts,
                max_length=self.max_length.current_state,
                summary_token_id=self.summary_token_ids["current_state"],
                segment_name="current_state",
            ),
            next_action=self._encode_segment(
                next_action_texts,
                max_length=self.max_length.next_action,
                summary_token_id=self.summary_token_ids["next_action"],
                segment_name="next_action",
            ),
            next_state=self._encode_segment(
                next_state_texts,
                max_length=self.max_length.next_state,
                summary_token_id=self.summary_token_ids["next_state"],
                segment_name="next_state",
            ),
        )
