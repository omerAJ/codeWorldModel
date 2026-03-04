from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from cwmodel.data.collate import SegmentBatch, TransitionBatch


@dataclass
class WorldModelOutput:
    pred_next_state: torch.Tensor
    pred_next_action: torch.Tensor
    target_next_state: torch.Tensor
    target_next_action: torch.Tensor
    loss: torch.Tensor


class LatentTransitionModel(nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        predictor: nn.Module,
        *,
        hidden_size: int,
        target_eval_mode: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.hidden_size = hidden_size
        self.target_eval_mode = target_eval_mode
        self._latent_source_module: nn.Module | None = None

    def _extract_summary_latent(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        summary_token_id: int,
    ) -> torch.Tensor:
        summary_mask = input_ids.eq(summary_token_id)
        if not bool(summary_mask.any(dim=1).all()):
            raise ValueError("Summary token missing in batch during latent extraction")

        positions = summary_mask.float().argmax(dim=1).to(dtype=torch.long)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    def _encode_segment(
        self,
        segment: SegmentBatch,
        *,
        requires_grad: bool,
        eval_mode_for_pass: bool,
    ) -> torch.Tensor:
        was_training = self.backbone.training
        if eval_mode_for_pass and self.target_eval_mode:
            self.backbone.eval()

        try:
            with torch.set_grad_enabled(requires_grad):
                candidate_modules = self._ordered_candidate_modules()
                for module in candidate_modules:
                    outputs = self._run_forward(module, segment)
                    last_hidden = self._extract_last_hidden(outputs)
                    if last_hidden is None:
                        continue
                    self._latent_source_module = module
                    return self._extract_summary_latent(
                        last_hidden,
                        segment.input_ids,
                        segment.summary_token_id,
                    )

                raise AttributeError(
                    "Unable to extract hidden states from backbone outputs. "
                    "Tried cached backbone and inner model candidates."
                )
        finally:
            if eval_mode_for_pass and self.target_eval_mode:
                self.backbone.train(was_training)

    def _ordered_candidate_modules(self) -> list[nn.Module]:
        candidates = []
        seen = set()

        def add(module: nn.Module | None) -> None:
            if module is None:
                return
            marker = id(module)
            if marker in seen:
                return
            seen.add(marker)
            candidates.append(module)

        add(self._latent_source_module)
        add(self.backbone)

        get_base_model = getattr(self.backbone, "get_base_model", None)
        if callable(get_base_model):
            try:
                add(get_base_model())
            except Exception:
                pass

        add(getattr(self.backbone, "model", None))

        # Some wrappers (for example PEFT + CausalLM) expose nested `.model`.
        for module in list(candidates):
            add(getattr(module, "model", None))

        return candidates

    def _run_forward(self, module: nn.Module, segment: SegmentBatch):
        kwargs = {
            "input_ids": segment.input_ids,
            "attention_mask": segment.attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        try:
            return module(**kwargs)
        except TypeError:
            kwargs.pop("output_hidden_states", None)
            kwargs.pop("return_dict", None)
            return module(**kwargs)

    @staticmethod
    def _extract_last_hidden(outputs) -> torch.Tensor | None:
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if isinstance(last_hidden, torch.Tensor):
            return last_hidden

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) > 0:
            candidate = hidden_states[-1]
            if isinstance(candidate, torch.Tensor):
                return candidate

        if isinstance(outputs, dict):
            value = outputs.get("last_hidden_state")
            if isinstance(value, torch.Tensor):
                return value
            hs = outputs.get("hidden_states")
            if hs is not None and len(hs) > 0 and isinstance(hs[-1], torch.Tensor):
                return hs[-1]

        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if isinstance(first, torch.Tensor) and first.dim() == 3:
                return first

        return None

    def forward(self, batch: TransitionBatch) -> WorldModelOutput:
        ctx_latent = self._encode_segment(
            batch.code_context,
            requires_grad=True,
            eval_mode_for_pass=False,
        )
        action_latent = self._encode_segment(
            batch.action,
            requires_grad=True,
            eval_mode_for_pass=False,
        )
        current_state_latent = self._encode_segment(
            batch.current_state,
            requires_grad=True,
            eval_mode_for_pass=False,
        )

        fused_latent = torch.cat(
            [ctx_latent, action_latent, current_state_latent],
            dim=-1,
        )
        predicted_concat = self.predictor(fused_latent)
        pred_next_state, pred_next_action = torch.split(
            predicted_concat,
            self.hidden_size,
            dim=-1,
        )

        with torch.no_grad():
            target_next_action = self._encode_segment(
                batch.next_action,
                requires_grad=False,
                eval_mode_for_pass=True,
            )
            target_next_state = self._encode_segment(
                batch.next_state,
                requires_grad=False,
                eval_mode_for_pass=True,
            )

        loss_state = F.mse_loss(pred_next_state, target_next_state)
        loss_action = F.mse_loss(pred_next_action, target_next_action)
        loss = loss_state + loss_action

        return WorldModelOutput(
            pred_next_state=pred_next_state,
            pred_next_action=pred_next_action,
            target_next_state=target_next_state,
            target_next_action=target_next_action,
            loss=loss,
        )

    def get_backbone_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def get_default_device(self) -> torch.device:
        return next(self.parameters()).device

    def predictor_device(self) -> torch.device:
        return next(self.predictor.parameters()).device

    def move_predictor_to_backbone_device(self) -> None:
        self.predictor.to(self.get_backbone_device())
