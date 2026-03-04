from __future__ import annotations

import torch

from cwmodel.config import MaxLengthConfig, ModelConfig, TokenConfig
from cwmodel.data.collate import TransitionCollator
from cwmodel.data.dataset import TransitionDataset
from cwmodel.modeling.hf_setup import (
    apply_lora_if_enabled,
    infer_hidden_size,
    load_backbone,
    load_tokenizer,
    register_special_tokens,
)
from cwmodel.modeling.predictor import ConcatLatentPredictor
from cwmodel.modeling.world_model import LatentTransitionModel


def _build_model_and_batch(tiny_hf_model_dir):
    model_cfg = ModelConfig(
        local_path=str(tiny_hf_model_dir),
        use_4bit=False,
        enable_lora=True,
        lora_target_modules=["query", "key", "value", "dense"],
    )
    token_cfg = TokenConfig()

    tokenizer = load_tokenizer(model_cfg)
    _, summary_token_ids = register_special_tokens(tokenizer, token_cfg)

    backbone, _ = load_backbone(model_cfg, device=torch.device("cpu"))
    backbone.resize_token_embeddings(len(tokenizer))
    backbone, _ = apply_lora_if_enabled(backbone, model_cfg, is_kbit=False)

    hidden_size = infer_hidden_size(backbone)
    predictor = ConcatLatentPredictor(hidden_size=hidden_size)
    model = LatentTransitionModel(
        backbone=backbone,
        predictor=predictor,
        hidden_size=hidden_size,
    )
    model.to(torch.device("cpu"))

    dataset = TransitionDataset("data/humaneval_transition_1task.jsonl", token_cfg.wrappers)
    collator = TransitionCollator(
        tokenizer,
        MaxLengthConfig(
            code_context=256,
            action=96,
            current_state=128,
            next_action=96,
            next_state=128,
        ),
        summary_token_ids,
    )
    batch = collator([dataset[0], dataset[1]])
    return model, batch, hidden_size


def test_forward_output_shapes_and_loss(tiny_hf_model_dir) -> None:
    model, batch, hidden_size = _build_model_and_batch(tiny_hf_model_dir)

    output = model(batch)
    assert output.pred_next_state.shape == (2, hidden_size)
    assert output.pred_next_action.shape == (2, hidden_size)
    assert output.target_next_state.shape == (2, hidden_size)
    assert output.target_next_action.shape == (2, hidden_size)
    assert torch.isfinite(output.loss).item()


def test_only_lora_and_predictor_are_trainable(tiny_hf_model_dir) -> None:
    model, _, _ = _build_model_and_batch(tiny_hf_model_dir)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert any("lora_" in name for name in trainable)
    assert any(name.startswith("predictor.") for name in trainable)

    unexpected = [
        name
        for name in trainable
        if "lora_" not in name and not name.startswith("predictor.")
    ]
    assert not unexpected
