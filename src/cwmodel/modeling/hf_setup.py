from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from cwmodel.config import ModelConfig, TokenConfig

LOGGER = logging.getLogger(__name__)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    normalized = dtype_name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'")
    return mapping[normalized]


def load_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer_kwargs = {
        "trust_remote_code": model_cfg.trust_remote_code,
        "use_fast": model_cfg.use_fast_tokenizer,
    }
    if model_cfg.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.local_path,
            **tokenizer_kwargs,
        )
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        tokenizer_kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.local_path,
            **tokenizer_kwargs,
        )
    return tokenizer


def register_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_cfg: TokenConfig,
) -> Tuple[int, Dict[str, int]]:
    tokens = list(_iter_special_tokens(token_cfg))

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    added_count = tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    summary_ids = {
        "code_context": tokenizer.convert_tokens_to_ids(token_cfg.summary.context),
        "action": tokenizer.convert_tokens_to_ids(token_cfg.summary.action),
        "current_state": tokenizer.convert_tokens_to_ids(token_cfg.summary.state),
        "next_action": tokenizer.convert_tokens_to_ids(token_cfg.summary.action),
        "next_state": tokenizer.convert_tokens_to_ids(token_cfg.summary.state),
    }

    unk_id = tokenizer.unk_token_id
    missing = []
    for name, token_id in summary_ids.items():
        if token_id is None or token_id < 0:
            missing.append(name)
            continue
        if unk_id is not None and token_id == unk_id:
            missing.append(name)

    if missing:
        raise ValueError(f"Failed to resolve summary token ids for segments: {missing}")

    return added_count, summary_ids


def _iter_special_tokens(token_cfg: TokenConfig) -> Iterable[str]:
    ordered = [
        token_cfg.summary.context,
        token_cfg.summary.action,
        token_cfg.summary.state,
        token_cfg.wrappers.context_bos,
        token_cfg.wrappers.action_bos,
        token_cfg.wrappers.state_bos,
        token_cfg.wrappers.next_action_bos,
        token_cfg.wrappers.next_state_bos,
        token_cfg.wrappers.segment_sep,
    ]
    seen = set()
    for token in ordered:
        if token not in seen:
            seen.add(token)
            yield token


def load_backbone(
    model_cfg: ModelConfig,
    *,
    device: torch.device,
) -> tuple[PreTrainedModel, bool]:
    dtype = resolve_dtype(model_cfg.torch_dtype, device)

    use_4bit = model_cfg.use_4bit and device.type == "cuda"
    if model_cfg.use_4bit and device.type != "cuda":
        LOGGER.warning("use_4bit was requested but CUDA is unavailable. Falling back to non-quantized load.")

    kwargs = {
        "trust_remote_code": model_cfg.trust_remote_code,
    }

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover - import behavior varies by runtime
            LOGGER.warning("bitsandbytes is unavailable (%s). Falling back to non-quantized load.", exc)
            use_4bit = False
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    if not use_4bit:
        kwargs["torch_dtype"] = dtype

    model = AutoModel.from_pretrained(model_cfg.local_path, **kwargs)
    return model, use_4bit


def maybe_enable_gradient_checkpointing(model: PreTrainedModel, model_cfg: ModelConfig) -> None:
    if model_cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()


def apply_lora_if_enabled(
    model: PreTrainedModel,
    model_cfg: ModelConfig,
    *,
    is_kbit: bool,
) -> tuple[PreTrainedModel, Sequence[str]]:
    if not model_cfg.enable_lora:
        return model, []

    if is_kbit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=model_cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if model_cfg.gradient_checkpointing
            else None,
        )

    target_modules = _resolve_target_modules(model, model_cfg.lora_target_modules)
    lora_cfg = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_cfg)
    return model, target_modules


def _resolve_target_modules(model: PreTrainedModel, requested: Sequence[str]) -> List[str]:
    available_suffixes = _discover_linear_like_suffixes(model)
    selected = [name for name in requested if name in available_suffixes]

    if selected:
        return selected

    fallback = sorted(available_suffixes)
    if not fallback:
        raise ValueError("Could not find any linear-like modules for LoRA target_modules.")

    LOGGER.warning(
        "None of requested LoRA target modules were found. Falling back to all discovered linear-like suffixes: %s",
        fallback,
    )
    return fallback


def _discover_linear_like_suffixes(model: PreTrainedModel) -> set[str]:
    suffixes: set[str] = set()
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "linear" in cls_name
        if is_linear_like and "." in module_name:
            suffixes.add(module_name.split(".")[-1])
    return suffixes


def infer_hidden_size(model: PreTrainedModel) -> int:
    cfg = model.config
    for attr in ("hidden_size", "n_embd", "d_model", "dim"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to infer hidden size from model config")


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total
