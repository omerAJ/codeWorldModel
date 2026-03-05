#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cwmodel.config import TrainConfig
from cwmodel.data.collate import TransitionCollator
from cwmodel.data.dataset import TransitionDataset
from cwmodel.modeling.hf_setup import (
    apply_lora_if_enabled,
    count_trainable_parameters,
    infer_hidden_size,
    load_backbone,
    load_tokenizer,
    maybe_enable_gradient_checkpointing,
    register_special_tokens,
)
from cwmodel.modeling.predictor import ConcatLatentPredictor
from cwmodel.modeling.world_model import LatentTransitionModel
from cwmodel.training.logging import MetricsLogger
from cwmodel.training.loop import train_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent transition predictor")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with dotted.path=value syntax",
    )
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: str, run_name: str | None) -> Path:
    root = Path(output_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_name = run_name or f"run_{timestamp}"
    run_dir = root / final_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config, overrides=args.set)

    seed_everything(cfg.training.seed)
    device = choose_device()

    run_dir = make_run_dir(cfg.output.root_dir, cfg.output.run_name)
    cfg.dump_yaml(run_dir / "resolved_config.yaml")

    tokenizer = load_tokenizer(cfg.model)
    _, summary_token_ids = register_special_tokens(tokenizer, cfg.tokens)

    backbone, is_kbit = load_backbone(cfg.model, device=device)
    if hasattr(backbone, "resize_token_embeddings"):
        backbone.resize_token_embeddings(len(tokenizer))

    maybe_enable_gradient_checkpointing(backbone, cfg.model)
    backbone, lora_targets = apply_lora_if_enabled(backbone, cfg.model, is_kbit=is_kbit)

    hidden_size = infer_hidden_size(backbone)
    predictor = ConcatLatentPredictor(
        hidden_size=hidden_size,
        hidden_multiplier=cfg.model.predictor_hidden_multiplier,
        dropout=cfg.model.predictor_dropout,
    )

    model = LatentTransitionModel(
        backbone=backbone,
        predictor=predictor,
        hidden_size=hidden_size,
        target_eval_mode=cfg.model.target_eval_mode,
    )

    if is_kbit:
        model.move_predictor_to_backbone_device()
        model_device = model.get_backbone_device()
    else:
        model.to(device)
        model_device = device

    trainable_count, total_count = count_trainable_parameters(model)

    dataset_paths = cfg.data.resolved_paths()
    dataset = TransitionDataset(dataset_paths, cfg.tokens.wrappers)
    overlength_filter_stats = None
    if cfg.data.filter_overlength:
        overlength_filter_stats = dataset.filter_overlength_examples(
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )
        print(
            "[train.py] Applied overlength filter: "
            f"dropped {overlength_filter_stats['dropped']} / {overlength_filter_stats['before']} "
            f"({100.0 * float(overlength_filter_stats['dropped_ratio']):.2f}%). "
            f"Remaining: {overlength_filter_stats['after']}."
        )
        if int(overlength_filter_stats["after"]) == 0:
            raise ValueError(
                "All samples were filtered by data.filter_overlength. "
                "Increase data.max_length.* or disable data.filter_overlength."
            )
    collator = TransitionCollator(tokenizer, cfg.data.max_length, summary_token_ids)
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": cfg.data.batch_size,
        "shuffle": cfg.data.shuffle,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory and model_device.type == "cuda",
        "collate_fn": collator,
    }

    if cfg.data.num_workers > 0:
        # Avoid unsafe fork behavior once CUDA/quantized modules are initialized.
        dataloader_kwargs["multiprocessing_context"] = "spawn"
        dataloader_kwargs["persistent_workers"] = True

    try:
        dataloader = DataLoader(**dataloader_kwargs)
    except Exception as exc:
        if cfg.data.num_workers <= 0:
            raise
        print(
            f"[train.py] DataLoader worker initialization failed with num_workers={cfg.data.num_workers} "
            f"(reason: {exc}). Falling back to num_workers=0 for stability."
        )
        dataloader_kwargs.pop("multiprocessing_context", None)
        dataloader_kwargs.pop("persistent_workers", None)
        dataloader_kwargs["num_workers"] = 0
        dataloader = DataLoader(**dataloader_kwargs)

    tokenizer.save_pretrained(run_dir / "tokenizer")

    logger = MetricsLogger(
        run_dir,
        enable_tensorboard=cfg.logging.tensorboard,
        enable_jsonl=cfg.logging.jsonl,
    )

    try:
        result = train_loop(
            model=model,
            dataloader=dataloader,
            config=cfg,
            device=model_device,
            output_dir=run_dir,
            metrics_logger=logger,
        )
    finally:
        logger.close()

    summary = {
        "run_dir": str(run_dir),
        "dataset_path": dataset_paths[0] if len(dataset_paths) == 1 else None,
        "dataset_paths": dataset_paths,
        "dataset_rows": len(dataset),
        "overlength_filter": overlength_filter_stats,
        "global_steps": result.global_steps,
        "final_loss": result.final_loss,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "lora_target_modules": list(lora_targets),
        "kbit": bool(is_kbit),
        "device": str(model_device),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
