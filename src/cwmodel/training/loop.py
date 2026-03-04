from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from cwmodel.config import TrainConfig
from cwmodel.modeling.world_model import LatentTransitionModel
from cwmodel.training.logging import MetricsLogger


@dataclass
class TrainResult:
    global_steps: int
    final_loss: float


def _save_checkpoint(
    model: LatentTransitionModel,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    step: int,
) -> Path:
    ckpt_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    backbone_dir = ckpt_dir / "backbone"
    model.backbone.save_pretrained(backbone_dir)
    torch.save(model.predictor.state_dict(), ckpt_dir / "predictor.pt")

    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        ckpt_dir / "trainer_state.pt",
    )
    return ckpt_dir


def train_loop(
    *,
    model: LatentTransitionModel,
    dataloader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    output_dir: str | Path,
    metrics_logger: Optional[MetricsLogger] = None,
) -> TrainResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found")

    optimizer = AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.optimizer_beta1, config.training.optimizer_beta2),
    )

    updates_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * config.training.epochs)
    warmup_steps = int(total_updates * config.training.warmup_ratio)

    scheduler = get_scheduler(
        config.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    amp_enabled = config.training.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    try:
        scaler = torch.amp.GradScaler(scaler_device, enabled=scaler_enabled)
    except TypeError:
        # Backward compatibility for older torch signatures.
        scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    last_loss = 0.0

    for epoch in range(config.training.epochs):
        model.train()

        for step_in_epoch, batch in enumerate(dataloader, start=1):
            batch = batch.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(batch)
                loss = output.loss
                loss_for_backward = loss / config.training.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_step = (
                step_in_epoch % config.training.gradient_accumulation_steps == 0
                or step_in_epoch == len(dataloader)
            )
            if not should_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            clip_grad_norm_(trainable_params, config.training.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            last_loss = float(loss.detach().item())

            if metrics_logger is not None and global_step % config.training.log_every_steps == 0:
                metrics_logger.log(
                    global_step,
                    {
                        "train/loss": last_loss,
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/epoch": float(epoch + 1),
                    },
                )

            if (
                config.training.save_every_steps > 0
                and global_step % config.training.save_every_steps == 0
            ):
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    out_dir,
                    step=global_step,
                )

    _save_checkpoint(model, optimizer, scheduler, out_dir, step=global_step)

    return TrainResult(global_steps=global_step, final_loss=last_loss)
