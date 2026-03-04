from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_train_script_smoke_run(tmp_path: Path, tiny_hf_model_dir: Path) -> None:
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "smoke_config.yaml"

    config = {
        "model": {
            "local_path": str(tiny_hf_model_dir),
            "trust_remote_code": True,
            "use_fast_tokenizer": True,
            "enable_lora": True,
            "use_4bit": False,
            "gradient_checkpointing": False,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "lora_target_modules": ["query", "key", "value", "dense"],
            "torch_dtype": "float32",
            "target_eval_mode": True,
            "predictor_dropout": 0.0,
            "predictor_hidden_multiplier": 1,
        },
        "tokens": {
            "summary": {
                "context": "<CTX_SUM>",
                "action": "<ACT_SUM>",
                "state": "<STATE_SUM>",
            },
            "wrappers": {
                "context_bos": "<CTX_BOS>",
                "action_bos": "<ACT_BOS>",
                "state_bos": "<STATE_BOS>",
                "next_action_bos": "<NEXT_ACT_BOS>",
                "next_state_bos": "<NEXT_STATE_BOS>",
                "segment_sep": "<SEG_SEP>",
            },
        },
        "data": {
            "path": "data/humaneval_transition_1task.jsonl",
            "batch_size": 4,
            "num_workers": 0,
            "shuffle": False,
            "pin_memory": False,
            "max_length": {
                "code_context": 256,
                "action": 96,
                "current_state": 128,
                "next_action": 96,
                "next_state": 128,
            },
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "log_every_steps": 1,
            "save_every_steps": 10,
            "seed": 123,
            "amp": False,
            "optimizer_beta1": 0.9,
            "optimizer_beta2": 0.999,
            "lr_scheduler": "linear",
            "warmup_ratio": 0.0,
        },
        "logging": {
            "tensorboard": True,
            "jsonl": True,
        },
        "output": {
            "root_dir": str(output_root),
            "run_name": "smoke",
        },
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "scripts/train.py", "--config", str(config_path)]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Smoke train failed (code {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    run_dir = output_root / "smoke"
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "tensorboard").exists()

    checkpoint_root = run_dir / "checkpoints"
    assert checkpoint_root.exists()
    assert any(path.is_dir() for path in checkpoint_root.iterdir())
