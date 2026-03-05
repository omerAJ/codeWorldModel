#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


def _default_harness_dir() -> str:
    env_value = os.environ.get("BIGCODE_HARNESS_DIR")
    if env_value:
        return env_value
    repo_candidates = (
        "tools/bigcode-evaluation-harness",
        "bigcode-evaluation-harness",
    )
    for candidate in repo_candidates:
        if Path(candidate).exists():
            return candidate
    return "tools/bigcode-evaluation-harness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bigcode-evaluation-harness with a local PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--harness-dir",
        default=_default_harness_dir(),
        help="Path to a local clone of bigcode-evaluation-harness (or set BIGCODE_HARNESS_DIR).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional training run directory under outputs/; used to auto-resolve checkpoint backbone.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Optional explicit checkpoint step to use from --run-dir (e.g., 30 for step_000030).",
    )
    parser.add_argument(
        "--peft-model",
        default=None,
        help="Path to adapter directory (contains adapter_config.json and adapter_model.safetensors).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Base model ID/path. If omitted, inferred from adapter_config.json.",
    )
    parser.add_argument("--tasks", default="humaneval", help="Harness task pattern(s), comma-separated.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of benchmark problems.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size per worker.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of generations per problem.")
    parser.add_argument(
        "--max-length-generation",
        type=int,
        default=512,
        help="Maximum total tokens (prompt + generation).",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter.")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling (equivalent to --do_sample False).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="Precision passed to harness.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit in harness.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit in harness.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust_remote_code to harness model/tokenizer loading.",
    )
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Enable benchmark code execution (required for pass@k metrics).",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Generate only; skip execution/evaluation.",
    )
    parser.add_argument(
        "--metric-output-path",
        default=None,
        help="Where to write harness metrics JSON. Defaults to outputs/evals/<timestamp>_metrics.json.",
    )
    parser.add_argument(
        "--save-generations",
        action="store_true",
        help="Save generations JSON via harness.",
    )
    parser.add_argument(
        "--save-generations-path",
        default=None,
        help="Path for generations JSON. Defaults to outputs/evals/<timestamp>_generations.json.",
    )
    parser.add_argument(
        "--save-references",
        action="store_true",
        help="Save references JSON via harness.",
    )
    parser.add_argument(
        "--save-references-path",
        default=None,
        help="Path for references JSON (only used with --save-references).",
    )
    parser.add_argument(
        "--left-padding",
        action="store_true",
        help="Force left padding tokenization in harness (useful for some chat models).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional generation prefix (e.g. InCoder style language prefix).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional harness prompt variant, e.g. for HumanEvalPack tasks.",
    )
    parser.add_argument(
        "--resize-to-adapter-vocab",
        action="store_true",
        default=True,
        help="Before adapter load, resize base embeddings to adapter vocab size if needed.",
    )
    parser.add_argument(
        "--no-resize-to-adapter-vocab",
        dest="resize_to_adapter_vocab",
        action="store_false",
        help="Disable automatic embedding resize before PEFT adapter load.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra raw argument token to forward to harness (repeat per token).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved harness argv and exit without running.",
    )
    return parser.parse_args()


def _resolve_checkpoint_backbone(run_dir: Path, step: Optional[int]) -> Path:
    checkpoints_root = run_dir / "checkpoints"
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"No checkpoints directory found: {checkpoints_root}")

    if step is not None:
        candidate = checkpoints_root / f"step_{step:06d}" / "backbone"
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint backbone not found: {candidate}")
        return candidate

    step_dirs = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            step_num = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        backbone_dir = child / "backbone"
        if backbone_dir.exists():
            step_dirs.append((step_num, backbone_dir))

    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint backbone directories found in {checkpoints_root}")

    step_dirs.sort(key=lambda item: item[0])
    return step_dirs[-1][1]


def _infer_base_model_from_adapter(adapter_dir: Path) -> Optional[str]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("base_model_name_or_path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _adapter_vocab_size(adapter_dir: Path) -> Optional[int]:
    model_path = adapter_dir / "adapter_model.safetensors"
    if not model_path.exists():
        return None

    try:
        from safetensors.torch import load_file
    except Exception:
        return None

    state = load_file(str(model_path))
    candidate_keys = (
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.embed_tokens.weight",
    )
    for key in candidate_keys:
        tensor = state.get(key)
        if tensor is not None and getattr(tensor, "ndim", None) == 2:
            return int(tensor.shape[0])
    return None


def _patch_peft_vocab_resize(enabled: bool) -> None:
    if not enabled:
        return

    try:
        from peft import PeftModel
    except Exception as exc:
        print(
            f"[eval_bigcode_harness] Could not import peft for adapter vocab auto-resize: {exc}",
            file=sys.stderr,
        )
        return

    original_from_pretrained = PeftModel.from_pretrained

    def patched_from_pretrained(model, model_id, *args, **kwargs):
        adapter_path = Path(str(model_id))
        target_vocab = _adapter_vocab_size(adapter_path)
        if (
            target_vocab is not None
            and hasattr(model, "get_input_embeddings")
            and hasattr(model, "resize_token_embeddings")
        ):
            embeddings = model.get_input_embeddings()
            current_vocab = int(embeddings.weight.shape[0])
            if current_vocab != target_vocab:
                print(
                    "[eval_bigcode_harness] Resizing base embeddings for adapter compatibility: "
                    f"{current_vocab} -> {target_vocab}"
                )
                model.resize_token_embeddings(target_vocab)

        return original_from_pretrained(model, model_id, *args, **kwargs)

    PeftModel.from_pretrained = patched_from_pretrained


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _append_if(args: list[str], flag: str, condition: bool) -> None:
    if condition:
        args.append(flag)


def _append_pair(args: list[str], key: str, value: Optional[str | int | float]) -> None:
    if value is None:
        return
    args.extend([key, str(value)])


def _validate_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd().resolve()

    harness_dir = Path(args.harness_dir).expanduser().resolve()
    harness_main = harness_dir / "main.py"
    _validate_paths([harness_dir, harness_main])

    peft_model: Optional[Path]
    if args.peft_model:
        peft_model = Path(args.peft_model).expanduser().resolve()
    elif args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        peft_model = _resolve_checkpoint_backbone(run_dir, args.checkpoint_step)
    else:
        raise ValueError("Provide either --peft-model or --run-dir.")

    _validate_paths([peft_model])

    model_name = args.model
    if not model_name:
        inferred = _infer_base_model_from_adapter(peft_model)
        if inferred:
            model_name = inferred
        else:
            raise ValueError(
                "Could not infer base model from adapter_config.json. Pass --model explicitly."
            )

    model_path_candidate = Path(str(model_name)).expanduser()
    if not model_path_candidate.is_absolute():
        local_model_path = (launch_cwd / model_path_candidate).resolve()
        if local_model_path.exists():
            model_name = str(local_model_path)

    eval_root = Path("outputs") / "evals"
    eval_root.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    metric_output_path = args.metric_output_path or str(eval_root / f"{ts}_metrics.json")
    save_generations_path = args.save_generations_path or str(eval_root / f"{ts}_generations.json")
    save_references_path = args.save_references_path or str(eval_root / f"{ts}_references.json")

    metric_output_path = str((launch_cwd / metric_output_path).resolve())
    save_generations_path = str((launch_cwd / save_generations_path).resolve())
    save_references_path = str((launch_cwd / save_references_path).resolve())

    harness_argv = [str(harness_main)]
    _append_pair(harness_argv, "--model", model_name)
    _append_pair(harness_argv, "--peft_model", str(peft_model))
    _append_pair(harness_argv, "--tasks", args.tasks)
    _append_pair(harness_argv, "--prefix", args.prefix)
    _append_pair(harness_argv, "--batch_size", args.batch_size)
    _append_pair(harness_argv, "--n_samples", args.n_samples)
    _append_pair(harness_argv, "--max_length_generation", args.max_length_generation)
    _append_pair(harness_argv, "--temperature", args.temperature)
    _append_pair(harness_argv, "--top_p", args.top_p)
    _append_pair(harness_argv, "--top_k", args.top_k)
    _append_pair(harness_argv, "--do_sample", False if args.greedy else True)
    _append_pair(harness_argv, "--precision", args.precision)
    _append_pair(harness_argv, "--metric_output_path", metric_output_path)
    _append_pair(harness_argv, "--save_generations_path", save_generations_path)
    _append_pair(harness_argv, "--save_references_path", save_references_path)
    _append_pair(harness_argv, "--prompt", args.prompt)
    _append_pair(harness_argv, "--limit", args.limit)
    _append_if(harness_argv, "--load_in_4bit", args.load_in_4bit)
    _append_if(harness_argv, "--load_in_8bit", args.load_in_8bit)
    _append_if(harness_argv, "--trust_remote_code", args.trust_remote_code)
    _append_if(harness_argv, "--allow_code_execution", args.allow_code_execution)
    _append_if(harness_argv, "--generation_only", args.generation_only)
    _append_if(harness_argv, "--save_generations", args.save_generations)
    _append_if(harness_argv, "--save_references", args.save_references)
    _append_if(harness_argv, "--left_padding", args.left_padding)
    harness_argv.extend(args.extra_arg)

    print("[eval_bigcode_harness] Harness dir:", harness_dir)
    print("[eval_bigcode_harness] Base model:", model_name)
    print("[eval_bigcode_harness] PEFT adapter:", peft_model)
    print("[eval_bigcode_harness] Tasks:", args.tasks)
    print("[eval_bigcode_harness] metric_output_path:", metric_output_path)
    if args.save_generations:
        print("[eval_bigcode_harness] save_generations_path:", save_generations_path)
    if args.save_references:
        print("[eval_bigcode_harness] save_references_path:", save_references_path)
    print("[eval_bigcode_harness] resize_to_adapter_vocab:", args.resize_to_adapter_vocab)
    print("[eval_bigcode_harness] argv:", " ".join(harness_argv))

    if args.dry_run:
        return

    _patch_peft_vocab_resize(args.resize_to_adapter_vocab)

    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        sys.argv = harness_argv
        os.chdir(harness_dir)
        runpy.run_path(str(harness_main), run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


if __name__ == "__main__":
    main()
