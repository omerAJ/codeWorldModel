#!/usr/bin/env python3
"""Run trace generation + viewer build with per-run artifact tracking."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(text: str) -> str:
    chars: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "dataset"


def _summary_from_dataset(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_task = Counter(str(r.get("task_id", "")) for r in rows)
    noop = sum(bool(r.get("is_noop_state")) for r in rows)
    branch = sum(bool(r.get("is_branch_noop")) for r in rows)
    identity = sum(bool(r.get("is_pure_identity_noop")) for r in rows)
    loop = sum(bool(r.get("is_loop_progress_only")) for r in rows)

    return {
        "rows": len(rows),
        "tasks": len(by_task),
        "max_rows_per_task": max(by_task.values()) if by_task else 0,
        "max_task_share": (max(by_task.values()) / len(rows)) if by_task and rows else 0.0,
        "state_changing_rows": len(rows) - noop,
        "noop_rows": noop,
        "branch_noop_rows": branch,
        "identity_noop_rows": identity,
        "loop_progress_rows": loop,
        "top_tasks_by_rows": by_task.most_common(10),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run transition dataset generation with tracked artifacts.",
    )
    parser.add_argument("--input", "-i", required=True, help="Input HumanEval-style JSONL")
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "humaneval", "mbpp"],
        default="auto",
        help="Input schema for generator",
    )
    parser.add_argument(
        "--run-root",
        default="data/runs",
        help="Directory where run folders are created",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name (default: <UTC timestamp>_<input stem>)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional viewer title (default: run name)",
    )
    parser.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Skip HTML viewer generation",
    )
    parser.add_argument("--max-examples-per-task", type=int, default=120)
    parser.add_argument("--quota-state-changing", type=int, default=70)
    parser.add_argument("--quota-branch-noop", type=int, default=25)
    parser.add_argument("--quota-loop-progress", type=int, default=20)
    parser.add_argument("--quota-identity-noop", type=int, default=5)
    parser.add_argument("--branch-noop-cap-per-path", type=int, default=2)
    parser.add_argument("--loop-progress-cap-per-path", type=int, default=2)
    parser.add_argument("--overflow-policy", choices=["priority", "proportional", "error"], default="priority")
    parser.add_argument("--sample-mode", choices=["first", "uniform"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    run_name = args.run_name or f"{_timestamp()}_{_safe_name(input_path.stem)}"
    run_root = Path(args.run_root).expanduser().resolve()
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "transition_dataset.jsonl"
    metadata_path = run_dir / "transition_dataset.meta.json"
    errors_path = run_dir / "transition_dataset.errors.log"
    viewer_path = run_dir / "transition_dataset.viewer.html"

    generate_cmd = [
        sys.executable,
        "dataPreparation/generate_humaneval_traces.py",
        "--input",
        str(input_path),
        "--dataset-format",
        str(args.dataset_format),
        "--output",
        str(dataset_path),
        "--metadata-output",
        str(metadata_path),
        "--error-log",
        str(errors_path),
        "--max-examples-per-task",
        str(args.max_examples_per_task),
        "--quota-state-changing",
        str(args.quota_state_changing),
        "--quota-branch-noop",
        str(args.quota_branch_noop),
        "--quota-loop-progress",
        str(args.quota_loop_progress),
        "--quota-identity-noop",
        str(args.quota_identity_noop),
        "--branch-noop-cap-per-path",
        str(args.branch_noop_cap_per_path),
        "--loop-progress-cap-per-path",
        str(args.loop_progress_cap_per_path),
        "--overflow-policy",
        str(args.overflow_policy),
        "--sample-mode",
        str(args.sample_mode),
        "--seed",
        str(args.seed),
        "--start",
        str(args.start),
    ]
    if args.end is not None:
        generate_cmd.extend(["--end", str(args.end)])
    if args.limit is not None:
        generate_cmd.extend(["--limit", str(args.limit)])

    start_time = datetime.now(timezone.utc)
    subprocess.run(generate_cmd, check=True)

    viewer_cmd: List[str] = []
    if not args.skip_viewer:
        viewer_cmd = [
            sys.executable,
            "dataPreparation/visualize_transition_dataset.py",
            "--input",
            str(dataset_path),
            "--output",
            str(viewer_path),
            "--title",
            args.title or run_name,
        ]
        subprocess.run(viewer_cmd, check=True)

    end_time = datetime.now(timezone.utc)
    summary = _summary_from_dataset(dataset_path)

    manifest = {
        "run_name": run_name,
        "started_at_utc": start_time.isoformat(),
        "finished_at_utc": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "input_dataset": str(input_path),
        "generator_args": {
            "max_examples_per_task": args.max_examples_per_task,
            "quota_state_changing": args.quota_state_changing,
            "quota_branch_noop": args.quota_branch_noop,
            "quota_loop_progress": args.quota_loop_progress,
            "quota_identity_noop": args.quota_identity_noop,
            "branch_noop_cap_per_path": args.branch_noop_cap_per_path,
            "loop_progress_cap_per_path": args.loop_progress_cap_per_path,
            "overflow_policy": args.overflow_policy,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "dataset_format": args.dataset_format,
            "start": args.start,
            "end": args.end,
            "limit": args.limit,
        },
        "commands": {
            "generate": generate_cmd,
            "viewer": viewer_cmd,
        },
        "artifacts": {
            "dataset_jsonl": str(dataset_path),
            "metadata_json": str(metadata_path),
            "error_log": str(errors_path),
            "viewer_html": str(viewer_path) if viewer_cmd else None,
        },
        "summary": summary,
    }

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Error log: {errors_path}")
    if viewer_cmd:
        print(f"Viewer: {viewer_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
