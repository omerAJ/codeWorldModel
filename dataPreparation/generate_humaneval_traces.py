#!/usr/bin/env python3
"""
Generate CWM-formatted execution traces for HumanEval tasks.

This script reads a HumanEval JSONL file, builds a temporary module for each
task (prompt + canonical_solution), extracts the first candidate call from the
test harness, and writes a trace file per task.
"""

import argparse
import json
import os
import re
import sys
import tempfile

from trace_generator import format_cwm_trace, load_function, trace_function

DEFAULT_INPUT = (
    "/home/maincoder/Documents/maincode/SDS - Synthetic Data Experiments/code/"
    "SDS-SyntheticCodeData/benchmarks/humaneval/human-eval/data/"
    "HumanEval.jsonl/human-eval-v2-20210705.jsonl"
)


class _CaptureCall(Exception):
    pass


def _sanitize_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_id)


def _build_module_code(prompt: str, canonical_solution: str) -> str:
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    code = prompt + canonical_solution
    if not code.endswith("\n"):
        code = code + "\n"
    return code


def _extract_first_call_args(test_src: str):
    test_globals = {}
    exec(test_src, test_globals)
    check = test_globals.get("check")
    if not callable(check):
        raise ValueError("test harness did not define check(candidate)")
    captured = {}

    def candidate_wrapper(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _CaptureCall()

    try:
        check(candidate_wrapper)
    except _CaptureCall:
        pass
    if "args" not in captured:
        raise ValueError("no candidate() call found in test harness")
    if captured.get("kwargs"):
        raise ValueError("keyword arguments are not supported")
    return list(captured["args"])


def _iter_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-task CWM traces for HumanEval JSONL tasks. "
            "Each output file contains the traced function definition followed "
            "by trace frames."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Path to HumanEval JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(os.path.dirname(__file__), "humaneval_traces"),
        help="Directory to write trace files",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based, inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (0-based, exclusive)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of traces to generate",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have output files",
    )
    parser.add_argument(
        "--include-call-context",
        action="store_true",
        help="Include a synthetic main() wrapper in the trace context",
    )
    parser.add_argument(
        "--error-log",
        default=None,
        help="Optional path to write error messages",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    error_f = open(args.error_log, "w", encoding="utf-8") if args.error_log else None

    written = 0
    errors = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, task in enumerate(_iter_tasks(args.input)):
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break
            if args.limit is not None and written >= args.limit:
                break

            task_id = task.get("task_id", f"task_{idx}")
            entry_point = task["entry_point"]
            prompt = task["prompt"]
            canonical_solution = task["canonical_solution"]
            test_src = task["test"]

            out_name = f"{_sanitize_task_id(task_id)}.trace"
            out_path = os.path.join(args.output_dir, out_name)
            if args.skip_existing and os.path.exists(out_path):
                continue

            try:
                func_args = _extract_first_call_args(test_src)
                module_code = _build_module_code(prompt, canonical_solution)
                module_path = os.path.join(
                    temp_dir, f"{_sanitize_task_id(task_id)}.py"
                )
                with open(module_path, "w", encoding="utf-8") as f:
                    f.write(module_code)

                func = load_function(module_path, entry_point)
                trace_data = trace_function(func, func_args, module_path)
                trace_text = format_cwm_trace(
                    trace_data["source"],
                    trace_data["function"],
                    trace_data["args"],
                    trace_data["events"],
                    include_call_context=args.include_call_context,
                )

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(trace_text)
                written += 1
            except Exception as exc:
                errors += 1
                msg = f"[{task_id}] {type(exc).__name__}: {exc}\n"
                if error_f:
                    error_f.write(msg)
                else:
                    sys.stderr.write(msg)

    if error_f:
        error_f.close()
    sys.stdout.write(f"Done. Wrote {written} traces with {errors} errors.\n")


if __name__ == "__main__":
    main()
