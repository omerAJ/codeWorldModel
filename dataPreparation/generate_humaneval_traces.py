#!/usr/bin/env python3
"""
Generate a HumanEval transition dataset for latent-state modeling.

For each executed line in each task, this script writes one JSONL record with:
- code_context: code before the current line (+ <CTX_SUM>)
- action: current line (+ <ACT_SUM>)
- current_state: locals before executing the line (+ <STATE_SUM>)
- next_state: state at the next trace event (+ <STATE_SUM>)
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, Iterable

from trace_generator import build_transition_examples, load_function, trace_function

DEFAULT_INPUT = (
    "/home/maincoder/Documents/maincode/SDS - Synthetic Data Experiments/code/"
    "SDS-SyntheticCodeData/benchmarks/humaneval/human-eval/data/"
    "HumanEval.jsonl/human-eval-v2-20210705.jsonl"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "humaneval_transition_dataset.jsonl",
)


class _CaptureCall(Exception):
    pass


def _build_module_code(prompt: str, canonical_solution: str) -> str:
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    code = prompt + canonical_solution
    if not code.endswith("\n"):
        code = code + "\n"
    return code


def _extract_first_call_args(test_src: str):
    test_globals: Dict[str, object] = {}
    exec(test_src, test_globals)

    check = test_globals.get("check")
    if not callable(check):
        raise ValueError("test harness did not define check(candidate)")

    captured: Dict[str, object] = {}

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


def _iter_tasks(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HumanEval line-transition datapoints with summary tokens.",
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="HumanEval JSONL file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--start", type=int, default=0, help="Start task index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End task index (exclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to process")
    parser.add_argument("--ctx-sum-token", default="<CTX_SUM>", help="Context summary token")
    parser.add_argument("--act-sum-token", default="<ACT_SUM>", help="Action summary token")
    parser.add_argument("--state-sum-token", default="<STATE_SUM>", help="State summary token")
    parser.add_argument("--error-log", default=None, help="Optional error log path")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata JSON output (counts and config)",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    error_f = open(args.error_log, "w", encoding="utf-8") if args.error_log else None

    selected_tasks = 0
    successful_tasks = 0
    written_examples = 0
    errors = 0

    with open(args.output, "w", encoding="utf-8") as out_f, tempfile.TemporaryDirectory() as temp_dir:
        for idx, task in enumerate(_iter_tasks(args.input)):
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break
            if args.limit is not None and selected_tasks >= args.limit:
                break

            selected_tasks += 1

            task_id = task.get("task_id", f"task_{idx}")
            entry_point = task["entry_point"]
            prompt = task["prompt"]
            canonical_solution = task["canonical_solution"]
            test_src = task["test"]

            try:
                func_args = _extract_first_call_args(test_src)
                module_code = _build_module_code(prompt, canonical_solution)
                module_path = os.path.join(temp_dir, f"task_{idx}.py")
                with open(module_path, "w", encoding="utf-8") as module_f:
                    module_f.write(module_code)

                func = load_function(module_path, entry_point)
                trace_data = trace_function(func, func_args, module_path)
                examples = build_transition_examples(
                    trace_data,
                    task_id=str(task_id),
                    entry_point=str(entry_point),
                    ctx_sum_token=args.ctx_sum_token,
                    act_sum_token=args.act_sum_token,
                    state_sum_token=args.state_sum_token,
                )

                for ex in examples:
                    out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

                written_examples += len(examples)
                successful_tasks += 1
            except Exception as exc:
                errors += 1
                msg = f"[{task_id}] {type(exc).__name__}: {exc}\n"
                if error_f:
                    error_f.write(msg)
                else:
                    sys.stderr.write(msg)

    if error_f:
        error_f.close()

    if args.metadata_output:
        metadata = {
            "input": args.input,
            "output": args.output,
            "selected_tasks": selected_tasks,
            "successful_tasks": successful_tasks,
            "written_examples": written_examples,
            "errors": errors,
            "ctx_sum_token": args.ctx_sum_token,
            "act_sum_token": args.act_sum_token,
            "state_sum_token": args.state_sum_token,
            "start": args.start,
            "end": args.end,
            "limit": args.limit,
        }
        with open(args.metadata_output, "w", encoding="utf-8") as meta_f:
            json.dump(metadata, meta_f, ensure_ascii=False, indent=2)
            meta_f.write("\n")

    sys.stdout.write(
        "Done. "
        f"Selected {selected_tasks} tasks, succeeded on {successful_tasks}, "
        f"wrote {written_examples} examples, errors: {errors}.\n"
    )


if __name__ == "__main__":
    main()
