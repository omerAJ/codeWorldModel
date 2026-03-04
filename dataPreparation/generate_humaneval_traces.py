#!/usr/bin/env python3
"""Generate HumanEval transition datapoints for latent-state modeling."""

import argparse
import ast
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

CATEGORY_STATE_CHANGING = "state_changing"
CATEGORY_BRANCH_NOOP = "branch_noop"
CATEGORY_LOOP_PROGRESS = "loop_progress"
CATEGORY_IDENTITY_NOOP = "identity_noop"

CATEGORY_ORDER = [
    CATEGORY_STATE_CHANGING,
    CATEGORY_BRANCH_NOOP,
    CATEGORY_LOOP_PROGRESS,
    CATEGORY_IDENTITY_NOOP,
]

CATEGORY_TO_QUOTA_ARG = {
    CATEGORY_STATE_CHANGING: "quota_state_changing",
    CATEGORY_BRANCH_NOOP: "quota_branch_noop",
    CATEGORY_LOOP_PROGRESS: "quota_loop_progress",
    CATEGORY_IDENTITY_NOOP: "quota_identity_noop",
}

FORMAT_HUMANEVAL = "humaneval"
FORMAT_MBPP = "mbpp"
FORMAT_AUTO = "auto"
FORMAT_CHOICES = [FORMAT_AUTO, FORMAT_HUMANEVAL, FORMAT_MBPP]


class _CaptureCall(Exception):
    pass


def _build_module_code(prompt: str, canonical_solution: str) -> str:
    if not prompt.endswith("\n"):
        prompt += "\n"
    code = prompt + canonical_solution
    if not code.endswith("\n"):
        code += "\n"
    return code


def _build_module_code_from_code(code: str) -> str:
    if not code.endswith("\n"):
        code += "\n"
    return code


def _extract_first_call_args(
    test_src: str, available_globals: Optional[Dict[str, object]] = None
) -> List[Any]:
    test_globals: Dict[str, object] = {}
    if available_globals:
        test_globals.update(available_globals)
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


def _function_names_in_source(module_code: str) -> List[str]:
    try:
        parsed = ast.parse(module_code)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def _extract_first_mbpp_call(
    test_list: List[str],
    test_setup_code: str,
    module_code: str,
    explicit_entry_point: Optional[str],
) -> Tuple[str, List[Any]]:
    module_globals: Dict[str, object] = {}
    exec(module_code, module_globals)

    candidate_names: List[str] = []
    if explicit_entry_point:
        candidate_names.append(str(explicit_entry_point))

    for name in _function_names_in_source(module_code):
        if name not in candidate_names:
            candidate_names.append(name)

    for name, value in module_globals.items():
        if (
            callable(value)
            and not name.startswith("__")
            and name not in candidate_names
        ):
            candidate_names.append(name)

    if not candidate_names:
        raise ValueError("MBPP task has no callable candidates in code")

    for candidate_name in candidate_names:
        captured: Dict[str, object] = {}
        test_globals: Dict[str, object] = dict(module_globals)

        if test_setup_code.strip():
            exec(test_setup_code, test_globals)

        def candidate_wrapper(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise _CaptureCall()

        test_globals[candidate_name] = candidate_wrapper

        try:
            for test_src in test_list:
                exec(test_src, test_globals)
        except _CaptureCall:
            if captured.get("kwargs"):
                raise ValueError("keyword arguments are not supported")
            return candidate_name, list(captured.get("args", ()))
        except Exception:
            continue

    raise ValueError("no callable MBPP test invocation found")


def _detect_format(task: Dict[str, Any], forced: str) -> str:
    if forced != FORMAT_AUTO:
        return forced
    if all(k in task for k in ("prompt", "canonical_solution", "test", "entry_point")):
        return FORMAT_HUMANEVAL
    if all(k in task for k in ("code", "test_list")):
        return FORMAT_MBPP
    raise ValueError("unable to auto-detect task format")


def _iter_tasks(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _sample_records(
    records: List[Dict[str, Any]],
    limit: int,
    sample_mode: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if limit < 0 or limit >= len(records):
        return list(records)
    if limit == 0:
        return []
    if sample_mode == "uniform":
        indices = sorted(rng.sample(range(len(records)), limit))
        return [records[i] for i in indices]
    return records[:limit]


def _category_of(example: Dict[str, Any]) -> str:
    if bool(example.get("is_pure_identity_noop")):
        return CATEGORY_IDENTITY_NOOP
    if bool(example.get("is_branch_noop")):
        return CATEGORY_BRANCH_NOOP
    if bool(example.get("is_loop_progress_only")):
        return CATEGORY_LOOP_PROGRESS
    if bool(example.get("is_noop_state")):
        return CATEGORY_IDENTITY_NOOP
    return CATEGORY_STATE_CHANGING


def _path_key(example: Dict[str, Any]) -> Tuple[int, str]:
    line_no = int(example.get("line_no", -1))
    next_line_no = example.get("next_line_no")
    edge = (
        str(next_line_no)
        if next_line_no is not None
        else str(example.get("next_event_type", "terminal")).upper()
    )
    return line_no, edge


def _apply_path_cap(
    records: List[Dict[str, Any]],
    cap: int,
    sample_mode: str,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], int]:
    if cap < 0:
        return list(records), 0

    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_path_key(record)].append(record)

    kept: List[Dict[str, Any]] = []
    for group_records in grouped.values():
        kept.extend(_sample_records(group_records, cap, sample_mode, rng))

    kept.sort(key=lambda row: int(row.get("step_index", 0)))
    return kept, len(records) - len(kept)


def _apply_max_overflow(
    grouped: Dict[str, List[Dict[str, Any]]],
    max_examples_per_task: int,
    overflow_policy: str,
    sample_mode: str,
    rng: random.Random,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    total = sum(len(v) for v in grouped.values())
    if max_examples_per_task <= 0 or total <= max_examples_per_task:
        return grouped, 0

    if overflow_policy == "error":
        raise ValueError(
            f"examples per task ({total}) exceed --max-examples-per-task={max_examples_per_task}"
        )

    trimmed: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CATEGORY_ORDER}

    if overflow_policy == "priority":
        remaining = max_examples_per_task
        for category in CATEGORY_ORDER:
            if remaining <= 0:
                break
            available = grouped.get(category, [])
            take = min(len(available), remaining)
            trimmed[category] = _sample_records(available, take, sample_mode, rng)
            remaining -= len(trimmed[category])
    else:  # proportional
        counts = {category: len(grouped.get(category, [])) for category in CATEGORY_ORDER}
        allocations: Dict[str, int] = {}
        fractions: List[Tuple[float, str]] = []
        used = 0

        for category in CATEGORY_ORDER:
            target = (counts[category] * max_examples_per_task) / total
            base = min(counts[category], int(target))
            allocations[category] = base
            used += base
            fractions.append((target - int(target), category))

        fractions.sort(reverse=True)
        idx = 0
        while used < max_examples_per_task:
            if idx >= len(fractions):
                idx = 0
            _, category = fractions[idx]
            if allocations[category] < counts[category]:
                allocations[category] += 1
                used += 1
            idx += 1

        for category in CATEGORY_ORDER:
            trimmed[category] = _sample_records(
                grouped.get(category, []),
                allocations[category],
                sample_mode,
                rng,
            )

    dropped = total - sum(len(v) for v in trimmed.values())
    return trimmed, dropped


def _compact_examples_for_task(
    examples: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {category: [] for category in CATEGORY_ORDER}
    for example in examples:
        grouped[_category_of(example)].append(example)

    dropped_by_rule = {
        "branch_path_cap": 0,
        "loop_progress_path_cap": 0,
        "quota_state_changing": 0,
        "quota_branch_noop": 0,
        "quota_loop_progress": 0,
        "quota_identity_noop": 0,
        "max_examples": 0,
    }

    grouped[CATEGORY_BRANCH_NOOP], dropped_by_rule["branch_path_cap"] = _apply_path_cap(
        grouped[CATEGORY_BRANCH_NOOP],
        args.branch_noop_cap_per_path,
        args.sample_mode,
        rng,
    )
    grouped[CATEGORY_LOOP_PROGRESS], dropped_by_rule["loop_progress_path_cap"] = _apply_path_cap(
        grouped[CATEGORY_LOOP_PROGRESS],
        args.loop_progress_cap_per_path,
        args.sample_mode,
        rng,
    )

    for category in CATEGORY_ORDER:
        quota = getattr(args, CATEGORY_TO_QUOTA_ARG[category])
        before = len(grouped[category])
        grouped[category] = _sample_records(grouped[category], quota, args.sample_mode, rng)
        dropped = before - len(grouped[category])
        if category == CATEGORY_STATE_CHANGING:
            dropped_by_rule["quota_state_changing"] = dropped
        elif category == CATEGORY_BRANCH_NOOP:
            dropped_by_rule["quota_branch_noop"] = dropped
        elif category == CATEGORY_LOOP_PROGRESS:
            dropped_by_rule["quota_loop_progress"] = dropped
        elif category == CATEGORY_IDENTITY_NOOP:
            dropped_by_rule["quota_identity_noop"] = dropped

    grouped, dropped_by_rule["max_examples"] = _apply_max_overflow(
        grouped,
        args.max_examples_per_task,
        args.overflow_policy,
        args.sample_mode,
        rng,
    )

    kept: List[Dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        kept.extend(grouped[category])
    kept.sort(key=lambda row: int(row.get("step_index", 0)))

    return kept, dropped_by_rule


def _add_compaction_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-examples-per-task",
        type=int,
        default=0,
        help="Hard cap per task after all quotas (0 = unlimited)",
    )
    parser.add_argument(
        "--quota-state-changing",
        type=int,
        default=-1,
        help="Max state-changing rows per task (-1 = unlimited)",
    )
    parser.add_argument(
        "--quota-branch-noop",
        type=int,
        default=-1,
        help="Max branch-noop rows per task (-1 = unlimited)",
    )
    parser.add_argument(
        "--quota-loop-progress",
        type=int,
        default=-1,
        help="Max loop-progress-only rows per task (-1 = unlimited)",
    )
    parser.add_argument(
        "--quota-identity-noop",
        type=int,
        default=-1,
        help="Max pure-identity-noop rows per task (-1 = unlimited)",
    )
    parser.add_argument(
        "--branch-noop-cap-per-path",
        type=int,
        default=-1,
        help="Cap branch-noop repeats per (line_no -> next edge) path (-1 = unlimited)",
    )
    parser.add_argument(
        "--loop-progress-cap-per-path",
        type=int,
        default=-1,
        help="Cap loop-progress-only repeats per (line_no -> next edge) path (-1 = unlimited)",
    )
    parser.add_argument(
        "--overflow-policy",
        choices=["priority", "proportional", "error"],
        default="priority",
        help="How to resolve conflicts when category totals exceed max-per-task",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "uniform"],
        default="first",
        help="Sampling mode for path caps and quotas",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sample-mode=uniform",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate line-transition datapoints from HumanEval/MBPP-style JSONL.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=FORMAT_CHOICES,
        default=FORMAT_AUTO,
        help="Input schema: auto (default), humaneval, or mbpp",
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input JSONL file")
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
    _add_compaction_flags(parser)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    error_f = open(args.error_log, "w", encoding="utf-8") if args.error_log else None

    selected_tasks = 0
    successful_tasks = 0
    written_examples = 0
    errors = 0
    raw_examples_total = 0
    format_counts = {
        FORMAT_HUMANEVAL: 0,
        FORMAT_MBPP: 0,
    }
    dropped_by_rule_totals = {
        "branch_path_cap": 0,
        "loop_progress_path_cap": 0,
        "quota_state_changing": 0,
        "quota_branch_noop": 0,
        "quota_loop_progress": 0,
        "quota_identity_noop": 0,
        "max_examples": 0,
    }

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
            task_text_raw = task.get("text")
            task_text: Optional[str] = None
            if task_text_raw is not None:
                normalized = str(task_text_raw).strip()
                if normalized:
                    task_text = normalized

            try:
                task_format = _detect_format(task, args.dataset_format)
                format_counts[task_format] += 1

                if task_format == FORMAT_HUMANEVAL:
                    entry_point = str(task["entry_point"])
                    module_code = _build_module_code(
                        str(task["prompt"]),
                        str(task["canonical_solution"]),
                    )
                    module_globals: Dict[str, object] = {}
                    exec(module_code, module_globals)
                    func_args = _extract_first_call_args(
                        str(task["test"]),
                        available_globals=module_globals,
                    )
                else:
                    module_code = _build_module_code_from_code(str(task["code"]))
                    test_list_raw = task.get("test_list", [])
                    if not isinstance(test_list_raw, list):
                        raise ValueError("MBPP task has non-list test_list")
                    test_list = [str(t) for t in test_list_raw if str(t).strip()]
                    if not test_list:
                        raise ValueError("MBPP task has empty test_list")
                    test_setup_code = str(task.get("test_setup_code", ""))
                    explicit_entry_point = task.get("entry_point")
                    entry_point, func_args = _extract_first_mbpp_call(
                        test_list=test_list,
                        test_setup_code=test_setup_code,
                        module_code=module_code,
                        explicit_entry_point=(
                            str(explicit_entry_point)
                            if explicit_entry_point is not None
                            else None
                        ),
                    )

                module_path = os.path.join(temp_dir, f"task_{idx}.py")
                with open(module_path, "w", encoding="utf-8") as module_f:
                    module_f.write(module_code)

                func = load_function(module_path, entry_point)
                trace_data = trace_function(func, func_args, module_path)
                raw_examples = build_transition_examples(
                    trace_data,
                    task_id=str(task_id),
                    entry_point=entry_point,
                    ctx_sum_token=args.ctx_sum_token,
                    act_sum_token=args.act_sum_token,
                    state_sum_token=args.state_sum_token,
                )

                task_rng = random.Random(args.seed + idx)
                examples, dropped_by_rule = _compact_examples_for_task(raw_examples, args, task_rng)

                raw_examples_total += len(raw_examples)
                for key in dropped_by_rule_totals:
                    dropped_by_rule_totals[key] += dropped_by_rule[key]

                for ex in examples:
                    if task_text is not None:
                        ex["task_text"] = task_text
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
            "dataset_format": args.dataset_format,
            "format_counts": format_counts,
            "selected_tasks": selected_tasks,
            "successful_tasks": successful_tasks,
            "raw_examples": raw_examples_total,
            "written_examples": written_examples,
            "dropped_examples": raw_examples_total - written_examples,
            "errors": errors,
            "ctx_sum_token": args.ctx_sum_token,
            "act_sum_token": args.act_sum_token,
            "state_sum_token": args.state_sum_token,
            "start": args.start,
            "end": args.end,
            "limit": args.limit,
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
            "dropped_by_rule": dropped_by_rule_totals,
        }
        with open(args.metadata_output, "w", encoding="utf-8") as meta_f:
            json.dump(metadata, meta_f, ensure_ascii=False, indent=2)
            meta_f.write("\n")

    sys.stdout.write(
        "Done. "
        f"Selected {selected_tasks} tasks, succeeded on {successful_tasks}, "
        f"raw examples {raw_examples_total}, wrote {written_examples}, errors: {errors}.\n"
    )


if __name__ == "__main__":
    main()
