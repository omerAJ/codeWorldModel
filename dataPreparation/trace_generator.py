#!/usr/bin/env python3
"""Helpers for generating state-transition examples from Python execution traces."""

import ast
import importlib.util
import inspect
import json
import linecache
import sys
from types import FrameType
from typing import Any, Dict, List, Optional, Set

TRACE_EVENTS = {"line", "return", "exception"}


def load_function(module_path: str, func_name: str):
    """Load a function object from a Python module file."""
    spec = importlib.util.spec_from_file_location("trace_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        func = getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"Function '{func_name}' not found in {module_path}") from exc

    if not callable(func):
        raise TypeError(f"'{func_name}' is not callable")
    return func


def trace_function(func: Any, func_args: List[Any], module_path: str) -> Dict[str, Any]:
    """Execute a function and capture line-level local-state snapshots."""
    try:
        raw_source_lines, source_start_lineno = inspect.getsourcelines(func)
        source_lines = [line.rstrip("\n") for line in raw_source_lines]
    except OSError:
        with open(module_path, "r", encoding="utf-8") as f:
            source_lines = f.read().splitlines()
        source_start_lineno = 1

    target_code = func.__code__
    events: List[Dict[str, Any]] = []

    def snapshot_locals(frame: FrameType) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for key, value in frame.f_locals.items():
            try:
                snapshot[key] = repr(value)
            except Exception:
                snapshot[key] = "<unrepr>"
        return snapshot

    def trace_lines(frame: FrameType, event: str, arg: Any):
        if frame.f_code is not target_code:
            return trace_lines

        line_src = linecache.getline(module_path, frame.f_lineno).rstrip("\n")
        if event == "line":
            events.append(
                {
                    "event": "line",
                    "lineno": frame.f_lineno,
                    "source": line_src,
                    "locals": snapshot_locals(frame),
                }
            )
        elif event == "return":
            events.append(
                {
                    "event": "return",
                    "lineno": frame.f_lineno,
                    "source": line_src,
                    "locals": snapshot_locals(frame),
                    "return_value": repr(arg),
                }
            )
        elif event == "exception":
            exc_type, exc_value, _ = arg
            events.append(
                {
                    "event": "exception",
                    "lineno": frame.f_lineno,
                    "source": line_src,
                    "locals": snapshot_locals(frame),
                    "exception_type": exc_type.__name__,
                    "exception_value": repr(exc_value),
                }
            )

        return trace_lines

    def trace_calls(frame: FrameType, event: str, arg: Any):
        if event == "call" and frame.f_code is target_code:
            return trace_lines
        return None

    exception_info: Optional[Dict[str, str]] = None
    sys.settrace(trace_calls)
    try:
        func(*func_args)
    except Exception as exc:
        exception_info = {"exception": repr(exc)}
    finally:
        sys.settrace(None)

    trace_data: Dict[str, Any] = {
        "module": module_path,
        "function": func.__name__,
        "args": [repr(a) for a in func_args],
        "source_lines": source_lines,
        "source_start_lineno": source_start_lineno,
        "events": events,
    }
    if exception_info is not None:
        trace_data.update(exception_info)
    return trace_data


def _append_summary_token(text: str, token: str) -> str:
    text = text.rstrip()
    if not text:
        return token
    return f"{text}\n{token}"


def _find_next_event(events: List[Dict[str, Any]], start_idx: int) -> Optional[Dict[str, Any]]:
    for idx in range(start_idx, len(events)):
        ev = events[idx]
        if ev.get("event") in TRACE_EVENTS:
            return ev
    return None


def _state_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": event["event"],
        "locals": event.get("locals", {}),
    }
    if event["event"] == "return":
        payload["return_value"] = event.get("return_value")
    elif event["event"] == "exception":
        payload["exception_type"] = event.get("exception_type")
        payload["exception_value"] = event.get("exception_value")
    return payload


def _context_for_line(source_lines: List[str], source_start_lineno: int, lineno: int) -> str:
    offset = lineno - source_start_lineno
    if offset <= 0:
        return ""
    if offset > len(source_lines):
        offset = len(source_lines)
    return "\n".join(source_lines[:offset]).rstrip("\n")


def _action_for_line(
    source_lines: List[str], source_start_lineno: int, event: Dict[str, Any]
) -> str:
    lineno = event["lineno"]
    offset = lineno - source_start_lineno
    if 0 <= offset < len(source_lines):
        return source_lines[offset].rstrip("\n")
    return event.get("source", "").rstrip("\n")


def _next_action_text(
    source_lines: List[str], source_start_lineno: int, next_event: Dict[str, Any]
) -> str:
    if next_event.get("event") == "line":
        return _action_for_line(source_lines, source_start_lineno, next_event)
    return f"<{str(next_event.get('event', 'terminal')).upper()}>"


def _is_control_flow_action(action_line: str) -> bool:
    stripped = action_line.strip()
    control_prefixes = (
        "if ",
        "if(",
        "elif ",
        "elif(",
        "for ",
        "while ",
        "match ",
        "case ",
        "try:",
        "except ",
        "finally:",
        "continue",
        "break",
        "return",
        "raise ",
    )
    return stripped.startswith(control_prefixes)


def _has_control_flow_effect(
    current_event: Dict[str, Any], next_event: Dict[str, Any], action_line: str
) -> bool:
    if _is_control_flow_action(action_line):
        return True
    if next_event.get("event") != "line":
        return True
    return int(next_event["lineno"]) != int(current_event["lineno"]) + 1


def _changed_local_keys(
    current_event: Dict[str, Any], next_event: Dict[str, Any]
) -> List[str]:
    current_locals = current_event.get("locals", {})
    next_locals = next_event.get("locals", {})
    all_keys = set(current_locals.keys()) | set(next_locals.keys())
    changed = [
        key for key in all_keys if current_locals.get(key) != next_locals.get(key)
    ]
    changed.sort()
    return changed


def _name_ids(node: ast.AST) -> List[str]:
    return sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})


def _source_to_function_node(source_lines: List[str]) -> Optional[ast.AST]:
    if not source_lines:
        return None
    try:
        module_node = ast.parse("\n".join(source_lines))
    except SyntaxError:
        return None
    for stmt in module_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return stmt
    return None


def _loop_header_names(loop_node: ast.AST) -> Set[str]:
    if isinstance(loop_node, (ast.For, ast.AsyncFor)):
        return set(_name_ids(loop_node.target))
    if isinstance(loop_node, ast.While):
        return set(_name_ids(loop_node.test))
    return set()


def _is_counter_step_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_counter_step_expr(node.operand)
    return False


def _self_update_targets(stmt: ast.AST) -> Set[str]:
    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.op, (ast.Add, ast.Sub)) and isinstance(stmt.target, ast.Name):
            return {stmt.target.id}
        return set()

    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return set()
        target = stmt.targets[0].id
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        if stmt.value is None or not isinstance(stmt.target, ast.Name):
            return set()
        target = stmt.target.id
        value = stmt.value
    else:
        return set()

    if not isinstance(value, ast.BinOp) or not isinstance(value.op, (ast.Add, ast.Sub)):
        return set()

    if isinstance(value.left, ast.Name) and value.left.id == target:
        return {target} if _is_counter_step_expr(value.right) else set()
    if isinstance(value.right, ast.Name) and value.right.id == target:
        if isinstance(value.op, ast.Add) and _is_counter_step_expr(value.left):
            return {target}
    return set()


def _build_loop_metadata(
    source_lines: List[str],
    source_start_lineno: int,
) -> Dict[str, Dict[int, Set[str]]]:
    metadata: Dict[str, Dict[int, Set[str]]] = {
        "header_vars": {},
        "enclosing_vars": {},
        "counter_update_vars": {},
    }

    function_node = _source_to_function_node(source_lines)
    if function_node is None:
        return metadata

    for node in ast.walk(function_node):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        rel_header_lineno = getattr(node, "lineno", None)
        rel_end_lineno = getattr(node, "end_lineno", rel_header_lineno)
        if rel_header_lineno is None or rel_end_lineno is None:
            continue

        header_names = _loop_header_names(node)
        abs_header_lineno = source_start_lineno + rel_header_lineno - 1
        metadata["header_vars"].setdefault(abs_header_lineno, set()).update(header_names)

        for rel_lineno in range(rel_header_lineno, rel_end_lineno + 1):
            abs_lineno = source_start_lineno + rel_lineno - 1
            metadata["enclosing_vars"].setdefault(abs_lineno, set()).update(header_names)

    for node in ast.walk(function_node):
        target_names = _self_update_targets(node)
        if not target_names:
            continue
        rel_lineno = getattr(node, "lineno", None)
        if rel_lineno is None:
            continue
        abs_lineno = source_start_lineno + rel_lineno - 1
        metadata["counter_update_vars"].setdefault(abs_lineno, set()).update(target_names)

    return metadata


def _for_target_names(action_line: str) -> List[str]:
    stripped = action_line.strip()
    if not (stripped.startswith("for ") or stripped.startswith("async for ")):
        return []

    parse_source = f"{stripped}\n    pass"
    stmt: Optional[ast.stmt] = None
    if stripped.startswith("async for "):
        parse_source = f"async def _f():\n    {stripped}\n        pass"

    try:
        parsed = ast.parse(parse_source)
    except SyntaxError:
        return []
    if not parsed.body:
        return []
    if stripped.startswith("async for "):
        func_node = parsed.body[0]
        if not isinstance(func_node, ast.AsyncFunctionDef) or not func_node.body:
            return []
        stmt = func_node.body[0]
    else:
        stmt = parsed.body[0]
    if not isinstance(stmt, (ast.For, ast.AsyncFor)):
        return []
    return _name_ids(stmt.target)


def _while_condition_names(action_line: str) -> List[str]:
    stripped = action_line.strip()
    if not stripped.startswith("while "):
        return []
    try:
        parsed = ast.parse(f"{stripped}\n    pass")
    except SyntaxError:
        return []
    if not parsed.body:
        return []
    stmt = parsed.body[0]
    if not isinstance(stmt, ast.While):
        return []
    return _name_ids(stmt.test)


def _is_loop_progress_only(
    action_line: str,
    current_event: Dict[str, Any],
    next_event: Dict[str, Any],
    changed_vars: List[str],
    loop_metadata: Dict[str, Dict[int, Set[str]]],
) -> bool:
    if not changed_vars:
        return False
    if current_event.get("event") != "line" or next_event.get("event") != "line":
        return False

    line_no = int(current_event["lineno"])
    changed_set = set(changed_vars)

    header_vars = loop_metadata.get("header_vars", {}).get(line_no, set())
    if header_vars and changed_set.issubset(header_vars):
        return True

    counter_update_vars = loop_metadata.get("counter_update_vars", {}).get(line_no, set())
    if counter_update_vars and changed_set.issubset(counter_update_vars):
        enclosing_vars = loop_metadata.get("enclosing_vars", {}).get(line_no, set())
        if not enclosing_vars or changed_set.issubset(enclosing_vars):
            return True

    # Fallback line-text checks for cases where AST metadata is unavailable.
    stripped = action_line.strip()
    if stripped.startswith("for ") or stripped.startswith("async for "):
        loop_vars = set(_for_target_names(action_line))
        return bool(loop_vars) and changed_set.issubset(loop_vars)
    if stripped.startswith("while "):
        condition_vars = set(_while_condition_names(action_line))
        return bool(condition_vars) and changed_set.issubset(condition_vars)
    return False


def build_transition_examples(
    trace_data: Dict[str, Any],
    task_id: str,
    entry_point: str,
    *,
    ctx_sum_token: str = "<CTX_SUM>",
    act_sum_token: str = "<ACT_SUM>",
    state_sum_token: str = "<STATE_SUM>",
) -> List[Dict[str, Any]]:
    """Convert a raw trace into per-line transition examples.

    Each example contains four model fields with trailing summary tokens:
    code_context, action, current_state, and next_state.
    Extra metadata fields include next_action and no-op tags.
    """
    source_lines: List[str] = trace_data.get("source_lines", [])
    source_start_lineno: int = int(trace_data.get("source_start_lineno", 1))
    events: List[Dict[str, Any]] = trace_data.get("events", [])
    loop_metadata = _build_loop_metadata(source_lines, source_start_lineno)

    examples: List[Dict[str, Any]] = []
    step_index = 0

    for idx, current_event in enumerate(events):
        if current_event.get("event") != "line":
            continue

        next_event = _find_next_event(events, idx + 1)
        if next_event is None:
            continue

        code_context = _context_for_line(
            source_lines,
            source_start_lineno,
            int(current_event["lineno"]),
        )
        action = _action_for_line(source_lines, source_start_lineno, current_event)
        next_action = _next_action_text(source_lines, source_start_lineno, next_event)
        is_noop_state = current_event.get("locals", {}) == next_event.get("locals", {})
        changed_vars = _changed_local_keys(current_event, next_event)
        has_control_flow_effect = _has_control_flow_effect(
            current_event, next_event, action
        )
        is_loop_progress_only = _is_loop_progress_only(
            action, current_event, next_event, changed_vars, loop_metadata
        )
        next_line_no = (
            int(next_event["lineno"]) if next_event.get("event") == "line" else None
        )

        current_state = json.dumps(
            _state_payload(current_event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        next_state = json.dumps(
            _state_payload(next_event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        examples.append(
            {
                "example_id": f"{task_id}:{step_index}",
                "task_id": task_id,
                "entry_point": entry_point,
                "step_index": step_index,
                "line_no": current_event["lineno"],
                "next_event_type": next_event["event"],
                "next_line_no": next_line_no,
                "code_context": _append_summary_token(code_context, ctx_sum_token),
                "action": _append_summary_token(action, act_sum_token),
                "next_action": _append_summary_token(next_action, act_sum_token),
                "current_state": _append_summary_token(current_state, state_sum_token),
                "next_state": _append_summary_token(next_state, state_sum_token),
                "changed_vars": changed_vars,
                "is_noop_state": is_noop_state,
                "is_branch_noop": is_noop_state and has_control_flow_effect,
                "is_pure_identity_noop": is_noop_state and not has_control_flow_effect,
                "is_loop_progress_only": is_loop_progress_only,
            }
        )
        step_index += 1

    return examples
