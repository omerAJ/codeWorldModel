#!/usr/bin/env python3
"""
Helpers for generating state-transition examples from Python execution traces.
"""

import importlib.util
import inspect
import json
import linecache
import sys
from types import FrameType
from typing import Any, Dict, List, Optional


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
        if ev.get("event") in {"line", "return", "exception"}:
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
    """
    source_lines: List[str] = trace_data.get("source_lines", [])
    source_start_lineno: int = int(trace_data.get("source_start_lineno", 1))
    events: List[Dict[str, Any]] = trace_data.get("events", [])

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
                "code_context": _append_summary_token(code_context, ctx_sum_token),
                "action": _append_summary_token(action, act_sum_token),
                "current_state": _append_summary_token(current_state, state_sum_token),
                "next_state": _append_summary_token(next_state, state_sum_token),
            }
        )
        step_index += 1

    return examples
