#!/usr/bin/env python3
"""
Minimal Python trace dataset generator.

This script demonstrates how to reproduce a simplified version of the
CWM function‑level tracing pipeline. Given a Python module file and a
function name with its arguments, it executes the function in a
controlled environment and records a line‑by‑line trace of local
variable states. The output is a single CWM-formatted text trace
containing the traced function definition followed by trace frames.

Usage:
    python trace_generator.py path/to/example.py compute_sum 4

The first argument is the path to the Python module containing the
function. The second argument is the function name. All remaining
arguments are passed to the function; they will be parsed using
ast.literal_eval so you can pass numbers, strings, lists, etc.

The resulting trace will be printed to stdout. Use `--output` (or shell
redirection) to save it to a file.

This code is intentionally simple and omits many complexities of the
full CWM tracing pipeline (e.g., handling nested calls, external
side effects, variable compression, etc.) but it illustrates the core
idea of capturing program state transitions.
"""

import argparse
import importlib.util
import inspect
import json
import linecache
import sys
from ast import literal_eval
from types import FrameType
from typing import Any, Dict, List, Optional


def load_function(module_path: str, func_name: str):
    """Load a function object from a given module file and function name."""
    spec = importlib.util.spec_from_file_location("trace_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        func = getattr(module, func_name)
    except AttributeError:
        raise AttributeError(f"Function '{func_name}' not found in {module_path}")
    if not callable(func):
        raise TypeError(f"'{func_name}' is not a callable object")
    return func


def trace_function(func: Any, func_args: List[Any], module_path: str) -> Dict[str, Any]:
    """Execute a function under a tracer and record a simplified execution trace.

    Returns a dictionary with the source context and a list of events.
    Each event includes the event type (call, line, return, exception),
    the line number, the source code line, and the local variables at
    that point. For return and exception events, the return value or
    exception info is also included.
    """
    # Capture just the target function's source for context (not the full module).
    # Fall back to the full module source only if we cannot retrieve the function.
    try:
        source_code = inspect.getsource(func)
    except OSError:
        with open(module_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    # We identify the target function's code object to filter events
    target_code = func.__code__
    events: List[Dict[str, Any]] = []

    def local_vars_snapshot(frame: FrameType) -> Dict[str, str]:
        """Return a copy of local variables with repr() for JSON encoding."""
        snapshot = {}
        for k, v in frame.f_locals.items():
            # Represent values using repr to preserve information like strings
            try:
                snapshot[k] = repr(v)
            except Exception:
                snapshot[k] = "<unrepr>"
        return snapshot

    def tracer(frame: FrameType, event: str, arg: Any):
        # We only care about events for the target function
        if event == "call":
            # Start tracing when we enter the target function
            if frame.f_code is target_code:
                # Record the call with initial locals (arguments)
                event_record = {
                    "event": "call",
                    "lineno": frame.f_lineno,
                    "source": linecache.getline(module_path, frame.f_lineno).rstrip(),
                    "locals": local_vars_snapshot(frame),
                }
                events.append(event_record)
                return trace_lines  # Return inner tracer for line/return events
            else:
                # Do not trace other function calls
                return None
        return None

    def trace_lines(frame: FrameType, event: str, arg: Any):
        # Only handle events for our target function
        if frame.f_code is not target_code:
            return trace_lines
        if event == "line":
            event_record = {
                "event": "line",
                "lineno": frame.f_lineno,
                "source": linecache.getline(module_path, frame.f_lineno).rstrip(),
                "locals": local_vars_snapshot(frame),
            }
            events.append(event_record)
        elif event == "return":
            event_record = {
                "event": "return",
                "lineno": frame.f_lineno,
                "source": linecache.getline(module_path, frame.f_lineno).rstrip(),
                "locals": local_vars_snapshot(frame),
                "return_value": repr(arg),
            }
            events.append(event_record)
        elif event == "exception":
            # arg is (exc_type, exc_value, traceback)
            exc_type, exc_value, _ = arg
            event_record = {
                "event": "exception",
                "lineno": frame.f_lineno,
                "source": linecache.getline(module_path, frame.f_lineno).rstrip(),
                "locals": local_vars_snapshot(frame),
                "exception_type": repr(exc_type.__name__),
                "exception_value": repr(exc_value),
            }
            events.append(event_record)
        # Continue tracing
        return trace_lines

    # Install the tracer
    sys.settrace(tracer)
    exception_info: Optional[Dict[str, Any]] = None
    try:
        func(*func_args)
    except Exception as e:
        # Exceptions within the function will trigger the exception event
        exception_info = {"exception": repr(e)}
    finally:
        # Disable tracing
        sys.settrace(None)

    # Compose final trace object
    trace_data = {
        "module": module_path,
        "function": func.__name__,
        "args": [repr(a) for a in func_args],
        "source": source_code,
        "events": events,
    }
    if exception_info is not None:
        trace_data.update(exception_info)
    return trace_data


def format_cwm_trace(
    context_code: str,
    func_name: str,
    args_repr: List[str],
    events: List[Dict[str, Any]],
    *,
    include_call_context: bool = False,
) -> str:
    """Format a trace in the CWM execution trace representation.

    Args:
        context_code: The Python source context (typically the function definition).
        func_name: The name of the traced function.
        args_repr: The list of arguments (as repr strings) passed to the function.
        events: A list of event dictionaries produced by trace_function().

    Returns:
        A single string containing the context code followed by the trace in
        CWM format, including the required separator tokens. The trace starts
        immediately after the `<|trace_context_start|>` token and ends with a
        final `<|frame_sep|>` token. Each event uses `<|frame_sep|>` followed
        by one of `<|call_sep|>`, `<|line_sep|>`, `<|return_sep|>`, or
        `<|exception_sep|>`.
    """
    context = context_code.strip("\n")
    if include_call_context:
        # Build a simple main function call context with a START_OF_TRACE marker.
        # This mimics how CWM examples embed the entry point of the trace.
        args_str = ", ".join(args_repr)
        main_code = f"def main(): # << START_OF_TRACE\n    return {func_name}({args_str})\n"
        context = context + "\n" + main_code
    # Begin trace with context and trace_context_start token
    # Build the trace content: context, trace context token, then event lines
    lines: List[str] = []
    # Include the context code first
    lines.append(context)
    # The trace_context_start token indicates that the code context has ended
    lines.append("<|trace_context_start|>")
    # Build each event as a single line string
    # Track previous locals to compress unchanged values with ".."
    prev_locals: Dict[str, str] = {}
    for ev in events:
        if ev["event"] in {"call", "line"}:
            # Build a compressed locals dictionary
            full_locals: Dict[str, str] = ev["locals"]  # as repr strings
            compressed: Dict[str, str] = {}
            for key, value in full_locals.items():
                if key in prev_locals and prev_locals[key] == value:
                    compressed[key] = ".."
                else:
                    compressed[key] = value
            # Update prev_locals for next comparison
            prev_locals = full_locals.copy()
            if ev["event"] == "call":
                line = (
                    "<|frame_sep|><|call_sep|>"
                    + json.dumps(compressed, ensure_ascii=False)
                    + "<|action_sep|>"
                    + ev["source"].rstrip()
                )
                lines.append(line)
            else:  # line event
                line = (
                    "<|frame_sep|><|line_sep|>"
                    + json.dumps(compressed, ensure_ascii=False)
                    + "<|action_sep|>"
                    + ev["source"].rstrip()
                )
                lines.append(line)
        elif ev["event"] == "return":
            # Return event resets prev_locals because execution ended
            prev_locals = {}
            line = (
                "<|frame_sep|><|return_sep|><|action_sep|>"
                + ev["source"].rstrip()
                + "<|arg_sep|>"
                + json.dumps(ev["return_value"], ensure_ascii=False)
            )
            lines.append(line)
        elif ev["event"] == "exception":
            # Exception event resets prev_locals
            prev_locals = {}
            line = (
                "<|frame_sep|><|exception_sep|><|action_sep|>"
                + ev["source"].rstrip()
                + "<|arg_sep|>"
                + json.dumps(ev["exception_value"], ensure_ascii=False)
            )
            lines.append(line)
        else:
            # Skip unknown events
            continue
    # Finally, append a closing frame separator to mark the end of the trace
    lines.append("<|frame_sep|>")
    # Join with newline separators for readability
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Python execution trace in CWM format. The output includes "
            "the source context (the traced function definition), a "
            "<|trace_context_start|> token, and frame separators (<|frame_sep|>) "
            "with event tags (<|call_sep|>, <|line_sep|>, <|return_sep|>, "
            "<|exception_sep|>)."
        )
    )
    parser.add_argument(
        "module_path",
        help="Path to the Python file containing the function to trace",
    )
    parser.add_argument(
        "function_name",
        help="Name of the function to trace",
    )
    parser.add_argument(
        "args",
        nargs='+',
        help=(
            "Arguments to pass to the function (literal eval). Example: python "
            "trace_generator.py example.py factorial 5"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Optional output file to save the CWM trace. If not provided, the "
            "trace is printed to stdout."
        ),
    )
    parser.add_argument(
        "--include-call-context",
        action="store_true",
        help=(
            "Include a synthetic main() wrapper that calls the traced function "
            "(adds a START_OF_TRACE marker)."
        ),
    )
    args = parser.parse_args()

    # Load function and parse arguments
    func = load_function(args.module_path, args.function_name)
    func_args = [literal_eval(a) for a in args.args]

    # Generate the raw trace data (source context and events)
    trace_data = trace_function(func, func_args, args.module_path)

    # Format the trace in CWM representation
    cwm_trace = format_cwm_trace(
        trace_data["source"],
        trace_data["function"],
        trace_data["args"],
        trace_data["events"],
        include_call_context=args.include_call_context,
    )

    # Output the trace either to file or stdout
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(cwm_trace)
    else:
        print(cwm_trace)


if __name__ == "__main__":
    main()
