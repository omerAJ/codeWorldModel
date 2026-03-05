from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


REQUIRED_FIELDS = [
    "example_id",
    "task_id",
    "step_index",
    "code_context",
    "action",
    "current_state",
    "next_action",
    "next_state",
]


@dataclass
class TransitionExample:
    example_id: str
    task_id: str
    step_index: int
    code_context: str
    action: str
    current_state: str
    next_action: str
    next_state: str
    task_text: Optional[str] = None
    entry_point: Optional[str] = None
    line_no: Optional[int] = None
    next_line_no: Optional[int] = None

    @classmethod
    def from_record(cls, record: Dict[str, Any], *, line_no: Optional[int] = None) -> "TransitionExample":
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            where = f" at line {line_no}" if line_no is not None else ""
            raise ValueError(f"Missing required fields {missing}{where}")

        return cls(
            example_id=str(record["example_id"]),
            task_id=str(record["task_id"]),
            step_index=int(record["step_index"]),
            code_context=str(record["code_context"]),
            action=str(record["action"]),
            current_state=str(record["current_state"]),
            next_action=str(record["next_action"]),
            next_state=str(record["next_state"]),
            task_text=(str(record["task_text"]) if record.get("task_text") is not None else None),
            entry_point=(str(record["entry_point"]) if record.get("entry_point") is not None else None),
            line_no=(int(record["line_no"]) if record.get("line_no") is not None else None),
            next_line_no=(int(record["next_line_no"]) if record.get("next_line_no") is not None else None),
        )
