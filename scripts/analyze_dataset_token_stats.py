#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cwmodel.config import TrainConfig
from cwmodel.modeling.hf_setup import load_tokenizer, register_special_tokens


@dataclass
class FieldTokenStats:
    token_lengths: List[int] = field(default_factory=list)
    missing: int = 0

    def add(self, length: int) -> None:
        self.token_lengths.append(int(length))

    def add_missing(self) -> None:
        self.missing += 1

    def to_summary(self) -> Dict[str, Any]:
        if not self.token_lengths:
            return {
                "count": 0,
                "missing": self.missing,
                "mean": None,
                "stdev": None,
                "min": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }

        values = sorted(self.token_lengths)
        count = len(values)
        return {
            "count": count,
            "missing": self.missing,
            "mean": round(float(statistics.mean(values)), 3),
            "stdev": round(float(statistics.pstdev(values)), 3),
            "min": int(values[0]),
            "p50": _percentile(values, 50.0),
            "p90": _percentile(values, 90.0),
            "p95": _percentile(values, 95.0),
            "p99": _percentile(values, 99.0),
            "max": int(values[-1]),
        }


def _percentile(sorted_values: List[int], pct: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    value = sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
    return int(round(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token-count stats for dataset fields.")
    parser.add_argument(
        "--config",
        default="configs/train.maincoder_1b.yaml",
        help="Path to training YAML; used to resolve model tokenizer and data paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Config overrides using dotted.path=value syntax (same as train.py).",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional explicit dataset JSONL path(s). Overrides config data.path/data.paths if provided.",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=[],
        help="Optional subset of fields to analyze. Defaults to all fields encountered.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional cap on rows processed per dataset.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output file for machine-readable stats JSON.",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path, max_rows: int | None) -> Iterable[Dict[str, Any]]:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)
            rows += 1
            if max_rows is not None and rows >= max_rows:
                return


def _to_tokenizable_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _summarize_dataset(
    path: Path,
    tokenizer,
    *,
    selected_fields: set[str] | None,
    max_rows: int | None,
) -> Dict[str, Any]:
    stats: Dict[str, FieldTokenStats] = defaultdict(FieldTokenStats)
    num_rows = 0
    all_fields = set()

    for record in _iter_jsonl(path, max_rows):
        num_rows += 1
        row_fields = set(record.keys())
        if selected_fields is None:
            new_fields = row_fields - all_fields
            for field_name in new_fields:
                # Backfill rows seen before this field first appears.
                stats[field_name].missing += (num_rows - 1)
        all_fields.update(row_fields)

        fields_to_check: Iterable[str]
        if selected_fields is None:
            fields_to_check = all_fields
        else:
            fields_to_check = selected_fields

        for field_name in fields_to_check:
            if field_name not in record:
                stats[field_name].add_missing()
                continue
            text = _to_tokenizable_text(record.get(field_name))
            if text is None:
                stats[field_name].add_missing()
                continue

            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            stats[field_name].add(token_count)

    field_summaries = {
        field_name: stats[field_name].to_summary()
        for field_name in sorted(stats.keys())
    }

    return {
        "path": str(path),
        "rows": num_rows,
        "fields_seen": sorted(all_fields),
        "field_token_stats": field_summaries,
    }


def _format_field_row(field_name: str, summary: Dict[str, Any]) -> str:
    return (
        f"{field_name:24} "
        f"count={summary['count']:8} "
        f"missing={summary['missing']:8} "
        f"mean={str(summary['mean']):>8} "
        f"p50={str(summary['p50']):>6} "
        f"p90={str(summary['p90']):>6} "
        f"p95={str(summary['p95']):>6} "
        f"p99={str(summary['p99']):>6} "
        f"max={str(summary['max']):>6}"
    )


def _summarize_combined(
    dataset_paths: List[Path],
    tokenizer,
    *,
    selected_fields: set[str] | None,
    max_rows: int | None,
) -> Dict[str, Any]:
    stats: Dict[str, FieldTokenStats] = defaultdict(FieldTokenStats)
    num_rows = 0
    all_fields = set()

    for path in dataset_paths:
        for record in _iter_jsonl(path, max_rows):
            num_rows += 1
            row_fields = set(record.keys())
            if selected_fields is None:
                new_fields = row_fields - all_fields
                for field_name in new_fields:
                    stats[field_name].missing += (num_rows - 1)
            all_fields.update(row_fields)

            fields_to_check: Iterable[str]
            if selected_fields is None:
                fields_to_check = all_fields
            else:
                fields_to_check = selected_fields

            for field_name in fields_to_check:
                if field_name not in record:
                    stats[field_name].add_missing()
                    continue
                text = _to_tokenizable_text(record.get(field_name))
                if text is None:
                    stats[field_name].add_missing()
                    continue
                token_count = len(tokenizer.encode(text, add_special_tokens=False))
                stats[field_name].add(token_count)

    field_summaries = {
        field_name: stats[field_name].to_summary()
        for field_name in sorted(stats.keys())
    }
    return {
        "rows": num_rows,
        "fields_seen": sorted(all_fields),
        "field_token_stats": field_summaries,
    }


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config, overrides=args.set)

    dataset_paths = [Path(p) for p in (args.path or cfg.data.resolved_paths())]
    if not dataset_paths:
        raise ValueError("No dataset paths resolved.")

    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

    selected_fields = set(args.fields) if args.fields else None

    tokenizer = load_tokenizer(cfg.model)
    register_special_tokens(tokenizer, cfg.tokens)

    per_dataset = []
    for path in dataset_paths:
        per_dataset.append(
            _summarize_dataset(
                path,
                tokenizer,
                selected_fields=selected_fields,
                max_rows=args.max_rows_per_dataset,
            )
        )

    combined = _summarize_combined(
        dataset_paths,
        tokenizer,
        selected_fields=selected_fields,
        max_rows=args.max_rows_per_dataset,
    )

    payload = {
        "config_path": args.config,
        "resolved_dataset_paths": [str(p) for p in dataset_paths],
        "tokenizer_path": cfg.model.local_path,
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "selected_fields": sorted(selected_fields) if selected_fields else None,
        "per_dataset": per_dataset,
        "combined": combined,
    }

    print("\n=== Token Count Summary (Per Dataset) ===")
    for ds in per_dataset:
        print(f"\n[{ds['path']}] rows={ds['rows']}")
        for field_name, summary in ds["field_token_stats"].items():
            print(_format_field_row(field_name, summary))

    print("\n=== Token Count Summary (Combined) ===")
    print(f"rows={combined['rows']}")
    for field_name, summary in combined["field_token_stats"].items():
        print(_format_field_row(field_name, summary))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
