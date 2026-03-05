#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cwmodel.config import TrainConfig
from cwmodel.modeling.hf_setup import load_tokenizer, register_special_tokens


@dataclass
class DatasetOverThresholdStats:
    dataset: str
    rows: int
    current_over: int
    next_over: int

    @property
    def current_pct(self) -> float:
        return (100.0 * self.current_over / self.rows) if self.rows else 0.0

    @property
    def next_pct(self) -> float:
        return (100.0 * self.next_over / self.rows) if self.rows else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "current_over": self.current_over,
            "next_over": self.next_over,
            "current_pct": self.current_pct,
            "next_pct": self.next_pct,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frequency of state fields over a token threshold.",
    )
    parser.add_argument(
        "--config",
        default="configs/train.maincoder_1b.yaml",
        help="Training config path; used for tokenizer and default data paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Config override(s), dotted.path=value.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional dataset JSONL paths. If omitted, uses data.path/data.paths from config.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=512,
        help="Token threshold for over-threshold counts.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix (without extension). Defaults to outputs/state_over_<threshold>.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional cap on rows processed per dataset.",
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


def _short_label(path: Path) -> str:
    if path.name == "transition_dataset.jsonl":
        return path.parent.name
    if path.name.endswith(".jsonl"):
        return path.name.replace(".jsonl", "")
    return path.name


def _compute_stats(
    dataset_paths: List[Path],
    tokenizer,
    threshold: int,
    *,
    max_rows_per_dataset: int | None,
) -> List[DatasetOverThresholdStats]:
    def token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    stats: List[DatasetOverThresholdStats] = []

    total_rows = 0
    total_current = 0
    total_next = 0
    for path in dataset_paths:
        rows = 0
        current_over = 0
        next_over = 0

        for record in _iter_jsonl(path, max_rows_per_dataset):
            rows += 1
            current_state = str(record.get("current_state", ""))
            next_state = str(record.get("next_state", ""))
            if token_len(current_state) > threshold:
                current_over += 1
            if token_len(next_state) > threshold:
                next_over += 1

        stats.append(
            DatasetOverThresholdStats(
                dataset=str(path),
                rows=rows,
                current_over=current_over,
                next_over=next_over,
            )
        )
        total_rows += rows
        total_current += current_over
        total_next += next_over

    stats.append(
        DatasetOverThresholdStats(
            dataset="COMBINED_ALL",
            rows=total_rows,
            current_over=total_current,
            next_over=total_next,
        )
    )
    return stats


def _plot(stats: List[DatasetOverThresholdStats], threshold: int, output_prefix: Path, dpi: int) -> None:
    labels = []
    current_pcts = []
    next_pcts = []
    rows = []
    current_counts = []
    next_counts = []

    for item in stats:
        if item.dataset == "COMBINED_ALL":
            labels.append("COMBINED_ALL")
        else:
            labels.append(_short_label(Path(item.dataset)))
        rows.append(item.rows)
        current_counts.append(item.current_over)
        next_counts.append(item.next_over)
        current_pcts.append(item.current_pct)
        next_pcts.append(item.next_pct)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    bars_current = ax.bar(x - width / 2, current_pcts, width, label="current_state > threshold")
    bars_next = ax.bar(x + width / 2, next_pcts, width, label="next_state > threshold")

    ax.set_title(f"State Length Frequency Over {threshold} Tokens")
    ax.set_ylabel("Percentage of rows (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right")

    ymax = max(max(current_pcts, default=0.0), max(next_pcts, default=0.0))
    ax.set_ylim(0, max(1.0, ymax * 1.25))

    def annotate(bars, counts):
        for bar, count, total in zip(bars, counts, rows):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.3f}%\n({count}/{total})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    annotate(bars_current, current_counts)
    annotate(bars_next, next_counts)

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cfg = TrainConfig.from_yaml(args.config, overrides=args.set)
    dataset_paths = [Path(p) for p in (args.path or cfg.data.resolved_paths())]
    if not dataset_paths:
        raise ValueError("No dataset paths resolved.")
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

    tokenizer = load_tokenizer(cfg.model)
    register_special_tokens(tokenizer, cfg.tokens)

    stats = _compute_stats(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        threshold=args.threshold,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )

    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else Path("outputs") / f"state_over_{args.threshold}"
    )
    _plot(stats, args.threshold, output_prefix, args.dpi)

    payload = {
        "threshold": args.threshold,
        "tokenizer_path": cfg.model.local_path,
        "dataset_paths": [str(p) for p in dataset_paths],
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "stats": [item.to_dict() for item in stats],
        "plot_png": str(output_prefix.with_suffix(".png")),
        "plot_svg": str(output_prefix.with_suffix(".svg")),
    }
    json_path = output_prefix.with_name(output_prefix.name + "_stats.json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
