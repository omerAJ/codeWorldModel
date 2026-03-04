from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        enable_tensorboard: bool,
        enable_jsonl: bool,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer: Optional[SummaryWriter] = None
        if enable_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

        self.jsonl_path: Optional[Path] = None
        if enable_jsonl:
            self.jsonl_path = self.run_dir / "metrics.jsonl"

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        payload = {
            "step": int(step),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **{k: float(v) for k, v in metrics.items()},
        }

        if self.tb_writer is not None:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, float(value), step)

        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
