#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def slugify_model_id(model_id: str) -> str:
    return model_id.strip().replace("/", "-").replace(" ", "-").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/update a Hugging Face model snapshot to a local directory",
    )
    parser.add_argument("--model-id", required=True, help="Hugging Face model ID")
    parser.add_argument("--local-dir", default=None, help="Local destination directory")
    parser.add_argument("--revision", default="main", help="Model revision (branch/tag/commit)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly refresh metadata and update files in local dir",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional; otherwise uses cached auth/env)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_dir = Path(args.local_dir or (Path("models") / slugify_model_id(args.model_id))).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    meta_path = local_dir / "snapshot_meta.json"
    if meta_path.exists() and not args.refresh:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        existing["skipped_download"] = True
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return

    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=args.token,
    )

    resolved_sha = None
    try:
        info = HfApi(token=args.token).model_info(args.model_id, revision=args.revision)
        resolved_sha = info.sha
    except Exception:
        resolved_sha = None

    meta = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_sha": resolved_sha,
        "snapshot_path": str(snapshot_path),
        "local_dir": str(local_dir),
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "refresh_requested": bool(args.refresh),
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
