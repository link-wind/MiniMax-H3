#!/usr/bin/env python3
"""Merge per-GPU continuation cache manifests into split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()

    for split in ("train", "validation", "test"):
        destination = root / split / "manifest.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for shard in sorted(root.glob("shard_*")):
                source = shard / split / "manifest.jsonl"
                if not source.exists():
                    continue
                with source.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        cache_path = record.get("cache_path")
                        if cache_path:
                            record["cache_path"] = str(
                                (source.parent / Path(cache_path).name).resolve()
                            )
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
        temporary.replace(destination)
        print(f"{split}: {count} records -> {destination}")


if __name__ == "__main__":
    main()
