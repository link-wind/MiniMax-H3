#!/usr/bin/env python3
"""Rebuild a shot-level index from the **deduplicated** per-source shot inventory.

The training index (`data_with_face_and_speech_and_caption.jsonl`) is built from
overlapping multi-shot windows of the same episodes, and every record was
re-annotated, so the same underlying shot appears in several records with
*different* captions.  Content hashing therefore cannot deduplicate it
(24682 shots in 5000 records produce 24682 distinct caption bodies); the only
reliable identity is the source-level shot range.

`plan_shot_pairs.py --shot-cache` already extracted that inventory: one row per
`source_id` with the ordered shot lengths in output frames.  This script turns
it back into a *synthetic* index whose records are single sources, so the real
training sampler (`iter_shot_continuation_samples`) can be replayed over the
duplicate-free shot sequences.  The output is for **statistics only** -- it has
no media paths and cannot build a latent cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIDEO_FPS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shot-cache", type=Path, required=True,
                        help="shot_lengths_cache.jsonl emitted by plan_shot_pairs.py")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--min-shots", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = 0
    shots = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.shot_cache.open() as source, args.output_jsonl.open("w", encoding="utf-8") as out:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lengths = [int(value) for value in row.get("shots_out") or () if int(value) > 0]
            if len(lengths) < args.min_shots:
                continue
            source_id = str(row["source_id"])
            cursor = 0
            entries = []
            for index, frames in enumerate(lengths, start=1):
                start_sec = cursor / VIDEO_FPS
                cursor += frames
                end_sec = cursor / VIDEO_FPS
                entries.append(f"[Shot {index}/{len(lengths)} | {start_sec:.2f}s-{end_sec:.2f}s]")
            records += 1
            shots += len(lengths)
            out.write(json.dumps({
                "sequence_id": f"dedup:{source_id}",
                "file_path": f"<synthetic>/{source_id}.mp4",
                "prompt": "\n".join(entries),
                "audio_path": None,
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output_jsonl), "sources": records, "shots": shots}))


if __name__ == "__main__":
    main()
