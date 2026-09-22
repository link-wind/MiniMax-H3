#!/usr/bin/env python3
"""Extract the *source-level* shot identity of every record in the 650k corpus.

Why this exists: the training index is built from overlapping multi-shot windows
of the same episodes, and each window was re-annotated, so the same underlying
shot carries a different caption in every record (5000 records -> 24682 shots,
24682 distinct caption bodies: caption hashing cannot deduplicate it).

The only stable identity is the shot's range on the **source episode** timeline.
The corpus stores it directly: `clips[].frame_range` gives the window span and
`cut_points[].next_start_frame` the internal boundaries, both in absolute source
frames.  A shot is then identified by `(source_id, start_frame, end_frame)`, and
two records that contain the same shot produce the same key.

Output is one row per record, small enough to keep in memory:

    {"sequence_id": ..., "source_id": ..., "shots": [[start, end], ...]}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50000)
    return parser.parse_args()


def shot_ranges(record: dict) -> list[list[int]]:
    """Ordered absolute shot ranges of one record, in source frames."""
    spans: list[tuple[int, int]] = []
    for clip in record.get("clips") or []:
        frame_range = clip.get("frame_range")
        if frame_range:
            spans.append((int(frame_range[0]), int(frame_range[1]) + 1))
    if not spans:
        return []
    lo = min(start for start, _ in spans)
    hi = max(end for _, end in spans)
    cuts = sorted({
        int(cut["next_start_frame"])
        for cut in record.get("cut_points") or []
        if isinstance(cut.get("next_start_frame"), (int, float)) and lo < int(cut["next_start_frame"]) < hi
    })
    bounds = [lo] + cuts + [hi]
    return [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = shots = 0
    next_report = args.progress_every
    with args.source_jsonl.open(encoding="utf-8") as handle, args.output_jsonl.open("w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sequence_id = str(record.get("sequence_id") or "").strip()
            source_id = str(record.get("source_id") or "").strip()
            if not sequence_id or not source_id:
                continue
            ranges = shot_ranges(record)
            records += 1
            shots += len(ranges)
            out.write(json.dumps(
                {"sequence_id": sequence_id, "source_id": source_id, "shots": ranges},
                ensure_ascii=False,
            ) + "\n")
            if records >= next_report:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(f"[ident] records={records} shots={shots} rate={records / elapsed:.0f} rec/s", flush=True)
                next_report += args.progress_every
    print(json.dumps({
        "output": str(args.output_jsonl), "records": records, "shots": shots,
        "elapsed_sec": round(time.monotonic() - started, 1),
    }))


if __name__ == "__main__":
    main()
