#!/usr/bin/env python3
"""Stage-1 supply statistics for shot-level sampling with a variable context.

Replays exactly the sampler used for training
(:func:`diffsynth.utils.continuation_lora.iter_shot_continuation_samples`) and
aggregates the result into the two tables the training plan needs:

* the **target** table -- how many samples each ``17k`` target bucket supplies;
* the **context** table and the joint target x context matrix -- what the
  per-sample overlap actually costs in window length, which is the thing a
  fixed context cannot express.

Because the sampler itself is replayed rather than re-derived, the numbers
cannot drift from the data the cache builder will consume.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.utils.continuation_lora import (  # noqa: E402
    MAX_SHOT_CONTEXT_FRAMES,
    MAX_SHOT_WINDOW_FRAMES,
    MIN_SHOT_TARGET_FRAMES,
    VIDEO_FPS,
    iter_shot_continuation_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-context-frames", type=int, default=MAX_SHOT_CONTEXT_FRAMES)
    parser.add_argument("--max-window-frames", type=int, default=MAX_SHOT_WINDOW_FRAMES)
    parser.add_argument("--min-target-frames", type=int, default=MIN_SHOT_TARGET_FRAMES)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument(
        "--shot-identities", type=Path, default=None,
        help=(
            "Deduplicate by **source-level shot identity** instead of by caption. "
            "Points at record_shot_identities.jsonl from "
            "extract_record_shot_identities.py; each shot is keyed by "
            "(source_id, start_frame, end_frame) on the source episode timeline, "
            "which is the only identity that survives the overlapping-window "
            "augmentation (every window was re-annotated, so captions differ)."
        ),
    )
    parser.add_argument(
        "--identity-report", type=Path, default=None,
        help="Write per-bucket counts for both the raw and the deduplicated view.",
    )
    parser.add_argument(
        "--dedup", action="store_true",
        help=(
            "Keep only the first occurrence of each adjacent shot pair. The training "
            "index is built from overlapping multi-shot windows of the same episodes, "
            "so the same shot pair appears in several records; without this flag the "
            "bucket counts are inflated and the epoch definition is wrong. The "
            "duplicate-free counts are reported under ``counts_dedup``."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats: dict[str, int] = {}
    target_hist: collections.Counter[int] = collections.Counter()
    context_hist: collections.Counter[int] = collections.Counter()
    joint: collections.Counter[tuple[int, int]] = collections.Counter()
    window_by_target: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    started = time.monotonic()
    next_report = args.progress_every
    identities: dict[str, tuple[str, list]] = {}
    if args.shot_identities is not None:
        with args.shot_identities.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identities[row["sequence_id"]] = (row["source_id"], row["shots"])
        print(f"[ident] loaded {len(identities)} records from {args.shot_identities}", flush=True)

    samples = 0
    samples_emitted = 0
    seen_pairs = set() if args.dedup else None
    seen_identities: set = set()
    identity_mismatch = 0
    for sample in iter_shot_continuation_samples(
        args.source_jsonl,
        max_context_frames=args.max_context_frames,
        max_window_frames=args.max_window_frames,
        min_target_frames=args.min_target_frames,
        include_plain=True,
        include_prompt=args.dedup,
        stats=stats,
        seen_pairs=seen_pairs,
    ):
        samples += 1
        samples_emitted += 1
        if sample.prefix_mode == "none":
            continue
        if identities:
            entry = identities.get(sample.sequence_id)
            if entry is None:
                identity_mismatch += 1
                continue
            source_id, ranges = entry
            if not 1 <= sample.shot_index <= len(ranges):
                identity_mismatch += 1
                continue
            previous = ranges[sample.shot_index - 2] if sample.shot_index >= 2 else None
            target_range = ranges[sample.shot_index - 1]
            key = None if previous is None else (
                source_id, previous[0], previous[1], target_range[0], target_range[1],
            )
            if key is None or key in seen_identities:
                continue
            seen_identities.add(key)
        target = sample.window_frames - sample.overlap_frames
        target_hist[target] += 1
        context_hist[sample.overlap_frames] += 1
        joint[(target, sample.overlap_frames)] += 1
        window_by_target[target][sample.window_frames] += 1
        if stats.get("records", 0) >= next_report:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[plan] records={stats['records']} samples={samples} "
                f"rate={stats['records'] / elapsed:.0f} rec/s",
                flush=True,
            )
            next_report += args.progress_every
    total = sum(target_hist.values())
    payload = {
        "source_jsonl": str(args.source_jsonl),
        "max_context_frames": args.max_context_frames,
        "max_window_frames": args.max_window_frames,
        "min_target_frames": args.min_target_frames,
        "video_fps": VIDEO_FPS,
        "stats": stats,
        "dedup": bool(args.dedup),
        "dedup_by_shot_identity": bool(identities),
        "identity_records": len(identities),
        "identity_mismatch": identity_mismatch,
        "unique_pairs": len(seen_identities) if identities else None,
        "counts": {
            "continuation_samples": total,
            "prefix_free_samples": stats.get("accepted_plain", 0),
            "paired_shots": stats.get("pairs", 0),
            "records": stats.get("records", 0),
        },
        "targets": [
            {
                "target_frames": frames,
                "target_seconds": round(frames / VIDEO_FPS, 3),
                "target_latent_steps": (frames // 17) * 5,
                "samples": count,
                "share": round(count / total, 6) if total else 0.0,
                "window_frames_observed": sorted(window_by_target[frames]),
            }
            for frames, count in sorted(target_hist.items())
        ],
        "contexts": [
            {
                "context_frames": frames,
                "context_seconds": round(frames / VIDEO_FPS, 3),
                "context_latent_steps": frames // 17 * 5 + 2,
                "samples": count,
                "share": round(count / total, 6) if total else 0.0,
            }
            for frames, count in sorted(context_hist.items())
        ],
        "joint": [
            {
                "target_frames": target,
                "context_frames": context,
                "window_frames": target + context,
                "samples": count,
            }
            for (target, context), count in sorted(joint.items())
        ],
        "counts_all": {
            "continuation_samples_emitted": samples_emitted,
            "prefix_free_samples": stats.get("accepted_plain", 0),
            "adjacent_shot_pairs": stats.get("pairs", 0),
            "duplicate_pairs_skipped": stats.get("duplicate_pairs", 0),
        },
        "elapsed_sec": round(time.monotonic() - started, 1),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: payload[k] for k in ("counts", "elapsed_sec")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
