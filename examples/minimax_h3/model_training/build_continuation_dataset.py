#!/usr/bin/env python3
"""Stream H3 continuation metadata from the LongVideoGen JSONL source."""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow direct execution from any working directory without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.utils.continuation_lora import (
    MAX_SHOT_CONTEXT_FRAMES,
    MAX_SHOT_WINDOW_FRAMES,
    MIN_SHOT_TARGET_FRAMES,
    iter_continuation_pairs,
    iter_continuation_samples,
    iter_shot_continuation_samples,
    iter_shot_memory_only_samples,
)


def _build_iterator(args, stats):
    """Select the sample iterator for the requested data contract."""
    if args.memory_only:
        return iter_shot_memory_only_samples(
            args.source_jsonl,
            max_window_frames=args.max_window_frames,
            min_target_frames=args.min_target_frames,
            split_seed=args.split_seed,
            require_paths=args.require_paths,
            stats=stats,
        )
    if args.shot_level:
        return iter_shot_continuation_samples(
            args.source_jsonl,
            max_context_frames=args.max_context_frames,
            include_plain=args.include_plain,
            split_seed=args.split_seed,
            require_paths=args.require_paths,
            conditioning_mode=args.mode,
            stats=stats,
        )
    if args.directional_pairs:
        return iter_continuation_pairs(
            args.source_jsonl,
            window_frames=args.window_frames,
            overlap_frames=args.overlap_frames,
            hard_core_frames=args.hard_core_frames,
            split_seed=args.split_seed,
            pair_stride_frames=args.window_stride_frames,
            require_paths=args.require_paths,
            conditioning_mode=args.mode,
            stats=stats,
        )
    return iter_continuation_samples(
        args.source_jsonl,
        window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
        hard_core_frames=args.hard_core_frames,
        split_seed=args.split_seed,
        window_stride_frames=args.window_stride_frames,
        require_paths=args.require_paths,
        conditioning_mode=args.mode,
        stats=stats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--window-frames", type=int, default=345)
    parser.add_argument("--overlap-frames", type=int, default=34)
    parser.add_argument("--hard-core-frames", type=int, default=17)
    parser.add_argument(
        "--mode", choices=("legacy", "masked-av-v14"), default="legacy",
        help="Data contract; masked-av-v14 uses native 39-frame context and no transition band.",
    )
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--window-stride-frames", type=int, default=None)
    parser.add_argument(
        "--directional-pairs", action="store_true",
        help="Emit v3 A/B records: A tail conditions a later B window in the same shot.",
    )
    parser.add_argument("--require-paths", action="store_true")
    parser.add_argument(
        "--shot-level", action="store_true",
        help=(
            "Emit stage-1 shot-level samples: one whole target shot conditioned on "
            "min(previous shot, --max-context-frames) frames of context. The overlap "
            "is then per-sample instead of the run-level --overlap-frames."
        ),
    )
    parser.add_argument(
        "--max-context-frames", type=int, default=MAX_SHOT_CONTEXT_FRAMES,
        help="Cap on the shot-level context, quantised down to the 17n+5 grid.",
    )
    parser.add_argument(
        "--memory-only", action="store_true",
        help=(
            "Emit stage-2 memory-only samples: one whole shot per sample, prefix_mode=none, "
            "no context rows at all. The entire window is the prediction target and the only "
            "cross-shot conditioning is the memory block attached by build_continuation_cache.py "
            "--memory-frames/--memory-anchor."
        ),
    )
    parser.add_argument(
        "--max-window-frames", type=int, default=MAX_SHOT_WINDOW_FRAMES,
        help="Cap on a memory-only target window, quantised down to the 17n+5 grid.",
    )
    parser.add_argument(
        "--min-target-frames", type=int, default=MIN_SHOT_TARGET_FRAMES,
        help="Drop memory-only shots whose window would be shorter than this.",
    )
    parser.add_argument(
        "--include-plain", action="store_true",
        help=(
            "With --shot-level, interleave one prefix-free row per record (its first "
            "shot, prefix_mode=none) as the anti-forgetting mixture."
        ),
    )
    parser.add_argument("--progress", action="store_true", help="Print scan progress and throughput periodically.")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "masked-av-v14":
        if args.overlap_frames == 34:
            args.overlap_frames = 39
        args.hard_core_frames = 0
    stats = {}
    limits = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
    }
    written = {split: 0 for split in limits}
    started_at = time.monotonic()
    next_progress = 10000
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if args.shot_level and args.mode != "masked-av-v14":
        parser.error("--shot-level requires --mode masked-av-v14")
    if args.shot_level and args.directional_pairs:
        parser.error("--shot-level and --directional-pairs are mutually exclusive")
    if args.memory_only and (args.shot_level or args.directional_pairs):
        parser.error("--memory-only is mutually exclusive with --shot-level and --directional-pairs")
    if args.memory_only and args.mode != "masked-av-v14":
        parser.error("--memory-only requires --mode masked-av-v14")
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        iterator = _build_iterator(args, stats)
        for record in iterator:
            split = record.split
            limit = limits.get(split)
            if limit is not None and written[split] >= limit:
                continue
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            written[split] += 1
            if args.progress and stats.get("records", 0) >= next_progress:
                elapsed = max(time.monotonic() - started_at, 1e-6)
                rate = stats["records"] / elapsed
                print(
                    f"[index] scanned={stats['records']} accepted={stats.get('accepted', 0)} "
                    f"written={sum(written.values())} rate={rate:.1f} records/s",
                    flush=True,
                )
                next_progress += 10000
            if all(limit is not None and written[split] >= limit for split, limit in limits.items()):
                break
    print(json.dumps({"output": args.output_jsonl, "written": written, **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
