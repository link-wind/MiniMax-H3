#!/usr/bin/env python3
"""Stage-1 shot-level autoregressive data plan: shot inventory + pair buckets.

Task definition this script prepares data for:

    p(target_shot | context_shot, text),   context : target = 1 : 1

Constraints taken from the training stack (do not re-derive):

* the H3 video VAE only accepts frame counts of ``17n + 5``
  (``VIDEO_CLIP_FRAMES=17``, ``VIDEO_LENGTH_REMAINDER=5``); the legal grid is
  22, 39, 56, 73, ..., 345 and the latent-step count is
  ``((frames - 5) // 17) * 5 + 2``; a 345-frame window is 102 latent steps and
  the shipped 39-frame overlap is 12;
* ``masked-av-v14`` forces ``hard_core == overlap`` and ``transition == 0``
  **and restricts the overlap to the native grid 39 + 51m** (39, 90, 141, 192,
  243, 294 -- enforced in ``iter_continuation_samples``);
* context and target therefore cannot share a length: with
  ``context = 39 + 51m`` (== 5 mod 17) and ``window = 17n+5``, the target must
  be a multiple of 17.  The pair is then latent-exact:
  ``context 39 = 12 steps``, ``target 17k = 5k steps``, ``window = 12 + 5k``;
* context takes the 39 frames immediately before the target (= the tail of the
  previous shot); target takes the shot from its own head, aligned down to the
  bucket.  A sample is usable iff the shot is long enough for the bucket.

This script only measures supply.  It writes no latents and no training index.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

VIDEO_FPS = 24.0
SOURCE_FPS = 25.0
CLIP_FRAMES = 17
LENGTH_REMAINDER = 5

# Target buckets in output frames.  Targets must be multiples of 17 because
# the context is fixed on the native grid (39 = 17*2+5) and window must be
# 17n+5.  Largest bucket 289 = 12.04 s keeps window = 39 + 289 = 328 <= 345.
CANDIDATE_BUCKETS = [34, 51, 68, 102, 136, 204, 289]
MIN_BUCKET = 34
CONTEXT_FRAMES = 39


def align_down(frames: int) -> int | None:
    """Largest multiple of 17 <= frames (None below the smallest bucket)."""
    if frames < MIN_BUCKET:
        return None
    return (frames // CLIP_FRAMES) * CLIP_FRAMES


def bucket_of(frames: int) -> int | None:
    """Largest candidate bucket <= frames."""
    best = None
    for B in CANDIDATE_BUCKETS:
        if frames >= B:
            best = B
    return best


def target_latent_steps(frames: int) -> int:
    """Latent steps of a target that starts on a latent boundary."""
    return (frames // CLIP_FRAMES) * 5


def window_latent_steps(frames: int) -> int:
    return ((frames - LENGTH_REMAINDER) // CLIP_FRAMES) * 5 + 2


def load_sources(path, max_lines=None, progress_every=0):
    clips_by: dict[str, list[tuple[int, int]]] = {}
    cuts_by: dict[str, set[int]] = {}
    n_lines = 0
    started = time.monotonic()
    with open(path) as handle:
        for line in handle:
            n_lines += 1
            if max_lines is not None and n_lines > max_lines:
                break
            if progress_every and n_lines % progress_every == 0:
                print(f"[load] lines={n_lines} sources={len(clips_by)} "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_id = record.get("source_id") or ""
            if not source_id:
                continue
            spans = clips_by.setdefault(source_id, [])
            cuts = cuts_by.setdefault(source_id, set())
            for clip in record.get("clips") or []:
                frame_range = clip.get("frame_range")
                if frame_range:
                    spans.append((int(frame_range[0]), int(frame_range[1]) + 1))
            for cut in record.get("cut_points") or []:
                frame = cut.get("next_start_frame")
                if isinstance(frame, (int, float)):
                    cuts.add(int(frame))
    return clips_by, cuts_by, n_lines


def build_shots(spans, cuts):
    lo = min(a for a, _ in spans)
    hi = max(b for _, b in spans)
    ordered = sorted(c for c in cuts if lo < c < hi)
    bounds = [lo] + ordered + [hi]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-stats", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--samples-per-bucket", type=int, default=5)
    parser.add_argument("--shot-cache", type=Path, default=None,
                        help="reuse a previously extracted per-source shot-length cache")
    args = parser.parse_args()

    started = time.monotonic()
    shots_by_source: dict[str, list[int]] = {}
    if args.shot_cache and args.shot_cache.exists():
        with args.shot_cache.open() as handle:
            for line in handle:
                row = json.loads(line)
                shots_by_source[row["source_id"]] = row["shots_out"]
        n_lines = 0
        print(f"[cache] sources={len(shots_by_source)} from {args.shot_cache}", flush=True)
    else:
        clips_by, cuts_by, n_lines = load_sources(args.source_jsonl, args.max_lines, args.progress_every)
        for source_id, spans in clips_by.items():
            shots = build_shots(spans, cuts_by.get(source_id, set()))
            shots_by_source[source_id] = [
                int(round((end - start) * VIDEO_FPS / SOURCE_FPS)) for start, end in shots
            ]
        print(f"[load] lines={n_lines} sources={len(shots_by_source)} elapsed={time.monotonic() - started:.0f}s", flush=True)
        if args.shot_cache:
            args.shot_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.shot_cache.open("w", encoding="utf-8") as handle:
                for source_id, out in shots_by_source.items():
                    handle.write(json.dumps({"source_id": source_id, "shots_out": out}, ensure_ascii=False) + "\n")
            print(f"[cache] wrote {args.shot_cache}", flush=True)

    buckets = {L: {"shots_ge_L": 0, "bucket_size": 0} for L in CANDIDATE_BUCKETS}
    shot_out_frames = []
    samples = {L: [] for L in CANDIDATE_BUCKETS}
    no_prefix = 0
    dropped_short = 0

    for i, (source_id, out) in enumerate(shots_by_source.items()):
        if len(out) < 2:
            continue
        shot_out_frames.extend(out)
        for j, f in enumerate(out):
            if f < MIN_BUCKET:
                dropped_short += 1
                continue
            if j == 0:
                no_prefix += 1          # first shot of a source: only usable in the no-prefix split
                continue
            L = bucket_of(f)            # largest candidate bucket <= f
            for B in CANDIDATE_BUCKETS:
                if f >= B:
                    buckets[B]["shots_ge_L"] += 1
            buckets[L]["bucket_size"] += 1
            if len(samples[L]) < args.samples_per_bucket:
                samples[L].append({
                    "source_id": source_id,
                    "shot_index": j,
                    "target_frames_out": f,
                    "bucket_frames": L,
                    "bucket_sec": round(L / VIDEO_FPS, 3),
                    "context_frames": CONTEXT_FRAMES,
                    "window_frames": CONTEXT_FRAMES + L,
                    "target_latent_steps": target_latent_steps(L),
                    "window_latent_steps": window_latent_steps(CONTEXT_FRAMES + L),
                })
        if (i + 1) % 4000 == 0:
            print(f"[scan] {i + 1}/{len(clips_by)} elapsed={time.monotonic() - started:.0f}s", flush=True)

    total_shots = len(shot_out_frames)
    stats = {
        "config": {
            "source_jsonl": str(args.source_jsonl),
            "source_fps": SOURCE_FPS,
            "output_fps": VIDEO_FPS,
            "frame_grid": f"17n+{LENGTH_REMAINDER}",
            "candidate_buckets": CANDIDATE_BUCKETS,
            "pair_rule": "target = shot aligned down to 17k; context = the 39 frames immediately before it",
            "note": "shot lengths converted from source fps to output fps before bucketing",
        },
        "corpus": {
            "scanned_lines": n_lines,
            "sources": len(shots_by_source),
            "shots_total": total_shots,
            "shots_ge_min_bucket": sum(1 for f in shot_out_frames if f >= MIN_BUCKET),
            "pair_rule": "usable = shot itself; context is taken from the preceding timeline",
        },
        "shot_len_out_frames": {
            "mean": round(sum(shot_out_frames) / max(total_shots, 1), 2),
            "median": sorted(shot_out_frames)[total_shots // 2] if total_shots else None,
            "p10": sorted(shot_out_frames)[int(0.1 * total_shots)] if total_shots else None,
            "p90": sorted(shot_out_frames)[int(0.9 * total_shots)] if total_shots else None,
        },
        "usable": {
            "with_history": sum(1 for _ in shot_out_frames) - no_prefix - dropped_short,
            "no_prefix_only": no_prefix,
            "dropped_below_min_bucket": dropped_short,
        },
        "buckets": {},
    }
    for L in CANDIDATE_BUCKETS:
        b = buckets[L]
        stats["buckets"][str(L)] = {
            "target_frames": L,
            "target_sec": round(L / VIDEO_FPS, 3),
            "context_frames": CONTEXT_FRAMES,
            "window_frames": CONTEXT_FRAMES + L,
            "target_latent_steps": target_latent_steps(L),
            "window_latent_steps": window_latent_steps(CONTEXT_FRAMES + L),
            "bucket_size": b["bucket_size"],
            "shots_ge_L": b["shots_ge_L"],
        }

    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    args.output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_samples:
        args.output_samples.parent.mkdir(parents=True, exist_ok=True)
        with args.output_samples.open("w", encoding="utf-8") as handle:
            for L in CANDIDATE_BUCKETS:
                for row in samples[L]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] {args.output_stats} elapsed={time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    main()
