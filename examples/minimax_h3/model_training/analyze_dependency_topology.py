#!/usr/bin/env python3
"""Quantify the dependency topology of long multi-shot video corpora.

One pass over the LongVideoGen annotation JSONL.  Rows are grouped by
``source_id`` exactly like ``plan_shot_windows.py --all-sources`` (a single
source is split across many rows; grouping is mandatory).

What is measured
----------------
1. shot-length distribution, rebuilt from ``cut_points[].next_start_frame``;
2. window-kind distribution (continuation / shot_cut / multi_cut);
3. how often the leading context band already contains a shot cut;
4. the *dependency radius* ``delta`` of every window: how many additional
   strides of history are needed before the start of the shot that owns the
   window's first frame becomes visible;
5. the context-cost curve: coverage of the leading-shot start as the context
   band is widened, together with the number of frames that must still be
   generated.

Definitions (printed into the report, no ambiguity left)
-------------------------------------------------------
``s``   window start frame (source frames, ``--fps``)
``p``   start frame of the shot that contains ``s``
``ctx`` context band length (``--context-frames``, 39 = the shipped overlap)
``L``   stride (``--stride``, default window - context)

    delta = max(0, ceil((s - p - ctx) / L))

so ``delta = 0`` means the current 39-frame context already reaches the start
of the leading shot, ``delta = k`` means k extra strides of history are needed.

Known traps, kept explicit
--------------------------
* source frames are 25 fps here, while the model contract is 345 output frames
  at 24 fps: 345 *source* frames is the legacy planning coordinate, not the
  contract;
* ``clips`` are contiguous fragments split by gaps *or* cuts, so ``len(clips)``
  is not a shot count -- shot counts only come from ``cut_points``;
* ``cut_frames`` written by different planner modes use different coordinate
  systems and must not be merged.  This script never merges them: it always
  rebuilds from the source JSONL.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# loading / timeline helpers
# --------------------------------------------------------------------------- #
def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def covered_in(merged, a: int, b: int) -> int:
    covered = 0
    for start, end in merged:
        if end <= a:
            continue
        if start >= b:
            break
        covered += max(0, min(end, b) - max(start, a))
    return covered


def load_sources(path: str | Path, max_lines: int | None = None, progress_every: int = 0):
    """Group the corpus by source_id -> (spans, cuts)."""
    clips_by: dict[str, dict[str, tuple[int, int]]] = {}
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
            clips = clips_by.setdefault(source_id, {})
            cuts = cuts_by.setdefault(source_id, set())
            for clip in record.get("clips") or []:
                video_path = clip.get("video_path")
                frame_range = clip.get("frame_range")
                if video_path and frame_range:
                    clips.setdefault(video_path, (int(frame_range[0]), int(frame_range[1]) + 1))
            for cut in record.get("cut_points") or []:
                frame = cut.get("next_start_frame")
                if isinstance(frame, (int, float)):
                    cuts.add(int(frame))
    return clips_by, cuts_by, n_lines


# --------------------------------------------------------------------------- #
# per-source analysis
# --------------------------------------------------------------------------- #
SHOT_EDGES = [0, 25, 50, 75, 100, 150, 200, 300, 450, 600, 900, 1200, 10 ** 9]
CUT_EDGES = [0, 1, 2, 3, 4, 5, 6, 8, 10 ** 9]
DELTA_EDGES = [0, 1, 2, 3, 4, 5, 10 ** 9]
COST_CURVE = [39, 78, 150, 234, 318, 434]


def bucket(value, edges, labels=None):
    idx = bisect.bisect_right(edges, value) - 1
    idx = max(0, min(idx, len(edges) - 2))
    if labels is None:
        hi = edges[idx + 1]
        return f"{edges[idx]}-{hi}" if hi < 10 ** 9 else f"{edges[idx]}+"
    return labels[idx]


def analyze_source(source_id, clips, cuts, args):
    if not clips:
        return None
    spans = merge_intervals([(a, b) for a, b in clips.values()])
    lo, hi = spans[0][0], spans[-1][1]
    if hi - lo < args.window_frames:
        return None
    sorted_cuts = sorted(c for c in cuts if lo < c < hi)

    bounds = [lo] + sorted_cuts + [hi]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    stride = args.stride or (args.window_frames - args.context_frames)
    ctx = args.context_frames

    out = {
        "source_id": source_id,
        "timeline": [lo, hi],
        "timeline_sec": (hi - lo) / args.fps,
        "clips": len(clips),
        "cuts": len(sorted_cuts),
        "shots": len(shots),
        "missing_frames": (hi - lo) - covered_in(spans, lo, hi),
        "shot_len": [],
        "shot_len_covered": [],
        "cuts_per_window": [],
        "window_kind": {},
        "overlap_has_cut": 0,
        "lead_start_before_window": 0,
        "lead_start_outside_ctx": 0,
        "lead_is_first_shot_of_source": 0,
        "delta": {},
        "cost_curve": {x: 0 for x in args.cost_curve},
        "windows": 0,
    }

    for start, end in shots:
        length = end - start
        cov = covered_in(spans, start, end) / max(length, 1)
        out["shot_len"].append(length)
        if cov >= 0.99:
            out["shot_len_covered"].append(length)

    shot_starts = [s for s, _ in shots]

    start = lo
    while start + args.window_frames <= hi:
        end = start + args.window_frames
        out["windows"] += 1
        n_cuts = bisect.bisect_left(sorted_cuts, end) - bisect.bisect_right(sorted_cuts, start)
        out["cuts_per_window"].append(n_cuts)

        if n_cuts == 0:
            kind = "continuation"
        elif n_cuts == 1:
            kind = "shot_cut"
        else:
            kind = "multi_cut"
        out["window_kind"][kind] = out["window_kind"].get(kind, 0) + 1

        if any(start < c < start + ctx for c in sorted_cuts):
            out["overlap_has_cut"] += 1

        idx = bisect.bisect_right(shot_starts, start) - 1
        idx = max(0, min(idx, len(shots) - 1))
        p = shots[idx][0]
        if idx == 0:
            out["lead_is_first_shot_of_source"] += 1
        if p < start:
            out["lead_start_before_window"] += 1
        if start - p > ctx:
            out["lead_start_outside_ctx"] += 1

        delta = max(0, math.ceil((start - p - ctx) / stride))
        key = bucket(delta, DELTA_EDGES, ["0", "1", "2", "3", "4", "5+"])
        out["delta"][key] = out["delta"].get(key, 0) + 1

        for x in args.cost_curve:
            if start - p <= x:
                out["cost_curve"][x] += 1

        start += stride

    return out


def hist(values, edges, labels=None):
    counts = {}
    for value in values:
        key = bucket(value, edges, labels)
        counts[key] = counts.get(key, 0) + 1
    return counts


def pct(value, total):
    return round(100.0 * value / total, 2) if total else 0.0


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-sources", type=Path, default=None,
                        help="optional per-source jsonl with the raw per-source counters")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--window-frames", type=int, default=345)
    parser.add_argument("--context-frames", type=int, default=39)
    parser.add_argument("--stride", type=int, default=None, help="defaults to window - context")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--source-contains", default=None,
                        help="only analyse source_ids containing this substring (debug / single-source check)")
    parser.add_argument("--top-sources", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--cost-curve", type=int, nargs="*", default=COST_CURVE)
    args = parser.parse_args()

    started = time.monotonic()
    clips_by, cuts_by, n_lines = load_sources(args.source_jsonl, args.max_lines,
                                               progress_every=args.progress_every)
    print(f"[load] lines={n_lines} sources={len(clips_by)} elapsed={time.monotonic() - started:.1f}s", flush=True)

    agg = {
        "sources": 0,
        "scanned_lines": n_lines,
        "clips": 0,
        "cuts": 0,
        "shots": 0,
        "windows": 0,
        "missing_frames": 0,
        "timeline_frames": 0,
        "shot_len": [],
        "shot_len_covered": [],
        "cuts_per_window": [],
        "window_kind": {},
        "overlap_has_cut": 0,
        "lead_start_before_window": 0,
        "lead_start_outside_ctx": 0,
        "lead_is_first_shot_of_source": 0,
        "delta": {},
        "cost_curve": {x: 0 for x in args.cost_curve},
    }
    per_source = []

    for i, (source_id, clips) in enumerate(clips_by.items()):
        if args.source_contains and args.source_contains not in source_id:
            continue
        stats = analyze_source(source_id, clips, cuts_by.get(source_id, set()), args)
        if stats is None:
            continue
        agg["sources"] += 1
        agg["clips"] += stats["clips"]
        agg["cuts"] += stats["cuts"]
        agg["shots"] += stats["shots"]
        agg["windows"] += stats["windows"]
        agg["missing_frames"] += stats["missing_frames"]
        agg["timeline_frames"] += stats["timeline"][1] - stats["timeline"][0]
        agg["shot_len"].extend(stats["shot_len"])
        agg["shot_len_covered"].extend(stats["shot_len_covered"])
        agg["cuts_per_window"].extend(stats["cuts_per_window"])
        agg["overlap_has_cut"] += stats["overlap_has_cut"]
        agg["lead_start_before_window"] += stats["lead_start_before_window"]
        agg["lead_start_outside_ctx"] += stats["lead_start_outside_ctx"]
        agg["lead_is_first_shot_of_source"] += stats["lead_is_first_shot_of_source"]
        for key, value in stats["window_kind"].items():
            agg["window_kind"][key] = agg["window_kind"].get(key, 0) + value
        for key, value in stats["delta"].items():
            agg["delta"][key] = agg["delta"].get(key, 0) + value
        for key, value in stats["cost_curve"].items():
            agg["cost_curve"][key] = agg["cost_curve"].get(key, 0) + value
        per_source.append({
            "source_id": source_id,
            "timeline_sec": round(stats["timeline_sec"], 1),
            "clips": stats["clips"],
            "cuts": stats["cuts"],
            "shots": stats["shots"],
            "windows": stats["windows"],
            "median_shot_sec": round(percentile(stats["shot_len_covered"] or stats["shot_len"], 0.5) / args.fps, 3),
            "overlap_has_cut_pct": pct(stats["overlap_has_cut"], stats["windows"]),
            "delta_ge2_pct": pct(sum(v for k, v in stats["delta"].items() if k in ("2", "3", "4", "5+")), stats["windows"]),
            "multi_cut_pct": pct(stats["window_kind"].get("multi_cut", 0), stats["windows"]),
        })
        if (i + 1) % args.progress_every == 0:
            print(f"[scan] {i + 1}/{len(clips_by)} sources, windows={agg['windows']}, "
                  f"elapsed={time.monotonic() - started:.0f}s", flush=True)

    total_shots = len(agg["shot_len"])
    total_covered = len(agg["shot_len_covered"])
    windows = agg["windows"]

    report = {
        "config": {
            "source_jsonl": str(args.source_jsonl),
            "fps": args.fps,
            "window_frames": args.window_frames,
            "context_frames": args.context_frames,
            "stride": args.stride or (args.window_frames - args.context_frames),
            "window_sec": args.window_frames / args.fps,
            "output_fps": 24.0,
            "output_frame_equivalent": args.window_frames / args.fps * 24.0,
            "note": "source frames @ fps; the shipped contract is 345 output frames @24fps",
        },
        "corpus": {
            "scanned_lines": n_lines,
            "sources": agg["sources"],
            "clips": agg["clips"],
            "cuts": agg["cuts"],
            "shots": agg["shots"],
            "timeline_hours": round(agg["timeline_frames"] / args.fps / 3600.0, 2),
            "missing_frame_pct": pct(agg["missing_frames"], agg["timeline_frames"]),
            "windows": windows,
            "windows_per_hour": round(windows / max(agg["timeline_frames"] / args.fps / 3600.0, 1e-9), 1),
        },
        "shot_length": {
            "n": total_shots,
            "n_fully_covered": total_covered,
            "fully_covered_pct": pct(total_covered, total_shots),
            "mean_sec": round(sum(agg["shot_len"]) / max(total_shots, 1) / args.fps, 3),
            "median_sec": round(percentile(agg["shot_len"], 0.5) / args.fps, 3),
            "p10_sec": round(percentile(agg["shot_len"], 0.1) / args.fps, 3),
            "p90_sec": round(percentile(agg["shot_len"], 0.9) / args.fps, 3),
            "p99_sec": round(percentile(agg["shot_len"], 0.99) / args.fps, 3),
            "max_sec": round(max(agg["shot_len"]) / args.fps, 3),
            "covered_mean_sec": round(sum(agg["shot_len_covered"]) / max(total_covered, 1) / args.fps, 3),
            "covered_median_sec": round(percentile(agg["shot_len_covered"], 0.5) / args.fps, 3),
            "covered_p90_sec": round(percentile(agg["shot_len_covered"], 0.9) / args.fps, 3),
            "longer_than_context_pct": pct(sum(1 for v in agg["shot_len_covered"] if v > args.context_frames), total_covered),
            "longer_than_stride_pct": pct(
                sum(1 for v in agg["shot_len_covered"] if v > (args.stride or args.window_frames - args.context_frames)),
                total_covered),
            "hist_frames": hist(agg["shot_len_covered"] or agg["shot_len"], SHOT_EDGES),
            "shots_per_window_by_median": round(
                args.window_frames / max(percentile(agg["shot_len_covered"], 0.5), 1), 2),
            "shots_per_window_by_mean": round(
                args.window_frames / max(sum(agg["shot_len_covered"]) / max(total_covered, 1), 1e-9), 2),
        },
        "window_kind": {k: {"n": v, "pct": pct(v, windows)} for k, v in sorted(agg["window_kind"].items())},
        "cuts_per_window": {
            "mean": round(sum(agg["cuts_per_window"]) / max(windows, 1), 3),
            "median": percentile(agg["cuts_per_window"], 0.5),
            "hist": hist(agg["cuts_per_window"], CUT_EDGES),
        },
        "context_band": {
            "contains_cut_pct": pct(agg["overlap_has_cut"], windows),
            "lead_shot_start_before_window_pct": pct(agg["lead_start_before_window"], windows),
            "lead_shot_start_outside_context_pct": pct(agg["lead_start_outside_ctx"], windows),
            "lead_is_source_first_shot_pct": pct(agg["lead_is_first_shot_of_source"], windows),
        },
        "dependency_radius": {
            "definition": "delta = max(0, ceil((s - p - ctx) / stride)); s=window start, p=start of the shot owning s, ctx=context frames, stride=window-ctx",
            "hist": {k: {"n": agg["delta"].get(k, 0), "pct": pct(agg["delta"].get(k, 0), windows)}
                     for k in ("0", "1", "2", "3", "4", "5+")},
            "delta_ge2_pct": pct(sum(v for k, v in agg["delta"].items() if k in ("2", "3", "4", "5+")), windows),
            "delta_ge3_pct": pct(sum(v for k, v in agg["delta"].items() if k in ("3", "4", "5+")), windows),
        },
        "context_cost_curve": {
            str(x): {
                "context_sec": round(x / args.fps, 3),
                "covered_pct": pct(agg["cost_curve"][x], windows),
                "new_frames_per_window": args.window_frames - x,
            }
            for x in args.cost_curve
        },
        "top_sources_by_windows": sorted(per_source, key=lambda r: -r["windows"])[: args.top_sources],
        "elapsed_sec": round(time.monotonic() - started, 1),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_sources:
        args.output_sources.parent.mkdir(parents=True, exist_ok=True)
        with args.output_sources.open("w", encoding="utf-8") as handle:
            for row in per_source:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({k: report[k] for k in ("config", "corpus", "shot_length", "window_kind",
                                             "cuts_per_window", "context_band", "dependency_radius",
                                             "context_cost_curve")},
                     ensure_ascii=False, indent=2))
    print(f"[done] {args.output_json} elapsed={report['elapsed_sec']}s")


if __name__ == "__main__":
    main()
