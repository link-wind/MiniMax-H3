#!/usr/bin/env python3
"""Plan trainable windows over a rebuilt long multi-shot video (per source_id).

Two input modes:

1. ``--source-jsonl`` + ``--source-contains``: use the 650k annotation clips and
   their absolute ``frame_range`` / ``cut_points``.  Some clips are missing from
   the annotation subset, so windows usually contain small gaps.
2. ``--clip-dir`` + ``--source-id``: rebuild a continuous timeline directly from
   the original 25 fps fragment files (one fragment == one shot), using ffprobe
   durations; there are no gaps, and every fragment boundary is a shot cut.

Window kinds:

* ``continuation`` : no shot cut inside the window (same shot);
* ``shot_cut``     : exactly one shot cut, placed at a controlled offset;
* ``multi_cut``    : two or more cuts (advanced / long-chain samples).

Only window plans are produced (frame ranges, cut positions, kind).  Frame
extraction / VAE encoding is a separate step.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-jsonl", type=str, default=None)
    p.add_argument("--source-contains", action="append", default=[])
    p.add_argument("--clip-dir", type=str, default=None,
                   help="Directory of original 25 fps fragment files (continuous rebuild mode).")
    p.add_argument("--source-id", type=str, default=None,
                   help="Exact source_id; required with --clip-dir.")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--window-frames", type=int, default=345)
    p.add_argument("--overlap-frames", type=int, default=39)
    p.add_argument("--stride", type=int, default=None, help="Defaults to window-overlap.")
    p.add_argument("--output-fps", type=float, default=24.0,
                   help="H3 output fps.  --window-frames/--overlap-frames/--stride and "
                        "--cut-positions are expressed in output frames, so a 345-frame "
                        "window is 14.375 s of source, not 345 source frames.")
    p.add_argument("--cut-positions", type=str, default="117,172,232",
                   help="Cut offsets (frames from window start) for cut-aligned sampling.")
    p.add_argument("--allow-gap", action="store_true")
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--all-sources", action="store_true",
                   help="Group the jsonl by source_id and aggregate window stats over every source.")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--output-stats", type=Path, default=None)
    return p.parse_args()


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip()
    return float(out) if out else 0.0


def clip_index(name: str) -> int:
    stem = name.rsplit(".", 1)[0]
    return int(stem.rsplit("_", 1)[1])


def build_from_clip_dir(clip_dir: str, source_id: str, fps: float, workers: int = 8):
    names = [n for n in os.listdir(clip_dir)
             if n.startswith(source_id + "_") and n.endswith(".mp4")]
    names.sort(key=clip_index)
    paths = [Path(clip_dir) / n for n in names]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        durations = list(pool.map(ffprobe_duration, paths))
    clips = {}
    cursor = 0
    for path, dur in zip(paths, durations):
        frames = int(round(dur * fps))
        clips[str(path)] = (cursor, cursor + frames, dur)
        cursor += frames
    cuts = [end for _, (_, end, _) in list(clips.items())[:-1]]
    return [source_id], clips, set(cuts), len(paths)


def load_from_jsonl(path: str | Path, contains: list[str], max_lines: int | None):
    path = Path(path)
    clips: dict[str, tuple[int, int, float]] = {}
    cuts: set[int] = set()
    sources: set[str] = set()
    n_lines = 0
    with path.open() as handle:
        for line in handle:
            n_lines += 1
            if max_lines is not None and n_lines > max_lines:
                break
            if contains and not all(token in line for token in contains):
                continue
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_id = record.get("source_id", "")
            if contains and not all(token in source_id for token in contains):
                continue
            sources.add(source_id)
            for clip in record.get("clips") or []:
                video_path = clip.get("video_path")
                frame_range = clip.get("frame_range")
                if video_path and frame_range:
                    clips.setdefault(
                        video_path,
                        # ``frame_range`` is inclusive of both ends; convert to a
                        # half-open span so adjacent clips tile without 1-frame holes.
                        (int(frame_range[0]), int(frame_range[1]) + 1,
                         float(clip.get("duration_sec") or 0.0)),
                    )
            for cut in record.get("cut_points") or []:
                frame = cut.get("next_start_frame")
                if isinstance(frame, (int, float)):
                    cuts.add(int(frame))
    return sorted(sources), clips, cuts, n_lines


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def gap_in(merged, a: int, b: int) -> int:
    covered = 0
    for start, end in merged:
        if end <= a:
            continue
        if start >= b:
            break
        covered += max(0, min(end, b) - max(start, a))
    return (b - a) - covered


def classify(n_cuts: int) -> str:
    if n_cuts == 0:
        return "continuation"
    if n_cuts == 1:
        return "shot_cut"
    return "multi_cut"


def cuts_in(sorted_cuts, a: int, b: int) -> list[int]:
    lo = bisect.bisect_right(sorted_cuts, a)
    hi = bisect.bisect_left(sorted_cuts, b)
    return sorted_cuts[lo:hi]


def load_all_sources(path: str | Path, max_lines: int | None):
    path = Path(path)
    clips_by: dict[str, dict[str, tuple[int, int, float]]] = {}
    cuts_by: dict[str, set[int]] = {}
    n_lines = 0
    with path.open() as handle:
        for line in handle:
            n_lines += 1
            if max_lines is not None and n_lines > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_id = record.get("source_id", "")
            if not source_id:
                continue
            clips = clips_by.setdefault(source_id, {})
            cuts = cuts_by.setdefault(source_id, set())
            for clip in record.get("clips") or []:
                video_path = clip.get("video_path")
                frame_range = clip.get("frame_range")
                if video_path and frame_range:
                    clips.setdefault(
                        video_path,
                        (int(frame_range[0]), int(frame_range[1]) + 1,
                         float(clip.get("duration_sec") or 0.0)),
                    )
            for cut in record.get("cut_points") or []:
                frame = cut.get("next_start_frame")
                if isinstance(frame, (int, float)):
                    cuts.add(int(frame))
    return clips_by, cuts_by, n_lines


def plan_one(source_id: str, clips, cuts, args, window_src, stride_src, cut_positions):
    spans = sorted((v[0], v[1]) for v in clips.values())
    merged = merge_intervals(spans)
    if not merged:
        return None
    lo, hi = merged[0][0], merged[-1][1]
    sorted_cuts = sorted(cuts)
    out = {"mode": "jsonl", "sources": [source_id], "clips": len(clips), "cuts": len(cuts),
           "timeline": [lo, hi], "window_frames": args.window_frames,
           "overlap_frames": args.overlap_frames, "stride": args.window_frames - args.overlap_frames,
           "window_sec": window_src / args.fps, "missing_frames": 0}
    for m in ("stride", "cut_aligned"):
        for kind in ("continuation", "shot_cut", "multi_cut"):
            out[f"{m}/{kind}"] = 0
            out[f"{m}/{kind}/no_gap"] = 0
        out[f"{m}/total"] = 0
    start = lo
    while start + window_src <= hi:
        end = start + window_src
        in_cuts = cuts_in(sorted_cuts, start, end)
        gap = gap_in(merged, start, end)
        kind = classify(len(in_cuts))
        out["stride/total"] += 1
        out[f"stride/{kind}"] += 1
        if gap == 0:
            out[f"stride/{kind}/no_gap"] += 1
        start += stride_src
    for cut in sorted_cuts:
        for offset in cut_positions:
            start = cut - offset
            end = start + window_src
            if start < lo or end > hi:
                continue
            in_cuts = cuts_in(sorted_cuts, start, end)
            if len(in_cuts) != 1:
                continue
            gap = gap_in(merged, start, end)
            if gap > 0 and not args.allow_gap:
                continue
            out["cut_aligned/total"] += 1
            out["cut_aligned/shot_cut"] += 1
            if gap == 0:
                out["cut_aligned/shot_cut/no_gap"] += 1
    out["missing_frames"] = gap_in(merged, lo, hi)
    return out


def main() -> None:
    args = parse_args()
    frames_per_sec = args.fps / args.output_fps
    window_src = args.window_frames * frames_per_sec
    stride_out = args.stride or (args.window_frames - args.overlap_frames)
    stride_src = stride_out * frames_per_sec
    cut_positions = [float(x) * frames_per_sec for x in args.cut_positions.split(",") if x.strip() != ""]

    if args.all_sources:
        clips_by, cuts_by, n_lines = load_all_sources(args.source_jsonl, args.max_lines)
        agg: dict[str, int] = {}
        n_src = 0
        for source_id, clips in clips_by.items():
            stats = plan_one(source_id, clips, cuts_by.get(source_id, set()), args,
                             window_src, stride_src, cut_positions)
            if stats is None:
                continue
            n_src += 1
            for key, value in stats.items():
                if isinstance(value, int):
                    agg[key] = agg.get(key, 0) + value
        agg["sources"] = n_src
        agg["scanned_lines"] = n_lines
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("", encoding="utf-8")
        print(json.dumps(agg, ensure_ascii=False, indent=2))
        if args.output_stats:
            args.output_stats.parent.mkdir(parents=True, exist_ok=True)
            args.output_stats.write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    if args.clip_dir:
        if not args.source_id:
            raise SystemExit("--source-id is required with --clip-dir")
        sources, clips, cuts, n_lines = build_from_clip_dir(args.clip_dir, args.source_id, args.fps)
        mode = "clip_dir"
    else:
        if not args.source_jsonl:
            raise SystemExit("either --clip-dir/--source-id or --source-jsonl/--source-contains is required")
        sources, clips, cuts, n_lines = load_from_jsonl(args.source_jsonl, args.source_contains, args.max_lines)
        mode = "jsonl"
    print(f"mode={mode} scanned_lines={n_lines} sources={len(sources)} clips={len(clips)} cuts={len(cuts)}")

    if not clips:
        raise SystemExit("no clips matched")

    spans = sorted((v[0], v[1]) for v in clips.values())
    merged = merge_intervals(spans)
    lo, hi = merged[0][0], merged[-1][1]
    sorted_cuts = sorted(cuts)
    print(f"timeline {lo}..{hi} ({(hi - lo) / args.fps:.1f}s) coverage_intervals={len(merged)}")
    print(f"window {args.window_frames} output frames @{args.output_fps:g}fps = "
          f"{args.window_frames / args.output_fps:.4f}s = {window_src:.3f} source frames @{args.fps:g}fps")

    windows = []
    start = lo
    while start + window_src <= hi:
        end = start + window_src
        in_cuts = cuts_in(sorted_cuts, start, end)
        gap = gap_in(merged, start, end)
        windows.append({"mode": "stride", "start_frame": start, "end_frame": end,
                        "cut_frames": in_cuts, "gap_frames": gap, "kind": classify(len(in_cuts))})
        start += stride_src

    for cut in sorted_cuts:
        for offset in cut_positions:
            start = cut - offset
            end = start + window_src
            if start < lo or end > hi:
                continue
            in_cuts = cuts_in(sorted_cuts, start, end)
            if len(in_cuts) != 1:
                continue
            gap = gap_in(merged, start, end)
            if gap > 0 and not args.allow_gap:
                continue
            windows.append({"mode": "cut_aligned", "start_frame": start, "end_frame": end,
                            "cut_frames": in_cuts, "gap_frames": gap, "kind": "shot_cut"})

    seen, unique = set(), []
    for w in windows:
        key = (w["start_frame"], w["end_frame"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(w)

    stats = {"mode": mode, "sources": sources, "clips": len(clips), "cuts": len(cuts),
             "timeline": [lo, hi], "window_frames": args.window_frames,
             "overlap_frames": args.overlap_frames, "stride": stride_out,
             "output_fps": args.output_fps, "source_fps": args.fps,
             "window_sec": args.window_frames / args.output_fps,
             "missing_frames": sum(w["gap_frames"] for w in unique)}
    for m in ("stride", "cut_aligned"):
        subset = [w for w in unique if w["mode"] == m]
        for kind in ("continuation", "shot_cut", "multi_cut"):
            sel = [w for w in subset if w["kind"] == kind]
            stats[f"{m}/{kind}"] = len(sel)
            stats[f"{m}/{kind}/no_gap"] = sum(1 for w in sel if w["gap_frames"] == 0)
        stats[f"{m}/total"] = len(subset)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    window_sec = args.window_frames / args.output_fps
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for w in unique:
            # Snap forward onto the output-frame grid so resolve_media_window()
            # cannot round the window start before the planned source range.
            start_ticks = math.ceil(w["start_frame"] / args.fps * args.output_fps - 1e-9)
            start_sec = start_ticks / args.output_fps
            record = {**w, "start_sec": start_sec, "end_sec": start_sec + window_sec}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.output_stats:
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(unique)} window plans -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
