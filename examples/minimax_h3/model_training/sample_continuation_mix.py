#!/usr/bin/env python3
"""Sample a mixed continuation training index with per-kind quotas.

Kinds:

* continuation : one-shot window with the 39-frame prefix (masked-av-v14);
                  seeded from the existing cache manifest when possible;
* shot_cut     : window containing exactly one shot cut, with a prefix;
* new_shot     : one-shot window, NO prefix (prompt carries the identity cue);
* plain        : one-shot window, NO prefix, plain caption prompt.

Cross-fragment windows carry ``segments`` with absolute frame ranges plus each
fragment's own start frame, so the cache builder can decode fragments and
stitch the window.  Original fragments keep their AAC audio; records are marked
``audio_from_video`` so the builder can pull the audio from the fragment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.minimax_h3.model_training.plan_shot_windows import (  # noqa: E402
    cuts_in,
    gap_in,
    merge_intervals,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-jsonl", required=True)
    p.add_argument("--base-continuation-manifest", type=str, default=None)
    p.add_argument("--output-index", type=Path, required=True)
    p.add_argument("--output-stats", type=Path, default=None)
    p.add_argument("--window-frames", type=int, default=345)
    p.add_argument("--overlap-frames", type=int, default=39)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--output-fps", type=float, default=24.0,
                   help="H3 output fps.  Windows are 345 output frames = 14.375 s of "
                        "source video, not 345 source frames (25 fps source).")
    p.add_argument("--gap-tolerance", type=int, default=39)
    p.add_argument("--cut-position-low", type=float, default=0.15)
    p.add_argument("--cut-position-high", type=float, default=0.85)
    p.add_argument("--cut-aligned-offsets", type=str, default="117,172,232",
                   help="Output-frame positions of the cut inside a cut-aligned shot_cut "
                        "window; used to top up the shot_cut pool when stride windows run out.")
    p.add_argument("--quota-continuation", type=int, default=45000)
    p.add_argument("--quota-shot-cut", type=int, default=28000)
    p.add_argument("--quota-new-shot", type=int, default=15000)
    p.add_argument("--quota-plain", type=int, default=12000)
    p.add_argument("--per-source-cap", type=int, default=60)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def load_all_with_captions(path: str, max_lines: int | None):
    clips_by: dict[str, dict[str, tuple[int, int, float]]] = {}
    cuts_by: dict[str, set[int]] = {}
    captions: dict[str, str] = {}
    n_lines = 0
    with open(path) as handle:
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
                if not video_path or not frame_range:
                    continue
                clips.setdefault(
                    video_path,
                    # Inclusive frame_range -> half-open span (no 1-frame holes).
                    (int(frame_range[0]), int(frame_range[1]) + 1,
                     float(clip.get("duration_sec") or 0.0)),
                )
                if video_path not in captions:
                    cap = clip.get("caption") or {}
                    text = cap.get("audio_video_description") if isinstance(cap, dict) else None
                    if text:
                        captions[video_path] = str(text)
            for cut in record.get("cut_points") or []:
                frame = cut.get("next_start_frame")
                if isinstance(frame, (int, float)):
                    cuts.add(int(frame))
    return clips_by, cuts_by, captions, n_lines


def make_record(sample_id, sequence_id, segments, start_sec, end_sec, args,
                kind, prefix_mode, prompt, reused=False, cache_path=None):
    leaf = segments[0]
    return {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "video_path": leaf["video_path"],
        "audio_path": None,
        "audio_missing": True,
        "audio_from_video": True,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "window_frames": args.window_frames,
        "overlap_frames": args.overlap_frames,
        "hard_core_frames": 0,
        "transition_frames": 0,
        "conditioning_mode": "masked-av-v14",
        "prefix_mode": prefix_mode,
        "sample_kind": kind,
        "prompt": prompt,
        "prompt_source": "650k_caption",
        "prompt_cleaning_version": "v1",
        "split": "train",
        "status": "accepted",
        "segments": segments,
        "reused": reused,
        "cache_path": cache_path,
    }


def build_segments(items, start_sec: float, end_sec: float, src_fps: float) -> list[dict]:
    """Intersect a [start_sec, end_sec) window with the source clips (source frames)."""
    segs = []
    for vp, (cs, ce, _dur) in items:
        a = max(cs / src_fps, start_sec)
        b = min(ce / src_fps, end_sec)
        if b <= a:
            continue
        segs.append({
            "video_path": vp,
            "clip_start_frame": cs,
            "clip_end_frame": ce,
            "start_frame": int(math.floor(a * src_fps)),
            "end_frame": int(math.ceil(b * src_fps)),
            "start_sec": a,
            "end_sec": b,
        })
    return segs


def snap_to_output_grid(start_frames: float, src_fps: float, out_fps: float) -> float:
    """Round a window start forward onto an exact output-frame boundary.

    ``resolve_media_window`` rounds ``start_sec * out_fps``, which can land up to
    half an output frame *before* the requested start.  Snapping forward keeps the
    encoded window inside the source intervals the segments describe.
    """
    ticks = math.ceil(start_frames / src_fps * out_fps - 1e-9)
    return ticks / out_fps * src_fps


def segments_gap(segments: list[dict], start_sec: float, end_sec: float) -> float:
    """Seconds of the window not covered by contiguous source (0 == fully covered)."""
    if not segments:
        return end_sec - start_sec
    gap = max(0.0, segments[0]["start_sec"] - start_sec)
    gap += max(0.0, end_sec - segments[-1]["end_sec"])
    gap += sum(abs(segments[i + 1]["start_sec"] - segments[i]["end_sec"])
               for i in range(len(segments) - 1))
    return gap


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    fps = 25.0
    frames_per_sec = fps / args.output_fps
    window_sec = args.window_frames / args.output_fps
    window = args.window_frames * frames_per_sec          # source frames covered
    stride = (args.stride or (args.window_frames - args.overlap_frames)) * frames_per_sec
    quotas = {
        "shot_cut": args.quota_shot_cut,
        "new_shot": args.quota_new_shot,
        "plain": args.quota_plain,
    }
    pools: dict[str, list[dict]] = {"continuation_reuse": [], "continuation_supp": [],
                                    "shot_cut": [], "new_shot": [], "plain": []}
    cut_aligned_offsets = [float(x) * frames_per_sec for x in args.cut_aligned_offsets.split(",")
                           if x.strip() != ""]
    seen_shot_cut: set[tuple[str, float]] = set()

    base_records: list[dict] = []
    if args.base_continuation_manifest:
        with Path(args.base_continuation_manifest).open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("split") != "train":
                    continue
                base_records.append(rec)
                pools["continuation_reuse"].append({
                    "sample_id": rec.get("sample_id"),
                    "sequence_id": rec.get("sequence_id"),
                    "video_path": rec.get("video_path"),
                    "prompt": rec.get("prompt"),
                    "sample_kind": "continuation",
                    "prefix_mode": "masked",
                    "reused": True,
                    "cache_path": rec.get("cache_path"),
                    "cache_metadata": rec.get("metadata"),
                })
        rng.shuffle(pools["continuation_reuse"])
        pools["continuation_reuse"] = pools["continuation_reuse"][:args.quota_continuation]
        print(f"reused continuation windows: {len(pools['continuation_reuse'])}")

    need_supp = max(0, args.quota_continuation - len(pools["continuation_reuse"]))
    clips_by, cuts_by, captions, n_lines = load_all_with_captions(args.source_jsonl, args.max_lines)
    sources = list(clips_by.keys())
    rng.shuffle(sources)
    print(f"scanned lines={n_lines} sources={len(sources)} captions={len(captions)} need_supp={need_supp}")

    def make_shot_cut(source_id, items, merged, captions, start, end, cuts_in_window, pools, per_source):
        start = snap_to_output_grid(start, fps, args.output_fps)
        end = start + window
        start_sec = start / fps
        end_sec = start_sec + window_sec
        segs = build_segments(items, start_sec, end_sec, fps)
        if len(segs) < 2 or segments_gap(segs, start_sec, end_sec) > 1e-6:
            return False
        cap_a = captions.get(segs[0]["video_path"], "")
        cap_b = captions.get(segs[-1]["video_path"], "")
        cut = cuts_in_window[0]
        dt = (cut - start) / fps
        prompt = (f"{cap_a}\nAt {dt:.2f}s the camera hard-cuts to a new shot. {cap_b}\n"
                  "Keep the same subject identity, appearance and scene across the cut.")
        pools["shot_cut"].append(make_record(
            f"{source_id}:shotcut:{start}", source_id, segs, start_sec, end_sec,
            args, "shot_cut", "masked", prompt,
        ))
        per_source["shot_cut"] += 1
        return True

    for source_id in sources:
        if (len(pools["shot_cut"]) >= quotas["shot_cut"]
                and len(pools["new_shot"]) >= quotas["new_shot"]
                and len(pools["plain"]) >= quotas["plain"]
                and len(pools["continuation_supp"]) >= need_supp):
            break
        clips = clips_by[source_id]
        cuts = sorted(cuts_by.get(source_id, set()))
        if not clips:
            continue
        items = sorted(clips.items(), key=lambda kv: kv[1][0])
        merged = merge_intervals([(v[0], v[1]) for _, v in items])
        lo, hi = merged[0][0], merged[-1][1]
        per_source = {"continuation_supp": 0, "shot_cut": 0, "new_shot": 0, "plain": 0}

        # One extra source frame of slack absorbs the forward grid snap.
        long_shots = [(vp, v[0], v[1]) for vp, v in items if v[1] - v[0] >= window + frames_per_sec]
        targets = {"continuation_supp": need_supp, "new_shot": quotas["new_shot"], "plain": quotas["plain"]}
        for vp, cs, ce in long_shots:
            open_kinds = [k for k in targets
                          if len(pools[k]) < targets[k] and per_source[k] < args.per_source_cap]
            if not open_kinds:
                break
            kind = max(open_kinds, key=lambda k: (targets[k] - len(pools[k])) / max(1, targets[k]))
            caption = captions.get(vp, "")
            if not caption:
                continue
            start_sec = snap_to_output_grid(cs, fps, args.output_fps) / fps
            end_sec = start_sec + window_sec
            segs = build_segments(items, start_sec, end_sec, fps)
            if len(segs) != 1 or segments_gap(segs, start_sec, end_sec) > 1e-6:
                continue
            if kind == "continuation_supp":
                prompt = caption
                prefix = "masked"
            elif kind == "new_shot":
                prompt = f"{caption}\nA new shot begins. Keep the same subject identity, appearance and scene as the previous shot."
                prefix = "none"
            else:
                prompt = caption
                prefix = "none"
            pools[kind].append(make_record(
                f"{source_id}:{kind}:{cs}", source_id, segs, start_sec, end_sec,
                args, "continuation" if kind == "continuation_supp" else kind, prefix, prompt,
            ))
            per_source[kind] += 1

        start = lo
        while start + window <= hi:
            end = start + window
            in_cuts = cuts_in(cuts, start, end)
            if len(in_cuts) == 1 and per_source["shot_cut"] < args.per_source_cap \
                    and len(pools["shot_cut"]) < quotas["shot_cut"]:
                cut = in_cuts[0]
                pos = (cut - start) / window
                gap = gap_in(merged, start, end)
                if args.cut_position_low <= pos <= args.cut_position_high and gap <= args.gap_tolerance:
                    if (source_id, start) not in seen_shot_cut:
                        seen_shot_cut.add((source_id, start))
                        make_shot_cut(source_id, items, merged, captions, start, end,
                                      in_cuts, pools, per_source)
            start += stride

        # Cut-aligned top-up: place a single cut at a controlled offset inside the
        # window, which keeps both sides long enough and diversifies root positions.
        for cut in cuts:
            if (per_source["shot_cut"] >= args.per_source_cap
                    or len(pools["shot_cut"]) >= quotas["shot_cut"]):
                break
            for offset in cut_aligned_offsets:
                start = cut - offset
                end = start + window
                if start < lo or end > hi:
                    continue
                in_cuts = cuts_in(cuts, start, end)
                if len(in_cuts) != 1:
                    continue
                if gap_in(merged, start, end) > args.gap_tolerance:
                    continue
                if (source_id, start) in seen_shot_cut:
                    continue
                seen_shot_cut.add((source_id, start))
                if make_shot_cut(source_id, items, merged, captions, start, end,
                                 in_cuts, pools, per_source):
                    break

    deficits = {
        "continuation": args.quota_continuation - len(pools["continuation_reuse"]) - len(pools["continuation_supp"]),
        "new_shot": args.quota_new_shot - len(pools["new_shot"]),
        "plain": args.quota_plain - len(pools["plain"]),
    }
    if any(v > 0 for v in deficits.values()) and base_records:
        cursor = 0
        for key, need in deficits.items():
            prefix = "masked" if key == "continuation" else "none"
            kind = "continuation" if key == "continuation" else key
            for _ in range(max(0, need)):
                src = base_records[cursor % len(base_records)]
                cursor += 1
                prompt = src.get("prompt") or ""
                if key == "new_shot":
                    prompt = prompt + "\nA new shot begins. Keep the same subject identity, appearance and scene as the previous shot."
                pools["continuation_reuse" if key == "continuation" else key].append({
                    "sample_id": f"{src.get('sample_id')}:{key}",
                    "sequence_id": src.get("sequence_id"),
                    "video_path": src.get("video_path"),
                    "audio_path": src.get("audio_path"),
                    "audio_missing": bool(src.get("audio_missing")),
                    "start_sec": src.get("start_sec"),
                    "end_sec": src.get("end_sec"),
                    "window_frames": args.window_frames,
                    "overlap_frames": args.overlap_frames,
                    "conditioning_mode": "masked-av-v14",
                    "prefix_mode": prefix,
                    "sample_kind": kind,
                    "prompt": prompt,
                    "prompt_source": src.get("prompt_source", "cache_manifest"),
                    "prompt_cleaning_version": src.get("prompt_cleaning_version", "v1"),
                    "split": "train",
                    "status": "accepted",
                    "segments": [{"video_path": src.get("video_path"), "clip_start_frame": None,
                                  "start_frame": None, "end_frame": None}],
                    "reused": True,
                    "cache_path": src.get("cache_path"),
                    "cache_metadata": src.get("metadata"),
                })

    print(json.dumps({k: len(v) for k, v in pools.items()}, ensure_ascii=False))
    out = []
    out.extend(pools["continuation_reuse"][:args.quota_continuation])
    out.extend(pools["continuation_supp"][:need_supp])
    for kind, quota in (("shot_cut", args.quota_shot_cut), ("new_shot", args.quota_new_shot), ("plain", args.quota_plain)):
        out.extend(pools[kind][:quota])
    rng.shuffle(out)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    with args.output_index.open("w", encoding="utf-8") as handle:
        for rec in out:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = {
        "total": len(out),
        "continuation": sum(1 for r in out if r.get("sample_kind") == "continuation"),
        "continuation_reused": sum(1 for r in out if r.get("sample_kind") == "continuation" and r.get("reused")),
        "shot_cut": sum(1 for r in out if r.get("sample_kind") == "shot_cut"),
        "new_shot": sum(1 for r in out if r.get("sample_kind") == "new_shot"),
        "plain": sum(1 for r in out if r.get("sample_kind") == "plain"),
        "sources": len(sources),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.output_stats:
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(out)} index records -> {args.output_index}")


if __name__ == "__main__":
    main()
