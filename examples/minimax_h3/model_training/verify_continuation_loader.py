#!/usr/bin/env python3
"""Verify the library multi-segment loaders against an independent decode.

Builds window segments from the original 25 fps fragment clips, runs
``load_video_window_segments`` / ``load_audio_window_segments``, and then
re-reads the expected source frames directly to check the output is
pixel/time-exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.utils.continuation_lora import (  # noqa: E402
    AUDIO_SAMPLE_RATE,
    VIDEO_FPS,
    load_audio_window_segments,
    load_video_window_segments,
    resolve_media_window,
)
from examples.minimax_h3.model_training.plan_shot_windows import (  # noqa: E402
    build_from_clip_dir,
)

SRC_FPS = 25.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip-dir", required=True)
    p.add_argument("--audio-dir", default=None)
    p.add_argument("--source-id", required=True)
    p.add_argument("--start-sec", type=float, required=True)
    p.add_argument("--window-frames", type=int, default=345)
    p.add_argument("--checks", type=int, default=4, help="Source frames to compare exactly.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _sources, clips, _cuts, _n = build_from_clip_dir(args.clip_dir, args.source_id, SRC_FPS)
    items = sorted(((v[0], v[1], p) for p, v in clips.items()))

    start_sec = args.start_sec
    end_sec = start_sec + args.window_frames / VIDEO_FPS
    segments = []
    for cs, ce, path in items:
        a, b = max(cs / SRC_FPS, start_sec), min(ce / SRC_FPS, end_sec)
        if b > a:
            segments.append({
                "video_path": path,
                "clip_start_frame": cs,
                "start_frame": int(np.floor(a * SRC_FPS)),
                "end_frame": int(np.ceil(b * SRC_FPS)),
                "start_sec": a,
                "end_sec": b,
            })
    print(f"window [{start_sec:.3f}, {end_sec:.3f})s across {len(segments)} segments")
    for seg in segments:
        print(f"  {Path(seg['video_path']).name}  {seg['start_sec']:.3f}->{seg['end_sec']:.3f}s")

    window = resolve_media_window(start_sec, end_sec)
    frames = load_video_window_segments(segments, window)
    print(f"frames={len(frames)} (expected {args.window_frames})")

    import imageio
    rng = np.random.default_rng(0)
    check_ids = sorted(rng.choice(len(frames), size=min(args.checks, len(frames)), replace=False).tolist())
    readers, max_diff = {}, 0
    for frame_id in check_ids:
        t = (window.video_start_frame + frame_id) / VIDEO_FPS
        seg = next(s for s in segments if s["start_sec"] <= t < s["end_sec"])
        readers.setdefault(seg["video_path"], imageio.get_reader(seg["video_path"], "ffmpeg"))
        src_id = int(round((t - seg["start_sec"]) * SRC_FPS))
        expected = np.asarray(readers[seg["video_path"]].get_data(src_id))
        diff = int(np.abs(expected.astype(np.int32) - np.asarray(frames[frame_id]).astype(np.int32)).max())
        max_diff = max(max_diff, diff)
        print(f"  frame {frame_id}: src#{src_id} from {Path(seg['video_path']).name[-9:]} max|diff|={diff}")
    for reader in readers.values():
        reader.close()

    audio, rate, missing = load_audio_window_segments(
        segments, window, audio_dir=args.audio_dir, target_rate=AUDIO_SAMPLE_RATE)
    expected_samples = round((end_sec - start_sec) * AUDIO_SAMPLE_RATE)
    print(f"audio shape={tuple(audio.shape)} rate={rate} missing={missing} "
          f"expected_samples={expected_samples} rms={float(audio.pow(2).mean().sqrt()):.5f}")

    ok = (len(frames) == args.window_frames and max_diff == 0
          and tuple(audio.shape) == (2, expected_samples) and not missing
          and not torch.isnan(audio).any())
    print("RESULT:", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
