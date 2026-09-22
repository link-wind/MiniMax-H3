#!/usr/bin/env python3
"""Verify cross-fragment window extraction (video + audio) on one source.

Source fragments are 25 fps shot clips; H3 windows are 24 fps (345 frames =
14.375 s).  A window is therefore defined in *seconds* and every output frame
is mapped to the fragment covering its timestamp, which also handles windows
spanning several fragments.  Audio is taken from the aligned per-fragment WAV
(``audio/raw/<stem>_origin.wav``), stitched on the same timeline and resampled
to 32 kHz, matching the training cache pipeline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

VIDEO_FPS = 24.0   # H3 output fps
SRC_FPS = 25.0     # original fragment fps
AUDIO_RATE = 32000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clip-dir", required=True)
    p.add_argument("--audio-dir", default=None,
                   help="dir with <stem>_origin.wav; falls back to mp4 audio via ffmpeg")
    p.add_argument("--source-id", required=True)
    p.add_argument("--window-frames", type=int, default=345)
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/multi_segment_check"))
    return p.parse_args()


def clip_index(name: str) -> int:
    return int(name.rsplit(".", 1)[0].rsplit("_", 1)[1])


def probe_fragment(path: str) -> tuple[int, float]:
    import json
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate",
        "-of", "json", path], text=True))
    stream = (info.get("streams") or [{}])[0]
    nb_frames = int(stream.get("nb_frames") or 0)
    num, _, den = str(stream.get("r_frame_rate") or f"{SRC_FPS:.0f}/1").partition("/")
    fps = float(num) / float(den or 1)
    if nb_frames <= 0:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            text=True).strip())
        nb_frames = int(round(dur * fps))
    return nb_frames, fps


def collect_fragments(clip_dir: str, source_id: str):
    names = [n for n in os.listdir(clip_dir) if n.startswith(source_id + "_") and n.endswith(".mp4")]
    names.sort(key=clip_index)
    frags = []
    cursor = 0.0
    for name in names:
        path = os.path.join(clip_dir, name)
        nb_frames, fps = probe_fragment(path)
        dur = nb_frames / fps
        frags.append({"path": path, "stem": name[:-4], "start_sec": cursor,
                      "end_sec": cursor + dur, "dur": dur, "frames": nb_frames})
        cursor += dur
    return frags, cursor


def _read_wav_window(path: str, start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    import soundfile as sf
    info = sf.info(path)
    rate = int(info.samplerate)
    data, rate = sf.read(path, always_2d=True, dtype="float32",
                         start=max(0, int(round(start_sec * rate))),
                         stop=int(round(end_sec * rate)))
    return np.asarray(data), int(rate)


def _read_mp4_window(path: str, start_sec: float, end_sec: float) -> tuple[np.ndarray, int]:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ss", f"{start_sec:.6f}",
           "-t", f"{max(0.0, end_sec - start_sec):.6f}",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(AUDIO_RATE), "-"]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    return audio, AUDIO_RATE


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.shape[0] == 0:
        return audio
    n_out = int(round(audio.shape[0] * dst_rate / src_rate))
    src_idx = np.arange(audio.shape[0], dtype=np.float64)
    dst_idx = np.linspace(0, audio.shape[0] - 1, n_out, dtype=np.float64)
    return np.stack([np.interp(dst_idx, src_idx, audio[:, c]) for c in range(audio.shape[1])], axis=1)


def main() -> None:
    args = parse_args()
    frags, total = collect_fragments(args.clip_dir, args.source_id)
    total_src_frames = sum(f["frames"] for f in frags)
    print(f"fragments={len(frags)} total_sec={total:.3f} total_src_frames={total_src_frames} "
          f"(= {total_src_frames / SRC_FPS:.3f}s @25fps)")
    if not frags:
        raise SystemExit("no fragments found")

    win_sec = args.window_frames / VIDEO_FPS
    picked = []
    for f in frags:
        for offset in (0.0, 0.35, 0.65):
            s = f["start_sec"] + offset * f["dur"]
            if s + win_sec > total:
                continue
            inside = [g for g in frags if s < g["start_sec"] < s + win_sec]
            if inside and s not in picked:
                picked.append(s)
                break
        if len(picked) >= args.examples:
            break
    print(f"picked windows: {[round(s, 3) for s in picked]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import imageio
    ok_all = True
    for wi, start in enumerate(picked):
        frames = []
        readers, frame_counts, src_counts = {}, {}, {}
        missing = 0
        for i in range(args.window_frames):
            t = start + i / VIDEO_FPS
            frag = next((f for f in frags if f["start_sec"] <= t < f["end_sec"]), None)
            if frag is None:
                missing += 1
                if frames:
                    frames.append(frames[-1].copy())
                continue
            if frag["path"] not in readers:
                readers[frag["path"]] = imageio.get_reader(frag["path"])
                frame_counts[frag["path"]] = frag["frames"]
            reader = readers[frag["path"]]
            src_frame = int(round((t - frag["start_sec"]) * SRC_FPS))
            src_frame = min(max(src_frame, 0), max(0, frame_counts[frag["path"]] - 1))
            frames.append(reader.get_data(src_frame))
            key = os.path.basename(frag["path"])[-9:]
            src_counts[key] = src_counts.get(key, 0) + 1
        for r in readers.values():
            r.close()

        # audio: stitch per-fragment audio on the same timeline
        chunks = []
        for frag in frags:
            a = max(start, frag["start_sec"])
            b = min(start + win_sec, frag["end_sec"])
            if b <= a:
                continue
            wav = None
            if args.audio_dir:
                cand = os.path.join(args.audio_dir, frag["stem"] + "_origin.wav")
                if os.path.exists(cand):
                    wav = cand
            try:
                data, rate = (_read_wav_window(wav, a - frag["start_sec"], b - frag["start_sec"])
                              if wav else _read_mp4_window(frag["path"], a - frag["start_sec"], b - frag["start_sec"]))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] audio read failed for {frag['stem']}: {exc}")
                continue
            chunks.append(resample_linear(np.asarray(data), rate, AUDIO_RATE))
        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype="float32")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        target = int(round(win_sec * AUDIO_RATE))
        if audio.shape[0] < target:
            audio = np.pad(audio, ((0, target - audio.shape[0]), (0, 0)))
        else:
            audio = audio[:target]

        ok = missing == 0 and len(frames) == args.window_frames and audio.shape[0] == target
        ok_all = ok_all and ok
        print(f"window {wi}: start={start:.3f}s frames={len(frames)} missing={missing} "
              f"audio_samples={audio.shape[0]} expected={target} "
              f"fragments_used={src_counts} {'OK' if ok else 'FAIL'}")
        out = args.output_dir / f"window_{wi}_start{start:.2f}.mp4"
        writer = imageio.get_writer(str(out), fps=VIDEO_FPS, macro_block_size=1)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  saved {out}")
    print("RESULT:", "ALL OK" if ok_all else "HAS FAILURES")


if __name__ == "__main__":
    main()
