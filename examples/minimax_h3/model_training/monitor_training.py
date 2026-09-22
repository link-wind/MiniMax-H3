#!/usr/bin/env python3
"""Monitor H3 continuation training (watch/incremental mode).

Two run modes:
  * snapshot: print current state once (best for one-off checks).
  * watch   : print only what CHANGED since the last invocation, using a JSON
              state file to persist the last-read offset of loss.csv.

Usage:
  # once-off
  python monitor_training.py --log-dir <out_dir>

  # incremental watch (agent-friendly): persists offsets in <state> and only
  # reports new rows since the previous call.
  python monitor_training.py --log-dir <out_dir> --watch --state /tmp/h3_monitor.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CRASH_MARKERS = [
    "FileNotFoundError",
    "RuntimeError",
    "OutOfMemory",
    "CUDA out of memory",
    "CUDA error",
    "Traceback (most recent call last)",
    "ChildFailedError",
    "ProcessExitedWithError",
    "NaN",
]


def read_by_key(log_dir: Path, wanted: str) -> list[tuple[int, float]]:
    csv_path = log_dir / "loss.csv"
    if not csv_path.exists():
        return []
    rows: list[tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("key") != wanted:
                continue
            try:
                rows.append((int(row["step"]), float(row["value"])))
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def read_loss(log_dir: Path) -> list[tuple[int, float]]:
    return read_by_key(log_dir, "loss")


def ema(values: list[float], alpha: float = 0.2) -> float | None:
    if not values:
        return None
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def trend(rows: list[tuple[int, float]], window: int = 20) -> str:
    if len(rows) < 3:
        return "insufficient-data"
    recent = rows[-window:]
    losses = [v for _, v in recent]
    half = max(1, len(losses) // 2)
    left = sum(losses[:half]) / half
    right = sum(losses[half:]) / max(1, len(losses) - half)
    rel = (left - right) / (left + 1e-9)
    if rel > 0.02:
        return "falling"
    if rel < -0.02:
        return "rising"
    return "flat"


def scan_logs(log_dir: Path) -> list[str]:
    hits: list[str] = []
    for pattern in ("train.log", "train.node*.log"):
        for log in sorted(log_dir.glob(pattern)):
            try:
                text = log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for marker in CRASH_MARKERS:
                if marker in text:
                    hits.append(f"{log.name}: found {marker!r}")
                    break
    return hits


def _read_csv_since(csv_path: Path, start_offset: int):
    if not csv_path.exists():
        return start_offset, []
    size = csv_path.stat().st_size
    if size <= start_offset:
        return start_offset, []
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        handle.seek(start_offset)
        for row in csv.DictReader(handle):
            try:
                rows.append({"step": int(row["step"]), "key": row["key"], "value": float(row["value"])})
            except (TypeError, ValueError, KeyError):
                continue
    return size, rows


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def snapshot(log_dir: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = read_loss(log_dir)
    if not rows:
        return f"[{now}] step=<model-loading/ none yet> (loss.csv not written or empty)"
    last_step, last_loss = rows[-1]
    line = (
        f"[{now}] step={last_step} loss={last_loss:.5f} "
        f"ema={ema([v for _, v in rows]):.5f} "
        f"avg_last10={sum(v for _, v in rows[-10:]) / min(10, len(rows)):.5f} "
        f"trend={trend(rows)} total_rows={len(rows)}"
    )
    vrows = read_by_key(log_dir, "video_loss")
    arows = read_by_key(log_dir, "audio_loss")
    if vrows and arows:
        by_step = {st: val for st, val in vrows}
        alatest = arows[-1]
        vlatest = by_step.get(alatest[0], vrows[-1][1])
        line += f" video_loss(last {vrows[-1][0]})={vrows[-1][1]:.5f} audio_loss(last {alatest[0]})={alatest[1]:.5f}"
        line += f" [video_trend={trend(vrows)} audio_trend={trend(arows)}]"
    crashes = scan_logs(log_dir)
    if crashes:
        line += " | CRASH: " + "; ".join(crashes[-3:])
    return line


def watch(log_dir: Path, state_path: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    csv_path = log_dir / "loss.csv"
    state = _load_state(state_path)
    prev_offset = int(state.get("csv_offset", 0))
    new_offset, new_rows = _read_csv_since(csv_path, prev_offset)

    if not csv_path.exists() or new_offset <= 0:
        # No data yet (still loading). Report that we're waiting.
        _save_state(state_path, {"csv_offset": prev_offset, "last_ts": now})
        return f"[{now}] watching... loss.csv not written yet (model still loading?)"

    new_loss = [r for r in new_rows if r["key"] == "loss"]
    total_rows = read_loss(log_dir)
    last_step = total_rows[-1][0] if total_rows else None

    def latest(key):
        vals = read_by_key(log_dir, key)
        return vals[-1][1] if vals else None

    # Change since last observation.
    delta = ""
    if state.get("csv_offset"):
        prev_last_step = state.get("last_step")
        if prev_last_step is not None and last_step is not None:
            delta = f"+{last_step - prev_last_step} steps since last check"
        elif new_loss:
            delta = f"+{len(new_loss)} new loss rows since last check"

    parts = [
        f"[{now}] step={last_step}",
        f"loss={latest('loss'):.5f}" if latest('loss') is not None else "loss=None",
    ]
    if delta:
        parts.append(delta)
    v, a = latest("video_loss"), latest("audio_loss")
    if v is not None and a is not None:
        vt = trend(read_by_key(log_dir, "video_loss"))
        at = trend(read_by_key(log_dir, "audio_loss"))
        parts.append(f"video_loss={v:.5f}({vt}) audio_loss={a:.5f}({at}) "
                     f"ratio(a/v)={a/max(v,1e-9):.3f}")
        if vt == "rising" and at == "falling":
            parts.append("**WARN: video rising while audio falling**")
    crashes = scan_logs(log_dir)
    if crashes:
        parts.append("CRASH: " + "; ".join(crashes[-3:]))

    _save_state(state_path, {"csv_offset": new_offset, "last_ts": now, "last_step": last_step})
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--watch", action="store_true",
                        help="Incremental mode: report only changes since last run.")
    parser.add_argument("--state", type=Path, default=None,
                        help="State file for --watch (persists the read offset).")
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        sys.exit(f"error: log dir not found: {args.log_dir}")

    if args.watch:
        state_path = args.state or (args.log_dir / ".monitor_state.json")
        print(watch(args.log_dir, state_path))
    else:
        print(snapshot(args.log_dir))


if __name__ == "__main__":
    main()
