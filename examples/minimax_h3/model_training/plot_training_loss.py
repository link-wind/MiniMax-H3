#!/usr/bin/env python3
"""Plot continuation training loss from ModelLogger's loss.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, help="Training output directory containing loss.csv")
    parser.add_argument("--output", default=None, help="PNG path; defaults to <log-dir>/loss.png")
    parser.add_argument("--smooth", type=int, default=25, help="Moving-average window")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    csv_path = log_dir / "loss.csv"
    output_path = Path(args.output) if args.output else log_dir / "loss.png"
    if not csv_path.exists():
        raise FileNotFoundError(f"loss log not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("key") != "loss":
                continue
            try:
                rows.append((int(row["step"]), float(row["value"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no loss rows found in {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [item[0] for item in rows]
    values = [item[1] for item in rows]
    window = max(1, min(args.smooth, len(values)))
    smooth = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        smooth.append(running / min(index + 1, window))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax_log) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax.plot(steps, values, color="#9aa4b2", alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(steps, smooth, color="#2563eb", linewidth=2.0, label=f"moving average ({window})")
    ax.set_ylabel("loss")
    ax.set_title("MiniMax-H3 mask-v14 continuation LoRA training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    ax_log.plot(steps, values, color="#9aa4b2", alpha=0.35, linewidth=0.8)
    ax_log.plot(steps, smooth, color="#dc2626", linewidth=2.0)
    ax_log.set_yscale("log")
    ax_log.set_xlabel("optimizer step")
    ax_log.set_ylabel("loss (log)")
    ax_log.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"saved {output_path} ({len(rows)} loss points)")


if __name__ == "__main__":
    main()
