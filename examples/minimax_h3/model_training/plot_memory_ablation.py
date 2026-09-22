#!/usr/bin/env python3
"""Plot the memory-slot ablation detail log produced by eval_memory_ablation.py.

Reads ``memory_ablation.details.jsonl`` and writes a four-panel figure:

  (a) per-arm mean loss with error bars
  (b) per-sample paired contrasts (the causal readout)
  (c) the contrast stratified by window length -- memory should matter more on
      long windows, so a flat curve here is itself a finding
  (d) both-vs-swap scatter, which shows whether the gap is systematic or a few
      outliers

Usage:
    python plot_memory_ablation.py --log-dir outputs/continuation_lora/memory_ablation_v2_full
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_ORDER = ["both", "stm", "ltm", "none", "swap"]
ARM_COLORS = {
    "both": "#1f77b4",
    "stm": "#2ca02c",
    "ltm": "#9467bd",
    "none": "#7f7f7f",
    "swap": "#d62728",
}


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _per_sample_means(rows: list[dict], key: str = "loss") -> dict[tuple[str, int], float]:
    buckets: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        if arm == "swap" and not row.get("swap_applied", False):
            continue
        value = row.get(key)
        if value is not None:
            buckets[(arm, int(row["index"]))].append(float(value))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def _mean_stderr(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, (var / len(values)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--details", default=None, help="Defaults to <log-dir>/memory_ablation.details.jsonl")
    parser.add_argument("--output", default=None, help="Defaults to <log-dir>/memory_ablation.png")
    parser.add_argument("--metric", default="loss", choices=("loss", "video_loss", "audio_loss"))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    details_path = Path(args.details) if args.details else log_dir / "memory_ablation.details.jsonl"
    output_path = Path(args.output) if args.output else log_dir / "memory_ablation.png"
    if not details_path.exists():
        raise FileNotFoundError(f"detail log not found: {details_path}")

    rows = _load(details_path)
    if not rows:
        raise SystemExit(f"no records in {details_path}")
    sample_mean = _per_sample_means(rows, args.metric)
    window_frames = {int(r["index"]): int(r.get("window_frames") or 0) for r in rows}
    arms = [a for a in ARM_ORDER if any(arm == a for arm, _ in sample_mean)]
    indices = sorted({i for _, i in sample_mean})

    def paired(a: str, b: str) -> list[tuple[int, float]]:
        out = []
        for index in indices:
            ka, kb = (a, index), (b, index)
            if ka in sample_mean and kb in sample_mean:
                out.append((index, sample_mean[ka] - sample_mean[kb]))
        return out

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0][0]
    means, errs = [], []
    for arm in arms:
        values = [sample_mean[(arm, i)] for i in indices if (arm, i) in sample_mean]
        mean, err = _mean_stderr(values)
        means.append(mean)
        errs.append(err)
    ax.bar(arms, means, yerr=errs, color=[ARM_COLORS.get(a, "0.5") for a in arms], capsize=5)
    for i, (arm, mean) in enumerate(zip(arms, means)):
        n = sum(1 for idx in indices if (arm, idx) in sample_mean)
        ax.text(i, mean, f"{mean:.4f}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"{args.metric} (lower is better)")
    ax.set_title(f"(a) per-arm mean {args.metric} ± stderr", loc="left")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0][1]
    contrasts = [("swap", "both"), ("none", "both"), ("stm", "both"), ("ltm", "both")]
    labels, cm, ce, wins = [], [], [], []
    for a, b in contrasts:
        # swap is already restricted to transplanted samples inside
        # _per_sample_means, so paired() never sees a fallback entry.
        diffs = [d for _, d in paired(a, b)]
        if not diffs:
            continue
        mean, err = _mean_stderr(diffs)
        labels.append(f"{a} − {b}")
        cm.append(mean)
        ce.append(err)
        wins.append(sum(1 for d in diffs if d > 0))
    ax.barh(labels, cm, xerr=ce, color="#d62728", capsize=5)
    ax.axvline(0, color="k", lw=1)
    for i, (mean, n) in enumerate(zip(cm, wins)):
        ax.text(mean, i, f"  {mean:+.4f} (first arm worse on {n})", va="center", fontsize=9)
    ax.set_xlabel("per-sample paired difference (positive = first arm worse)")
    ax.set_title("(b) paired contrasts", loc="left")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1][0]
    none_diffs = paired("none", "both")
    if none_diffs:
        # Build the bins once so the bar heights and their error bars cannot
        # drift apart.
        panels = []
        for lo, hi in ((0, 60), (60, 100), (100, 160), (160, 220), (220, 10 ** 6)):
            group = [d for idx, d in none_diffs if lo <= window_frames.get(idx, 0) < hi]
            if not group:
                continue
            mean, err = _mean_stderr(group)
            panels.append((f"{lo}–{hi if hi < 10 ** 6 else '+'}", mean, err, len(group)))
        if panels:
            ax.bar([p[0] for p in panels], [p[1] for p in panels],
                   yerr=[p[2] for p in panels], color="#7f7f7f", capsize=5)
            for i, (_, mean, _, n) in enumerate(panels):
                ax.text(i, mean, f"{mean:+.3f}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("window length (frames)")
    ax.set_ylabel("none − both")
    ax.set_title("(c) memory benefit vs window length", loc="left")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    xs = [sample_mean[("both", i)] for i in indices if ("both", i) in sample_mean and ("swap", i) in sample_mean]
    ys = [sample_mean[("swap", i)] for i in indices if ("both", i) in sample_mean and ("swap", i) in sample_mean]
    ax.scatter(xs, ys, s=12, alpha=0.5, color="#d62728")
    if xs:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (slot ignored)")
        ax.legend(loc="upper left")
    ax.set_xlabel("both (own slots)")
    ax.set_ylabel("swap (foreign slots)")
    ax.set_title("(d) does the output follow the slot?", loc="left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"MiniMax-H3 memory-slot ablation — {details_path.stem}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    print(f"saved {output_path} ({len(sample_mean)} arm/sample points)")


if __name__ == "__main__":
    main()
