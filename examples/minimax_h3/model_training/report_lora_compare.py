#!/usr/bin/env python3
"""Collapse a ``run_lora_compare_eval_16gpu.sh`` output tree into one table.

Each job writes ``<OUTPUT_ROOT>/<arm>/<plan>/rollout_ablation_<arm>.json``.  Read
individually those are twelve JSON documents and the comparison lives in the
reader's head; this turns them into rows.

Three things the table refuses to hide:

* **Identity drift and motion are printed together.**  ``head_drift`` is an
  absolute frame difference against the sequence head, so an arm can improve it
  by simply moving less.  A row whose motion collapsed relative to base is
  flagged, because that is a change in the task, not a improvement in it.
* **Seam metrics sit next to the long-range ones.**  A run can be seamless at
  every cut and still drift away from the subject over a minute; those are
  different failures and averaging them loses the distinction.
* **The checkpoint behind each arm is printed.**  v5 and v9 differ in training
  recipe as well as in data volume, so a "winner" is a winner of the whole
  recipe.

Usage:
    python3 report_lora_compare.py --root outputs/continuation_lora/lora_compare
    python3 report_lora_compare.py --root ... --base base --out table.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Metric -> (column, lower_is_better).  Direction matters: reporting a raw number
# without it is how "the model stopped moving" gets read as an improvement.
METRICS: list[tuple[str, str, bool]] = [
    ("head_drift_mean", "identity_drift_mean", True),
    ("head_drift_last", "identity_drift_last", True),
    ("join_video_mad_mean", "seam_video_mad", True),
    ("join_audio_energy_mean", "seam_audio_energy", True),
    ("long_range_video_mad", "long_range_mad", True),
    ("motion_scale_mean", "motion_scale", False),
]

#: Below this fraction of the base arm's motion the row is flagged: the drift
#: number is no longer comparable to base's, because the task changed.
MOTION_COLLAPSE_RATIO = 0.7


def _load_jobs(root: Path) -> list[dict[str, Any]]:
    """One row per (plan, arm) pair found under the tree."""
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/rollout_ablation_*.json")):
        plan = summary_path.parent.name
        arm_dir = summary_path.parent.parent.name
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rows.append({"plan": plan, "arm": arm_dir, "error": f"unreadable: {exc}"})
            continue
        config = payload.get("config", {})
        for arm, entry in payload.get("arms", {}).items():
            row: dict[str, Any] = {
                "plan": plan, "arm": arm, "summary": str(summary_path),
                "lora_path": (config.get("lora_arms") or {}).get(arm, {}).get("lora_path"),
                "lora_scale": (config.get("lora_arms") or {}).get(arm, {}).get("lora_scale"),
                "peak_allocated_gib": entry.get("peak_allocated_gib"),
                "video_path": entry.get("video_path"),
            }
            for key, column, _lower in METRICS:
                row[column] = entry.get(key)
            joins = _spectral_from_joins(summary_path.parent, arm)
            row["seam_audio_spectral"] = joins.get("audio_spectral_jump")
            row["sampled_motion_mad"] = joins.get("sampled_motion_mad")
            rows.append(row)
    return rows


def _spectral_from_joins(job_dir: Path, arm: str) -> dict[str, float | None]:
    """The summary keeps a subset; the per-arm joins file has the rest."""
    path = job_dir / f"joins_{arm}.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    spectral = [
        item["audio_spectral_jump"] for item in payload.get("joins", [])
        if isinstance(item.get("audio_spectral_jump"), (int, float))
    ]
    return {
        "audio_spectral_jump": sum(spectral) / len(spectral) if spectral else None,
        "sampled_motion_mad": payload.get("video_metrics", {}).get(
            "sampled_motion_mean_absolute_difference"
        ),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _flag_motion(row: dict[str, Any], base: dict[str, Any] | None) -> str:
    """Name the failure a single number cannot: less drift because less happens."""
    if base is None or row is base:
        return ""
    motion, base_motion = row.get("motion_scale"), base.get("motion_scale")
    if not isinstance(motion, (int, float)) or not isinstance(base_motion, (int, float)):
        return ""
    if base_motion <= 0:
        return ""
    if motion < base_motion * MOTION_COLLAPSE_RATIO:
        return "MOTION↓"
    return ""


def _verdict(rows: list[dict[str, Any]], row: dict[str, Any]) -> str:
    """Best-on-drift note, stated per plan and never across plans."""
    drifts = [
        other["identity_drift_mean"] for other in rows
        if other["plan"] == row["plan"]
        and isinstance(other.get("identity_drift_mean"), (int, float))
    ]
    value = row.get("identity_drift_mean")
    if not drifts or not isinstance(value, (int, float)):
        return ""
    return "drift-min" if value == min(drifts) else ""


def _with_deltas(rows: list[dict[str, Any]], base_arm: str | None) -> None:
    """Attach ``<metric>_vs_base`` for every plan that has the base arm."""
    by_plan: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_plan.setdefault(row["plan"], {})[row["arm"]] = row
    for _plan, arms in by_plan.items():
        base = arms.get(base_arm) if base_arm else None
        if base is None:
            continue
        for row in arms.values():
            for _key, column, _lower in METRICS:
                value, base_value = row.get(column), base.get(column)
                if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                    row[f"{column}_vs_base"] = value - base_value
            row["motion_ratio_vs_base"] = (
                row["motion_scale"] / base["motion_scale"]
                if isinstance(row.get("motion_scale"), (int, float))
                and isinstance(base.get("motion_scale"), (int, float))
                and base["motion_scale"] > 0 else None
            )


def render_markdown(rows: list[dict[str, Any]], base_arm: str | None) -> str:
    by_plan: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_plan.setdefault(row["plan"], {})[row["arm"]] = row

    lines = ["# 长视频接续生成：LoRA 对比", ""]
    if base_arm:
        lines += [
            f"Δ 列均为相对 `{base_arm}` 臂的差（同 plan 内比较，跨 plan 不可比）。",
            f"`MOTION↓` 表示 motion_scale 低于 `{base_arm}` 的 {MOTION_COLLAPSE_RATIO:.0%}，",
            "此时 drift 的改善可能只是「画面动得少了」，需回看视频确认。", "",
        ]
    header = ["plan", "arm", "id_drift_mean", "id_drift_last", "Δdrift", "seam_vid",
              "Δseam_vid", "long_range", "seam_audio", "seam_aud_sp", "motion",
              "motion/base", "peak_GiB", "flag", "note"]
    for plan in sorted(by_plan):
        lines += [f"## {plan}", "", "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        arms = by_plan[plan]
        for arm in sorted(arms):
            row = arms[arm]
            flag = " ".join(filter(None, [_flag_motion(row, arms.get(base_arm) if base_arm else None),
                                          _verdict(list(arms.values()), row)]))
            lines.append("| " + " | ".join([
                plan, arm,
                _fmt(row.get("identity_drift_mean")),
                _fmt(row.get("identity_drift_last")),
                _fmt(row.get("identity_drift_mean_vs_base")),
                _fmt(row.get("seam_video_mad")),
                _fmt(row.get("seam_video_mad_vs_base")),
                _fmt(row.get("long_range_mad")),
                _fmt(row.get("seam_audio_energy")),
                _fmt(row.get("seam_audio_spectral")),
                _fmt(row.get("motion_scale")),
                _fmt(row.get("motion_ratio_vs_base")),
                _fmt(row.get("peak_allocated_gib")),
                flag,
                row.get("error", "") or "",
            ]) + " |")
        lines.append("")

    checkpoints = {
        row["arm"]: (row.get("lora_path"), row.get("lora_scale"))
        for row in rows if row.get("lora_path") or row.get("lora_scale") is not None
    }
    if checkpoints:
        lines += ["## 各臂对应的 checkpoint", "",
                  "| arm | lora_path | scale |", "|---|---|---|"]
        for arm in sorted(checkpoints):
            path, scale = checkpoints[arm]
            lines.append(f"| {arm} | `{path or '(base, no LoRA)'}` | {_fmt(scale)} |")
        lines.append("")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a LoRA comparison run tree.")
    parser.add_argument("--root", type=Path, required=True,
                        help="OUTPUT_ROOT of run_lora_compare_eval_16gpu.sh")
    parser.add_argument("--base", type=str, default="base",
                        help="Arm to use as the reference for the Δ columns.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Markdown path (default <root>/lora_compare_table.md).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV path (default <root>/lora_compare_table.csv).")
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"[error] no such directory: {root}")
    rows = _load_jobs(root)
    if not rows:
        raise SystemExit(
            f"[error] {root} has no */<plan>/rollout_ablation_*.json -- "
            "the eval has not produced anything yet"
        )
    base = args.base if any(row["arm"] == args.base for row in rows) else None
    if args.base and base is None:
        print(f"[warn] base arm {args.base!r} is absent; Δ columns are omitted")
    _with_deltas(rows, base)

    out = args.out or root / "lora_compare_table.md"
    out.write_text(render_markdown(rows, base), encoding="utf-8")
    csv_path = args.csv or root / "lora_compare_table.csv"
    write_csv(rows, csv_path)
    print(f"rows={len(rows)} plans={len({r['plan'] for r in rows})} "
          f"arms={sorted({r['arm'] for r in rows})}")
    print(f"markdown -> {out}")
    print(f"csv      -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
