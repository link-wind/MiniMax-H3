#!/usr/bin/env python3
"""Collapse a ``run_vbench_long_16gpu.sh`` output tree into one table.

Each job runs the official VBench-Long CLI on one (arm, plan) video for one
dimension, so the tree is ``<root>/<dim>/<arm>-<plan>/results_<ts>_eval_results.json``.
Read one at a time those are hundreds of JSON documents and the comparison lives
in the reader's head; this turns them into rows.

Three things the table refuses to hide:

* **How many clips each number rests on.**  A dimension whose clips were all
  filtered out (``temporal_flickering`` with the static filter on a locked-off
  talking head) reports a score of exactly 0, which is indistinguishable from a
  real bad score unless the clip count is printed next to it.
* **``temporal_style`` and ``overall_consistency`` are the same computation** in
  custom-input mode -- same function body upstream, only the name differs.  Two
  identical columns are not two pieces of evidence, so equal values are marked.
* **The video is split into 2 s clips before scoring**, so a consistency score is
  a within-clip average plus a cross-clip term, not a statement about the whole
  minute.  The clip counts expose how thin that is.

Usage:
    python3 report_vbench_long.py --root outputs/continuation_lora/lora_compare/vbench_long
    python3 report_vbench_long.py --root ... --base base --out table.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

#: Every VBench dimension is scored so that higher is better.  ``dynamic_degree``
#: is the one worth reading twice: it counts the fraction of clips judged
#: "dynamic", so it is not a quality score -- a still video scores low on it and
#: that is not automatically a defect.
HIGHER_IS_BETTER = True

#: Dimensions whose values are the same computation in custom-input mode.
DUPLICATE_PAIRS: list[tuple[str, str]] = [("temporal_style", "overall_consistency")]

#: ``human_action`` compares the classifier's top-1 against a label parsed out of
#: the *filename*, so on our clips (named after the arm) it is always 0.
LABEL_FROM_FILENAME_DIMS = ["human_action"]


def _newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: (p.stat().st_mtime, p.name))


def _read_payload(path: Path) -> tuple[str, Any, list[dict], list[dict]]:
    """Return ``(dimension, aggregate, per_video, per_clip)``.

    The long wrapper around every dimension ends in
    ``reorganize_clips_results``, which returns
    ``(average, detailed_clip_results, per_long_video_averages)``.  Older /
    non-long dimensions return a 2-tuple, so both shapes are accepted.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("empty payload")
    dimension = next(iter(payload))
    value = payload[dimension]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{dimension}: unexpected value shape")
    aggregate = value[0]
    per_clip = [x for x in (value[1] if len(value) > 1 and isinstance(value[1], list) else [])
                if isinstance(x, dict) and "video_results" in x]
    per_video = [x for x in (value[2] if len(value) > 2 and isinstance(value[2], list) else [])
                 if isinstance(x, dict) and "video_results" in x]
    return dimension, aggregate, per_video, per_clip


def _split_set_name(name: str, known_arms: list[str]) -> tuple[str, str] | None:
    """``<arm>-<plan>`` -> ``(arm, plan)``; ``None`` if the arm is not known."""
    for arm in sorted(known_arms, key=len, reverse=True):
        if name == arm:
            return None
        if name.startswith(arm + "-"):
            return arm, name[len(arm) + 1:]
    return None


def _load_rows(root: Path, known_arms: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for dim_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        set_dirs = sorted(p for p in dim_dir.iterdir() if p.is_dir())
        if not set_dirs:
            problems.append(f"{dim_dir.name}: 一个 job 的输出目录都没有")
            continue
        for set_dir in set_dirs:
            summaries = sorted(set_dir.glob("*_eval_results.json"))
            job_file = set_dir / "job.json"
            meta: dict[str, Any] = {}
            if job_file.is_file():
                try:
                    meta = json.loads(job_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
            arm, plan = meta.get("arm"), meta.get("plan")
            if not arm or not plan:
                parsed = _split_set_name(set_dir.name, known_arms)
                if parsed is None:
                    problems.append(
                        f"{dim_dir.name}/{set_dir.name}: 目录名不是 <arm>-<plan> 且没有 job.json，跳过"
                    )
                    continue
                arm, plan = parsed
            row: dict[str, Any] = {
                "dimension": dim_dir.name, "plan": plan, "arm": arm,
                "summary": str(summaries and _newest(summaries) or ""),
                "score": None, "n_videos": 0, "n_clips": 0, "error": "",
            }
            if not summaries:
                row["error"] = "no *_eval_results.json"
                rows.append(row)
                continue
            try:
                dimension, aggregate, per_video, per_clip = _read_payload(_newest(summaries))
            except (ValueError, json.JSONDecodeError) as exc:
                row["error"] = f"unreadable: {exc}"
                rows.append(row)
                continue
            if dimension != dim_dir.name:
                problems.append(
                    f"{dim_dir.name}/{set_dir.name}: 文件里的维度是 {dimension!r}，和目录名不一致"
                )
            row.update({
                "score": aggregate if isinstance(aggregate, (int, float)) else None,
                "n_videos": len(per_video), "n_clips": len(per_clip),
            })
            rows.append(row)
    return rows, problems


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(value: Any, base: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
        return None
    if base == 0:
        return None
    return (value - base) / abs(base) * 100.0


def render_markdown(rows: list[dict[str, Any]], base_arm: str | None,
                    problems: list[str]) -> str:
    by_plan: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for row in rows:
        by_plan.setdefault(row["plan"], {}).setdefault(row["dimension"], {})[row["arm"]] = row

    dims = sorted({row["dimension"] for row in rows})
    lines = [
        "# VBench-Long（custom input）对比",
        "",
        "全部维度**越高越好**（`dynamic_degree` 例外，见文末）。每个数是该 (臂, plan) 视频",
        "切成 2 s clip 后的平均值：一致性维度是「clip 内平均 + 跨 clip 项」按 `slow_fast_params.yaml`",
        "的 0.5 / 0.5 融合（`--dev_flag` 打开的那条支路）。",
        "",
        "**`clips` 列必须看**：`temporal_flickering` 开了 static filter，锁定机位的对话/唱歌镜头",
        "可能被整段滤掉，此时分数是 0、而 0 和「真的很差」在数字上无法区分；`clips=0` 就说明这个",
        "数没有意义。`vids` 列是参与平均的长视频条数。",
        "",
    ]
    if base_arm is None:
        lines += [f"（没有找到基准臂 `{base_arm}`，省略 Δ 列）", ""]

    for plan in sorted(by_plan):
        header = ["dimension", "arm", "score", "Δ%", "clips", "vids", "flag"]
        lines += [f"## {plan}", "", "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        per_dim = by_plan[plan]
        for dim in sorted(per_dim):
            arms = per_dim[dim]
            base_row = arms.get(base_arm) if base_arm else None
            base_score = base_row.get("score") if base_row else None
            # Only rows that actually measured something compete for "best".
            scored = [r for r in arms.values()
                      if isinstance(r.get("score"), (int, float))
                      and r.get("n_videos", 0) > 0 and r.get("n_clips", 0) > 0]
            best = None
            if scored:
                best = max(scored, key=lambda r: r["score"])["arm"] if HIGHER_IS_BETTER \
                    else min(scored, key=lambda r: r["score"])["arm"]
            for arm in sorted(arms):
                row = arms[arm]
                flags: list[str] = []
                if row.get("error"):
                    flags.append(row["error"])
                # A score of exactly 0 with no clips left behind it is "not
                # measured", and it is numerically indistinguishable from a real
                # 0 -- so it gets a flag rather than a place in the ranking.
                if isinstance(row.get("score"), (int, float)) and not row.get("n_clips"):
                    flags.append("NO-CLIPS")
                if arm == best and len(scored) > 1:
                    flags.append("best")
                if dim in LABEL_FROM_FILENAME_DIMS:
                    flags.append("LABEL-FROM-FILENAME")
                for left, right in DUPLICATE_PAIRS:
                    if dim == right and left in per_dim:
                        other = per_dim[left].get(arm, {}).get("score")
                        if isinstance(other, (int, float)) and other == row.get("score"):
                            flags.append(f"≡{left}")
                score = row.get("score")
                lines.append("| " + " | ".join([
                    dim, arm,
                    ("**" + _fmt(score) + "**") if arm == best and len(scored) > 1 else _fmt(score),
                    _fmt(_delta(score, base_score), 1),
                    str(row.get("n_clips", 0)), str(row.get("n_videos", 0)),
                    " ".join(dict.fromkeys(flags)),
                ]) + " |")
        lines.append("")

    lines += ["## 读这张表之前必须知道的三件事", ""]
    lines += [
        "1. **`clips=0` 的格子不是分数，是「没测到」。** 典型是 `temporal_flickering`：它先用 RAFT",
        "   算光流、把静态 clip 滤掉再打分，我们的镜头全是锁定机位，很可能被滤空。",
        "2. **`temporal_style` 与 `overall_consistency` 在 custom input 下是同一个计算**",
        "   （上游两个文件的函数体逐行相同，只差函数名和 `dimension=` 字符串），标 `≡` 的格子说明",
        "   两边取值一致 —— 这不是两条独立证据，是同一个数印了两遍。",
        "3. **`dynamic_degree` 不是质量分**，它是「被判为动态的 clip 占比」。数值高只说明画面在动，",
        "   一致性好不好得看 `subject_consistency` / `background_consistency`。",
    ]
    if LABEL_FROM_FILENAME_DIMS:
        lines += [
            "",
            f"`{'` / `'.join(LABEL_FROM_FILENAME_DIMS)}` 的标签是从**文件名**里解析出来的动作类别，",
            "我们按臂命名视频（`base-trainstyle65s.mp4`）→ 标签永远是 `base`，分数恒为 0，"
            "这一维在 custom input 下没有意义，默认不跑。",
        ]
    if problems:
        lines += ["", "## 汇总过程中的告警", ""]
        lines += [f"- {item}" for item in problems]
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
    parser = argparse.ArgumentParser(description="Summarize a VBench-Long run tree.")
    parser.add_argument("--root", type=Path, required=True,
                        help="VBENCH_OUTPUT of run_vbench_long_16gpu.sh")
    parser.add_argument("--base", type=str, default="base",
                        help="Arm used as the reference for the Δ column.")
    parser.add_argument("--arms", type=str, default="base,v5,v9",
                        help="Comma-separated arm names (to parse <arm>-<plan> dirs).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Markdown path (default <root>/vbench_long_table.md).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV path (default <root>/vbench_long_table.csv).")
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"[error] no such directory: {root}")
    known_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows, problems = _load_rows(root, known_arms)
    if not rows:
        raise SystemExit(
            f"[error] {root} has no <dim>/<arm>-<plan>/*_eval_results.json -- "
            "the VBench stage has not produced anything yet"
        )
    base = args.base if any(r["arm"] == args.base for r in rows) else None
    if args.base and base is None:
        print(f"[warn] base arm {args.base!r} is absent; the Δ column is omitted")

    out = args.out or root / "vbench_long_table.md"
    out.write_text(render_markdown(rows, base, problems), encoding="utf-8")
    csv_path = args.csv or root / "vbench_long_table.csv"
    write_csv(rows, csv_path)
    print(f"rows={len(rows)} dims={len({r['dimension'] for r in rows})} "
          f"plans={len({r['plan'] for r in rows})} arms={sorted({r['arm'] for r in rows})}")
    print(f"markdown -> {out}")
    print(f"csv      -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
