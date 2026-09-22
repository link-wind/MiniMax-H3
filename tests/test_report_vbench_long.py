"""Tests for report_vbench_long.py -- the VBench-Long run summary.

The report's whole job is to stop a reader from believing a number that was never
measured, so most of these tests are about the flags rather than the arithmetic.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "report_vbench_long",
        REPO / "examples/minimax_h3/model_training/report_vbench_long.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_module()


def _write(root: Path, dim: str, set_name: str, *, score, clips: int, vids: int = 1,
           extra_dims: dict | None = None, meta: dict | None = None) -> Path:
    """Drop one ``<dim>/<set>/results_*_eval_results.json`` into the tree."""
    set_dir = root / dim / set_name
    set_dir.mkdir(parents=True, exist_ok=True)
    clip_list = [{"video_path": f"/x/clip{i}.mp4", "video_results": score} for i in range(clips)]
    video_list = [{"video_path": f"/x/vid{i}", "video_results": score} for i in range(vids)]
    payload = {dim: [score, clip_list, video_list]}
    if extra_dims:
        payload.update(extra_dims)
    (set_dir / "results_20260922_eval_results.json").write_text(
        json.dumps(payload), encoding="utf-8")
    if meta is not None:
        (set_dir / "job.json").write_text(json.dumps(meta), encoding="utf-8")
    return set_dir


def test_read_payload_accepts_three_tuple():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.json"
        path.write_text(json.dumps({"aesthetic_quality": [0.5, [{"video_results": 1}],
                                                          [{"video_results": 1}]]}))
        dim, agg, per_video, per_clip = report._read_payload(path)
        assert dim == "aesthetic_quality"
        assert agg == 0.5
        assert len(per_video) == 1 and len(per_clip) == 1


def test_read_payload_accepts_two_tuple():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.json"
        path.write_text(json.dumps({"motion_smoothness": [0.9, [{"video_results": 0.9}]]}))
        dim, agg, per_video, per_clip = report._read_payload(path)
        assert (dim, agg) == ("motion_smoothness", 0.9)
        assert per_video == []
        assert len(per_clip) == 1


def test_rows_carry_clip_and_video_counts(tmp_path):
    _write(tmp_path, "subject_consistency", "base-planA", score=0.6, clips=4)
    _write(tmp_path, "subject_consistency", "v5-planA", score=0.7, clips=3, vids=1)
    rows, problems = report._load_rows(tmp_path, ["base", "v5"])
    assert not problems
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["base"]["n_clips"] == 4
    assert by_arm["v5"]["n_clips"] == 3
    assert by_arm["v5"]["plan"] == "planA"


def test_arm_and_plan_come_from_job_json_when_present(tmp_path):
    # Directory name says one thing, job.json another -- job.json wins, because it
    # is written by the runner and cannot be broken by a rename.
    _write(tmp_path, "subject_consistency", "base-planA", score=0.6, clips=2,
           meta={"arm": "base", "plan": "planA"})
    rows, _ = report._load_rows(tmp_path, ["base"])
    assert rows[0]["arm"] == "base" and rows[0]["plan"] == "planA"


def test_unknown_arm_directory_is_reported_not_guessed(tmp_path):
    _write(tmp_path, "subject_consistency", "mystery-planA", score=0.6, clips=2)
    rows, problems = report._load_rows(tmp_path, ["base", "v5"])
    assert rows == []
    assert any("mystery-planA" in p for p in problems)


def test_empty_dimension_directory_is_reported(tmp_path):
    (tmp_path / "subject_consistency").mkdir(parents=True)
    rows, problems = report._load_rows(tmp_path, ["base"])
    assert rows == []
    assert any("一个 job 的输出目录都没有" in p for p in problems)


def test_zero_clip_row_is_flagged_and_does_not_win(tmp_path):
    # v5's 0.99 beats base's 0.60, but it rests on no clips at all: the only
    # eligible row left is base, and a lone survivor gets no "best" mark either.
    _write(tmp_path, "temporal_flickering", "base-planA", score=0.60, clips=4)
    _write(tmp_path, "temporal_flickering", "v5-planA", score=0.99, clips=0)
    rows, _ = report._load_rows(tmp_path, ["base", "v5"])
    text = report.render_markdown(rows, "base", [])
    assert "NO-CLIPS" in text
    assert "**" not in text.split("## 读这张表")[0].split("| v5 |")[1].splitlines()[0]
    assert "**" not in [ln for ln in text.splitlines() if "| base |" in ln][0]


def test_highest_row_among_measured_arms_is_bolded(tmp_path):
    _write(tmp_path, "subject_consistency", "base-planA", score=0.60, clips=4)
    _write(tmp_path, "subject_consistency", "v5-planA", score=0.72, clips=4)
    rows, _ = report._load_rows(tmp_path, ["base", "v5"])
    text = report.render_markdown(rows, "base", [])
    assert "**0.7200**" in text
    assert "**0.6000**" not in text


def test_delta_is_relative_to_base(tmp_path):
    _write(tmp_path, "subject_consistency", "base-planA", score=0.5, clips=3)
    _write(tmp_path, "subject_consistency", "v5-planA", score=0.6, clips=3)
    _write(tmp_path, "subject_consistency", "v9-planA", score=0.4, clips=3)
    rows, _ = report._load_rows(tmp_path, ["base", "v5", "v9"])
    text = report.render_markdown(rows, "base", [])
    assert "20.0" in text      # v5: +20%
    assert "-20.0" in text     # v9: -20%


def test_duplicate_dimension_pair_is_marked(tmp_path):
    _write(tmp_path, "temporal_style", "base-planA", score=0.31, clips=3)
    _write(tmp_path, "overall_consistency", "base-planA", score=0.31, clips=3)
    rows, _ = report._load_rows(tmp_path, ["base"])
    text = report.render_markdown(rows, "base", [])
    assert "≡temporal_style" in text


def test_distinct_values_are_not_marked_as_duplicates(tmp_path):
    _write(tmp_path, "temporal_style", "base-planA", score=0.31, clips=3)
    _write(tmp_path, "overall_consistency", "base-planA", score=0.32, clips=3)
    rows, _ = report._load_rows(tmp_path, ["base"])
    text = report.render_markdown(rows, "base", [])
    assert "≡temporal_style" not in text


def test_human_action_is_flagged_as_label_from_filename(tmp_path):
    _write(tmp_path, "human_action", "base-planA", score=0.0, clips=4)
    rows, _ = report._load_rows(tmp_path, ["base"])
    text = report.render_markdown(rows, "base", [])
    assert "LABEL-FROM-FILENAME" in text


def test_missing_base_arm_drops_the_delta_column(tmp_path):
    _write(tmp_path, "subject_consistency", "v5-planA", score=0.7, clips=3)
    rows, _ = report._load_rows(tmp_path, ["base", "v5"])
    text = report.render_markdown(rows, None, [])
    assert "省略 Δ 列" in text
    assert "Δ%" in text  # header stays, values are "-"


def test_unreadable_summary_becomes_an_error_row_not_a_crash(tmp_path):
    set_dir = tmp_path / "subject_consistency" / "base-planA"
    set_dir.mkdir(parents=True)
    (set_dir / "results_x_eval_results.json").write_text("{not json", encoding="utf-8")
    rows, _ = report._load_rows(tmp_path, ["base"])
    assert len(rows) == 1
    assert rows[0]["error"].startswith("unreadable")


def test_csv_roundtrip(tmp_path):
    import csv
    _write(tmp_path, "subject_consistency", "base-planA", score=0.6, clips=3)
    rows, _ = report._load_rows(tmp_path, ["base"])
    out = tmp_path / "t.csv"
    report.write_csv(rows, out)
    with out.open() as handle:
        got = list(csv.DictReader(handle))
    assert len(got) == 1
    assert got[0]["arm"] == "base"
