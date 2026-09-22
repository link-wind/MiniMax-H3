"""CPU-checkable logic of the LoRA comparison report.

The report is where twelve per-job JSON files become a decision, so the ways it
can mislead are the thing worth pinning: comparing across plans, printing an
identity-drift win next to a collapsed motion scale without saying so, and
losing an arm because two jobs shared one summary path.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "examples/minimax_h3/model_training/report_lora_compare.py"
    spec = importlib.util.spec_from_file_location("h3_report_lora_compare_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load_module()


def _write_job(root: Path, arm: str, plan: str, metrics: dict, *, lora_path=None, joins=None):
    job = root / arm / plan
    job.mkdir(parents=True, exist_ok=True)
    config = {
        "plan": plan,
        "arms": [arm],
        "lora_arms": {arm: {"lora_path": lora_path, "lora_scale": 1.0}},
    }
    (job / f"rollout_ablation_{arm}.json").write_text(
        json.dumps({"arms": {arm: metrics}, "config": config}), encoding="utf-8",
    )
    if joins is not None:
        (job / f"joins_{arm}.json").write_text(json.dumps(joins), encoding="utf-8")
    return job


BASE = {"head_drift_mean": 1.0, "head_drift_last": 1.1, "motion_scale_mean": 0.8,
        "join_video_mad_mean": 1.4, "join_audio_energy_mean": 0.02,
        "long_range_video_mad": 2.0, "peak_allocated_gib": 70.0}


def test_load_jobs_reads_one_row_per_plan_and_arm(tmp_path):
    _write_job(tmp_path, "base", "p1", dict(BASE))
    _write_job(tmp_path, "v9", "p1", dict(BASE, head_drift_mean=0.5))
    rows = report._load_jobs(tmp_path)
    assert sorted((r["plan"], r["arm"]) for r in rows) == [("p1", "base"), ("p1", "v9")]
    v9 = next(r for r in rows if r["arm"] == "v9")
    assert v9["identity_drift_mean"] == 0.5
    assert v9["seam_video_mad"] == 1.4


def test_joins_file_supplies_the_metrics_the_summary_omits(tmp_path):
    _write_job(tmp_path, "v9", "p1", dict(BASE), joins={
        "joins": [{"audio_spectral_jump": 0.4}, {"audio_spectral_jump": 0.6}],
        "video_metrics": {"sampled_motion_mean_absolute_difference": 0.9},
    })
    row = report._load_jobs(tmp_path)[0]
    assert row["seam_audio_spectral"] == pytest.approx(0.5)
    assert row["sampled_motion_mad"] == pytest.approx(0.9)


def test_deltas_are_taken_within_a_plan_and_never_across_plans(tmp_path):
    """Cross-plan subtraction would compare two different tasks."""
    _write_job(tmp_path, "base", "p1", dict(BASE))
    _write_job(tmp_path, "v9", "p1", dict(BASE, head_drift_mean=0.7))
    _write_job(tmp_path, "base", "p2", dict(BASE, head_drift_mean=5.0))
    _write_job(tmp_path, "v9", "p2", dict(BASE, head_drift_mean=4.0))
    rows = report._load_jobs(tmp_path)
    report._with_deltas(rows, "base")
    for row in rows:
        if row["arm"] == "v9":
            expected = -0.3 if row["plan"] == "p1" else -1.0
            assert row["identity_drift_mean_vs_base"] == pytest.approx(expected)


def test_motion_collapse_is_flagged_even_when_drift_improves(tmp_path):
    _write_job(tmp_path, "base", "p1", dict(BASE))
    _write_job(tmp_path, "v9", "p1", dict(BASE, head_drift_mean=0.2, motion_scale_mean=0.1))
    rows = report._load_jobs(tmp_path)
    report._with_deltas(rows, "base")
    arms = {row["arm"]: row for row in rows}
    assert report._flag_motion(arms["v9"], arms["base"]) == "MOTION↓"
    assert report._flag_motion(arms["base"], arms["base"]) == ""


def test_mild_motion_change_is_not_flagged(tmp_path):
    """Otherwise every row carries a warning and the warning stops meaning anything."""
    _write_job(tmp_path, "base", "p1", dict(BASE))
    _write_job(tmp_path, "v5", "p1", dict(BASE, motion_scale_mean=0.7))
    rows = report._load_jobs(tmp_path)
    arms = {row["arm"]: row for row in rows}
    assert report._flag_motion(arms["v5"], arms["base"]) == ""


def test_verdict_is_per_plan(tmp_path):
    _write_job(tmp_path, "base", "p1", dict(BASE, head_drift_mean=1.0))
    _write_job(tmp_path, "v9", "p1", dict(BASE, head_drift_mean=0.5))
    _write_job(tmp_path, "base", "p2", dict(BASE, head_drift_mean=0.1))
    rows = report._load_jobs(tmp_path)
    p2_base = next(r for r in rows if r["plan"] == "p2" and r["arm"] == "base")
    # p2's base beats p1's v9 on the raw number, but they are not comparable.
    assert report._verdict([r for r in rows if r["plan"] == "p2"], p2_base) == "drift-min"


def test_missing_base_omits_deltas_and_says_so(tmp_path, capsys, monkeypatch):
    _write_job(tmp_path, "v5", "p1", dict(BASE))
    assert report.main.__module__ is not None
    monkeypatch.setattr(sys, "argv", [
        "report_lora_compare.py", "--root", str(tmp_path), "--base", "base",
    ])
    assert report.main() == 0
    out = capsys.readouterr().out
    assert "absent" in out
    rows = report._load_jobs(tmp_path)
    assert "identity_drift_mean_vs_base" not in rows[0]


def test_render_prints_the_checkpoint_behind_each_arm(tmp_path):
    _write_job(tmp_path, "base", "p1", dict(BASE), lora_path=None)
    _write_job(tmp_path, "v9", "p1", dict(BASE), lora_path="/ckpt/v9.safetensors")
    rows = report._load_jobs(tmp_path)
    report._with_deltas(rows, "base")
    markdown = report.render_markdown(rows, "base")
    assert "/ckpt/v9.safetensors" in markdown
    assert "(base, no LoRA)" in markdown


def test_empty_tree_is_reported_rather_than_rendered_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["report_lora_compare.py", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="has not produced anything"):
        report.main()


def test_csv_carries_every_column_it_saw(tmp_path):
    _write_job(tmp_path, "base", "p1", dict(BASE))
    rows = report._load_jobs(tmp_path)
    out = tmp_path / "table.csv"
    report.write_csv(rows, out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "identity_drift_mean" in header and "motion_scale" in header
