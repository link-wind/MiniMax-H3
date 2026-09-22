import json

import pytest
import torch

from diffsynth.metrics.h3_continuation_lora import (
    ContinuationEvaluationCase,
    RegressionGates,
    augment_seam_report,
    check_regression_gates,
    run_paired_continuation_evaluation,
    write_ablation_report,
)
from diffsynth.pipelines.minimax_h3_continuation import (
    ContinuationState,
    H3ContinuationConfig,
    H3ContinuationResult,
    H3Segment,
    H3SegmentPlan,
    H3WindowStateRecord,
    resolve_continuation_window,
)


def _fake_result(seed: int) -> H3ContinuationResult:
    config = H3ContinuationConfig()
    first = resolve_continuation_window(config, segment_index=0)
    second = resolve_continuation_window(
        config,
        segment_index=1,
        prior_timeline_video_frames=first.timeline_end_video_frame,
    )
    state = ContinuationState(
        plan_id="paired-plan",
        global_prompt="A continuous scene.",
        config=config,
        base_seed=seed,
        emitted_video_frames=second.timeline_end_video_frame,
    )
    state.records = [
        H3WindowStateRecord(window=first, seed=seed, prompt="A continuous scene."),
        H3WindowStateRecord(window=second, seed=seed, prompt="A continuous scene."),
    ]
    state.next_segment_index = 2
    frames = [
        torch.zeros(1, 8, 8) + (index % 3) * 0.02
        for index in range(second.timeline_end_video_frame)
    ]
    audio = torch.zeros(2, round(second.timeline_end_video_frame / 24 * 32000))
    return H3ContinuationResult(video=frames, audio=audio, state=state)


def test_augment_seam_report_adds_required_diagnostics():
    result = _fake_result(42)
    report = augment_seam_report(result)
    assert report["joins"]
    join = report["joins"][0]
    assert "brightness_color_difference" in join
    assert "person_box" in join
    assert "motion_difference" in join
    assert "flicker" in join
    assert "audio_energy" in join
    assert "audio_spectral_discontinuity" in join
    assert "audio_phase_waveform" in join
    assert "av_boundary_offset_ms" in join


def test_paired_evaluation_reuses_same_plan_and_seed():
    base_result = _fake_result(7)
    lora_result = _fake_result(7)

    class FakeRunner:
        def __init__(self, result):
            self.result = result

        def run(self, plan, base_seed, pipeline_kwargs=None):
            assert plan.plan_id == "paired-plan"
            assert base_seed == 7
            return self.result

    plan = H3SegmentPlan(
        global_prompt="A continuous scene.",
        segments=(H3Segment(), H3Segment()),
        plan_id="paired-plan",
    )
    base = ContinuationEvaluationCase(config_id="base", method="base", seed=7)
    lora = ContinuationEvaluationCase(config_id="lora", method="lora", seed=7)
    paired = run_paired_continuation_evaluation(
        plan,
        base,
        lora,
        lambda case: FakeRunner(base_result if case.method == "base" else lora_result),
    )
    assert set(paired) == {"base", "lora"}
    assert paired["base"]["base_seed"] == paired["lora"]["base_seed"] == 7


def test_regression_gates_flag_non_improving_lora():
    base_report = {
        "joins": [
            {
                "segment_index": 1,
                "video_mean_absolute_difference": 0.5,
                "audio_energy_jump": 0.4,
                "motion_difference": {"frame_mean_absolute_difference": 0.3},
            }
        ]
    }
    lora_report = {
        "joins": [
            {
                "segment_index": 1,
                "video_mean_absolute_difference": 0.45,
                "audio_energy_jump": 0.35,
                "motion_difference": {"frame_mean_absolute_difference": 0.3},
            }
        ]
    }
    result = check_regression_gates(
        {"base": base_report, "lora": lora_report},
        gates=RegressionGates(min_boundary_improvement=0.05),
    )
    assert result["passed"] is False


def test_ablation_report_writes_json_and_markdown(tmp_path):
    report = augment_seam_report(_fake_result(42))
    path = tmp_path / "ablation.json"
    summary = write_ablation_report(
        {"base": report},
        path,
        base_config_id="base",
    )
    assert summary["plan_id"] == "paired-plan"
    assert json.loads(path.read_text(encoding="utf-8"))["plan_id"] == "paired-plan"
    assert path.with_suffix(".md").exists()
