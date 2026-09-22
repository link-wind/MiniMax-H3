import json
import ast
from pathlib import Path

import pytest
import torch
from PIL import Image

from diffsynth.pipelines.minimax_h3_continuation import (
    ContinuationState,
    H3ContinuationConfig,
    h3_continuation_video_latent_steps,
    is_masked_av_context_frames,
    H3Segment,
    H3SegmentPlan,
    H3ContinuationRunner,
    H3PipelineResult,
    H3VideoLatentContinuation,
    H3AudioLatentContinuation,
    apply_audio_latent_continuation,
    apply_video_latent_continuation,
    H3WindowStateRecord,
    assemble_audio_suffix,
    assemble_video_suffix,
    compare_continuation_reports,
    cp_window_metadata,
    evaluate_continuation_joins,
    is_designated_cp_writer,
    load_h3_segment_plan,
    normalize_audio_to_timeline,
    project_video_motion_anchor,
    video_motion_anchor,
    resolve_continuation_window,
    resolve_h3_video_frames,
    validate_audio_latent_continuation,
    validate_video_latent_continuation,
    validate_cp_window_metadata,
    write_continuation_evaluation,
)


def test_native_masked_av_context_uses_joint_grid():
    assert is_masked_av_context_frames(39)
    assert is_masked_av_context_frames(90)
    assert not is_masked_av_context_frames(34)
    assert h3_continuation_video_latent_steps(39) == 12
    assert h3_continuation_video_latent_steps(90) == 27

try:
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_NoiseInitializer
except ImportError:  # The project-wide CPU environment may not ship H3's Qwen dependency.
    MiniMaxH3Unit_NoiseInitializer = None


class _FakeH3Pipeline:
    def __init__(self, *, emit_video=True, emit_audio=True):
        self.calls = []
        self.emit_video = emit_video
        self.emit_audio = emit_audio

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = kwargs["num_frames"]
        video = [f"{len(self.calls)}:{index}" for index in range(frames)] if self.emit_video else None
        samples = round(frames / 24 * 32_000)
        audio = torch.full((2, samples), float(len(self.calls))) if self.emit_audio else None
        return video, audio


class _ImageFakeH3Pipeline(_FakeH3Pipeline):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = kwargs["num_frames"]
        video = [Image.new("RGB", (2, 2), (index % 256, 0, 0)) for index in range(frames)]
        samples = round(frames / 24 * 32_000)
        return video, torch.zeros(2, samples)


class _LatentFakeH3Pipeline(_FakeH3Pipeline):
    def __call__(self, **kwargs):
        video, audio = super().__call__(**kwargs)
        frames = kwargs["num_frames"]
        video_steps = (frames - 5) // 17 * 5 + 2
        audio_steps = round(frames / 24 * 40)
        if not kwargs.get("return_latents"):
            return video, audio
        return H3PipelineResult(
            video=video,
            audio=audio,
            video_latents=torch.ones(1, 24, video_steps, 1, 1),
            audio_latents=torch.ones(2, 32, audio_steps),
            resolved_video_frames=frames,
            video_fps=24,
            audio_sample_rate=32_000,
            video_latent_steps=video_steps,
            audio_latent_steps=audio_steps,
        )


class _RejectingLatentFakeH3Pipeline(_LatentFakeH3Pipeline):
    def __call__(self, **kwargs):
        if "continuation_video_latents" in kwargs or "continuation_audio_latents" in kwargs:
            self.calls.append(kwargs)
            raise ValueError("incompatible latent tail")
        return super().__call__(**kwargs)


class _ContextDecodeFakeH3Pipeline(_LatentFakeH3Pipeline):
    def __call__(self, **kwargs):
        if kwargs.get("decode_output") is False:
            self.calls.append(kwargs)
            frames = kwargs["num_frames"]
            video_steps = (frames - 5) // 17 * 5 + 2
            audio_steps = round(frames / 24 * 40)
            return H3PipelineResult(
                video=None,
                audio=None,
                video_latents=torch.ones(1, 24, video_steps, 1, 1),
                audio_latents=torch.ones(2, 32, audio_steps),
                resolved_video_frames=frames,
                video_fps=24,
                audio_sample_rate=32_000,
                video_latent_steps=video_steps,
                audio_latent_steps=audio_steps,
            )
        return super().__call__(**kwargs)

    def decode_continuation_suffix(self, **kwargs):
        latent_steps = kwargs["current_video_latents"].shape[2]
        resolved_frames = (latent_steps - 2) // 5 * 17 + 5
        frames = resolved_frames - kwargs["overlap_video_frames"]
        samples = round(frames / 24 * 32_000)
        return [f"context:{i}" for i in range(frames)], torch.zeros(2, samples)


class _BridgeDecodeFakeH3Pipeline(_LatentFakeH3Pipeline):
    def decode_continuation_bridge(self, **kwargs):
        frames = kwargs["bridge_video_frames"]
        return [f"bridge:{i}" for i in range(frames)], None


def test_segment_plan_requires_ordered_nonempty_segments_and_resolves_prompt():
    plan = H3SegmentPlan(
        global_prompt="A pianist in a rain-lit cafe.",
        segments=(H3Segment(prompt="The camera slowly moves closer.", segment_id="intro"),),
    )
    assert plan.resolved_prompt(0) == "A pianist in a rain-lit cafe.\n\nThe camera slowly moves closer."
    with pytest.raises(ValueError, match="at least one"):
        H3SegmentPlan(global_prompt="x", segments=())


def test_h3_length_and_shared_timeline_resolution():
    config = H3ContinuationConfig(requested_window_frames=240, overlap_frames=34)
    first = resolve_continuation_window(config, segment_index=0)
    second = resolve_continuation_window(
        config, segment_index=1, prior_timeline_video_frames=first.timeline_end_video_frame
    )
    assert resolve_h3_video_frames(240) == 243
    assert first.resolved_video_frames == 243
    assert first.overlap_video_frames == 0
    assert second.timeline_start_video_frame == 209
    assert second.timeline_end_video_frame == 452
    assert second.overlap_audio_samples == round(34 / 24 * 32_000)
    assert second.overlap_audio_latents == round(34 / 24 * 40)
    assert first.resolved_audio_samples + second.resolved_audio_samples - second.overlap_audio_samples == round(452 / 24 * 32_000)


def test_partial_clip_overlap_is_rejected_before_generation():
    config = H3ContinuationConfig(overlap_frames=33)
    with pytest.raises(ValueError, match="complete H3 Video VAE clips"):
        resolve_continuation_window(config, segment_index=1, prior_timeline_video_frames=243)


def test_suffix_assembly_keeps_prior_overlap_as_owner():
    first = ["f0", "f1", "f2", "f3"]
    second = ["wrong-2", "wrong-3", "f4", "f5"]
    assert assemble_video_suffix(first, second, 2) == ["f0", "f1", "f2", "f3", "f4", "f5"]


def test_audio_hard_splice_and_crossfade_preserve_duration():
    previous = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    current = torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 3.0, 3.0]])
    hard = assemble_audio_suffix(previous, current, overlap_samples=2)
    faded = assemble_audio_suffix(previous, current, overlap_samples=2, crossfade_samples=2)
    assert hard.shape[-1] == faded.shape[-1] == 12
    assert torch.equal(hard[..., :6], previous)
    assert torch.equal(hard[..., 6:], current[..., 2:])
    # The final retained-overlap sample equals the current prefix's final sample,
    # so it connects to the suffix without inserting or removing samples.
    assert faded[0, 5].item() == pytest.approx(current[0, 1].item())
    assert torch.equal(faded[..., 6:], current[..., 2:])


def test_audio_timeline_normalization_reconciles_latent_tick_rounding():
    audio = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(normalize_audio_to_timeline(audio, 2), torch.tensor([[1.0, 2.0]]))
    assert torch.equal(normalize_audio_to_timeline(audio, 5), torch.tensor([[1.0, 2.0, 3.0, 3.0, 3.0]]))


def test_state_manifest_is_json_safe_and_never_embeds_tensor_payloads():
    config = H3ContinuationConfig()
    window = resolve_continuation_window(config, segment_index=0)
    state = ContinuationState(
        plan_id="unit-test",
        global_prompt="test prompt",
        config=config,
        base_seed=7,
        video_latent_tail=torch.ones(1, 24, 12, 2, 2),
        audio_latent_tail=torch.ones(2, 32, 10),
    )
    state.records.append(H3WindowStateRecord(window=window, seed=7, prompt="test prompt"))
    payload = state.to_manifest().to_json()
    decoded = json.loads(payload)
    assert decoded["replay_only"] is True
    assert "video_latent_tail" not in decoded
    assert decoded["records"][0]["window"]["resolved_video_frames"] == 243


def test_runner_uses_retake_prefixes_and_appends_only_new_suffix(tmp_path):
    fake = _FakeH3Pipeline()
    plan = H3SegmentPlan(
        global_prompt="A cinematic city at night.",
        segments=(H3Segment(prompt="Rain begins."), H3Segment(prompt="A taxi turns the corner.")),
    )
    runner = H3ContinuationRunner(
        fake, H3ContinuationConfig(audio_crossfade_ms=100), model_identity="fake-h3", manifest_directory=tmp_path
    )
    result = runner.run(plan, base_seed=100, pipeline_kwargs={"height": 64, "width": 64})
    assert len(fake.calls) == 2
    assert fake.calls[0]["seed"] == 100
    assert "retake_video" not in fake.calls[0]
    assert fake.calls[1]["seed"] == 101
    assert len(fake.calls[1]["retake_video"]) == 34
    assert fake.calls[1]["frame_regions_to_retake"] == [(34, 243)]
    assert fake.calls[1]["seconds_regions_to_retake"] == [(34 / 24, 243 / 24)]
    assert "A taxi turns" in fake.calls[1]["prompt"]
    assert len(result.video) == 243 + 243 - 34
    assert result.video[242] == "1:242"
    assert result.video[243] == "2:34"
    assert result.audio.shape[-1] == round(len(result.video) / 24 * 32_000)
    assert result.state.next_segment_index == 2
    manifest = json.loads((tmp_path / "continuation_state.json").read_text())
    assert manifest["records"][1]["seed"] == 101
    assert manifest["records"][1]["continuation_mode"] == "retake-hard"


def test_runner_uses_first_window_middle_frames_as_later_global_references():
    fake = _ImageFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment(), H3Segment()))
    user_reference = {"type": "image", "image": Image.new("RGB", (2, 2), "white")}
    H3ContinuationRunner(fake, global_reference_frames=3).run(
        plan, pipeline_kwargs={"references": [user_reference]}
    )
    assert fake.calls[0]["references"] == [user_reference]
    for call in fake.calls[1:]:
        references = call["references"]
        assert references[0] is user_reference
        assert len(references) == 4
        anchors = references[1:]
        assert [anchor["type"] for anchor in anchors] == ["image", "image", "image"]
        assert [anchor["image"].getpixel((0, 0))[0] for anchor in anchors] == [60, 121, 182]
    assert fake.calls[1]["references"][1]["image"] is not fake.calls[0].get("references", [None])[0]


def test_runner_global_references_are_disabled_by_default_and_reject_global_latent_decode():
    fake = _ImageFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    H3ContinuationRunner(fake).run(plan)
    assert "references" not in fake.calls[1]
    with pytest.raises(ValueError, match="global reference frames cannot be combined"):
        H3ContinuationRunner(fake, global_reference_frames=1, global_latent_decode=True)


@pytest.mark.parametrize("emit_video,emit_audio,expected_key", [(True, False, "retake_video"), (False, True, "retake_audio")])
def test_runner_allows_single_modality_retake_context(emit_video, emit_audio, expected_key):
    fake = _FakeH3Pipeline(emit_video=emit_video, emit_audio=emit_audio)
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    H3ContinuationRunner(fake).run(plan)
    assert expected_key in fake.calls[1]
    absent_key = "retake_audio" if expected_key == "retake_video" else "retake_video"
    assert absent_key not in fake.calls[1]


def test_latent_continuation_validation_accepts_tail_and_rejects_mismatch():
    target_video = torch.zeros(1, 24, 72, 2, 3)
    target_audio = torch.zeros(2, 32, 405)
    video = H3VideoLatentContinuation(torch.zeros(1, 24, 10, 2, 3), overlap_frames=34)
    audio = H3AudioLatentContinuation(torch.zeros(2, 32, 57), overlap_frames=34)
    assert validate_video_latent_continuation(video, target_video, resolved_video_frames=243, expected_overlap_frames=34) == 10
    assert validate_audio_latent_continuation(audio, target_audio, resolved_video_frames=243, expected_overlap_frames=34) == 57
    with pytest.raises(ValueError, match="spatial shape"):
        validate_video_latent_continuation(
            H3VideoLatentContinuation(torch.zeros(1, 24, 10, 2, 2), overlap_frames=34),
            target_video, resolved_video_frames=243, expected_overlap_frames=34,
        )


def test_latent_prefix_application_preserves_prefix_and_marks_only_suffix_for_denoising():
    target_video = torch.zeros(1, 24, 72, 1, 1)
    target_audio = torch.zeros(2, 32, 405)
    video_tail = H3VideoLatentContinuation(torch.full((1, 24, 10, 1, 1), 4.0), overlap_frames=34)
    audio_tail = H3AudioLatentContinuation(torch.full((2, 32, 57), 5.0), overlap_frames=34)
    conditioned_video, video_mask = apply_video_latent_continuation(video_tail, target_video, resolved_video_frames=243)
    conditioned_audio, audio_mask = apply_audio_latent_continuation(audio_tail, target_audio, resolved_video_frames=243)
    assert torch.all(conditioned_video[:, :, :10] == 4)
    assert torch.all(video_mask[:10] == 0) and torch.all(video_mask[10:] == 1)
    assert torch.all(conditioned_audio[..., :57] == 5)
    assert torch.all(audio_mask[:57] == 0) and torch.all(audio_mask[57:] == 1)
    with pytest.raises(ValueError, match="time extent"):
        validate_audio_latent_continuation(
            H3AudioLatentContinuation(torch.zeros(2, 32, 56), overlap_frames=34),
            target_audio, resolved_video_frames=243, expected_overlap_frames=34,
        )


@pytest.mark.skipif(MiniMaxH3Unit_NoiseInitializer is None, reason="MiniMax H3 inference dependencies are unavailable")
def test_v3_absolute_positions_are_invariant_to_later_prompt_length():
    """The same global latent coordinate must keep the same H3 RoPE time."""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    frame_rows = 4  # latent_h=latent_w=4, patch size 2
    first = builder._build_packed_fl2va(
        100, 10, 4, 4, 20, [],
        video_source_indices=tuple(range(0, 10)),
        audio_source_indices=tuple(range(0, 20)),
        global_video_latent_length=20,
        global_audio_latent_length=40,
        temporal_origin=100,
    )
    second = builder._build_packed_fl2va(
        120, 10, 4, 4, 20, [],
        video_source_indices=tuple(range(5, 15)),
        audio_source_indices=tuple(range(10, 30)),
        global_video_latent_length=20,
        global_audio_latent_length=40,
        temporal_origin=100,
    )
    first_video = first["img_position_ids"][0, first["img_pos"], 0].reshape(10, frame_rows)
    second_video = second["img_position_ids"][0, second["img_pos"], 0].reshape(10, frame_rows)
    assert torch.equal(first_video[5:], second_video[:5])
    # Audio rows are channel-major: [all time steps for channel 0, then
    # channel 1], so compare the two channels' physical overlap separately.
    first_audio = first["img_position_ids"][0, first["audio_pos"], 0]
    second_audio = second["img_position_ids"][0, second["audio_pos"], 0]
    assert torch.equal(first_audio[10:20], second_audio[:10])
    assert torch.equal(first_audio[30:40], second_audio[20:30])


def test_motion_anchor_projects_only_after_overlap():
    class _Scheduler:
        def add_noise(self, clean, noise, timestep):
            return clean + noise * float(timestep)

    target = torch.zeros(1, 1, 14, 1, 1)
    anchor = torch.full((1, 1, 3, 1, 1), 4.0)
    noise = torch.ones_like(anchor)
    projected = project_video_motion_anchor(
        target, _Scheduler(), torch.tensor([0.5]), anchor, noise,
        torch.tensor([0.5, 0.3, 0.1]), start_step=10,
    )
    assert torch.all(projected[:, :, :10] == 0)
    assert projected[0, 0, 10, 0, 0].item() == pytest.approx(2.25)
    assert projected[0, 0, 12, 0, 0].item() == pytest.approx(0.45)


def test_runner_prefers_compatible_latent_handoff_after_first_window():
    fake = _LatentFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    result = H3ContinuationRunner(fake, prefer_latent_handoff=True).run(plan)
    assert fake.calls[0]["return_latents"] is True
    assert "continuation_video_latents" in fake.calls[1]
    assert "continuation_audio_latents" in fake.calls[1]
    assert "retake_video" not in fake.calls[1]
    assert result.state.records[1].continuation_mode == "latent-handoff"


def test_runner_uses_context_decode_for_later_latent_windows():
    fake = _ContextDecodeFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    result = H3ContinuationRunner(fake, prefer_latent_handoff=True).run(plan)
    assert fake.calls[0].get("decode_output") is None
    assert fake.calls[1]["decode_output"] is False
    assert len(result.video) == 452
    assert result.audio.shape[-1] == 602667


def test_runner_falls_back_to_decoded_retake_when_pipeline_rejects_latent_tail():
    fake = _RejectingLatentFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    result = H3ContinuationRunner(fake, prefer_latent_handoff=True).run(plan)
    # first window, rejected latent attempt, decoded-Retake fallback
    assert len(fake.calls) == 3
    fallback_request = fake.calls[-1]
    assert "continuation_video_latents" not in fallback_request
    assert "retake_video" in fallback_request and "retake_audio" in fallback_request
    assert result.state.records[1].continuation_mode == "retake-hard"


def test_plan_loader_cp_metadata_and_writer_contract(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"global_prompt": "global", "segments": [{"segment_id": "a"}]}))
    assert load_h3_segment_plan(plan_file).segments[0].segment_id == "a"
    window = resolve_continuation_window(H3ContinuationConfig(), segment_index=0)
    metadata = cp_window_metadata(window, seed=3, num_inference_steps=50, continuation_mode="retake-hard")
    validate_cp_window_metadata(metadata, [metadata, dict(metadata)])
    with pytest.raises(ValueError, match="rank 1"):
        validate_cp_window_metadata(metadata, [metadata, {**metadata, "seed": 4}])
    assert is_designated_cp_writer(0)
    assert not is_designated_cp_writer(1)


def test_evaluation_report_contains_joint_boundaries_and_cpu_metrics(tmp_path):
    fake = _FakeH3Pipeline()
    result = H3ContinuationRunner(fake).run(H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment())))
    report = evaluate_continuation_joins(result)
    assert len(report["joins"]) == 1
    join = report["joins"][0]
    assert join["video_frame"] == 243
    assert join["audio_sample"] == round(243 / 24 * 32_000)
    assert isinstance(join["audio_energy_jump"], float)
    assert isinstance(join["audio_phase_jump"], float)
    assert report["video_metrics"]["sampled_frame_count"] == 0
    assert isinstance(report["audio_metrics"]["rms"], float)
    destination = tmp_path / "report.json"
    write_continuation_evaluation(result, destination)
    assert json.loads(destination.read_text())["plan_id"] == result.state.plan_id


def test_evaluation_report_measures_pil_video_without_optional_models():
    result = H3ContinuationRunner(_ImageFakeH3Pipeline()).run(
        H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    )
    report = evaluate_continuation_joins(result)
    assert report["joins"][0]["video_mean_absolute_difference"] == pytest.approx(208.0 / 3.0)
    assert report["video_metrics"]["sampled_frame_count"] > 1
    assert report["video_metrics"]["long_range_mean_absolute_difference"] is not None


def test_ablation_comparison_keeps_plan_and_seed_explicit(tmp_path):
    result = H3ContinuationRunner(_FakeH3Pipeline()).run(
        H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment())), base_seed=11
    )
    baseline = evaluate_continuation_joins(result)
    changed_overlap = {**baseline, "configuration": {**baseline["configuration"], "overlap_frames": 17}}
    summary = compare_continuation_reports({"hard": baseline, "overlap-17": changed_overlap}, output_path=tmp_path / "ablations.json")
    assert set(summary["runs"]) == {"hard", "overlap-17"}
    with pytest.raises(ValueError, match="planning intent"):
        compare_continuation_reports({"hard": baseline, "wrong-seed": {**baseline, "base_seed": 12}})


def test_real_pipeline_static_api_keeps_default_tuple_and_exposes_opt_in_latents():
    """Avoid importing H3 here: the CPU test environment lacks Qwen3-VL."""
    source_path = Path(__file__).parents[1] / "diffsynth/pipelines/minimax_h3_audio_video.py"
    module = ast.parse(source_path.read_text())
    pipeline = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Pipeline")
    call = next(node for node in pipeline.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__call__")
    args = {arg.arg for arg in call.args.args + call.args.kwonlyargs}
    assert {"return_latents", "continuation_video_latents", "continuation_audio_latents"} <= args
    defaults = list(call.args.defaults) + list(call.args.kw_defaults)
    assert any(isinstance(default, ast.Constant) and default.value is False for default in defaults)
    source = ast.unparse(call)
    assert "continuation_video_latents and retake_video are mutually exclusive" in source
    assert "continuation_audio_latents and retake_audio are mutually exclusive" in source
    assert "return (video, audio)" in source
    assert "return H3PipelineResult" in source


