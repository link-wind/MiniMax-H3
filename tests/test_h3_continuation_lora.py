import json
import copy
import os

import pytest
import torch
from types import SimpleNamespace
import diffsynth.utils.continuation_lora as continuation_lora

from diffsynth.utils.continuation_lora import (
    build_continuation_samples,
    build_continuation_pairs,
    build_latent_cache,
    clean_prompt,
    extract_shot_prompt,
    ContinuationSample,
    ContinuationPair,
    PAIR_MANIFEST_SCHEMA_VERSION,
    ContinuationRegionConfig,
    collate_continuation_latents,
    load_latent_cache,
    load_continuation_pair_cache,
    make_latent_cache_metadata,
    parse_shot_ranges,
    region_weights,
    resolve_h3_video_frames,
    resolve_timeline,
    save_latent_cache,
    v3_continuation_inputs,
    continuation_flow_loss,
    weighted_flow_loss,
    resolve_media_window,
    expected_video_latent_steps,
    expected_audio_latent_steps,
    load_audio_window,
    ContinuationLatentDataset,
    apply_continuation_lora,
    load_continuation_training_checkpoint,
    load_lora_safetensors,
    manifest_sha256,
    save_continuation_training_checkpoint,
    save_lora_safetensors,
)
from diffsynth.diffusion.loss import ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss
from diffsynth.utils.lora.general import GeneralLoRALoader
from examples.minimax_h3.model_training.build_continuation_cache import resize_h3_frame
from examples.minimax_h3.model_training.build_mult_window_manifest import build_manifest


def test_h3_lengths_and_audio_timeline():
    assert resolve_h3_video_frames(240) == 243
    timeline = resolve_timeline(243, 34)
    assert timeline.audio_samples == round(243 / 24 * 32000)
    assert timeline.overlap_audio_latent_steps == round(34 / 24 * 40)


def test_cache_resize_matches_h3_geometry_and_rejects_invalid_size():
    from PIL import Image

    frame = Image.new("RGB", (1920, 1080), (12, 34, 56))
    resized = resize_h3_frame(frame, height=480, width=832)
    assert resized.size == (832, 480)
    with pytest.raises(ValueError, match="multiples of 16"):
        resize_h3_frame(frame, height=481, width=832)


def test_media_window_uses_absolute_coordinates_without_boundary_drift():
    first = resolve_media_window(0.0, 243 / 24)
    second = resolve_media_window((243 - 34) / 24, (243 - 34 + 243) / 24)
    assert first.video_end_frame == 243
    assert second.video_start_frame == 209
    assert first.audio_end_sample == second.audio_start_sample + round(34 / 24 * 32000)
    assert expected_video_latent_steps(243) == 72


def test_missing_audio_is_explicit_silence(tmp_path):
    window = resolve_media_window(0.0, 243 / 24)
    waveform, rate, missing = load_audio_window(None, window)
    assert missing is True and rate == 32000
    assert waveform.shape == (2, window.audio_samples)
    assert torch.count_nonzero(waveform) == 0


def test_parse_shots_and_prompt_fallback():
    shots = parse_shot_ranges("[Shot 1/2 | 0.0s-12.5s] foo\n[Shot 2/2 | 13s-20s]")
    assert [(s.shot_index, s.start_sec, s.end_sec) for s in shots] == [(1, 0.0, 12.5), (2, 13.0, 20.0)]
    prompt, source = clean_prompt({"prompt": "integrated_multimodal_description " * 5, "asr_result": [[0, 1, "S", "继续说话"]]})
    assert source == "asr_fallback" and "继续说话" in prompt


def test_clean_prompt_extracts_only_requested_shot():
    prompt = (
        "[Shot 1/2 | 0s-4s]\n<SUBJECT>: A\n<Event>: first\n\n"
        "--- cut: frame_gap=1 ---\n\n"
        "[Shot 2/2 | 4s-9s]\n<SUBJECT>: B\n<Scene>: room\n<Event>: second"
    )
    body = extract_shot_prompt(prompt, 2)
    assert body is not None
    assert "<SUBJECT>: B" in body and "first" not in body
    assert "--- cut:" not in body and "[Shot 2/2" not in body
    cleaned, source = clean_prompt({"prompt": prompt}, shot_index=2)
    assert source == "prompt_shot" and cleaned == body


def test_streaming_index_rejects_short_shot_and_keeps_sequence_split(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [
        {"sequence_id": "a", "file_path": "/v/a.mp4", "audio_path": {"clean_wav_path": "/a.wav"}, "prompt": "[Shot 1/1 | 0s-12s] scene"},
        {"sequence_id": "b", "file_path": "/v/b.mp4", "prompt": "[Shot 1/1 | 0s-3s] too short"},
        {"sequence_id": "a", "file_path": "/v/a.mp4", "prompt": "[Shot 1/1 | 0s-12s] scene"},
    ]
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n{bad\n", encoding="utf-8")
    samples, stats = build_continuation_samples(source, window_frames=243, overlap_frames=34, hard_core_frames=17)
    assert samples and stats["short_shot"] == 1 and stats["bad_json"] == 1
    assert {sample.split for sample in samples} == {samples[0].split}
    assert all(sample.sequence_id == "a" for sample in samples)


def test_directional_pair_index_uses_a_tail_for_later_b_inside_one_shot(tmp_path):
    source = tmp_path / "source.jsonl"
    required_seconds = (2 * 345 - 34) / 24
    source.write_text(
        json.dumps({
            "sequence_id": "a", "file_path": "/v/a.mp4",
            "prompt": f"[Shot 1/1 | 0s-{required_seconds + 1:.6f}s] scene",
        }) + "\n" + json.dumps({
            "sequence_id": "b", "file_path": "/v/b.mp4",
            "prompt": "[Shot 1/1 | 0s-20s] too short",
        }) + "\n",
        encoding="utf-8",
    )
    pairs, stats = build_continuation_pairs(source, window_frames=345, overlap_frames=34)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.history.sequence_id == pair.target.sequence_id == "a"
    assert pair.target.start_sec == pytest.approx(pair.history.start_sec + (345 - 34) / 24)
    assert pair.history.end_sec - pair.target.start_sec == pytest.approx(34 / 24)
    assert pair.history.split == pair.target.split == pair.split
    assert stats["short_pair_shot"] == 1


def test_pair_cache_resolves_only_a_tail_as_b_history(tmp_path):
    common = dict(
        sequence_id="seq", video_path="v.mp4", audio_path=None, shot_index=1,
        shot_start_sec=0.0, shot_end_sec=40.0, window_frames=345,
        overlap_frames=34, hard_core_frames=17, transition_frames=17,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    history = ContinuationSample(
        sample_id="history", start_sec=0.0, end_sec=345 / 24, **common,
    )
    target = ContinuationSample(
        sample_id="target", start_sec=(345 - 34) / 24,
        end_sec=(2 * 345 - 34) / 24, **common,
    )
    video_history = torch.arange(102, dtype=torch.float32).view(1, 1, 102, 1, 1)
    video_target = torch.arange(102, dtype=torch.float32).view(1, 1, 102, 1, 1) + 1000
    history_meta = make_latent_cache_metadata(history, video_history, None)
    target_meta = make_latent_cache_metadata(target, video_target, None)
    save_latent_cache(tmp_path / "history.pt", video_history, None, history_meta)
    save_latent_cache(tmp_path / "target.pt", video_target, None, target_meta)
    pair = ContinuationPair("pair", "seq", "train", history, target)
    record = {
        **pair.to_dict(), "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "history_cache_path": "history.pt", "target_cache_path": "target.pt",
    }
    loaded_target, loaded_tail, loaded_audio, loaded_audio_tail, _, target_metadata = load_continuation_pair_cache(
        record, manifest_directory=tmp_path,
    )
    assert torch.equal(loaded_target, video_target)
    assert torch.equal(loaded_tail, video_history[:, :, -10:])
    assert loaded_audio is None and loaded_audio_tail is None
    assert target_metadata["sample_id"] == "target"
    record["target"] = {**record["target"], "split": "validation"}
    with pytest.raises(ValueError, match="crosses a sequence, split, or shot"):
        load_continuation_pair_cache(record, manifest_directory=tmp_path)


def test_region_weights():
    weights = region_weights(15, 5, 5, first_suffix_clip=2)
    assert torch.equal(weights[:5], torch.zeros(5))
    assert weights[5] > weights[6] > weights[9]
    assert torch.all(weights[10:12] == 3)
    assert torch.all(weights[12:] == 1)


def test_v3_training_input_uses_shared_noise_for_a_tail_without_clean_hard_core():
    def add_noise(clean, noise, timestep):
        return clean + noise * float(timestep)

    target = torch.zeros(1, 1, 12, 1, 1)
    history = torch.full((1, 1, 4, 1, 1), 4.0)
    shared_noise = torch.ones_like(target)
    inputs, flow_target, main_noisy = v3_continuation_inputs(
        target, history, shared_noise, 0.5, add_noise,
        hard_core=2, transition=2, transition_start_weight=0.8,
        transition_end_weight=0.5, time_dim=-3,
    )
    assert torch.all(main_noisy[:, :, :4] == 0.5)
    assert torch.all(inputs[:, :, :2] == 4.5)
    assert inputs[0, 0, 2, 0, 0].item() == pytest.approx(3.7)
    assert inputs[0, 0, 3, 0, 0].item() == pytest.approx(2.5)
    assert flow_target[0, 0, 0, 0, 0].item() == pytest.approx(-3.0)


def test_weighted_loss_excludes_hard_core():
    prediction = torch.zeros(1, 1, 4)
    target = torch.zeros_like(prediction)
    prediction[..., :1] = 100
    prediction[..., 1:] = 1
    loss, info = weighted_flow_loss(prediction, target, torch.tensor([0.0, 1.0, 1.0, 1.0]))
    assert loss == pytest.approx(1.0)
    assert info["weighted_tokens"] == pytest.approx(3.0)


def test_cache_metadata_roundtrip_and_validation(tmp_path):
    sample = ContinuationSample(
        sample_id="sample-1", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=10.125,
        window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    video = torch.zeros(1, 24, 12, 2, 2)
    metadata = make_latent_cache_metadata(sample, video, None, video_height=480, video_width=832)
    path = tmp_path / "sample.pt"
    save_latent_cache(path, video, None, metadata)
    loaded_video, loaded_audio, loaded_meta = load_latent_cache(path, expected_sample_id="sample-1", expected_window_frames=243, expected_overlap_frames=34)
    assert torch.equal(loaded_video, video)
    assert loaded_audio is None and loaded_meta["schema_version"] == "h3-continuation-v1"
    assert loaded_meta["video_height"] == 480 and loaded_meta["video_width"] == 832
    with pytest.raises(ValueError, match="sample_id mismatch"):
        load_latent_cache(path, expected_sample_id="wrong")


def test_latent_cache_cpu_prefetch_uses_preloaded_media(tmp_path, monkeypatch):
    sample = ContinuationSample(
        sample_id="prefetch-sample", sequence_id="seq", video_path="v.mp4", audio_path="a.wav",
        shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=243 / 24,
        window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=False, split="train",
    )
    calls = {"video": 0, "audio": 0}

    def load_video(*_args, **_kwargs):
        calls["video"] += 1
        return ["frame"] * 243

    def load_audio(*_args, **_kwargs):
        calls["audio"] += 1
        return torch.zeros(2, 324000), 32000, False

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)
    monkeypatch.setattr(continuation_lora, "load_audio_window", load_audio)

    def video_encoder(frames):
        assert len(frames) == 243
        return torch.zeros(1, 24, 72, 2, 2)

    def audio_encoder(waveform, sample_rate):
        assert sample_rate == 32000
        # Mirrors the audio VAE's hop: 800 samples per latent step.
        return torch.zeros(2, waveform.shape[-1] // 800)

    stats = build_latent_cache(
        [sample], tmp_path, video_encoder=video_encoder, audio_encoder=audio_encoder,
        cpu_prefetch=1, cpu_prefetch_workers=2,
    )
    assert stats == {"written": 1, "skipped": 0, "failed": 0, "reused": 0}
    assert calls == {"video": 1, "audio": 1}
    assert (tmp_path / "train" / "prefetch-sample.pt").exists()


def test_memory_block_is_cut_from_the_frames_before_the_window(tmp_path, monkeypatch):
    """M1-fast: the memory block is an independent 17n+5 clip ending at the window start."""
    sample = ContinuationSample(
        sample_id="memory-sample", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=2, shot_start_sec=20.0, shot_end_sec=40.0, start_sec=20.0, end_sec=20.0 + 243 / 24,
        window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    requested = []

    def load_video(_path, window, **_kwargs):
        requested.append((window.video_start_frame, window.video_end_frame))
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)

    def video_encoder(frames):
        assert len(frames) in (39, 243)
        steps = ((len(frames) - 5) // 17) * 5 + 2
        return torch.ones(1, 24, steps, 2, 2) * len(frames)

    stats = build_latent_cache([sample], tmp_path, video_encoder=video_encoder, memory_frames=39)
    assert stats["written"] == 1
    # The window itself plus the 39 frames immediately preceding it.
    assert requested == [(480, 723), (441, 480)]

    video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "memory-sample.pt", expect_memory=True,
    )
    assert tuple(video.shape) == (1, 24, 72, 2, 2)
    # A single slot loads as a one-entry slot list, so the training path has one
    # representation to handle whether the cache is legacy or dual-slot.
    assert len(memory) == 1 and memory[0]["name"] == "stm"
    assert memory[0]["lead_steps"] is None  # flush against the window
    assert tuple(memory[0]["tensor"].shape) == (1, 24, 12, 2, 2)
    assert float(memory[0]["tensor"].flatten()[0]) == 39.0
    assert metadata["memory_frames"] == 39 and metadata["memory_video_steps"] == 12
    assert metadata["memory_start_frame"] == 441

    manifest = tmp_path / "train" / "manifest.jsonl"
    manifest.write_text(json.dumps({"cache_path": "memory-sample.pt", **sample.to_dict()}) + "\n", encoding="utf-8")
    item = ContinuationLatentDataset(manifest, split="train")[0]
    assert torch.equal(item["continuation_memory_video_latents"][0]["tensor"], memory[0]["tensor"])
    assert tuple(item["continuation_memory_video_latents"][0]["tensor"].shape) == (1, 24, 12, 2, 2)
    assert (tmp_path / "train" / "memory-sample.pt").exists()


def test_dual_memory_slots_are_cut_from_two_anchors_and_kept_separate(tmp_path, monkeypatch):
    """Memory-only layout: contiguous STM + constant-anchored LTM in one entry."""
    sample = ContinuationSample(
        sample_id="dual-sample", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=3, shot_start_sec=20.0, shot_end_sec=40.0, start_sec=20.0, end_sec=20.0 + 243 / 24,
        window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    requested = []

    def load_video(_path, window, **_kwargs):
        requested.append((window.video_start_frame, window.video_end_frame))
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)

    def video_encoder(frames):
        steps = ((len(frames) - 5) // 17) * 5 + 2
        return torch.ones(1, 24, steps, 2, 2) * len(frames)

    specs = continuation_lora.dual_memory_slot_specs(39, 39, ltm_lead_steps=36)
    assert [(spec.name, spec.anchor, spec.lead_steps) for spec in specs] == [
        ("stm", "window-start", None), ("ltm", "sequence-head", 36),
    ]
    build_latent_cache([sample], tmp_path, video_encoder=video_encoder, memory_specs=specs)
    # STM is the 39 frames before the window; LTM is the opening 39 frames.
    assert requested == [(480, 723), (441, 480), (0, 39)]

    _video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "dual-sample.pt", expect_memory=True,
    )
    assert [slot["name"] for slot in memory] == ["stm", "ltm"]
    assert [slot["lead_steps"] for slot in memory] == [None, 36]
    assert float(memory[0]["tensor"].flatten()[0]) == 39.0
    assert float(memory[1]["tensor"].flatten()[0]) == 39.0
    assert [slot["start_frame"] for slot in metadata["memory_slots"]] == [441, 0]
    assert [slot["anchor"] for slot in metadata["memory_slots"]] == ["window-start", "sequence-head"]
    # The legacy mirror only covers the single-slot case, so a legacy reader
    # cannot silently train on one slot out of two.
    payload = torch.load(tmp_path / "train" / "dual-sample.pt", map_location="cpu", weights_only=False)
    assert payload["memory_latents"] is None and len(payload["memory_slots"]) == 2


def test_memory_slot_is_dropped_when_it_overlaps_the_window(tmp_path, monkeypatch):
    """A clean copy of the target is not a memory: overlapping slots are dropped.

    The sequence-head anchor of the first shot is that shot itself, and a window
    that starts *inside* the opening frames overlaps for the same reason.  Both
    must degrade to a slot-free sample rather than leak target frames in as clean
    conditioning the model could simply copy.
    """
    def load_video(_path, window, **_kwargs):
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)
    encoder = lambda frames: torch.zeros(1, 24, ((len(frames) - 5) // 17) * 5 + 2, 2, 2)
    specs = continuation_lora.dual_memory_slot_specs(39, 39, ltm_lead_steps=36)

    for sample_id, start_sec in (("leak-head", 0.0), ("leak-partial", 1.0), ("leak-clear", 2.0)):
        sample = ContinuationSample(
            sample_id=sample_id, sequence_id="seq", video_path="v.mp4", audio_path=None,
            shot_index=1, shot_start_sec=start_sec, shot_end_sec=start_sec + 20.0,
            start_sec=start_sec, end_sec=start_sec + 243 / 24,
            window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
            prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
            audio_missing=True, split="train",
        )
        build_latent_cache([sample], tmp_path, video_encoder=encoder, memory_specs=specs)

    # Window is the first shot itself (frames 0..243): neither slot exists.
    _video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "leak-head.pt",
    )
    assert memory is None and metadata["memory_slots"] == ()

    # Window starts at frame 24, so the opening 39 frames are partly inside it.
    _video, _audio, memory, _metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "leak-partial.pt",
    )
    assert memory is None

    # Window starts at frame 48, clear of the opening 39 frames and with 39
    # frames of its own history: both slots exist again.
    _video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "leak-clear.pt",
    )
    assert [slot["name"] for slot in memory] == ["stm", "ltm"]
    assert [slot["start_frame"] for slot in metadata["memory_slots"]] == [9, 0]

    # Structural consequence of the two guards: with equal slot lengths, a
    # window starts before the sequence-head block exactly when it has no
    # preceding frames at all, so a sample carries either both slots or none.
    # "LTM only" cannot happen in this layout -- it would need an LTM shorter
    # than the STM.


def test_dual_memory_slots_drop_the_missing_one_at_a_sequence_head(tmp_path, monkeypatch):
    """The first shot has no preceding frames, so only the LTM slot survives."""
    sample = ContinuationSample(
        sample_id="dual-head", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=0, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=243 / 24,
        window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )

    def load_video(_path, window, **_kwargs):
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)
    build_latent_cache(
        [sample], tmp_path,
        video_encoder=lambda frames: torch.zeros(1, 24, ((len(frames) - 5) // 17) * 5 + 2, 2, 2),
        memory_specs=continuation_lora.dual_memory_slot_specs(39, 39, ltm_lead_steps=36),
    )
    _video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "dual-head.pt",
    )
    # The head anchor of the first shot is the shot itself, so it is dropped as
    # a leak and the sample trains with no memory at all.
    assert memory is None and metadata["memory_slots"] == ()

    # A short LTM (22 frames, not 39) is the only way to get a single slot: its
    # head anchor stops before the window starts, while the STM still has no
    # frames to reach back to.
    short_ltm = continuation_lora.dual_memory_slot_specs(39, 22, ltm_lead_steps=36)
    mid = ContinuationSample(
        sample_id="dual-mid", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=1.0, end_sec=1.0 + 243 / 24,
        window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    build_latent_cache(
        [mid], tmp_path,
        video_encoder=lambda frames: torch.zeros(1, 24, ((len(frames) - 5) // 17) * 5 + 2, 2, 2),
        memory_specs=short_ltm,
    )
    _video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "dual-mid.pt",
    )
    assert [slot["name"] for slot in memory] == ["ltm"]
    assert metadata["memory_frames"] == 22 and metadata["memory_start_frame"] == 0


def test_memory_block_is_skipped_at_a_sequence_head(tmp_path, monkeypatch):
    """A window that starts at frame 0 has nothing in front of it to remember."""
    sample = ContinuationSample(
        sample_id="head-sample", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=0, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=243 / 24,
        window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )

    def load_video(_path, window, **_kwargs):
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)
    requested = []

    def video_encoder(frames):
        requested.append(len(frames))
        return torch.zeros(1, 24, ((len(frames) - 5) // 17) * 5 + 2, 2, 2)

    build_latent_cache([sample], tmp_path, video_encoder=video_encoder, memory_frames=39)
    assert requested == [243]
    _video, _audio, memory, metadata = continuation_lora.load_latent_cache_full(
        tmp_path / "train" / "head-sample.pt",
    )
    assert memory is None and metadata["memory_frames"] == 0


def test_check_memory_only_sample_guards_the_contract():
    """The memory-only contract is checked, because both failures are silent."""
    stm = {"name": "stm"}
    ltm = {"name": "ltm"}
    with pytest.raises(ValueError, match="prefix-free"):
        continuation_lora.check_memory_only_sample("masked", [stm, ltm])
    # A sample with nothing to remember is legal; the run-level audit catches a
    # cache that has no slots at all.
    assert continuation_lora.check_memory_only_sample("none", []) == []
    assert continuation_lora.check_memory_only_sample("none", None) == []
    with pytest.raises(ValueError, match="unknown memory slots"):
        continuation_lora.check_memory_only_sample("none", [{"name": "v3-absolute"}])
    # Absence of one slot is legal: the first shot has nothing in front of it.
    assert continuation_lora.check_memory_only_sample("none", [ltm]) == ["ltm"]
    assert continuation_lora.check_memory_only_sample("none", [stm, ltm]) == ["stm", "ltm"]


def test_audit_memory_slots_separates_a_memory_free_cache(tmp_path):
    """A slot-free sample is legal; a slot-free *cache* must fail the run."""
    manifest = tmp_path / "train"
    manifest.mkdir()
    rows = [
        {"cache_path": "a.pt", "split": "train", "metadata": {"memory_slots": [{"name": "stm"}]}},
        {"cache_path": "b.pt", "split": "train", "metadata": {"memory_slots": [{"name": "stm"}, {"name": "ltm"}]}},
        {"cache_path": "c.pt", "split": "train", "metadata": {"memory_slots": []}},
        {"cache_path": "d.pt", "split": "train", "metadata": {}},
    ]
    (manifest / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    audit = continuation_lora.audit_memory_slots(tmp_path)
    assert audit["records"] == 4 and audit["records_with_slots"] == 2
    assert audit["slot_combinations"] == {"stm": 1, "stm+ltm": 1}

    (manifest / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[2:]), encoding="utf-8",
    )
    assert continuation_lora.audit_memory_slots(tmp_path)["records_with_slots"] == 0


def test_select_memory_slots_drives_the_ablation_arms():
    """One cache serves every arm: the arm only decides what the model may see."""
    stm = {"name": "stm", "tensor": "stm-block"}
    ltm = {"name": "ltm", "tensor": "ltm-block"}
    both = [stm, ltm]
    assert continuation_lora.select_memory_slots(both, ()) == both
    assert continuation_lora.select_memory_slots(both, ("stm", "ltm")) == both
    assert continuation_lora.select_memory_slots(both, ("stm",)) == [stm]
    assert continuation_lora.select_memory_slots(both, ("ltm",)) == [ltm]
    # Order follows the arm, and a slot the cache lacks is simply absent.
    assert continuation_lora.select_memory_slots(both, ("ltm", "stm")) == [ltm, stm]
    assert continuation_lora.select_memory_slots([ltm], ("stm",)) == []


def test_select_memory_slots_accepts_a_slot_free_sample():
    """A slot-free sample arrives as ``None`` and must not be iterated.

    ``load_latent_cache_full`` returns ``slots or None``, so the first shot of a
    sequence -- which has nothing to remember -- reaches the arm filter as
    ``None``.  Every arm then has to yield "no conditioning" instead of raising
    on the iteration; 2,842 of the 12,000 training records are this case.
    """
    assert continuation_lora.select_memory_slots(None, ("stm", "ltm")) == []
    assert continuation_lora.select_memory_slots(None, ()) == []
    assert continuation_lora.select_memory_slots((), ("stm", "ltm")) == []


def test_memory_only_index_to_packed_sequence(tmp_path, monkeypatch):
    """End-to-end: memory-only index -> dual-slot cache -> packed sequence.

    This is the whole point of the memory-only objective: every sample is one
    shot, the window is entirely prediction target, and the only cross-shot
    conditioning is the two clean slots, each at its own anchor.
    """
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_PackedSequenceBuilder, normalize_memory_slots,
    )

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "sequence_id": "seq-1", "file_path": "v.mp4", "audio_path": None,
        "prompt": (
            "[Shot 1/3 | 0.0s-4.0s] a wide shot\n"
            "[Shot 2/3 | 4.0s-8.0s] a closer shot\n"
            "[Shot 3/3 | 8.0s-12.0s] a close-up"
        ),
    }) + "\n", encoding="utf-8")

    samples = list(continuation_lora.iter_shot_memory_only_samples(source))
    assert len(samples) == 3
    assert all(sample.prefix_mode == "none" for sample in samples)
    # The whole window is the target, so there is no 17k / 17k+5 split.
    assert all((sample.window_frames - 5) % 17 == 0 for sample in samples)

    requested = []

    def load_video(_path, window, **_kwargs):
        requested.append((window.video_start_frame, window.video_end_frame))
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)
    specs = continuation_lora.dual_memory_slot_specs(39, 39, ltm_lead_steps=36)
    build_latent_cache(
        samples, tmp_path,
        video_encoder=lambda frames: torch.zeros(1, 24, ((len(frames) - 5) // 17) * 5 + 2, 2, 2),
        memory_specs=specs,
    )

    dataset = ContinuationLatentDataset(tmp_path / "train" / "manifest.jsonl", split="train")
    assert len(dataset) == 3
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    for index, sample in enumerate(samples):
        item = dataset[index]
        slots = item["continuation_memory_video_latents"]
        names = continuation_lora.check_memory_only_sample(item["continuation_prefix_mode"], slots)
        # Shot 1 is the sequence head: its head anchor is the shot itself, so it
        # has nothing to remember and trains as plain text-to-shot.
        assert names == ([] if index == 0 else ["stm", "ltm"])
        # The ablation arm is a filter over the same cache: same cache, no re-encode.
        arm = continuation_lora.select_memory_slots(slots or (), ("stm", "ltm"))
        if index == 0:
            assert slots is None and arm == []
            continue  # nothing to remember: the layout is the no-memory one
        assert arm == slots
        assert continuation_lora.select_memory_slots(slots, ("ltm",)) == slots[-1:]
        normalized = normalize_memory_slots(arm)
        latent_t = int(item["input_latents"].shape[2])
        packed = builder._build_packed_fl2va(
            text_len=8, latent_t=latent_t, latent_h=2, latent_w=2, audio_t=2,
            keyframe_indices=[], memory_slots=normalized,
        )
        grid = packed["img_position_ids"][0]
        window_t = float(grid[packed["img_pos"][0], 0])
        slice_of = dict(zip(packed["mem_slot_names"], packed["mem_slot_slices"]))
        if "stm" in slice_of:
            start, stop = slice_of["stm"]
            assert stop - start == 12 * 1  # 12 latent steps of 2x2 patches
            assert abs(float(grid[start, 0]) - (window_t - builder._video_t_span(12))) < 1e-9
            assert float(grid[stop - 1, 0]) < window_t
        start, stop = slice_of["ltm"]
        assert abs(float(grid[start, 0]) - (window_t - builder._video_t_span(36))) < 1e-9
        # Every memory row is a clean condition, never a prediction target.
        assert not (set(packed["mem_pos"].tolist()) & set(packed["img_pos"].tolist()))
        assert (packed["token_tags"][packed["mem_pos"]] == 0).all()


def test_memory_frames_must_land_on_the_vae_grid(tmp_path):
    sample = ContinuationSample(
        sample_id="bad-memory", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=243 / 24,
        window_frames=243, overlap_frames=39, hard_core_frames=39, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    with pytest.raises(ValueError, match="17n\\+5"):
        build_latent_cache([sample], tmp_path, video_encoder=lambda frames: torch.zeros(1, 24, 72, 2, 2),
                           memory_frames=40)


def test_continuation_dataset_marks_cache_and_resolves_relative_paths(tmp_path):
    sample = ContinuationSample(
        sample_id="relative-sample", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=10.125,
        window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    video = torch.zeros(1, 24, 12, 2, 2)
    metadata = make_latent_cache_metadata(sample, video, None)
    cache_path = tmp_path / "train" / "relative-sample.pt"
    save_latent_cache(cache_path, video, None, metadata)
    manifest = cache_path.parent / "manifest.jsonl"
    manifest.write_text(json.dumps({"cache_path": "relative-sample.pt", **sample.to_dict()}) + "\n", encoding="utf-8")
    dataset = ContinuationLatentDataset(manifest, split="train")
    assert dataset.load_from_cache is True
    item = dataset[0]
    assert torch.equal(item["input_latents"], video)
    assert item["continuation_history_video_latents"] is not item["input_latents"]


def test_continuation_dataset_accepts_reused_cache_alias_metadata(tmp_path):
    sample = ContinuationSample(
        sample_id="origin", sequence_id="seq", video_path="v.mp4", audio_path=None,
        shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=10.125,
        window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )
    video = torch.zeros(1, 24, 72, 2, 2)
    metadata = make_latent_cache_metadata(sample, video, None)
    cache_path = tmp_path / "train" / "origin:plain.pt"
    save_latent_cache(cache_path, video, None, metadata)

    record = {
        **sample.to_dict(),
        "sample_id": "origin:plain",
        "overlap_frames": 17,
        "prefix_mode": "none",
        "cache_path": cache_path.name,
        "metadata": metadata.to_dict(),
    }
    manifest = cache_path.parent / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    item = ContinuationLatentDataset(manifest, split="train")[0]
    assert item["continuation_prefix_mode"] == "none"
    assert item["continuation_history_video_latents"] is None
    assert item["num_frames"] == 243
    assert torch.equal(item["input_latents"], video)


def test_continuation_dataset_drops_history_for_prefix_none_windows(tmp_path):
    def build(sample_id: str, prefix_mode: str):
        sample = ContinuationSample(
            sample_id=sample_id, sequence_id="seq", video_path="v.mp4", audio_path=None,
            shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=10.125,
            window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
            prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
            audio_missing=True, split="train", prefix_mode=prefix_mode,
        )
        video = torch.zeros(1, 24, 72, 2, 2)
        audio = torch.zeros(2, 405)
        metadata = make_latent_cache_metadata(sample, video, audio)
        path = tmp_path / "train" / f"{sample_id}.pt"
        save_latent_cache(path, video, audio, metadata)
        return {"cache_path": f"{sample_id}.pt", **sample.to_dict()}

    (tmp_path / "train").mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "train" / "manifest.jsonl"
    manifest.write_text(
        json.dumps(build("masked-sample", "masked")) + "\n"
        + json.dumps(build("plain-sample", "none")) + "\n",
        encoding="utf-8",
    )
    dataset = ContinuationLatentDataset(manifest, split="train")
    masked_item, plain_item = dataset[0], dataset[1]
    assert masked_item["continuation_prefix_mode"] == "masked"
    assert torch.equal(masked_item["continuation_history_video_latents"], masked_item["input_latents"])
    assert torch.equal(masked_item["continuation_history_audio_latents"], masked_item["audio_input_latents"])
    # "none" windows keep the same cached latents but must not carry a prefix.
    assert plain_item["continuation_prefix_mode"] == "none"
    assert plain_item["continuation_history_video_latents"] is None
    assert plain_item["continuation_history_audio_latents"] is None
    assert torch.equal(plain_item["input_latents"], masked_item["input_latents"])


def test_latent_cache_reuse_skips_media_prefetch_for_pointer_records(tmp_path):
    def make(sample_id, **kwargs):
        return ContinuationSample(
            sample_id=sample_id, sequence_id="seq", video_path="v.mp4", audio_path=None,
            shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=10.125,
            window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
            prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
            audio_missing=True, split="train", **kwargs,
        )

    origin = make("origin")
    video = torch.zeros(1, 24, 72, 2, 2)
    metadata = make_latent_cache_metadata(origin, video, None)
    old_path = tmp_path / "old" / "origin.pt"
    save_latent_cache(old_path, video, None, metadata)

    def video_encoder(frames):  # pragma: no cover - must never run
        raise AssertionError("reused records must not be decoded")

    # Reused records keep only a pointer plus placeholder segments; prefetch must
    # not try to decode them, and the copied entry reuses the stored metadata.
    reused = make(
        "origin:plain", prefix_mode="none", reused_cache_path=str(old_path),
        cache_metadata=metadata.to_dict(),
        segments=({"video_path": "v.mp4", "clip_start_frame": None,
                   "start_frame": None, "end_frame": None},),
    )
    stats = build_latent_cache(
        [reused], tmp_path / "new", video_encoder=video_encoder,
        cpu_prefetch=1, cpu_prefetch_workers=1,
    )
    assert stats == {"written": 0, "skipped": 0, "failed": 0, "reused": 1}
    destination = tmp_path / "new" / "train" / "origin:plain.pt"
    assert destination.exists()
    assert torch.equal(torch.load(destination, weights_only=False)["video_latents"], video)
    record = json.loads(
        (tmp_path / "new" / "train" / "manifest.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["cache_path"] == "origin:plain.pt"
    assert record["prefix_mode"] == "none"
    assert record["metadata"]["video_shape"] == list(video.shape)
    assert "cache_metadata" not in record


def test_latent_cache_reuse_idempotent_when_destination_already_linked(tmp_path):
    """Re-running a cache build (e.g. with OVERWRITE=1) must not fail when a
    reused window's destination is already a hard link to the source inode."""
    def make(sample_id, **kwargs):
        return ContinuationSample(
            sample_id=sample_id, sequence_id="seq", video_path="v.mp4", audio_path=None,
            shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0, start_sec=0.0, end_sec=10.125,
            window_frames=243, overlap_frames=34, hard_core_frames=17, transition_frames=17,
            prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
            audio_missing=True, split="train", **kwargs,
        )

    origin = make("origin")
    video = torch.zeros(1, 24, 72, 2, 2)
    metadata = make_latent_cache_metadata(origin, video, None)
    source = tmp_path / "old" / "origin.pt"
    save_latent_cache(source, video, None, metadata)

    # First build links source -> dest.
    reused = make("origin:plain", prefix_mode="none", reused_cache_path=str(source),
                  cache_metadata=metadata.to_dict())
    out = tmp_path / "new"
    build_latent_cache([reused], out, video_encoder=lambda f: (_ for _ in ()).throw(AssertionError()),
                       cpu_prefetch=0, cpu_prefetch_workers=1)
    dest = out / "train" / "origin:plain.pt"
    assert dest.exists()
    assert os.path.samefile(source, dest)

    # Rebuild with overwrite (overwrite=True so eligible_samples does not skip
    # the existing destination): reuse must be idempotent, not raise.
    stats = build_latent_cache([reused], out, video_encoder=lambda f: (_ for _ in ()).throw(AssertionError()),
                               overwrite=True, cpu_prefetch=0, cpu_prefetch_workers=1)
    assert stats["reused"] == 1
    assert stats["failed"] == 0
    assert dest.exists()
    assert torch.equal(torch.load(dest, weights_only=False)["video_latents"], video)


def test_joint_loss_normalizes_video_and_audio_independently():
    video_pred = torch.ones(1, 2, 3, 2, 2)
    video_target = torch.zeros_like(video_pred)
    audio_pred = torch.ones(2, 4)
    audio_target = torch.zeros_like(audio_pred)
    total, info = continuation_flow_loss(
        video_pred, video_target, torch.tensor([0.0, 1.0, 1.0]),
        audio_prediction=audio_pred, audio_target=audio_target, audio_weights=torch.ones(4), lambda_audio=0.5,
    )
    assert total == pytest.approx(1.5)
    assert info["video_weighted_tokens"] == pytest.approx(2.0)
    assert info["audio_weighted_tokens"] == pytest.approx(4.0)


def test_mask_v14_uses_39_frame_overlap_mapping():
    video = torch.zeros(1, 24, 102, 2, 2)
    audio = torch.zeros(2, 575)
    result = collate_continuation_latents(
        video, audio_latents=audio, config=ContinuationRegionConfig()
    )
    assert torch.all(result["video_loss_mask"][:12] == 0)
    assert torch.all(result["video_loss_mask"][12:17] == 3)
    assert result["video_history_clean"].shape[-3] == 12
    assert int(result["audio_overlap_steps"]) == 65
    assert torch.all(result["audio_loss_mask"][:65] == 0)
    assert result["audio_history_clean"].shape[-1] == 65


def test_continuation_loss_calls_joint_model_with_masked_latents():
    class Scheduler:
        timesteps = torch.tensor([0.5])
        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()
        def model_fn(self, **kwargs):
            assert kwargs["video_latents"].shape == (1, 2, 20, 1, 1)
            assert kwargs["audio_latents"].shape == (2, 100)
            return torch.zeros_like(kwargs["video_latents"]), torch.zeros_like(kwargs["audio_latents"])

    clean_video = torch.zeros(1, 2, 20, 1, 1)
    clean_audio = torch.zeros(2, 100)
    loss = ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        Pipe(), input_latents=clean_video, audio_input_latents=clean_audio,
    )
    assert torch.isfinite(loss)


def test_mask_v14_clean_prefix_injects_clean_latents_with_denoise_mask():
    class Scheduler:
        timesteps = torch.tensor([0.5])

        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()

        def model_fn(self, **kwargs):
            self.calls.append(kwargs)
            video = torch.zeros_like(kwargs["video_latents"])
            audio = torch.zeros_like(kwargs["audio_latents"])
            return video, audio

    pipe = Pipe()
    pipe.calls = []
    clean_video = torch.randn(1, 2, 20, 1, 1)
    clean_audio = torch.randn(2, 32, 100)
    config = ContinuationRegionConfig(prefix_present_mode="clean")
    loss = ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        pipe,
        region_config=config,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
    )
    assert torch.isfinite(loss)
    assert len(pipe.calls) == 1
    kwargs = pipe.calls[0]
    # Clean form must go through the same input_latents/denoise_mask path as
    # the masked-av-v14 inference embedder.
    assert "input_latents_video" in kwargs and "denoise_mask_video" in kwargs
    assert "input_latents_audio" in kwargs and "denoise_mask_audio" in kwargs
    assert kwargs["input_latents_video"].shape == clean_video.shape
    assert torch.all(kwargs["input_latents_video"][:, :, :12] == clean_video[:, :, :12])
    assert torch.all(kwargs["denoise_mask_video"][0, 0, :12] == 0)
    assert torch.all(kwargs["denoise_mask_video"][0, 0, 12:] == 1)
    assert kwargs["denoise_mask_video"].shape == (1, 1, 20, 1, 1)
    assert kwargs["input_latents_audio"].shape == clean_audio.shape
    assert torch.all(kwargs["input_latents_audio"][:, :, :65] == clean_audio[:, :, :65])
    assert torch.all(kwargs["denoise_mask_audio"][:, :, :65] == 0)
    assert torch.all(kwargs["denoise_mask_audio"][:, :, 65:] == 1)
    assert kwargs["denoise_mask_audio"].shape == (2, 1, 100)


def test_mask_v14_noised_prefix_keeps_v3_inputs():
    class Scheduler:
        timesteps = torch.tensor([0.5])

        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()

        def model_fn(self, **kwargs):
            self.calls.append(kwargs)
            video = torch.zeros_like(kwargs["video_latents"])
            audio = torch.zeros_like(kwargs["audio_latents"])
            return video, audio

    pipe = Pipe()
    pipe.calls = []
    clean_video = torch.randn(1, 2, 20, 1, 1)
    clean_audio = torch.randn(2, 32, 100)
    # Default prefix_present_mode="noised" must reproduce the v3/v5 objective
    # exactly: no clean-injection keys reach model_fn.
    loss = ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        pipe,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
    )
    assert torch.isfinite(loss)
    assert len(pipe.calls) == 1
    kwargs = pipe.calls[0]
    assert "input_latents_video" not in kwargs
    assert "denoise_mask_video" not in kwargs
    assert "input_latents_audio" not in kwargs
    assert "denoise_mask_audio" not in kwargs


def test_mask_v14_prefix_none_is_plain_flow_matching_with_uniform_weights():
    class Scheduler:
        timesteps = torch.tensor([0.5])

        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()

        def model_fn(self, **kwargs):
            self.calls.append(kwargs)
            return torch.zeros_like(kwargs["video_latents"]), torch.zeros_like(kwargs["audio_latents"])

    pipe = Pipe()
    pipe.calls = []
    clean_video = torch.zeros(1, 2, 20, 1, 1)
    clean_audio = torch.zeros(2, 32, 100)
    config = ContinuationRegionConfig(lambda_audio=1.0)
    loss = ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        pipe,
        region_config=config,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
        continuation_history_video_latents=None,
        continuation_history_audio_latents=None,
        continuation_prefix_mode="none",
    )
    assert torch.isfinite(loss)
    assert len(pipe.calls) == 1
    kwargs = pipe.calls[0]
    # A no-prefix window is never presented in the clean-injection form.
    assert "input_latents_video" not in kwargs
    assert "denoise_mask_video" not in kwargs
    assert "input_latents_audio" not in kwargs
    assert "denoise_mask_audio" not in kwargs
    # add_noise(zero_clean, noise, 0.5) exposes the drawn noise, so the expected
    # uniform-weight flow-matching loss can be recomputed exactly.
    noise_video = kwargs["video_latents"] / 0.5
    noise_audio = kwargs["audio_latents"] / 0.5
    expected = noise_video.square().mean() + config.lambda_audio * noise_audio.square().mean()
    assert loss.item() == pytest.approx(float(expected), rel=1e-5)


def test_mask_v14_prefix_none_ignores_history_and_masked_still_requires_it():
    class Scheduler:
        timesteps = torch.tensor([0.5])

        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()

        def model_fn(self, **kwargs):
            return (torch.zeros_like(kwargs["video_latents"]),
                    torch.zeros_like(kwargs["audio_latents"]))

    pipe = Pipe()
    clean_video = torch.randn(1, 2, 20, 1, 1)
    clean_audio = torch.randn(2, 32, 100)
    # The dataset sends the fully cached window for masked windows and None for
    # "none"; a truncated history proves the "none" branch never touches it.
    short_history = clean_video.narrow(-3, 0, 12)
    short_audio = clean_audio.narrow(-1, 0, 65)
    for history_video, history_audio in ((None, None), (short_history, short_audio)):
        loss = ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
            pipe,
            input_latents=clean_video,
            audio_input_latents=clean_audio,
            continuation_history_video_latents=history_video,
            continuation_history_audio_latents=history_audio,
            continuation_prefix_mode="none",
        )
        assert torch.isfinite(loss)
    # The masked path still consumes the 39-frame (12-step) prefix contract.
    with pytest.raises(ValueError):
        ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
            pipe,
            input_latents=clean_video,
            audio_input_latents=clean_audio,
            continuation_history_video_latents=clean_video.narrow(-3, 0, 5),
            continuation_prefix_mode="masked",
        )
    with pytest.raises(ValueError, match="prefix_mode"):
        ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
            pipe,
            input_latents=clean_video,
            continuation_prefix_mode="sometimes",
        )


def test_prefix_present_mode_validate():
    with pytest.raises(ValueError):
        ContinuationRegionConfig(
            conditioning_mode="legacy", prefix_present_mode="clean",
        ).validate(video_steps=20)
    with pytest.raises(ValueError):
        ContinuationRegionConfig(prefix_present_mode="unknown").validate(video_steps=20)
    ContinuationRegionConfig(prefix_present_mode="mixed").validate(video_steps=20)


def test_minimax_h3_parser_keeps_existing_sft_defaults():
    from examples.minimax_h3.model_training.train import minimax_h3_parser
    parser = minimax_h3_parser()
    args = parser.parse_args(["--dataset_base_path", ".", "--model_paths", "[]"])
    assert args.task == "sft"
    assert args.training_cfg_scale == 1.0
    assert args.cp_world_size == 1
    assert args.use_gradient_checkpointing is False
    assert args.bf16 is False
    assert args.seed == 42
    assert args.continuation_checkpoint_save_path is None
    assert args.continuation_resume_path is None


def test_continuation_loss_keeps_cfg_cp_scheduler_and_gradient_flags_compatible():
    class Scheduler:
        timesteps = torch.tensor([0.5])

        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()

        def model_fn(self, **kwargs):
            self.calls.append(kwargs)
            video = torch.zeros_like(kwargs["video_latents"])
            audio = torch.zeros_like(kwargs["audio_latents"])
            return video, audio

    pipe = Pipe()
    pipe.calls = []
    clean_video = torch.zeros(1, 2, 20, 1, 1)
    clean_audio = torch.zeros(2, 100)
    loss = ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        pipe,
        training_cfg_scale=2.0,
        inputs_nega={"negative_prompt": " "},
        cp_rank=0,
        cp_world_size=1,
        cp_group=None,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
        continuation_history_video_latents=clean_video,
        continuation_history_audio_latents=clean_audio,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    )
    assert torch.isfinite(loss)
    assert len(pipe.calls) == 2
    assert pipe.calls[0]["use_gradient_checkpointing"] is False
    assert pipe.calls[1]["use_gradient_checkpointing"] is True
    assert "timestep_video" in pipe.calls[1]
    assert "timestep_audio" in pipe.calls[1]


def test_continuation_checkpoint_roundtrip_and_mismatch_validation(tmp_path):
    from diffsynth.utils.continuation_lora import ContinuationRegionConfig
    config = ContinuationRegionConfig(
        overlap_video_steps=10,
        hard_core_video_steps=5,
        transition_video_steps=5,
        first_suffix_clip_steps=5,
        transition_weight=0.5,
        first_suffix_weight=3.0,
        suffix_weight=1.0,
        lambda_audio=0.5,
    )
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    path = tmp_path / "continuation.pt"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    digest = manifest_sha256(manifest)
    save_continuation_training_checkpoint(
        path,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        global_step=7,
        seed=123,
        manifest_hash=digest,
        region_config=config,
        rng_state=torch.get_rng_state(),
        random_state=__import__("random").getstate(),
    )
    restored = load_continuation_training_checkpoint(
        path,
        expected_manifest_hash=digest,
        expected_region_config=config,
    )
    assert restored["global_step"] == 7
    assert restored["seed"] == 123
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_continuation_training_checkpoint(path, expected_manifest_hash="wrong")
    # Schedule-only changes (weights, lengths) no longer invalidate a resume:
    # shot-level runs vary the overlap per sample and anneal the weights.
    retuned = ContinuationRegionConfig(
        **{**config.to_dict(), "transition_weight": 0.25, "overlap_video_steps": 42}
    )
    load_continuation_training_checkpoint(
        path, expected_region_config=retuned,
    )
    # A different conditioning contract still does.
    rechannelled = ContinuationRegionConfig(
        **{**config.to_dict(), "conditioning_mode": "legacy"}
    )
    with pytest.raises(ValueError, match="conditioning contract mismatch"):
        load_continuation_training_checkpoint(
            path,
            expected_manifest_hash=digest,
            expected_region_config=rechannelled,
        )


def test_lora_export_scale_zero_is_base_and_positive_changes_output(tmp_path):
    class LinearWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(3)
    base = LinearWrapper()
    lora_a = torch.randn(2, 4)
    lora_b = torch.randn(4, 2)
    lora_state = {
        "linear.lora_A.default.weight": lora_a,
        "linear.lora_B.default.weight": lora_b,
    }
    lora_path = tmp_path / "continuation_lora.safetensors"
    save_lora_safetensors(lora_state, lora_path, metadata={"rank": 2})
    loaded = load_lora_safetensors(lora_path)
    assert set(loaded) == set(lora_state)

    x = torch.randn(1, 4)
    expected = base(x).detach().clone()
    zero_model = copy.deepcopy(base)
    GeneralLoRALoader().fuse_lora_to_base_model(zero_model, loaded, alpha=0.0)
    assert torch.allclose(zero_model(x), expected)

    scaled_model = copy.deepcopy(base)
    GeneralLoRALoader().fuse_lora_to_base_model(scaled_model, loaded, alpha=2.0)
    assert not torch.allclose(scaled_model(x), expected)

    class FakePipe:
        def __init__(self):
            self.dit = object()
            self.loaded = []

        def load_lora(self, module, path, alpha):
            self.loaded.append((module, path, alpha))

    pipe = FakePipe()
    assert apply_continuation_lora(pipe, lora_path, 0.0)["applied"] is False
    assert pipe.loaded == []
    assert apply_continuation_lora(pipe, lora_path, 1.5)["applied"] is True
    assert pipe.loaded[-1][2] == 1.5


def test_build_mult_window_manifest_links_same_shot_only():
    common = {
        "sequence_id": "seq", "shot_index": 1, "window_frames": 345,
        "split": "train", "shot_start_sec": 0.0, "shot_end_sec": 40.0,
    }
    records = [
        {"sample_id": "a", "start_sec": 0.0, "end_sec": 345 / 24, **common},
        {"sample_id": "b", "start_sec": 306 / 24, "end_sec": (306 + 345) / 24, **common},
        {"sample_id": "c", "start_sec": 612 / 24, "end_sec": (612 + 345) / 24, **common},
    ]
    manifest, stats = build_manifest(records, overlap_frames=39, min_chain_length=2, tolerance_sec=1e-4)
    assert stats["groups_with_multiple_windows"] == 1
    assert stats["max_chain_length"] == 3
    assert [item["history_sample_id"] for item in manifest] == [None, "a", "b"]
    assert all(item["anchor_sample_id"] == "a" for item in manifest)


def test_cache_filename_keeps_both_final_and_temp_names_under_limit():
    """cache_filename must guarantee not only <id>.pt but also the atomic
    <id>.pt.tmp temp name fits in 255 bytes."""
    from diffsynth.utils.continuation_lora import cache_filename

    # A real long id whose .pt form is 252 bytes (OK) but whose .pt.tmp form is
    # 256 bytes (over the limit) - previously crashed with Errno 36.
    long_id = (
        "L0031_绝对不可能_侦探_上水流涼子的解析__"
        "e8f6a878880284b4767aabd97149b240_绝对不可能_侦探_上水流涼子的解析__"
        "Logically_Impossible__Detective_Ryoko_Kamizuru_Is_on_the_Case_2023_S01E02_"
        "WEB-DL_1080p_H264_AAC:shotcut:18168.75"
    )
    assert len(f"{long_id}.pt".encode("utf-8")) <= 255
    assert len(f"{long_id}.pt.tmp".encode("utf-8")) > 255

    for sid in (long_id, "380594d161fe191f", "origin:plain"):
        base = cache_filename(sid)
        final = base if base.endswith(".pt") else f"{base}.pt"
        temp = f"{final}.tmp"
        assert len(final.encode("utf-8")) <= 255, sid
        assert len(temp.encode("utf-8")) <= 255, sid
    # The over-long id must be shortened deterministically, not kept.
    assert cache_filename(long_id) != f"{long_id}.pt"
    assert cache_filename(long_id) == cache_filename(long_id)
    # Short ids keep their original name for resumability.
    assert cache_filename("380594d161fe191f") == "380594d161fe191f.pt"


def test_h3_frame_step_grid_round_trips_and_quantises_down():
    """The 17n+5 <-> 5n+2 grid is the contract shot-level sampling relies on."""
    for frames in (22, 39, 56, 73, 90, 141, 243, 345):
        steps = continuation_lora.video_frames_to_steps(frames)
        assert continuation_lora.video_steps_to_frames(steps) == frames
    assert continuation_lora.video_frames_to_steps(39) == 12
    assert continuation_lora.video_frames_to_steps(141) == 42
    # Rounding is always down, so the context never bleeds into the shot before
    # the previous one, and a request below 22 frames has no legal overlap.
    assert continuation_lora.resolve_h3_overlap_frames(141) == 141
    assert continuation_lora.resolve_h3_overlap_frames(140) == 124
    assert continuation_lora.resolve_h3_overlap_frames(39) == 39
    assert continuation_lora.resolve_h3_overlap_frames(38) == 22
    assert continuation_lora.resolve_h3_overlap_frames(21) == 0


def test_audio_overlap_is_anchored_on_frames_not_on_video_tokens():
    """39 frames = 65 audio steps exactly; token-linear scaling would drift."""
    assert continuation_lora.overlap_frames_to_audio_steps(39) == 65
    assert continuation_lora.overlap_frames_to_audio_steps(141) == 235
    # The frame anchor and the old ``steps * 65 / 12`` rule agree on the native
    # anchor and disagree by 7 audio steps (0.175 s) at a 141-frame context.
    assert round(12 * 65 / 12) == 65
    assert continuation_lora.overlap_frames_to_audio_steps(141) - round(42 * 65 / 12) == 7


def test_audio_latent_is_conformed_to_the_frame_derived_raster():
    """The audio grid is indexed by frames, not by the VAE's ``ceil`` output."""
    from diffsynth.utils.continuation_lora import (
        conform_audio_latent_steps, expected_audio_latent_steps,
    )

    # Frames are the anchor, so the step count is the frame-derived rounding.
    assert expected_audio_latent_steps(39) == 65
    assert expected_audio_latent_steps(345) == 575
    assert expected_audio_latent_steps(56) == 93  # ceil(74666 / 800) = 94
    with pytest.raises(ValueError, match="must be positive"):
        expected_audio_latent_steps(0)

    exact = torch.zeros(2, 32, 93)
    assert conform_audio_latent_steps(exact, 56) is exact
    assert conform_audio_latent_steps(None, 56) is None

    # A one-step overshoot is the audio VAE's right-padding; trim it.
    trimmed = conform_audio_latent_steps(torch.arange(2 * 32 * 94, dtype=torch.float32).reshape(2, 32, 94), 56)
    assert tuple(trimmed.shape) == (2, 32, 93)
    assert torch.equal(trimmed, torch.arange(2 * 32 * 94, dtype=torch.float32).reshape(2, 32, 94)[..., :93])
    # A one-step undershoot is padded, never resampled.
    padded = conform_audio_latent_steps(torch.ones(2, 32, 92), 56)
    assert tuple(padded.shape) == (2, 32, 93)
    assert torch.equal(padded[..., :92], torch.ones(2, 32, 92))
    assert float(padded[..., 92].abs().sum()) == 0.0
    # Anything further off is an encoding error, not a rounding artefact.
    with pytest.raises(ValueError, match="refusing to conform"):
        conform_audio_latent_steps(torch.zeros(2, 32, 100), 56)


def test_cache_writer_conforms_audio_to_the_frame_derived_raster(tmp_path, monkeypatch):
    """A cache is born on the audio raster the packed position grid assumes."""
    from diffsynth.utils.continuation_lora import build_latent_cache, load_latent_cache_full

    class VideoEncoder:
        def __call__(self, frames):
            return torch.zeros(1, 24, ((len(frames) - 5) // 17) * 5 + 2, 2, 2)

    class CeilAudioEncoder:
        """Mirrors the audio VAE: ``ceil(samples / 800)`` steps."""

        def __call__(self, waveform, sample_rate=None):
            import math

            return torch.zeros(2, 32, math.ceil(waveform.shape[-1] / 800))

    # 56 frames = 74666 samples at 32 kHz: 93.33 steps, so the VAE returns 94.
    sample = ContinuationSample(
        sample_id="off-lattice", sequence_id="seq", video_path="v.mp4", audio_path="a.wav",
        shot_index=1, shot_start_sec=15.583333333333334, shot_end_sec=17.916666666666668,
        start_sec=15.583333333333334, end_sec=17.916666666666668,
        window_frames=56, overlap_frames=22, hard_core_frames=0, transition_frames=0,
        prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
        audio_missing=False, split="train", prefix_mode="none",
    )
    def load_video(_path, window, **_kwargs):
        return ["frame"] * (window.video_end_frame - window.video_start_frame)

    # The real loader hands the VAE the sample interval of the window:
    # round(32000 * 17.9166667) - round(32000 * 15.5833333) = 74666 samples.
    def load_audio(_path, window, **_kwargs):
        return torch.zeros(2, window.audio_samples), 32000, False

    monkeypatch.setattr(continuation_lora, "load_video_window", load_video)
    monkeypatch.setattr(continuation_lora, "load_audio_window", load_audio)
    assert resolve_media_window(
        sample.start_sec, sample.end_sec,
    ).audio_samples == 74666
    build_latent_cache(
        [sample], tmp_path, video_encoder=VideoEncoder(), audio_encoder=CeilAudioEncoder(),
    )
    record = json.loads((tmp_path / "train" / "manifest.jsonl").read_text(encoding="utf-8").strip())
    video, audio, _memory, metadata = load_latent_cache_full(tmp_path / "train" / record["cache_path"])
    assert tuple(video.shape)[-3] == expected_video_latent_steps(56) == 17
    assert tuple(audio.shape) == (2, 32, 93) == (2, 32, expected_audio_latent_steps(56))
    assert tuple(metadata["audio_shape"]) == (2, 32, 93)


def test_continuation_dataset_conforms_legacy_ceil_audio_on_read(tmp_path):
    """A 43 GB cache built before the contract fix trains without re-encoding."""
    def build(sample_id: str, prefix_mode: str, audio_steps: int):
        sample = ContinuationSample(
            sample_id=sample_id, sequence_id="seq", video_path="v.mp4", audio_path=None,
            shot_index=1, shot_start_sec=0.0, shot_end_sec=20.0,
            start_sec=15.583333333333334, end_sec=17.916666666666668,
            window_frames=56, overlap_frames=22, hard_core_frames=0, transition_frames=0,
            prompt="scene", prompt_source="prompt", prompt_cleaning_version="v1",
            audio_missing=True, split="train", prefix_mode=prefix_mode,
        )
        video = torch.zeros(1, 24, 17, 2, 2)
        audio = torch.arange(2 * 32 * audio_steps, dtype=torch.float32).reshape(2, 32, audio_steps)
        metadata = make_latent_cache_metadata(sample, video, audio)
        path = tmp_path / "train" / f"{sample_id}.pt"
        save_latent_cache(path, video, audio, metadata)
        return {"cache_path": f"{sample_id}.pt", **sample.to_dict()}

    (tmp_path / "train").mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "train" / "manifest.jsonl"
    manifest.write_text(
        json.dumps(build("legacy-ceil", "none", 94)) + "\n"
        + json.dumps(build("legacy-ceil-masked", "masked", 94)) + "\n",
        encoding="utf-8",
    )
    plain, masked = ContinuationLatentDataset(manifest, split="train")[0], ContinuationLatentDataset(manifest, split="train")[1]
    assert tuple(plain["audio_input_latents"].shape) == (2, 32, 93)
    assert plain["continuation_history_audio_latents"] is None
    # The prefix is trimmed with the target, so the 22-frame overlap stays at the head.
    assert tuple(masked["continuation_history_audio_latents"].shape) == (2, 32, 93)
    assert torch.equal(masked["continuation_history_audio_latents"], masked["audio_input_latents"])
    assert masked["num_frames"] == 56


def test_continuation_layout_accepts_any_h3_aligned_overlap():
    for overlap in (22, 39, 56, 73, 90, 107, 124, 141):
        continuation_lora.validate_continuation_layout(345, overlap, 0)
    # Legacy whole-clip overlaps stay legal so old caches keep loading.
    continuation_lora.validate_continuation_layout(243, 34, 17)
    with pytest.raises(ValueError):
        continuation_lora.validate_continuation_layout(345, 40, 0)
    with pytest.raises(ValueError):
        continuation_lora.validate_continuation_layout(345, 250, 0)


def test_per_sample_overlap_resolution_falls_back_for_legacy_caches():
    resolve = continuation_lora.resolve_overlap_video_steps
    assert resolve({"overlap_frames": 141}, 12) == 42
    assert resolve({"overlap_frames": 22}, 12) == 7
    assert resolve({"overlap_video_steps": 27, "overlap_frames": 90}, 12) == 27
    # Legacy 34-frame caches are off the H3-aligned grid and keep the run value.
    assert resolve({"overlap_frames": 34}, 12) == 12
    assert resolve({}, 12) == 12
    assert resolve(None, 12) == 12


def test_region_fingerprint_ignores_schedule_but_not_conditioning():
    base = ContinuationRegionConfig()
    shot_level = ContinuationRegionConfig(
        overlap_video_steps=42, hard_core_video_steps=42, transition_video_steps=0,
    )
    assert continuation_lora.continuation_region_fingerprint(base) == \
        continuation_lora.continuation_region_fingerprint(shot_level)
    other = ContinuationRegionConfig(
        overlap_video_steps=12, hard_core_video_steps=4, transition_video_steps=8,
        conditioning_mode="legacy",
    )
    assert continuation_lora.continuation_region_fingerprint(base) != \
        continuation_lora.continuation_region_fingerprint(other)


def test_checkpoint_resume_tolerates_a_different_overlap(tmp_path):
    """Shot-level runs resume across overlap changes; only the contract is frozen."""
    path = tmp_path / "ckpt.pt"
    trained_with = ContinuationRegionConfig(
        overlap_video_steps=12, hard_core_video_steps=12,
    )
    save_continuation_training_checkpoint(
        path, model_state_dict={"w": torch.zeros(1)},
        optimizer_state_dict={}, scheduler_state_dict={}, global_step=3,
        region_config=trained_with,
    )
    resumed_with = ContinuationRegionConfig(
        overlap_video_steps=42, hard_core_video_steps=42,
    )
    payload = load_continuation_training_checkpoint(
        path, expected_region_config=resumed_with,
    )
    assert payload["global_step"] == 3
    with pytest.raises(ValueError):
        load_continuation_training_checkpoint(
            path,
            expected_region_config=ContinuationRegionConfig(
                overlap_video_steps=12, hard_core_video_steps=4,
                transition_video_steps=8, conditioning_mode="legacy",
            ),
        )


def _shot_record(sequence_id, shot_spans, frame_path):
    prompt = "\n".join(
        f"[Shot {i + 1}/{len(shot_spans)} | {start:.2f}s-{end:.2f}s]\nbody {i + 1}"
        for i, (start, end) in enumerate(shot_spans)
    )
    return {
        "sequence_id": sequence_id,
        "file_path": str(frame_path),
        "prompt": prompt,
        "audio_path": {"clean_wav_path": str(frame_path) + ".wav"},
    }


def test_shot_level_sampler_emits_variable_context_windows(tmp_path):
    source = tmp_path / "index.jsonl"
    video = tmp_path / "seq.mp4"
    video.write_bytes(b"")
    (tmp_path / "seq.mp4.wav").write_bytes(b"")
    # shot 1 is long (141+ frames -> full context), shot 2 is short, shot 3 is
    # long again; the last shot must be dropped only if it is too short.
    spans = [(0.0, 8.0), (8.0, 10.0), (10.0, 18.0), (18.0, 24.0)]
    with source.open("w") as handle:
        handle.write(json.dumps(_shot_record("seq_a", spans, video)) + "\n")
    stats = {}
    samples = list(continuation_lora.iter_shot_continuation_samples(
        source, stats=stats, require_paths=True,
    ))
    assert stats["single_shot"] == 0
    assert len(samples) == 3
    for sample in samples:
        continuation_lora.validate_continuation_layout(
            sample.window_frames, sample.overlap_frames, sample.hard_core_frames,
        )
        assert sample.window_frames <= continuation_lora.MAX_SHOT_WINDOW_FRAMES
        target = sample.window_frames - sample.overlap_frames
        assert target % 17 == 0 and target >= continuation_lora.MIN_SHOT_TARGET_FRAMES
        assert continuation_lora.is_h3_aligned_frames(sample.overlap_frames)
        # The context is the tail of the previous shot: the window must start
        # exactly `overlap` frames before the target shot's own start.
        assert round(sample.start_sec * 24) + sample.overlap_frames == round(
            sample.shot_start_sec * 24
        )
    assert [sample.shot_index for sample in samples] == [2, 3, 4]
    # The 8 s first shot gives the next sample the full context cap, the 2 s
    # second shot caps its successor's context to 39 frames, and the 8 s third
    # shot restores the cap.
    assert samples[0].overlap_frames == continuation_lora.MAX_SHOT_CONTEXT_FRAMES
    assert samples[1].overlap_frames == 39
    assert samples[2].overlap_frames == continuation_lora.MAX_SHOT_CONTEXT_FRAMES


def test_shot_level_sampler_interleaves_prefix_free_rows(tmp_path):
    source = tmp_path / "index.jsonl"
    video = tmp_path / "seq.mp4"
    video.write_bytes(b"")
    (tmp_path / "seq.mp4.wav").write_bytes(b"")
    spans = [(0.0, 6.0), (6.0, 12.0)]
    with source.open("w") as handle:
        handle.write(json.dumps(_shot_record("seq_b", spans, video)) + "\n")
    stats = {}
    samples = list(continuation_lora.iter_shot_continuation_samples(
        source, include_plain=True, stats=stats,
    ))
    assert stats["accepted_plain"] == 1
    # one prefix-free row + one continuation pair (two shots -> one pair)
    assert len(samples) == 2
    assert samples[0].prefix_mode == "none"
    assert samples[0].shot_index == 1
    continuation_lora.validate_continuation_layout(
        samples[0].window_frames, samples[0].overlap_frames, samples[0].hard_core_frames,
    )
    assert samples[0].window_frames % 17 == 5
    assert samples[1].prefix_mode == "masked"


def test_shot_level_sampler_rejects_non_v14_conditioning(tmp_path):
    with pytest.raises(ValueError):
        list(continuation_lora.iter_shot_continuation_samples(
            tmp_path / "missing.jsonl", conditioning_mode="legacy",
        ))


def test_loss_takes_the_overlap_from_the_sample_cache_metadata(monkeypatch):
    """Shot-level samples carry their own overlap; the loss must honour it."""
    from diffsynth.diffusion import loss as loss_module

    class Scheduler:
        timesteps = torch.tensor([0.5])

        def add_noise(self, clean, noise, timestep):
            return clean + noise * timestep

    class Pipe:
        scheduler = Scheduler()
        scheduler_audio = Scheduler()
        in_iteration_models = ()

        def model_fn(self, **kwargs):
            video = torch.zeros_like(kwargs["video_latents"])
            audio = torch.zeros_like(kwargs["audio_latents"])
            return video, audio

    calls = []
    original = loss_module.v3_continuation_inputs

    def spy(target, history, noise, timestep, add_noise, **kwargs):
        calls.append(kwargs)
        return original(target, history, noise, timestep, add_noise, **kwargs)

    monkeypatch.setattr(loss_module, "v3_continuation_inputs", spy)
    clean_video = torch.randn(1, 2, 27, 1, 1)
    clean_audio = torch.randn(2, 32, 200)
    # Run-level default is the shipped 39-frame / 12-step contract.
    run_config = ContinuationRegionConfig()
    ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        Pipe(),
        region_config=run_config,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
        continuation_cache_metadata={"window_frames": 243, "overlap_frames": 73},
    )
    assert calls[0]["hard_core"] == 22
    # 73 output frames = 121.7 s * 40 steps/s -> 122 audio latent steps.
    assert calls[1]["hard_core"] == 122

    calls.clear()
    ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        Pipe(),
        region_config=run_config,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
        continuation_cache_metadata={"window_frames": 243, "overlap_frames": 39},
    )
    assert calls[0]["hard_core"] == 12

    # Without metadata the run-level default still applies (legacy caches).
    calls.clear()
    ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
        Pipe(),
        region_config=run_config,
        input_latents=clean_video,
        audio_input_latents=clean_audio,
    )
    assert calls[0]["hard_core"] == 12


class _CallCounter(torch.nn.Module):
    """Stand-in for the shared video projection; counts every invocation."""

    def __init__(self, in_dim: int, out_dim: int = 4):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)
        self.calls = 0

    def forward(self, rows):
        self.calls += 1
        return self.proj(rows)


def _memory_embed_stub(hidden: int = 8, text_dim: int = 5, x_dim: int = 96, audio_dim: int = 32):
    """Duck-typed stand-in carrying only what ``MiniMaxH3DiT._embed`` touches."""

    class Stub:
        pass

    stub = Stub()
    stub.hidden_size = hidden
    stub.video_patch_proj = _CallCounter(x_dim, hidden)
    stub.audio_patch_proj = torch.nn.Linear(audio_dim, hidden)
    stub.condition_proj = torch.nn.Linear(text_dim, hidden)
    stub.token_refiner = lambda embed, cu_seqlens=None, max_seqlen=None: embed
    stub.time_embedder = lambda timesteps, dtype=None: torch.zeros(
        timesteps.shape[0], hidden, dtype=dtype
    )
    return stub


def test_memory_embedding_call_is_not_data_dependent():
    """The memory projection is invoked even with zero memory rows.

    Under ZeRO-3 a data-dependent module call site is a data-dependent
    collective: the rank that owns no memory row would issue one gather fewer
    than its peers and hang the group.  The projection therefore has to run on
    every rank; only the ``index_add_`` may be conditional.
    """
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT

    hidden = 8
    x = torch.randn(1, 6, 96)      # 24 latents x (1, 2, 2) patch
    audio_x = torch.randn(1, 4, 32)
    text_embeddings = torch.randn(2, 5)

    # One stub, so every call sees the same weights and only ``mem_pos`` varies.
    stub = _memory_embed_stub(hidden)

    def run(mem_pos):
        embeddings, _ = MiniMaxH3DiT._embed(
            stub,
            x=x, audio_x=audio_x, text_embeddings_selected=text_embeddings,
            unique_timesteps=torch.zeros(1), img_pos=torch.tensor([2, 3]),
            audio_pos=torch.tensor([0, 1]), text_pos=torch.tensor([4, 5]),
            text_embed_select=None, refiner_cu_seqlens=torch.tensor([0, 2]),
            refiner_max_seqlen=2, seq_len=6, device=x.device, mem_pos=mem_pos,
        )
        return embeddings

    # Long window, no memory at all: still two invocations of the shared projection.
    embeddings_without = run(None)
    assert stub.video_patch_proj.calls == 2
    # Idempotent: the discarded projection must not perturb the window's rows.
    assert torch.equal(run(None), embeddings_without)
    assert stub.video_patch_proj.calls == 4

    # Empty memory index (this rank owns none of the memory rows): same count.
    run(torch.tensor([], dtype=torch.long))
    assert stub.video_patch_proj.calls == 6

    # Real memory rows: same count, and the rows reach the embedding.
    embeddings_with = run(torch.tensor([0, 1]))
    assert stub.video_patch_proj.calls == 8
    assert not torch.equal(embeddings_with, embeddings_without)
