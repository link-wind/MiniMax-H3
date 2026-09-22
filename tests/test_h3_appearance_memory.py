import json

import pytest
import torch
from PIL import Image

from diffsynth.pipelines.h3_appearance_memory import (
    H3AppearanceMemoryBank,
    H3AppearanceMemoryConfig,
    appearance_memory_config_from_legacy,
)
from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationConfig,
    H3ContinuationRunner,
    H3Segment,
    H3SegmentPlan,
)


def _solid_image(color):
    return Image.new("RGB", (2, 2), color)


def _image_frames(count, mode="red"):
    if mode == "green":
        return [_solid_image((0, index % 256, 0)) for index in range(count)]
    return [_solid_image((index % 256, 0, 0)) for index in range(count)]


class _ImageFakeH3Pipeline:
    def __init__(self, *, emit_video=True, emit_audio=True):
        self.calls = []
        self.emit_video = emit_video
        self.emit_audio = emit_audio

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = kwargs["num_frames"]
        video = _image_frames(frames) if self.emit_video else None
        samples = round(frames / 24 * 32_000)
        audio = torch.zeros(2, samples) if self.emit_audio else None
        return video, audio


def test_appearance_memory_config_validates_and_maps_legacy_flags():
    with pytest.raises(ValueError, match="mode"):
        H3AppearanceMemoryConfig(mode="unknown")
    with pytest.raises(ValueError, match="static"):
        H3AppearanceMemoryConfig(mode="static", memory_frame_budget=1)
    with pytest.raises(ValueError, match="dynamic.*trusted"):
        H3AppearanceMemoryConfig(mode="dynamic", trusted_anchor_frames=0)
    with pytest.raises(ValueError, match="max_visual_references"):
        H3AppearanceMemoryConfig(
            mode="static",
            trusted_anchor_frames=4,
            boundary_reference_frames=True,
            max_visual_references=4,
        )

    legacy = appearance_memory_config_from_legacy(3, 1)
    assert legacy.mode == "static"
    assert legacy.trusted_anchor_frames == 3
    assert legacy.boundary_reference_frames is True
    assert appearance_memory_config_from_legacy().mode == "disabled"


def test_bank_initializes_trusted_anchors_and_preserves_user_references():
    config = H3AppearanceMemoryConfig(
        mode="static",
        trusted_anchor_frames=2,
        boundary_reference_frames=True,
        max_visual_references=3,
    )
    bank = H3AppearanceMemoryBank(config)
    frames = _image_frames(9)
    bank.initialize_anchors(frames, segment_index=0)
    bank.set_boundary(frames[-1], segment_index=0, frame_index=8)

    user = {"type": "image", "image": _solid_image((255, 255, 255))}
    references = bank.build_references([user])
    assert references[0] is user
    assert [ref["type"] for ref in references[1:]] == ["image", "image", "image"]
    assert len(bank.anchors) == 2
    assert bank.anchors[0].frame_index == 3
    assert bank.anchors[1].frame_index == 6
    assert bank.boundary is not None
    assert bank.active_reference_count() == 3
    assert bank.provenance_metadata()["anchors"] == [
        {"role": "trusted", "segment_index": 0, "frame_index": 3},
        {"role": "trusted", "segment_index": 0, "frame_index": 6},
    ]


def test_dynamic_bank_skips_near_duplicates_and_adds_diverse_memory():
    config = H3AppearanceMemoryConfig(
        mode="dynamic",
        trusted_anchor_frames=1,
        memory_frame_budget=2,
        boundary_reference_frames=False,
        max_visual_references=3,
        diversity_threshold=0.05,
        candidate_frames_per_window=2,
    )
    bank = H3AppearanceMemoryBank(config)
    bank.initialize_anchors(_image_frames(9, mode="red"), segment_index=0)
    assert len(bank.anchors) == 1

    bank.update_from_window(
        _image_frames(9, mode="red"),
        segment_index=1,
        overlap_frames=2,
    )
    assert not bank.memory_frames

    bank.update_from_window(
        [_solid_image((0, 128, 0)) for _ in range(9)],
        segment_index=2,
        overlap_frames=2,
    )
    assert 0 < len(bank.memory_frames) <= 2
    assert all(item.role == "memory" for item in bank.memory_frames)
    assert bank.memory_frames[0].frame.getpixel((0, 0))[1] > 0


def test_legacy_global_references_are_injected_and_recorded_in_state():
    fake = _ImageFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    runner = H3ContinuationRunner(fake, global_reference_frames=2)
    result = runner.run(plan)

    assert "references" not in fake.calls[0]
    assert len(fake.calls[1]["references"]) == 2
    assert result.state.records[1].global_reference_frames == 2
    manifest = json.loads(result.manifest.to_json())
    assert manifest["appearance_memory_metadata"]["mode"] == "static"
    assert len(manifest["appearance_memory_metadata"]["anchors"]) == 2


def test_default_runner_does_not_inject_appearance_references():
    fake = _ImageFakeH3Pipeline()
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    runner = H3ContinuationRunner(fake)
    result = runner.run(plan)

    assert "references" not in fake.calls[1]
    assert result.state.records[1].appearance_memory_mode == "disabled"
    assert result.state.records[1].appearance_memory_frames == 0


def test_explicit_dynamic_bank_runs_masked_av_v14_with_local_positions():
    fake = _ImageFakeH3Pipeline()
    config = H3ContinuationConfig(requested_window_frames=345, overlap_frames=39)
    bank_config = H3AppearanceMemoryConfig(
        mode="dynamic",
        trusted_anchor_frames=1,
        memory_frame_budget=1,
        boundary_reference_frames=True,
        max_visual_references=3,
        candidate_frames_per_window=2,
    )
    plan = H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    runner = H3ContinuationRunner(
        fake,
        config,
        continuation_mode="masked-av-v14",
        prefer_latent_handoff=True,
        appearance_memory_config=bank_config,
    )
    result = runner.run(plan)

    call = fake.calls[1]
    assert call["references"]
    for key in (
        "video_source_indices",
        "audio_source_indices",
        "global_video_latent_length",
        "global_audio_latent_length",
        "global_temporal_position_origin",
    ):
        assert key not in call
    assert result.state.records[1].appearance_memory_mode == "dynamic"
    assert result.state.records[1].appearance_memory_frames >= 2
    assert result.manifest.appearance_memory_metadata["boundary"]["role"] == "boundary"


def test_explicit_bank_rejects_non_masked_mode_and_global_latent_decode():
    config = H3AppearanceMemoryConfig(mode="dynamic", trusted_anchor_frames=1)
    with pytest.raises(ValueError, match="masked-av-v14"):
        H3ContinuationRunner(
            _ImageFakeH3Pipeline(),
            continuation_mode="latent-handoff",
            prefer_latent_handoff=True,
            appearance_memory_config=config,
        )
    with pytest.raises(ValueError, match="global latent decode"):
        H3ContinuationRunner(
            _ImageFakeH3Pipeline(),
            H3ContinuationConfig(requested_window_frames=345, overlap_frames=39),
            continuation_mode="masked-av-v14",
            prefer_latent_handoff=True,
            global_latent_decode=True,
            appearance_memory_config=config,
        )
