"""Rollout-side long-horizon memory must reproduce the training cut.

The training cache builder cuts two slots out of the source media
(``diffsynth.utils.continuation_lora.encode_memory_slot``).  A rollout has to
make the *same* two cuts out of the frames it generated itself -- same frame
count, same anchor, same lead, and the same cases where a slot is dropped instead
of faked.  These tests pin that agreement directly by driving
``encode_memory_slot`` with a stub loader and comparing it against the buffer.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationConfig,
    H3ContinuationRunner,
    H3Segment,
    H3SegmentPlan,
)
from diffsynth.utils import continuation_lora as cl
from diffsynth.utils.continuation_lora import (
    ContinuationSample,
    dual_memory_slot_specs,
    encode_memory_slot,
    expected_video_latent_steps,
)
from diffsynth.utils.h3_memory_rollout import H3MemoryRolloutPlan, H3MemorySlotBuffer


STM_FRAMES = 39
LTM_FRAMES = 39
OVERLAP_FRAMES = 39


def _tagged_frames(count, *, tag=0):
    return [Image.new("RGB", (2, 2), (tag % 256, index % 256, 0)) for index in range(count)]


def _tagged_frame(window, index):
    """Tag one rollout frame with (window, index); the index spans two channels."""
    return Image.new("RGB", (2, 2), (window % 256, (index // 256) % 256, index % 256))


def _tag_of(window, index):
    return (window % 256, (index // 256) % 256, index % 256)


def _encoder(tensor_shape=(1, 24, 12, 2, 2)):
    seen = []

    def encode(frames):
        seen.append(list(frames))
        return torch.zeros(tensor_shape)

    encode.seen = seen
    return encode


def _sample(start_sec, end_sec):
    return ContinuationSample(
        sample_id="s", sequence_id="q", video_path="v.mp4", audio_path=None,
        shot_index=0, shot_start_sec=start_sec, shot_end_sec=end_sec,
        start_sec=start_sec, end_sec=end_sec,
        window_frames=345, overlap_frames=39, hard_core_frames=0, transition_frames=0,
        prompt="p", prompt_source="t", prompt_cleaning_version="v1",
        audio_missing=True, split="train",
    )


def _media_window(start_frame, end_frame):
    return cl.MediaWindow(
        start_sec=start_frame / 24, end_sec=end_frame / 24,
        video_start_frame=start_frame, video_end_frame=end_frame,
        audio_start_sample=0, audio_end_sample=0, audio_start_latent_step=0, audio_end_latent_step=0,
    )


def _training_slot_read(spec, *, window_start_frame, window_end_frame, monkeypatch):
    """Run the real training cutter; report which frames it read, if any."""
    read = {}

    def fake_loader(path, media_window, frame_processor=None):
        read["window"] = media_window
        return _tagged_frames(media_window.video_end_frame - media_window.video_start_frame)

    monkeypatch.setattr(cl, "load_video_window", fake_loader)
    tensor, description = encode_memory_slot(
        _sample(window_start_frame / 24, window_end_frame / 24),
        spec,
        video_encoder=lambda frames, dtype=None: torch.zeros(
            1, 24, expected_video_latent_steps(spec.frames), 2, 2
        ),
        window=_media_window(window_start_frame, window_end_frame),
    )
    return tensor, description, read.get("window")


def _buffer(observed_frames=0, *, plan=None):
    plan = plan or H3MemoryRolloutPlan(stm_frames=STM_FRAMES, ltm_frames=LTM_FRAMES)
    buffer = H3MemorySlotBuffer(plan, overlap_frames=OVERLAP_FRAMES)
    if observed_frames:
        buffer.observe(_tagged_frames(observed_frames, tag=7), start_frame=0)
    return buffer


def test_plan_validates_frames_are_on_the_vae_grid():
    with pytest.raises(ValueError, match="17n\\+5"):
        H3MemoryRolloutPlan(stm_frames=40)
    with pytest.raises(ValueError, match="17n\\+5"):
        H3MemoryRolloutPlan(ltm_frames=23)
    with pytest.raises(ValueError, match="positive"):
        H3MemoryRolloutPlan(stm_frames=0)
    with pytest.raises(ValueError, match=">= 0"):
        H3MemoryRolloutPlan(stm_frames=39, ltm_lead_steps=-1)
    assert H3MemoryRolloutPlan(stm_frames=22, ltm_frames=22).enabled
    assert H3MemoryRolloutPlan(stm_frames=39, ltm_frames=39).slot_names() == ("stm", "ltm")
    assert H3MemoryRolloutPlan().slot_names() == ()


def test_rollout_keeps_and_drops_exactly_where_training_did(monkeypatch):
    """Sweep window positions; the buffer and the cache builder must agree.

    ``encode_memory_slot`` drops a slot when its source frames do not exist
    (sequence head) or would overlap the window they condition.  A rollout that
    kept such a slot would condition the model on something it never saw while
    training, so the decision table has to match, not merely the tensor shapes.
    """
    specs = dual_memory_slot_specs(STM_FRAMES, LTM_FRAMES)
    assert [spec.name for spec in specs] == ["stm", "ltm"]

    for window_start in (0, 5, 38, 39, 40, 100, 306, 612, 1500):
        window_end = window_start + 345
        # A rollout observes frames as it emits them, so by the time this window
        # runs, exactly ``[0, window_start)`` frames exist.
        buffer = _buffer(window_start)
        slots = {slot["name"]: slot for slot in buffer.slots(
            window_start_frame=window_start, window_end_frame=window_end, encode=_encoder(),
        )}
        for spec in specs:
            _tensor, description, media_window = _training_slot_read(
                spec, window_start_frame=window_start, window_end_frame=window_end,
                monkeypatch=monkeypatch,
            )
            assert (spec.name in slots) is (media_window is not None), (spec.name, window_start)
            if media_window is None:
                continue
            assert media_window.video_start_frame == (
                0 if spec.anchor == "sequence-head" else window_start - spec.frames
            ), (spec.name, window_start)
            assert media_window.video_end_frame - media_window.video_start_frame == spec.frames
            assert description["frames"] == spec.frames
            assert slots[spec.name]["lead_steps"] == spec.lead_steps


def test_stm_reads_the_frames_immediately_before_the_window():
    frames = _tagged_frames(345, tag=3)
    buffer = _buffer()
    buffer.observe(frames, start_frame=0)
    encode = _encoder()

    slots = buffer.slots(window_start_frame=306, window_end_frame=651, encode=encode)
    stm = next(slot for slot in slots if slot["name"] == "stm")
    assert stm["lead_steps"] is None
    assert len(encode.seen[0]) == STM_FRAMES
    assert encode.seen[0] == frames[306 - STM_FRAMES:306]
    assert stm["tensor"].shape == (1, 24, 12, 2, 2)


def test_ltm_reads_the_sequence_head_and_is_encoded_once():
    buffer = _buffer()
    head_frames = _tagged_frames(345, tag=5)
    buffer.observe(head_frames, start_frame=0)
    encode = _encoder()

    first = buffer.slots(window_start_frame=306, window_end_frame=651, encode=encode)
    buffer.observe(_tagged_frames(306, tag=6), start_frame=345)
    second = buffer.slots(window_start_frame=612, window_end_frame=957, encode=encode)

    ltm = next(slot for slot in first if slot["name"] == "ltm")
    assert ltm["lead_steps"] == cl.DEFAULT_LTM_LEAD_STEPS
    assert encode.seen[1] == head_frames[:LTM_FRAMES]
    head_reads = [block for block in encode.seen if block and block[0] is head_frames[0]]
    assert len(head_reads) == 1, (
        "the head slot is a constant anchor; re-encoding it per window would both waste a VAE "
        "pass and let its content drift between windows"
    )
    assert next(slot for slot in second if slot["name"] == "ltm")["tensor"] is ltm["tensor"]
    assert buffer.provenance_metadata()["ltm_encodes"] == 1
    assert buffer.provenance_metadata()["stm_encodes"] == 2


def test_sequence_head_window_conditions_on_nothing():
    buffer = _buffer()
    buffer.observe(_tagged_frames(345, tag=1), start_frame=0)
    assert buffer.slots(window_start_frame=0, window_end_frame=345, encode=_encoder()) == []


def test_retention_still_covers_the_next_stm_cut():
    """A long rollout must not prune below what the next window's STM reaches to."""
    buffer = _buffer()
    buffer.observe(_tagged_frames(345, tag=2), start_frame=0)
    for index in range(1, 8):
        buffer.observe(_tagged_frames(306, tag=index), start_frame=345 + (index - 1) * 306)
    window_start = buffer.emitted_frames - OVERLAP_FRAMES
    encode = _encoder()
    slots = buffer.slots(window_start_frame=window_start, window_end_frame=window_start + 345, encode=encode)
    assert [slot["name"] for slot in slots] == ["stm", "ltm"]
    assert len(encode.seen[0]) == STM_FRAMES


def test_observe_rejects_gapped_history_and_absorbs_redecoded_overlap():
    buffer = _buffer()
    buffer.observe(_tagged_frames(100), start_frame=0)
    with pytest.raises(ValueError, match="only 100 frames exist"):
        buffer.observe(_tagged_frames(10), start_frame=150)
    # A context decode re-emits the overlap region the previous window already
    # covered; absorbing it must not duplicate frames or move the timeline.
    buffer.observe(_tagged_frames(50), start_frame=80)
    assert buffer.emitted_frames == 130
    # A fully redundant re-observation is a no-op rather than an error.
    buffer.observe(_tagged_frames(30), start_frame=100)
    assert buffer.emitted_frames == 130


def test_retain_frames_must_cover_the_stm_reach():
    with pytest.raises(ValueError, match="cannot cover"):
        H3MemorySlotBuffer(
            H3MemoryRolloutPlan(stm_frames=STM_FRAMES), overlap_frames=OVERLAP_FRAMES, retain_frames=64
        )


class _TaggedFakeH3Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        window = len(self.calls)
        frames = kwargs["num_frames"]
        video = [Image.new("RGB", (2, 2), (window % 256, index % 256, 0)) for index in range(frames)]
        return video, torch.zeros(2, round(frames / 24 * 32_000))


def _runner(fake, plan, **kwargs):
    return H3ContinuationRunner(
        fake,
        H3ContinuationConfig(requested_window_frames=345, overlap_frames=OVERLAP_FRAMES),
        continuation_mode="masked-av-v14",
        prefer_latent_handoff=True,
        memory_rollout_plan=plan,
        memory_slot_encoder=lambda frames, height=None, width=None: torch.zeros(1, 24, 12, 2, 2),
        **kwargs,
    )


def test_runner_conditions_each_window_on_the_slots_the_rollout_built():
    plan = H3MemoryRolloutPlan(stm_frames=STM_FRAMES, ltm_frames=LTM_FRAMES)
    fake = _TaggedFakeH3Pipeline()
    result = _runner(fake, plan).run(
        H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    )

    # Window 0 is the sequence head, so training dropped both slots there; a
    # rollout that conditioned on them anyway would be off the trained support.
    assert "memory_latents" not in fake.calls[0]
    assert [slot["name"] for slot in fake.calls[1]["memory_latents"]] == ["stm", "ltm"]
    assert fake.calls[1]["memory_latents"][0]["lead_steps"] is None
    assert fake.calls[1]["memory_latents"][1]["lead_steps"] == cl.DEFAULT_LTM_LEAD_STEPS

    provenance = result.manifest.experiment_metadata["long_horizon_memory"]
    assert provenance["source"] == "rollout-generated-frames"
    assert provenance["slot_names"] == ["stm", "ltm"]
    assert provenance["emitted_frames"] == 651


def test_runner_rejects_memory_slots_outside_the_trained_contract():
    plan = H3MemoryRolloutPlan(stm_frames=STM_FRAMES)
    with pytest.raises(ValueError, match="masked-av-v14"):
        H3ContinuationRunner(
            _TaggedFakeH3Pipeline(),
            H3ContinuationConfig(requested_window_frames=345, overlap_frames=OVERLAP_FRAMES),
            continuation_mode="latent-handoff", memory_rollout_plan=plan,
        )
    with pytest.raises(ValueError, match="global latent decode"):
        _runner(_TaggedFakeH3Pipeline(), plan, global_latent_decode=True)
    with pytest.raises(TypeError, match="H3MemoryRolloutPlan"):
        H3ContinuationRunner(
            _TaggedFakeH3Pipeline(),
            H3ContinuationConfig(requested_window_frames=345, overlap_frames=OVERLAP_FRAMES),
            memory_rollout_plan={"stm_frames": 39},
        )


def test_disabled_plan_leaves_the_request_untouched():
    fake = _TaggedFakeH3Pipeline()
    _runner(fake, H3MemoryRolloutPlan()).run(
        H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment()))
    )
    for call in fake.calls:
        assert "memory_latents" not in call


class _VaeFakeH3Pipeline:
    """A fake pipeline that owns a VAE, so the runner's built-in slot path runs."""

    def __init__(self):
        self.calls = []
        self.encoded = []
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.loaded = []
        self.video_vae = self
        self.video_vae_calls = []
        self.encoded_frames = []

    def load_models_to_device(self, names):
        self.loaded.append(tuple(names))

    def preprocess_video(self, frames, torch_dtype=None, min_value=None, device=None):
        arrays = [np.asarray(frame).copy() for frame in frames]
        # Every frame carries (window, in-window index) so a test can name exactly
        # which source frames a slot was cut from.
        self.encoded_frames.append([tuple(int(value) for value in arr[0, 0]) for arr in arrays])
        return torch.stack([torch.from_numpy(arr) for arr in arrays]).permute(3, 0, 1, 2)

    def encode_video(self, tensor, dtype=None, **kwargs):
        self.video_vae_calls.append(int(tensor.shape[1]))
        return torch.zeros(1, 24, (tensor.shape[1] - 5) // 17 * 5 + 2, 2, 2)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        window = len(self.calls)
        frames = kwargs["num_frames"]
        video = [_tagged_frame(window, index) for index in range(frames)]
        return video, torch.zeros(2, round(frames / 24 * 32_000))
def test_builtin_slot_path_feeds_the_vae_and_reports_via_the_observer():
    """With no custom encoder the runner must encode slots itself, once per slot."""
    fake = _VaeFakeH3Pipeline()
    observed = []
    runner = H3ContinuationRunner(
        fake,
        H3ContinuationConfig(requested_window_frames=345, overlap_frames=OVERLAP_FRAMES),
        continuation_mode="masked-av-v14",
        prefer_latent_handoff=True,
        memory_rollout_plan=H3MemoryRolloutPlan(stm_frames=STM_FRAMES, ltm_frames=LTM_FRAMES),
        memory_slot_observer=lambda name, tensor: observed.append((name, tuple(tensor.shape))),
    )
    runner.run(H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment())))

    # Window 0 is the head, so only window 1 contributes: one STM and one LTM.
    assert [name for name, _shape in observed] == ["stm", "ltm"]
    assert fake.video_vae_calls == [STM_FRAMES, LTM_FRAMES]
    assert fake.loaded and all(names == ("video_vae",) for names in fake.loaded)
    assert observed[0][1] == (1, 24, 12, 2, 2)


def test_three_window_rollout_cuts_stm_from_the_previous_shot_tail():
    """End-to-end proof that the slot's frames are the ones training would have cut.

    Window 1's STM is the last 39 frames of window 0, and window 2's is the last
    39 frames of window 1.  With per-window tagged frames the assertion is exact:
    the slot must start at in-window index 267 of the *previous* window, because
    306 - 39 = 267.
    """
    fake = _VaeFakeH3Pipeline()
    observed = []
    runner = H3ContinuationRunner(
        fake,
        H3ContinuationConfig(requested_window_frames=345, overlap_frames=OVERLAP_FRAMES),
        continuation_mode="masked-av-v14",
        prefer_latent_handoff=True,
        memory_rollout_plan=H3MemoryRolloutPlan(stm_frames=STM_FRAMES, ltm_frames=LTM_FRAMES),
        memory_slot_observer=lambda name, tensor: observed.append(name),
    )
    runner.run(H3SegmentPlan(global_prompt="test", segments=(H3Segment(), H3Segment(), H3Segment())))

    # One LTM encode plus one STM per continuation window.
    assert observed == ["stm", "ltm", "stm"]
    assert fake.video_vae_calls == [STM_FRAMES, LTM_FRAMES, STM_FRAMES]
    # The LTM is the sequence head, encoded once, from window 0's first 39 frames.
    # The fake tags window w's frames with red = w + 1 (it appends the call first).
    assert fake.encoded_frames[1] == [_tag_of(1, index) for index in range(LTM_FRAMES)]
    # Window 1 starts at frame 306 with 39 frames of overlap, so its STM is the
    # previous window's [267, 306) -- never any of window 1's own frames.
    assert fake.encoded_frames[0] == [_tag_of(1, index) for index in range(267, 306)]
    # Window 2 starts at 612, so its STM is window 1's [573, 612) = in-window 267.
    assert fake.encoded_frames[2] == [_tag_of(2, index) for index in range(267, 306)]
