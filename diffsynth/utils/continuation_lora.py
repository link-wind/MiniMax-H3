"""CPU-safe data and objective helpers for MiniMax-H3 continuation LoRA.

The module intentionally contains no model or media decoder imports.  It is used by
the streaming index builder, cache jobs, and unit tests before a H3 checkpoint is
available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch


VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32_000
AUDIO_LATENT_RATE = 40
VIDEO_CLIP_FRAMES = 17
VIDEO_LENGTH_REMAINDER = 5
CACHE_SCHEMA_VERSION = "h3-continuation-v1"
PAIR_MANIFEST_SCHEMA_VERSION = "h3-continuation-pair-v1"
TRAINING_CHECKPOINT_KIND = "h3-continuation-training-v1"
LORA_FORMAT_VERSION = "h3-continuation-lora-v1"


# The H3 video VAE timeline is a two-way grid: ``17n + 5`` output frames encode
# to ``5n + 2`` latent steps (22/39/56/... frames <-> 7/12/17/... steps).  Both
# directions are needed once a sample carries its own overlap, because the
# training stack stores overlaps as frames while the model consumes steps.
MIN_OVERLAP_FRAMES = VIDEO_CLIP_FRAMES + VIDEO_LENGTH_REMAINDER


def is_h3_aligned_frames(frames: int) -> bool:
    """True when ``frames`` lies on the ``17n+5`` grid the H3 VAE accepts."""
    return (
        frames >= MIN_OVERLAP_FRAMES
        and frames % VIDEO_CLIP_FRAMES == VIDEO_LENGTH_REMAINDER % VIDEO_CLIP_FRAMES
    )


def video_frames_to_steps(video_frames: int) -> int:
    """Latent steps of an H3-aligned frame count (``17n+5``)."""
    if resolve_h3_video_frames(video_frames) != video_frames:
        raise ValueError(f"video_frames must satisfy 17n+5, got {video_frames}")
    return 5 * ((video_frames - VIDEO_LENGTH_REMAINDER) // VIDEO_CLIP_FRAMES) + 2


def video_steps_to_frames(video_steps: int) -> int:
    """Inverse of :func:`video_frames_to_steps` (``5n+2`` -> ``17n+5``)."""
    delta = video_steps - 2
    if delta < 0 or delta % 5:
        raise ValueError(f"video_steps must satisfy 5n+2, got {video_steps}")
    return VIDEO_CLIP_FRAMES * (delta // 5) + VIDEO_LENGTH_REMAINDER


def resolve_h3_overlap_frames(requested_frames: int) -> int:
    """Round an arbitrary context length down onto the ``17n+5`` overlap grid.

    Context length is a data-side choice (the previous shot's length), so it
    must be quantised onto the same grid the VAE accepts.  Rounding *down*
    keeps the context inside the previous shot instead of bleeding into the
    shot before it.  Returns 0 when the request is below the smallest usable
    overlap.
    """
    if requested_frames < MIN_OVERLAP_FRAMES:
        return 0
    n = (requested_frames - VIDEO_LENGTH_REMAINDER) // VIDEO_CLIP_FRAMES
    return VIDEO_CLIP_FRAMES * n + VIDEO_LENGTH_REMAINDER


def overlap_frames_to_audio_steps(overlap_frames: int) -> int:
    """Map an overlap length to audio latent steps through the frame anchor.

    The native contract is exact in frames: 39 output frames = 1.625 s = 65
    audio latent steps at 40 steps/s.  Scaling the 12-token anchor linearly
    instead would use ``12 * 65 / 12`` for the anchor but drift afterwards,
    because 12 video latent tokens already over-cover 39 frames
    (12 * 17/5 = 40.8).  At a 141-frame overlap the two rules differ by 7
    audio steps (0.175 s), which is the whole lip-sync budget, so the frame
    anchor is the one that is kept.
    """
    return round(overlap_frames / VIDEO_FPS * AUDIO_LATENT_RATE)


def resolve_overlap_video_steps(
    metadata: Mapping[str, Any] | None, fallback: int,
) -> int:
    """Per-sample overlap in video latent steps, defaulting to a run-level value.

    Shot-level training gives every sample the overlap it actually needs, so
    the run-level ``overlap_video_steps`` is only a fallback.  The per-sample
    value is derived from the cache's ``overlap_frames`` (already validated at
    load time) rather than stored separately, so existing caches stay valid.
    """
    if not isinstance(metadata, Mapping):
        return fallback
    raw = metadata.get("overlap_video_steps")
    if isinstance(raw, int) and raw > 0:
        return raw
    frames = metadata.get("overlap_frames")
    if not isinstance(frames, int) or frames <= 0:
        return fallback
    if not is_h3_aligned_frames(frames):
        return fallback
    return video_frames_to_steps(frames)


@dataclass(frozen=True)
class ShotRange:
    shot_index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class ContinuationSample:
    sample_id: str
    sequence_id: str
    video_path: str
    audio_path: str | None
    shot_index: int
    shot_start_sec: float
    shot_end_sec: float
    start_sec: float
    end_sec: float
    window_frames: int
    overlap_frames: int
    hard_core_frames: int
    transition_frames: int
    prompt: str
    prompt_source: str
    prompt_cleaning_version: str
    audio_missing: bool
    split: str
    status: str = "accepted"
    conditioning_mode: str = "legacy"
    # Cross-fragment windows: ordered source ranges the window is cut from, each
    # with ``video_path`` plus ``start_sec``/``end_sec``.  Empty for the classic
    # single-file windows, which only need ``video_path``/``start_sec``/``end_sec``.
    segments: tuple[Mapping[str, Any], ...] = ()
    # Audio for fragment sources lives in a sibling wav rather than a separate
    # ``audio_path``; resolve it per segment when true.
    audio_from_video: bool = False
    # "masked" keeps the trained masked-av-v14 prefix contract; "none" trains the
    # same window with no history at all (hard-cut / new-shot windows).
    prefix_mode: str = "masked"
    # Reuse path: an already-encoded ``.pt`` from an earlier cache (plus its
    # metadata) that the new cache links instead of re-encoding the window.
    reused_cache_path: str | None = None
    cache_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationPair:
    """Directional continuation training record: A tail conditions later B."""

    pair_id: str
    sequence_id: str
    split: str
    history: ContinuationSample
    target: ContinuationSample
    schema_version: str = PAIR_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "sequence_id": self.sequence_id,
            "split": self.split,
            "history": self.history.to_dict(),
            "target": self.target.to_dict(),
        }


@dataclass(frozen=True)
class TimelineResolution:
    video_frames: int
    video_fps: int
    audio_samples: int
    audio_latent_steps: int
    duration_sec: float
    overlap_audio_samples: int
    overlap_audio_latent_steps: int


@dataclass(frozen=True)
class MediaWindow:
    """Integer media coordinates derived from one physical time interval."""

    start_sec: float
    end_sec: float
    video_start_frame: int
    video_end_frame: int
    audio_start_sample: int
    audio_end_sample: int
    audio_start_latent_step: int
    audio_end_latent_step: int

    @property
    def video_frames(self) -> int:
        return self.video_end_frame - self.video_start_frame

    @property
    def audio_samples(self) -> int:
        return self.audio_end_sample - self.audio_start_sample

    @property
    def audio_latent_steps(self) -> int:
        return self.audio_end_latent_step - self.audio_start_latent_step


def resolve_media_window(start_sec: float, end_sec: float, *, video_fps: int = VIDEO_FPS,
                         audio_sample_rate: int = AUDIO_SAMPLE_RATE,
                         audio_latent_rate: int = AUDIO_LATENT_RATE) -> MediaWindow:
    """Resolve all media boundaries from the same physical interval.

    Rounding is performed on absolute coordinates, rather than independently on
    duration and offset, so adjacent windows cannot accumulate one-sample drift.
    """
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("media window must satisfy 0 <= start_sec < end_sec")
    video_start = round(start_sec * video_fps)
    video_end = round(end_sec * video_fps)
    audio_start = round(start_sec * audio_sample_rate)
    audio_end = round(end_sec * audio_sample_rate)
    latent_start = round(start_sec * audio_latent_rate)
    latent_end = round(end_sec * audio_latent_rate)
    if video_end <= video_start or audio_end <= audio_start:
        raise ValueError("media window is shorter than one output tick")
    return MediaWindow(start_sec, end_sec, video_start, video_end, audio_start,
                       audio_end, latent_start, latent_end)


@dataclass(frozen=True)
class LatentCacheMetadata:
    schema_version: str
    sample_id: str
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...] | None
    video_dtype: str
    audio_dtype: str | None
    video_fps: int
    audio_sample_rate: int
    audio_latent_rate: int
    window_frames: int
    overlap_frames: int
    source_video: str
    source_audio: str | None
    source_sha256: str | None = None
    audio_missing: bool = False
    start_sec: float = 0.0
    end_sec: float = 0.0
    video_start_frame: int = 0
    video_end_frame: int = 0
    audio_start_sample: int = 0
    audio_end_sample: int = 0
    audio_start_latent_step: int = 0
    audio_end_latent_step: int = 0
    video_height: int | None = None
    video_width: int | None = None
    video_resize_mode: str | None = None
    # Overlap expressed on the model's latent axis.  0 means "not recorded"
    # (legacy cache); readers then re-derive it from ``overlap_frames``.
    overlap_video_steps: int = 0
    # Long-horizon memory block (M1-fast): a short clean video-latent block cut
    # from the frames immediately preceding the window, i.e. the tail of the
    # shot the window's own context no longer reaches.  0 frames = no memory.
    memory_frames: int = 0
    memory_video_steps: int = 0
    memory_start_frame: int = 0
    # Full multi-slot memory layout (STM + anchored LTM).  The three ``memory_*``
    # fields above keep describing the first slot, so every legacy reader still
    # sees a consistent single-slot cache.
    memory_slots: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationRegionConfig:
    """Latent-axis region contract shared by collator, loss, and evaluation."""

    # mask-v14 uses the 39-frame H3 overlap (12 video latent tokens) as a
    # fully protected prefix.  The old taper settings remain selectable via
    # explicit CLI values for backwards-compatible experiments.
    overlap_video_steps: int = 12
    hard_core_video_steps: int = 12
    transition_video_steps: int = 0
    first_suffix_clip_steps: int = 5
    transition_weight: float = 0.5
    first_suffix_weight: float = 3.0
    suffix_weight: float = 1.0
    lambda_audio: float = 0.5
    conditioning_mode: str = "masked-av-v14"
    prefix_present_mode: str = "noised"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, video_steps: int) -> None:
        if self.overlap_video_steps != self.hard_core_video_steps + self.transition_video_steps:
            raise ValueError("overlap must equal hard-core plus transition steps")
        if self.overlap_video_steps <= 0 or self.overlap_video_steps >= video_steps:
            raise ValueError("overlap_video_steps must be positive and smaller than video_steps")
        if self.hard_core_video_steps <= 0 or (self.transition_video_steps < 0):
            raise ValueError("hard-core must be positive and transition non-negative")
        if self.first_suffix_clip_steps <= 0 or self.lambda_audio < 0:
            raise ValueError("suffix clip steps must be positive and lambda_audio non-negative")
        if self.conditioning_mode != "masked-av-v14":
            raise ValueError("conditioning_mode must be masked-av-v14")
        if self.conditioning_mode == "masked-av-v14" and (
            self.hard_core_video_steps != self.overlap_video_steps or self.transition_video_steps != 0
        ):
            raise ValueError("masked-av-v14 requires hard_core=overlap and transition=0")
        if self.prefix_present_mode not in ("noised", "clean", "mixed"):
            raise ValueError("prefix_present_mode must be one of noised/clean/mixed")
        if self.prefix_present_mode != "noised" and self.conditioning_mode != "masked-av-v14":
            raise ValueError("clean/mixed prefix presentation requires conditioning_mode=masked-av-v14")


# Resume-time identity of a continuation run.  Only the *form* of the objective
# is frozen: the same weights must have been trained against the same
# conditioning contract.  Lengths and per-term weights are deliberately absent
# -- shot-level training makes the overlap per-sample (the run-level value is
# only a fallback), and curriculum annealing changes the weights on purpose, so
# neither may invalidate a resume.
CONTINUATION_REGION_FINGERPRINT_FIELDS = (
    "conditioning_mode",
    "prefix_present_mode",
)


def continuation_region_fingerprint(
    config: "ContinuationRegionConfig | None",
) -> dict[str, Any] | None:
    """Stable, resume-relevant subset of a region config."""
    if config is None:
        return None
    fields = config.to_dict()
    return {key: fields.get(key) for key in CONTINUATION_REGION_FINGERPRINT_FIELDS}


def collate_continuation_latents(
    video_latents: torch.Tensor,
    *,
    audio_latents: torch.Tensor | None = None,
    history_video_latents: torch.Tensor | None = None,
    history_audio_latents: torch.Tensor | None = None,
    config: ContinuationRegionConfig | None = None,
    video_time_dim: int = -3,
    audio_time_dim: int = -1,
) -> dict[str, torch.Tensor | None]:
    """Create region masks and clean history views from one teacher-forced window."""
    config = config or ContinuationRegionConfig()
    video_time_dim = video_time_dim % video_latents.ndim
    config.validate(video_latents.shape[video_time_dim])

    video_mask = region_weights(
        video_latents.shape[video_time_dim],
        config.hard_core_video_steps,
        config.transition_video_steps,
        transition_weight=config.transition_weight,
        first_suffix_clip=config.first_suffix_clip_steps,
        first_suffix_weight=config.first_suffix_weight,
        suffix_weight=config.suffix_weight,
        device=video_latents.device,
        dtype=video_latents.dtype,
    )
    expected_history_steps = config.overlap_video_steps
    video_history = (
        video_latents.narrow(video_time_dim, 0, expected_history_steps).clone()
        if history_video_latents is None else history_video_latents
    )
    if (
        video_history.ndim != video_latents.ndim
        or video_history.shape[:video_time_dim] != video_latents.shape[:video_time_dim]
        or video_history.shape[video_time_dim + 1:] != video_latents.shape[video_time_dim + 1:]
        or video_history.shape[video_time_dim] != expected_history_steps
    ):
        raise ValueError("video history must match target geometry and the selected continuation mode")

    result: dict[str, torch.Tensor | None] = {
        "video_latents": video_latents,
        "video_history_clean": video_history,
        "video_loss_mask": video_mask,
        "audio_latents": audio_latents,
        "audio_loss_mask": None,
    }
    if audio_latents is not None:
        audio_time_dim = audio_time_dim % audio_latents.ndim
        # H3 video VAE exposes five latent steps for each 17 frame clip.
        # The mask-v14 contract is exact in frames (39 output frames = 65
        # audio steps); anchor on frames, exactly like the loss and the
        # masked-av-v14 inference pipeline.
        audio_overlap = overlap_frames_to_audio_steps(
            video_steps_to_frames(config.overlap_video_steps)
        )
        audio_hard = overlap_frames_to_audio_steps(
            video_steps_to_frames(config.hard_core_video_steps)
        )
        audio_transition = max(0, audio_overlap - audio_hard)
        audio_first_suffix = round(config.first_suffix_clip_steps / 5 * VIDEO_CLIP_FRAMES / VIDEO_FPS * AUDIO_LATENT_RATE)
        audio_mask = region_weights(
            audio_latents.shape[audio_time_dim], audio_hard, audio_transition,
            transition_weight=config.transition_weight,
            first_suffix_clip=max(1, audio_first_suffix),
            first_suffix_weight=config.first_suffix_weight,
            suffix_weight=config.suffix_weight,
            device=audio_latents.device,
            dtype=audio_latents.dtype,
        )
        result["audio_loss_mask"] = audio_mask
        result["audio_overlap_steps"] = torch.tensor(audio_overlap, device=audio_latents.device)
        expected_audio_history = audio_overlap
        audio_history = (
            audio_latents.narrow(audio_time_dim, 0, expected_audio_history).clone()
            if history_audio_latents is None else history_audio_latents
        )
        if (
            audio_history.ndim != audio_latents.ndim
            or audio_history.shape[:audio_time_dim] != audio_latents.shape[:audio_time_dim]
            or audio_history.shape[audio_time_dim + 1:] != audio_latents.shape[audio_time_dim + 1:]
            or audio_history.shape[audio_time_dim] != expected_audio_history
        ):
            raise ValueError("audio history must match target geometry and the selected continuation mode")
        result["audio_history_clean"] = audio_history
    return result


def make_latent_cache_metadata(
    sample: ContinuationSample,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor | None,
    *,
    source_sha256: str | None = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
    video_height: int | None = None,
    video_width: int | None = None,
    video_resize_mode: str | None = None,
    memory_latents: torch.Tensor | None = None,
    memory_start_frame: int = 0,
    memory_slots: Sequence[Mapping[str, Any]] = (),
) -> LatentCacheMetadata:
    """Build JSON-safe metadata for a cached H3 VAE result."""
    resolve_timeline(sample.window_frames, sample.overlap_frames)
    slots = [dict(slot) for slot in memory_slots]
    if not slots and memory_latents is not None:
        slots = [{
            "name": "stm",
            "kind": "latent",
            "frames": int(memory_latents.shape[-3] - 2) // 5 * 17 + VIDEO_LENGTH_REMAINDER,
            "video_steps": int(memory_latents.shape[-3]),
            "start_frame": int(memory_start_frame),
            "lead_steps": None,
        }]
    media = resolve_media_window(sample.start_sec, sample.end_sec)
    return LatentCacheMetadata(
        schema_version=schema_version,
        sample_id=sample.sample_id,
        video_shape=tuple(video_latents.shape),
        audio_shape=None if audio_latents is None else tuple(audio_latents.shape),
        video_dtype=str(video_latents.dtype),
        audio_dtype=None if audio_latents is None else str(audio_latents.dtype),
        video_fps=VIDEO_FPS,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_latent_rate=AUDIO_LATENT_RATE,
        window_frames=sample.window_frames,
        overlap_frames=sample.overlap_frames,
        source_video=sample.video_path,
        source_audio=sample.audio_path,
        source_sha256=source_sha256,
        audio_missing=sample.audio_missing,
        start_sec=sample.start_sec,
        end_sec=sample.end_sec,
        video_start_frame=media.video_start_frame,
        video_end_frame=media.video_end_frame,
        audio_start_sample=media.audio_start_sample,
        audio_end_sample=media.audio_end_sample,
        audio_start_latent_step=media.audio_start_latent_step,
        audio_end_latent_step=media.audio_end_latent_step,
        video_height=video_height,
        video_width=video_width,
        video_resize_mode=video_resize_mode,
        memory_frames=0 if not slots else int(slots[0]["frames"]),
        memory_video_steps=0 if not slots else int(slots[0]["video_steps"]),
        memory_start_frame=0 if not slots else int(slots[0]["start_frame"]),
        memory_slots=tuple(slots),
        overlap_video_steps=(
            video_frames_to_steps(sample.overlap_frames)
            if is_h3_aligned_frames(sample.overlap_frames)
            else 0
        ),
    )


def validate_latent_cache_metadata(
    metadata: Mapping[str, Any], *, expected_sample_id: str | None = None,
    expected_window_frames: int | None = None, expected_overlap_frames: int | None = None,
    expected_video_height: int | None = None, expected_video_width: int | None = None,
) -> None:
    required = ("schema_version", "sample_id", "video_shape", "video_dtype", "video_fps", "audio_sample_rate", "window_frames", "overlap_frames")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"latent cache metadata missing fields: {', '.join(missing)}")
    if metadata["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported latent cache schema {metadata['schema_version']!r}; expected {CACHE_SCHEMA_VERSION!r}")
    if expected_sample_id is not None and metadata["sample_id"] != expected_sample_id:
        raise ValueError(f"latent cache sample_id mismatch: {metadata['sample_id']!r} != {expected_sample_id!r}")
    if expected_window_frames is not None and int(metadata["window_frames"]) != expected_window_frames:
        raise ValueError("latent cache window_frames mismatch; rebuild cache")
    if expected_overlap_frames is not None and int(metadata["overlap_frames"]) != expected_overlap_frames:
        raise ValueError("latent cache overlap_frames mismatch; rebuild cache")
    if expected_video_height is not None and metadata.get("video_height") not in (None, expected_video_height):
        raise ValueError("latent cache video_height mismatch; rebuild cache")
    if expected_video_width is not None and metadata.get("video_width") not in (None, expected_video_width):
        raise ValueError("latent cache video_width mismatch; rebuild cache")
    if int(metadata["video_fps"]) != VIDEO_FPS or int(metadata["audio_sample_rate"]) != AUDIO_SAMPLE_RATE:
        raise ValueError("latent cache media rates mismatch; expected 24 fps and 32000 Hz")
    if "audio_latent_rate" in metadata and int(metadata["audio_latent_rate"]) != AUDIO_LATENT_RATE:
        raise ValueError("latent cache audio latent rate mismatch; expected 40 steps/s")
    validate_continuation_layout(int(metadata["window_frames"]), int(metadata["overlap_frames"]), VIDEO_CLIP_FRAMES)


def save_latent_cache(path: str | Path, video_latents: torch.Tensor, audio_latents: torch.Tensor | None, metadata: LatentCacheMetadata,
                      memory_latents: torch.Tensor | None = None,
                      memory_slots: Sequence[Mapping[str, Any]] | None = None) -> None:
    """Atomically save CPU tensors and metadata; no GPU tensor is embedded in JSON.

    ``memory_slots`` is the optional list of long-horizon memory slots (each a
    dict with at least ``tensor``).  ``memory_latents`` remains the single-slot
    shorthand.  A single slot is *also* mirrored under the legacy
    ``memory_latents`` key so old readers keep working; a multi-slot cache
    deliberately does not, so a legacy reader fails loudly instead of silently
    training on one slot out of two.
    """
    slots = [dict(slot) for slot in (memory_slots or [])]
    if memory_latents is not None:
        if slots:
            raise ValueError("pass either memory_latents or memory_slots, not both")
        slots = [{"name": "stm", "tensor": memory_latents, "lead_steps": None}]
    tensors = [slot["tensor"].detach().cpu() for slot in slots]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_latents": video_latents.detach().cpu(), "audio_latents": None if audio_latents is None else audio_latents.detach().cpu(),
               "memory_latents": tensors[0] if len(tensors) == 1 else None,
               "memory_slots": tensors or None,
               "metadata": metadata.to_dict()}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_latent_cache_full(
    path: str | Path, *, expected_sample_id: str | None = None,
    expected_window_frames: int | None = None, expected_overlap_frames: int | None = None,
    expect_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]] | None, dict[str, Any]]:
    """Load a cache entry including its optional long-horizon memory slots.

    The third element is an ordered slot list (``tensor`` + ``name`` +
    ``lead_steps``), or ``None`` when the entry has no memory block.  A legacy
    single-block cache loads as a one-slot list, so the training path has exactly
    one representation to handle.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "metadata" not in payload or "video_latents" not in payload:
        raise ValueError(f"invalid latent cache payload: {path}")
    metadata = payload["metadata"]
    validate_latent_cache_metadata(metadata, expected_sample_id=expected_sample_id, expected_window_frames=expected_window_frames, expected_overlap_frames=expected_overlap_frames)
    video = payload["video_latents"]
    audio = payload.get("audio_latents")
    if not isinstance(video, torch.Tensor) or (audio is not None and not isinstance(audio, torch.Tensor)):
        raise ValueError(f"latent cache tensors are invalid: {path}")
    if tuple(video.shape) != tuple(metadata["video_shape"]):
        raise ValueError("video latent shape differs from cache metadata")
    if str(video.dtype) != str(metadata["video_dtype"]):
        raise ValueError("video latent dtype differs from cache metadata")
    if audio is not None and tuple(audio.shape) != tuple(metadata.get("audio_shape", ())):
        raise ValueError("audio latent shape differs from cache metadata")
    if audio is not None and str(audio.dtype) != str(metadata.get("audio_dtype")):
        raise ValueError("audio latent dtype differs from cache metadata")
    descriptions = list(metadata.get("memory_slots") or ())
    tensors = payload.get("memory_slots")
    if tensors is not None and not isinstance(tensors, (list, tuple)):
        raise ValueError(f"latent cache memory slot list is invalid: {path}")
    if tensors is None:
        legacy = payload.get("memory_latents")
        if legacy is not None and not isinstance(legacy, torch.Tensor):
            raise ValueError(f"latent cache memory tensor is invalid: {path}")
        tensors = None if legacy is None else [legacy]
    slots = []
    for index, tensor in enumerate(tensors or ()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"latent cache memory slot {index} is not a tensor: {path}")
        description = dict(descriptions[index]) if index < len(descriptions) else {}
        description.setdefault("name", f"mem{index}")
        description.setdefault("lead_steps", None)
        expected_steps = int(description.get("video_steps") or 0)
        if expected_steps and int(tensor.shape[-3]) != expected_steps:
            raise ValueError(f"memory slot {index} latent shape differs from cache metadata")
        slots.append({"name": str(description["name"]), "tensor": tensor, "lead_steps": description["lead_steps"]})
    if expect_memory and not slots:
        raise ValueError(f"latent cache has no memory block; rebuild it with --memory-frames: {path}")
    return video, audio, (slots or None), dict(metadata)


def load_latent_cache(path: str | Path, *, expected_sample_id: str | None = None, expected_window_frames: int | None = None, expected_overlap_frames: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    video, audio, _memory, metadata = load_latent_cache_full(
        path, expected_sample_id=expected_sample_id,
        expected_window_frames=expected_window_frames,
        expected_overlap_frames=expected_overlap_frames,
    )
    return video, audio, metadata


#: Default anchor distance, in video latent steps, for a memory slot whose
#: content comes from the head of the sequence.  Such a slot is arbitrarily old
#: in wall-clock terms, so pinning it to the true (unbounded) distance would walk
#: it straight out of the trained position range; the constant anchor keeps it
#: where the model can see it, and how stale it actually is travels through the
#: recency embedding instead.
DEFAULT_LTM_LEAD_STEPS = 36


@dataclass(frozen=True)
class MemorySlotSpec:
    """One planned memory slot: where its frames come from and where it sits.

    ``anchor`` picks the source frames (``window-start`` = the frames directly
    before the window, ``sequence-head`` = the opening frames of the clip) and
    ``lead_steps`` picks the position: ``None`` puts the slot flush against the
    window, an integer pins it to that constant distance in video latent steps.
    """

    name: str
    frames: int
    anchor: str = "window-start"
    lead_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def memory_slot_specs(memory_frames: int | None, memory_anchor: str = "window-start") -> list[MemorySlotSpec]:
    """The single-slot CLI shorthand -> one slot plan."""
    if not memory_frames or int(memory_frames) <= 0:
        return []
    frames = int(memory_frames)
    if resolve_h3_video_frames(frames) != frames:
        raise ValueError(f"memory_frames must satisfy 17n+5, got {frames}")
    if memory_anchor not in ("window-start", "sequence-head"):
        raise ValueError(f"unsupported memory_anchor={memory_anchor!r}")
    if memory_anchor == "sequence-head":
        return [MemorySlotSpec(name="ltm", frames=frames, anchor="sequence-head", lead_steps=DEFAULT_LTM_LEAD_STEPS)]
    return [MemorySlotSpec(name="stm", frames=frames, anchor="window-start", lead_steps=None)]


def dual_memory_slot_specs(
    stm_frames: int | None, ltm_frames: int | None, *,
    ltm_lead_steps: int = DEFAULT_LTM_LEAD_STEPS,
) -> list[MemorySlotSpec]:
    """The memory-only layout: a contiguous STM slot plus an anchored LTM slot.

    The two slots answer different questions -- "what did the previous shot look
    like" (recency, contiguous) and "which subject/scene is this" (identity,
    anchored) -- so they are kept as separate blocks with separate positions
    instead of being concatenated into one indistinct memory tensor.
    """
    specs = []
    if stm_frames and int(stm_frames) > 0:
        specs.extend(memory_slot_specs(int(stm_frames), "window-start"))
    if ltm_frames and int(ltm_frames) > 0:
        specs.extend(memory_slot_specs(int(ltm_frames), "sequence-head"))
        specs[-1] = MemorySlotSpec(name="ltm", frames=specs[-1].frames, anchor="sequence-head", lead_steps=int(ltm_lead_steps))
    return specs


#: Every slot name the cache builder can emit.  A cache carrying anything else
#: was built by a different code version, and training on it would silently
#: change the conditioning contract.
KNOWN_MEMORY_SLOT_NAMES = ("stm", "ltm")


def check_memory_only_sample(
    prefix_mode: str, memory_slots: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """Guard the memory-only contract on one sample; return the cache's slot names.

    Memory-only training is only meaningful if the memory slots are the *only*
    cross-shot conditioning, so three things are checked here:

    * the sample is prefix-free (a prefix would keep the model looking
      continuous while the memory does nothing, and the loss would not show it);
    * every slot present is a name this version understands.

    A sample with no slots at all is legal -- see ``audit_memory_slots`` for the
    run-level check that the cache was built with slots in the first place.

    What each *arm* uses is a separate question, handled by
    :func:`select_memory_slots`; absence of a specific slot is legal because the
    first shot of a sequence has nothing in front of it.
    """
    if prefix_mode != "none":
        raise ValueError(
            f"memory-only needs prefix-free samples, got prefix_mode={prefix_mode!r}; "
            "rebuild the index with --memory-only"
        )
    if not memory_slots:
        # Legal: the first shot of a sequence has nothing to remember, so both
        # slots are dropped and the sample is a plain text-to-shot objective.
        # ``audit_memory_slots`` catches the case that matters -- a whole cache
        # built without slots, which would make the run silently memory-free.
        return []
    names = [str(slot.get("name")) for slot in memory_slots]
    unknown = [name for name in names if name not in KNOWN_MEMORY_SLOT_NAMES]
    if unknown:
        raise ValueError(
            f"cache provides unknown memory slots {unknown}; this version understands "
            f"{list(KNOWN_MEMORY_SLOT_NAMES)} -- the cache was built by a different code version"
        )
    return names


def audit_memory_slots(manifest: str | Path) -> dict[str, Any]:
    """Count how the memory slots are distributed over a cache manifest.

    A memory-only run must not be silently memory-free: the per-sample contract
    legitimately allows a sample to carry no slot (the first shot of a sequence
    has nothing to remember), so "the cache has slots at all" has to be checked
    once over the whole manifest instead of per sample.
    """
    manifest = Path(manifest)
    if manifest.is_dir():
        manifest = manifest / "train" / "manifest.jsonl"
    records = 0
    with_slots = 0
    slot_counts: dict[str, int] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records += 1
        metadata = record.get("metadata")
        slots = (metadata or {}).get("memory_slots") or ()
        if not slots:
            continue
        with_slots += 1
        # A sample may carry a subset of the layout (the second shot has no
        # sequence-head problem, the first has no preceding shot, ...), so count
        # each slot name separately rather than assuming a fixed arity.
        key = "+".join(str(slot.get("name")) for slot in slots)
        slot_counts[key] = slot_counts.get(key, 0) + 1
    return {
        "manifest": str(manifest),
        "records": records,
        "records_with_slots": with_slots,
        "slot_combinations": dict(sorted(slot_counts.items())),
    }


def select_memory_slots(
    memory_slots: Sequence[Mapping[str, Any]] | None, arm_names: Sequence[str] = (),
) -> list[Mapping[str, Any]]:
    """Keep only the slots this arm uses, in the configured order.

    The slots are independent conditioning blocks, so an ablation arm is just a
    different subset of them.  Filtering here -- rather than rebuilding the cache
    per arm -- means one GPU-encoded cache serves every arm, and the arms differ
    only in what the model is allowed to see.  An empty ``arm_names`` keeps
    everything; a slot the cache does not have is simply absent from the result.

    A slot-free sample is legal -- the first shot of a sequence has nothing to
    remember -- and ``load_latent_cache_full`` stores that case as ``None``
    rather than ``[]``.  The empty check therefore has to come before anything
    iterates the argument, and it returns ``[]`` so the caller's ``or None``
    still sees "no conditioning" instead of crashing on the iteration.
    """
    if not memory_slots:
        return []
    names = [str(name) for name in arm_names]
    if not names:
        return list(memory_slots)
    by_name = {str(slot.get("name")): slot for slot in memory_slots}
    return [by_name[name] for name in names if name in by_name]


def encode_memory_slot(
    sample: ContinuationSample, spec: MemorySlotSpec, *, video_encoder,
    frame_processor: Callable[[Any], Any] | None = None,
    video_preprocessor: Callable[[Any], torch.Tensor] | None = None,
    window: MediaWindow | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Encode one memory slot with the window's own VAE geometry.

    Returns ``(tensor, description)``.  The tensor is ``None`` when the slot has
    nothing to remember -- a ``window-start`` slot on the sequence's first shot
    has no frames before it -- in which case the caller simply drops the slot.
    """
    if resolve_h3_video_frames(spec.frames) != spec.frames:
        raise ValueError(f"memory slot {spec.name!r} frames must satisfy 17n+5, got {spec.frames}")
    window = window if window is not None else resolve_media_window(sample.start_sec, sample.end_sec)
    if spec.anchor == "sequence-head":
        start_frame = 0
    elif spec.anchor == "window-start":
        start_frame = window.video_start_frame - int(spec.frames)
    else:
        raise ValueError(f"unsupported memory anchor={spec.anchor!r}")
    description = {
        "name": spec.name,
        "kind": "latent",
        "anchor": spec.anchor,
        "frames": int(spec.frames),
        "video_steps": expected_video_latent_steps(int(spec.frames)),
        "start_frame": 0,
        "lead_steps": spec.lead_steps,
    }
    if start_frame < 0:
        # Nothing precedes this window, so the slot does not exist.  The caller
        # drops it rather than padding it with a fake or repeated block.
        return None, description
    end_frame = start_frame + int(spec.frames)
    if start_frame < window.video_end_frame and end_frame > window.video_start_frame:
        # The slot's frames are (partly) inside the window it is supposed to
        # condition.  A clean block of the target is not a memory -- the model
        # could simply copy it -- so the slot is dropped instead of clipped.
        # This is the first shot of a sequence: its sequence-head anchor is the
        # shot itself, so neither slot exists and the sample degrades to plain
        # text-to-shot, which is the honest objective for "nothing to remember".
        return None, description
    memory_window = MediaWindow(
        start_sec=start_frame / VIDEO_FPS, end_sec=end_frame / VIDEO_FPS,
        video_start_frame=start_frame, video_end_frame=end_frame,
        audio_start_sample=window.audio_start_sample, audio_end_sample=window.audio_end_sample,
        audio_start_latent_step=window.audio_start_latent_step,
        audio_end_latent_step=window.audio_end_latent_step,
    )
    if sample.segments:
        memory_raw = load_video_window_segments(sample.segments, memory_window, frame_processor=frame_processor)
    else:
        memory_raw = load_video_window(sample.video_path, memory_window, frame_processor=frame_processor)
    memory_input = video_preprocessor(memory_raw) if video_preprocessor else memory_raw
    encode_video = getattr(video_encoder, "encode_video", video_encoder)
    try:
        memory_latents = encode_video(memory_input)
    except TypeError:
        memory_latents = encode_video(memory_input, dtype=None)
    if not isinstance(memory_latents, torch.Tensor):
        raise TypeError("video encoder must return a torch.Tensor")
    if memory_latents.shape[-3] != description["video_steps"]:
        raise ValueError(
            f"memory slot {spec.name!r} VAE latent time mismatch: got {memory_latents.shape[-3]}, "
            f"expected {description['video_steps']}"
        )
    description["start_frame"] = int(start_frame)
    return memory_latents, description


def expected_video_latent_steps(video_frames: int) -> int:
    """Temporal length emitted by the H3 video VAE for a valid frame count."""
    if resolve_h3_video_frames(video_frames) != video_frames:
        raise ValueError(f"video_frames must satisfy 17n+5, got {video_frames}")
    return ((video_frames - VIDEO_LENGTH_REMAINDER) // VIDEO_CLIP_FRAMES) * 5 + 2


def expected_audio_latent_steps(video_frames: int) -> int:
    """Audio latent steps the H3 audio raster pairs with ``video_frames``.

    The audio axis advances at 40 steps/s and the video axis at 24 fps, so one
    output frame is ``40/24`` audio steps.  This frame-derived rule is the one
    the model itself uses everywhere (noise initialiser, packed audio position
    grid, frame-anchored overlap mapping), which makes it the contract every
    cached audio latent has to satisfy.
    """
    if video_frames <= 0:
        raise ValueError(f"video_frames must be positive, got {video_frames}")
    return round(video_frames / VIDEO_FPS * AUDIO_LATENT_RATE)


def conform_audio_latent_steps(
    audio_latents: torch.Tensor | None, video_frames: int, *, tolerance: int = 2,
) -> torch.Tensor | None:
    """Force a cached audio latent onto the frame-derived audio raster.

    The audio VAE emits ``ceil(samples / 800)`` steps for the sample interval it
    is handed, while the model sizes the audio axis from the *frame* count.  On
    the 51-frame lattice the two agree, but a shot-level window that starts off
    that lattice makes them differ by one step, and then the packed audio rows
    and the latents handed to the model describe different timelines.

    Frames are the anchor (they are what the video VAE and the loss both count),
    so the audio latent is trimmed -- or, defensively, zero-padded -- to the
    frame-derived length instead.  A block that is further off than ``tolerance``
    steps is a real encoding error and is refused rather than silently reshaped.
    """
    if audio_latents is None:
        return None
    target = expected_audio_latent_steps(video_frames)
    current = int(audio_latents.shape[-1])
    if current == target:
        return audio_latents
    if abs(current - target) > tolerance:
        raise ValueError(
            f"audio latent has {current} steps but window_frames={video_frames} implies {target}; "
            "refusing to conform an audio block this far off the frame-derived raster"
        )
    if current > target:
        return audio_latents[..., :target].contiguous()
    pad = torch.zeros(
        (*audio_latents.shape[:-1], target - current),
        dtype=audio_latents.dtype, device=audio_latents.device,
    )
    return torch.cat([audio_latents, pad], dim=-1)


def load_video_window(path: str | Path, window: MediaWindow, *, output_fps: int = VIDEO_FPS,
                      frame_processor: Callable[[Any], Any] | None = None) -> list[Any]:
    """Read exact output-frame coordinates from a source video.

    ``imageio`` is imported lazily so indexing and CPU-only tests do not require
    a video codec. Source frames are selected by timestamp, which also handles
    videos whose native frame rate differs from 24 fps.
    """
    import imageio
    from PIL import Image

    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data(index=...)
        source_fps = float(meta.get("fps") or output_fps)
        # ``count_frames()`` forces a full decode pass for many ffmpeg-backed
        # readers.  Shot ranges already guarantee that the requested window is
        # inside the source clip, so use finite metadata when available and
        # otherwise let get_data() decode directly.
        reported_total = meta.get("nframes")
        total = int(reported_total) if reported_total not in (None, float("inf")) and math.isfinite(float(reported_total)) else None
        frames = []
        for frame_id in range(window.video_start_frame, window.video_end_frame):
            source_id = max(0, round(frame_id / output_fps * source_fps))
            if total is not None:
                source_id = min(source_id, int(total) - 1)
            frame = Image.fromarray(reader.get_data(source_id)).convert("RGB")
            frames.append(frame_processor(frame) if frame_processor else frame)
        return frames
    finally:
        reader.close()


def load_audio_window(path: str | Path | None, window: MediaWindow, *, target_rate: int = AUDIO_SAMPLE_RATE,
                      channels: int = 2) -> tuple[torch.Tensor, int, bool]:
    """Load/resample an audio interval, padding missing tracks with explicit silence."""
    if not path:
        return torch.zeros(channels, window.audio_samples), target_rate, True
    try:
        waveform, sample_rate = _read_audio_file(path)
    except Exception as exc:
        raise RuntimeError(f"failed to read audio {path}: {exc}") from exc
    waveform = _resample_to_stereo(waveform, int(sample_rate), target_rate)
    start, end = window.audio_start_sample, window.audio_end_sample
    if start >= waveform.shape[-1]:
        clip = torch.zeros(2, end - start, dtype=waveform.dtype)
    else:
        clip = waveform[:, start:min(end, waveform.shape[-1])]
        if clip.shape[-1] < end - start:
            clip = torch.nn.functional.pad(clip, (0, end - start - clip.shape[-1]))
    return clip[:channels], target_rate, False


def _read_audio_file(path: str | Path) -> tuple[torch.Tensor, int]:
    """Decode one audio file to a float waveform, tolerating missing backends."""
    try:
        from ..utils.data.audio import read_audio
        waveform, sample_rate = read_audio(str(path), backend="torchcodec")
    except Exception:
        try:
            # ``torchaudio.load`` delegates to torchcodec in recent builds and
            # may require an unavailable CUDA runtime.  soundfile handles the
            # PCM WAV files used by the annotation corpus without that runtime.
            import soundfile as sf
            import numpy as np
            samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(np.asarray(samples).T.copy())
        except Exception:
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(str(path))
            except Exception as exc:
                raise RuntimeError(f"failed to read audio {path}: {exc}") from exc
    return waveform.float(), int(sample_rate)


def _resample_to_stereo(waveform: torch.Tensor, source_rate: int, target_rate: int,
                        channels: int = 2) -> torch.Tensor:
    from ..utils.data.audio import resample_waveform, convert_to_stereo
    waveform = convert_to_stereo(waveform.float())
    waveform = resample_waveform(waveform, int(source_rate), int(target_rate))
    return waveform[:channels]


def resolve_segment_audio_path(video_path: str | Path, *, audio_dir: str | Path | None = None,
                               audio_suffix: str = "_origin.wav") -> str | None:
    """Locate the time-aligned wav that accompanies a 25 fps fragment clip.

    Fragment corpora keep audio next to the video (``<clip_dir>/../audio/raw/<stem>_origin.wav``);
    ``audio_dir`` overrides that location explicitly.
    """
    stem = Path(video_path).stem
    name = f"{stem}{audio_suffix}"
    if audio_dir is not None:
        candidates = [Path(audio_dir) / name]
    else:
        # Fragments live in ``<dataset>/video_25fps/data``; the aligned wavs are
        # in ``<dataset>/audio/raw``.  Try both plausible depths so the layout
        # does not have to be passed explicitly.
        clip_dir = Path(video_path).parent
        candidates = [clip_dir.parent / "audio" / "raw" / name,
                      clip_dir.parent.parent / "audio" / "raw" / name]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_video_window_segments(segments: Sequence[Mapping[str, Any]], window: MediaWindow, *,
                               output_fps: int = VIDEO_FPS,
                               frame_processor: Callable[[Any], Any] | None = None) -> list[Any]:
    """Read a window stitched from several fragments, one output frame at a time.

    Each output frame is mapped to the fragment that covers its timestamp, so a
    window may freely span shot boundaries and fragment-length changes.
    """
    import imageio
    from PIL import Image

    for seg in segments:
        missing = [key for key in ("video_path", "start_sec", "end_sec") if key not in seg]
        if missing:
            raise ValueError(
                "each segment needs video_path/start_sec/end_sec, missing "
                f"{', '.join(missing)} (reused records carry no media window)"
            )
    spans = sorted((float(seg["start_sec"]), float(seg["end_sec"]), str(seg["video_path"]), seg)
                   for seg in segments)
    readers: dict[str, Any] = {}
    source_fps: dict[str, float] = {}
    frame_totals: dict[str, int | None] = {}
    frames: list[Any] = []
    try:
        for frame_id in range(window.video_start_frame, window.video_end_frame):
            timestamp = frame_id / output_fps
            span = next((s for s in spans if s[0] <= timestamp < s[1]), None)
            if span is None:
                # The sampler rejects windows that straddle un-annotated source,
                # so a miss here is at most a sub-frame rounding sliver at either
                # edge.  Read the nearest segment and clamp the frame index
                # rather than failing a long cache job.
                if spans:
                    span = min(spans, key=lambda s: min(abs(timestamp - s[0]), abs(timestamp - s[1])))
                elif frames:
                    frames.append(frames[-1].copy())
                    continue
                else:
                    raise ValueError(f"no source segment covers t={timestamp:.4f}s")
            start_sec, _end_sec, path, segment = span
            if path not in readers:
                readers[path] = imageio.get_reader(path)
                meta = readers[path].get_meta_data(index=...)
                source_fps[path] = float(meta.get("fps") or output_fps)
                # Prefer the segment's own frame span (free and exact); fall back
                # to container metadata, then to decoding the count once.
                clip_start = segment.get("clip_start_frame")
                clip_end = segment.get("clip_end_frame")
                total = None
                if clip_start is not None and clip_end is not None:
                    total = int(clip_end) - int(clip_start)
                if total is None or total <= 0:
                    reported = meta.get("nframes")
                    total = (int(reported) if reported not in (None, float("inf"))
                             and math.isfinite(float(reported)) else None)
                if total is None or total <= 0:
                    try:
                        total = int(readers[path].count_frames())
                    except Exception:
                        total = None
                frame_totals[path] = total
            source_id = max(0, round((timestamp - start_sec) * source_fps[path]))
            if frame_totals[path] is not None:
                source_id = min(source_id, int(frame_totals[path]) - 1)
            frame = Image.fromarray(readers[path].get_data(source_id)).convert("RGB")
            frames.append(frame_processor(frame) if frame_processor else frame)
        return frames
    finally:
        for reader in readers.values():
            reader.close()


def load_audio_window_segments(segments: Sequence[Mapping[str, Any]], window: MediaWindow, *,
                               target_rate: int = AUDIO_SAMPLE_RATE, channels: int = 2,
                               audio_dir: str | Path | None = None,
                               audio_suffix: str = "_origin.wav") -> tuple[torch.Tensor, int, bool]:
    """Stitch per-fragment audio on the window timeline, resampled to ``target_rate``.

    Boundaries are rounded on absolute sample coordinates, matching
    :func:`resolve_media_window`, so stitching cannot accumulate drift.
    """
    total = window.audio_samples
    output = torch.zeros(channels, total, dtype=torch.float32)
    filled = torch.zeros(total, dtype=torch.bool)
    for seg in segments:
        seg_start = float(seg["start_sec"])
        seg_end = float(seg["end_sec"])
        a = max(window.start_sec, seg_start)
        b = min(window.end_sec, seg_end)
        if b <= a:
            continue
        path = seg.get("audio_path") or resolve_segment_audio_path(
            seg["video_path"], audio_dir=audio_dir, audio_suffix=audio_suffix)
        if not path or not Path(path).exists():
            continue
        waveform, sample_rate = _read_audio_file(path)
        waveform = _resample_to_stereo(waveform, sample_rate, target_rate, channels=channels)
        lo = round(a * target_rate) - window.audio_start_sample
        hi = round(b * target_rate) - window.audio_start_sample
        lo = max(0, min(lo, total))
        hi = max(0, min(hi, total))
        if hi <= lo:
            continue
        source_lo = max(0, round(a * target_rate) - round(seg_start * target_rate))
        chunk = waveform[:, source_lo:source_lo + (hi - lo)]
        if chunk.shape[-1] < hi - lo:
            chunk = torch.nn.functional.pad(chunk, (0, hi - lo - chunk.shape[-1]))
        output[:, lo:hi] = chunk[:, :hi - lo]
        filled[lo:hi] = True
    return output[:channels], target_rate, bool((~filled).any())


def preload_continuation_sample(
    sample: ContinuationSample,
    *,
    frame_processor: Callable[[Any], Any] | None = None,
    load_audio: bool = True,
    audio_dir: str | Path | None = None,
    audio_suffix: str = "_origin.wav",
) -> tuple[list[Any], torch.Tensor | None, int | None, bool]:
    """Perform the CPU-bound media work needed for one cache entry.

    This deliberately returns CPU-resident frames and audio.  Moving a 345-frame
    RGB clip through a multiprocessing queue would duplicate hundreds of MB per
    sample, so the cache builder uses a bounded background thread instead.
    """
    window = resolve_media_window(sample.start_sec, sample.end_sec)
    if sample.segments:
        frames = load_video_window_segments(sample.segments, window, frame_processor=frame_processor)
    else:
        frames = load_video_window(sample.video_path, window, frame_processor=frame_processor)
    if not load_audio:
        return frames, None, None, sample.audio_missing
    if sample.segments:
        waveform, sample_rate, audio_missing = load_audio_window_segments(
            sample.segments, window, audio_dir=audio_dir, audio_suffix=audio_suffix)
    else:
        waveform, sample_rate, audio_missing = load_audio_window(sample.audio_path, window)
    return frames, waveform, sample_rate, audio_missing


def encode_continuation_sample(
    sample: ContinuationSample,
    *,
    video_encoder: Callable[..., torch.Tensor] | Any,
    audio_encoder: Callable[..., torch.Tensor] | Any | None = None,
    frame_processor: Callable[[Any], Any] | None = None,
    video_preprocessor: Callable[[Any], torch.Tensor] | None = None,
    source_sha256: str | None = None,
    video_height: int | None = None,
    video_width: int | None = None,
    video_resize_mode: str | None = None,
    preloaded_inputs: tuple[list[Any], torch.Tensor | None, int | None, bool] | None = None,
    audio_dir: str | Path | None = None,
    audio_suffix: str = "_origin.wav",
    memory_frames: int | None = None,
    memory_anchor: str = "window-start",
    memory_specs: Sequence[MemorySlotSpec] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any]] | None, LatentCacheMetadata]:
    """Crop one sample and encode it with injected H3 VAE objects.

    Injecting encoders keeps this function usable with fake CPU encoders and
    avoids importing MiniMax-H3 weights in the dataset indexing process.
    """
    if preloaded_inputs is None:
        preloaded_inputs = preload_continuation_sample(
            sample, frame_processor=frame_processor, load_audio=audio_encoder is not None,
            audio_dir=audio_dir, audio_suffix=audio_suffix,
        )
    frames, waveform, sample_rate, audio_missing = preloaded_inputs
    video_input = video_preprocessor(frames) if video_preprocessor else frames
    encode_video = getattr(video_encoder, "encode_video", video_encoder)
    try:
        video_latents = encode_video(video_input)
    except TypeError:
        video_latents = encode_video(video_input, dtype=None)
    if not isinstance(video_latents, torch.Tensor):
        raise TypeError("video encoder must return a torch.Tensor")
    expected_steps = expected_video_latent_steps(sample.window_frames)
    if video_latents.shape[-3] != expected_steps:
        raise ValueError(f"video VAE latent time mismatch: got {video_latents.shape[-3]}, expected {expected_steps}")

    audio_latents = None
    if audio_encoder is not None:
        if waveform is None or sample_rate is None:
            raise RuntimeError("audio prefetch did not provide waveform data")
        encode_audio = getattr(audio_encoder, "encode_audio", audio_encoder)
        try:
            audio_latents = encode_audio(waveform, sample_rate)
        except TypeError:
            audio_latents = encode_audio(waveform)
        if not isinstance(audio_latents, torch.Tensor):
            raise TypeError("audio encoder must return a torch.Tensor")
        # The VAE hands back ``ceil(samples / 800)`` steps for whatever sample
        # interval it was given, which disagrees with the frame-derived audio
        # raster by one step as soon as the window starts off the 51-frame
        # lattice.  Conform here so every cache on disk is born consistent with
        # the packed audio position grid.
        audio_latents = conform_audio_latent_steps(audio_latents, sample.window_frames)
    # Long-horizon memory slots: independent ``17n+5`` clips encoded with the
    # same VAE geometry as the window, so their tokens are directly comparable
    # to the window's.  A slot carries its own source anchor (what to remember)
    # and its own positional lead (where to place it), which is what lets a
    # contiguous STM slot and a constant-anchored LTM slot coexist in one
    # sequence without either of them drifting with the rollout length.
    specs = list(memory_specs) if memory_specs is not None else memory_slot_specs(memory_frames, memory_anchor)
    window = resolve_media_window(sample.start_sec, sample.end_sec)
    memory_slots: list[dict[str, Any]] = []
    slot_descriptions: list[dict[str, Any]] = []
    for spec in specs:
        tensor, description = encode_memory_slot(
            sample, spec, video_encoder=video_encoder, frame_processor=frame_processor,
            video_preprocessor=video_preprocessor, window=window,
        )
        if tensor is None:
            continue
        memory_slots.append({"name": spec.name, "tensor": tensor, "lead_steps": spec.lead_steps})
        slot_descriptions.append(description)
    metadata = make_latent_cache_metadata(
        sample, video_latents, audio_latents, source_sha256=source_sha256,
        video_height=video_height, video_width=video_width,
        video_resize_mode=video_resize_mode,
        memory_slots=slot_descriptions,
    )
    if metadata.audio_missing != audio_missing:
        metadata = LatentCacheMetadata(**{**metadata.to_dict(), "audio_missing": audio_missing})
    return video_latents, audio_latents, (memory_slots or None), metadata


def cache_filename(sample_id: str) -> str:
    """Return a filesystem-safe ``.pt`` cache filename for a sample id.

    Short, plain ids keep their ``<sample_id>.pt`` name so on-disk caches from
    earlier runs are discovered and resumed.  Over-long ids (e.g. descriptive
    ids built from a video filename plus shot/cut markers) are reduced to a short
    deterministic SHA-1 name so the produced path never exceeds the filesystem's
    255-byte limit.
    """
    # The cache is written atomically to ``<base>.pt.tmp`` and then renamed to
    # ``<base>.pt``, so ``base`` must leave room for the 4-char ``.tmp`` suffix:
    # a name whose ``.pt`` form fits but whose ``.pt.tmp`` form does not would
    # still exceed the filesystem limit.  Check against the longer temp suffix.
    candidate = f"{sample_id}.pt.tmp"
    if len(candidate.encode("utf-8", "surrogatepass")) <= 255:
        # Resolve to the final ``.pt`` name while guaranteeing the temp variant
        # also fits: ``sample_id.pt`` is 4 bytes shorter than ``sample_id.pt.tmp``.
        assert len(f"{sample_id}.pt".encode("utf-8", "surrogatepass")) <= 255
        return f"{sample_id}.pt"
    digest = hashlib.sha1(sample_id.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{digest}.pt"


def build_latent_cache(
    samples: Iterable[ContinuationSample], output_dir: str | Path, *,
    video_encoder: Callable[..., torch.Tensor] | Any,
    audio_encoder: Callable[..., torch.Tensor] | Any | None = None,
    frame_processor: Callable[[Any], Any] | None = None,
    video_preprocessor: Callable[[Any], torch.Tensor] | None = None,
    split: str | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
    source_hash_fn: Callable[[str], str | None] | None = None,
    video_height: int | None = None,
    video_width: int | None = None,
    video_resize_mode: str | None = None,
    cpu_prefetch: int = 0,
    cpu_prefetch_workers: int = 1,
    audio_dir: str | Path | None = None,
    audio_suffix: str = "_origin.wav",
    reuse_cache: bool = True,
    memory_frames: int | None = None,
    memory_anchor: str = "window-start",
    memory_specs: Sequence[MemorySlotSpec] | None = None,
) -> dict[str, int]:
    """Stream samples into ``<split>/<cache_filename(sample_id)>.pt`` and a JSONL manifest."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "skipped": 0, "failed": 0, "reused": 0}
    manifest_handles: dict[str, Any] = {}
    if memory_frames is not None and int(memory_frames) > 0:
        memory_frames = int(memory_frames)
        if resolve_h3_video_frames(memory_frames) != memory_frames:
            raise ValueError(f"memory_frames must satisfy 17n+5, got {memory_frames}")
    if cpu_prefetch < 0:
        raise ValueError("cpu_prefetch must be non-negative")
    if cpu_prefetch_workers <= 0:
        raise ValueError("cpu_prefetch_workers must be positive")

    def eligible_samples() -> Iterator[ContinuationSample]:
        for sample in samples:
            if split is not None and sample.split != split:
                continue
            if max_samples is not None and stats["written"] >= max_samples:
                break
            split_dir = root / sample.split
            split_dir.mkdir(parents=True, exist_ok=True)
            destination = split_dir / cache_filename(sample.sample_id)
            if destination.exists() and not overwrite:
                stats["skipped"] += 1
                continue
            yield sample

    def prefetched_samples() -> Iterator[tuple[ContinuationSample, tuple[list[Any], torch.Tensor | None, int | None, bool] | None]]:
        if cpu_prefetch == 0:
            for sample in eligible_samples():
                yield sample, None
            return
        # CPU workers overlap the next media decode with the current VAE encode.
        # Keep the current item plus `cpu_prefetch` future items: a depth of 1
        # therefore retains at most two 345-frame clips and actually overlaps.
        worker_count = min(cpu_prefetch + 1, cpu_prefetch_workers)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="h3-media-prefetch") as executor:
            pending: deque[tuple[ContinuationSample, Any]] = deque()
            source = iter(eligible_samples())
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < cpu_prefetch + 1:
                    try:
                        sample = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    if sample.reused_cache_path:
                        # Already encoded by an earlier cache job: there is no
                        # media to prefetch, and the placeholder segments that
                        # point at the existing entry must not be decoded.
                        pending.append((sample, None))
                        continue
                    future = executor.submit(
                        preload_continuation_sample,
                        sample,
                        frame_processor=frame_processor,
                        load_audio=audio_encoder is not None,
                        audio_dir=audio_dir,
                        audio_suffix=audio_suffix,
                    )
                    pending.append((sample, future))
                if pending:
                    sample, future = pending.popleft()
                    yield sample, None if future is None else future.result()

    def reuse_cached_sample(sample: ContinuationSample, destination: Path) -> dict[str, Any] | None:
        """Link an already-encoded window instead of decoding it again.

        A reused record points at latents that were cut from a different physical
        window than ``video_path``/``start_sec`` describe, so silently re-encoding
        it would train on the wrong content; fail loudly instead.
        """
        if not sample.reused_cache_path:
            return None
        if not reuse_cache:
            raise RuntimeError(f"reuse disabled but {sample.sample_id} requires {sample.reused_cache_path}")
        source = Path(sample.reused_cache_path)
        if not source.exists():
            raise FileNotFoundError(f"reused cache missing for {sample.sample_id}: {source}")
        # Idempotent reuse: if an earlier run already linked the source inode into
        # this cache, the destination is the same file — treat it as complete
        # rather than re-linking (os.link would FileExistsError / copy2 SameFileError).
        if destination.exists() and os.path.samefile(source, destination):
            return dict(sample.cache_metadata or {})
        if destination.exists():
            # Stale destination from a different encoding: replace before linking.
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return dict(sample.cache_metadata or {})

    try:
        for sample, preloaded_inputs in prefetched_samples():
            # Prefetch pulls a whole window of samples ahead, so stop here rather
            # than relying on the generator: --max-samples must bound the number
            # of *written* entries, not the number of decoded ones.
            if max_samples is not None and stats["written"] >= max_samples:
                break
            split_dir = root / sample.split
            destination = split_dir / cache_filename(sample.sample_id)
            try:
                reused_metadata = reuse_cached_sample(sample, destination)
                if reused_metadata is not None:
                    payload = sample.to_dict()
                    payload.pop("cache_metadata", None)
                    manifest_path = root / sample.split / "manifest.jsonl"
                    if sample.split not in manifest_handles:
                        mode = "w" if overwrite else "a"
                        manifest_handles[sample.split] = open(manifest_path, mode, encoding="utf-8")
                    handle = manifest_handles[sample.split]
                    handle.write(json.dumps({"cache_path": destination.name, **payload,
                                             "metadata": reused_metadata}, ensure_ascii=False) + "\n")
                    handle.flush()
                    stats["reused"] += 1
                    continue
                source_hash = source_hash_fn(sample.video_path) if source_hash_fn else None
                video, audio, memory, metadata = encode_continuation_sample(
                    sample, video_encoder=video_encoder, audio_encoder=audio_encoder,
                    frame_processor=frame_processor, video_preprocessor=video_preprocessor,
                    source_sha256=source_hash,
                    video_height=video_height, video_width=video_width,
                    video_resize_mode=video_resize_mode,
                    preloaded_inputs=preloaded_inputs,
                    audio_dir=audio_dir, audio_suffix=audio_suffix,
                    memory_frames=memory_frames, memory_anchor=memory_anchor,
                    memory_specs=memory_specs,
                )
                save_latent_cache(destination, video, audio, metadata, memory_slots=memory)
                manifest_path = root / sample.split / "manifest.jsonl"
                if sample.split not in manifest_handles:
                    mode = "w" if overwrite else "a"
                    manifest_handles[sample.split] = open(manifest_path, mode, encoding="utf-8")
                handle = manifest_handles[sample.split]
                # Store paths relative to the split manifest.  This keeps a
                # cache portable and avoids resolving ``outputs/.../train``
                # twice when ContinuationLatentDataset loads the record.
                payload = sample.to_dict()
                payload.pop("cache_metadata", None)
                handle.write(json.dumps({"cache_path": destination.name, **payload,
                                         "metadata": metadata.to_dict()}, ensure_ascii=False) + "\n")
                handle.flush()
                stats["written"] += 1
            except Exception:
                stats["failed"] += 1
                raise
    finally:
        for handle in manifest_handles.values():
            handle.close()
    return stats


def build_latent_pair_cache(
    pairs: Iterable[ContinuationPair], output_dir: str | Path, *,
    video_encoder: Callable[..., torch.Tensor] | Any,
    audio_encoder: Callable[..., torch.Tensor] | Any | None = None,
    frame_processor: Callable[[Any], Any] | None = None,
    video_preprocessor: Callable[[Any], torch.Tensor] | None = None,
    split: str | None = None,
    max_pairs: int | None = None,
    overwrite: bool = False,
    source_hash_fn: Callable[[str], str | None] | None = None,
    video_height: int | None = None,
    video_width: int | None = None,
    video_resize_mode: str | None = None,
) -> dict[str, int]:
    """Cache A and B separately, then record their directional relationship.

    Saving the two VAE windows under their own sample ids permits future pairs
    to reuse a window without serializing an unbounded history tensor.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stats = {"written_pairs": 0, "written_windows": 0, "skipped_pairs": 0, "failed": 0}
    manifests: dict[str, Any] = {}
    try:
        for pair in pairs:
            if split is not None and pair.split != split:
                continue
            if max_pairs is not None and stats["written_pairs"] >= max_pairs:
                break
            if pair.history.sequence_id != pair.target.sequence_id or pair.history.split != pair.target.split:
                raise ValueError("directional pair must keep history and target in one sequence split")
            split_dir = root / pair.split
            split_dir.mkdir(parents=True, exist_ok=True)
            pair_path = split_dir / f"{pair.pair_id}.pair.json"
            if pair_path.exists() and not overwrite:
                stats["skipped_pairs"] += 1
                continue
            try:
                cached: dict[str, tuple[Path, LatentCacheMetadata]] = {}
                for role, sample in (("history", pair.history), ("target", pair.target)):
                    destination = split_dir / cache_filename(sample.sample_id)
                    if destination.exists() and not overwrite:
                        _, _, metadata_dict = load_latent_cache(
                            destination, expected_sample_id=sample.sample_id,
                            expected_window_frames=sample.window_frames,
                            expected_overlap_frames=sample.overlap_frames,
                        )
                        metadata = LatentCacheMetadata(**metadata_dict)
                    else:
                        source_hash = source_hash_fn(sample.video_path) if source_hash_fn else None
                        video, audio, memory, metadata = encode_continuation_sample(
                            sample, video_encoder=video_encoder, audio_encoder=audio_encoder,
                            frame_processor=frame_processor, video_preprocessor=video_preprocessor,
                            source_sha256=source_hash, video_height=video_height,
                            video_width=video_width, video_resize_mode=video_resize_mode,
                        )
                        save_latent_cache(destination, video, audio, metadata, memory_slots=memory)
                        stats["written_windows"] += 1
                    cached[role] = (destination, metadata)
                history_path, history_metadata = cached["history"]
                target_path, target_metadata = cached["target"]
                record = {
                    **pair.to_dict(),
                    "history_cache_path": history_path.name,
                    "target_cache_path": target_path.name,
                    "history_metadata": history_metadata.to_dict(),
                    "target_metadata": target_metadata.to_dict(),
                }
                temporary = pair_path.with_suffix(pair_path.suffix + ".tmp")
                temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                temporary.replace(pair_path)
                manifest_path = split_dir / "pair_manifest.jsonl"
                if pair.split not in manifests:
                    mode = "w" if overwrite else "a"
                    manifests[pair.split] = open(manifest_path, mode, encoding="utf-8")
                manifests[pair.split].write(json.dumps(record, ensure_ascii=False) + "\n")
                manifests[pair.split].flush()
                stats["written_pairs"] += 1
            except Exception:
                stats["failed"] += 1
                raise
    finally:
        for handle in manifests.values():
            handle.close()
    return stats


def load_continuation_pair_cache(
    record: Mapping[str, Any],
    *,
    manifest_directory: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any], dict[str, Any]]:
    """Load a validated A-tail/B-target pair from a pair manifest record."""
    if record.get("schema_version") != PAIR_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported continuation pair manifest schema")
    history_record = record.get("history")
    target_record = record.get("target")
    if not isinstance(history_record, Mapping) or not isinstance(target_record, Mapping):
        raise ValueError("continuation pair must contain history and target records")
    if (
        history_record.get("sequence_id") != target_record.get("sequence_id")
        or history_record.get("split") != target_record.get("split")
        or history_record.get("split") != record.get("split")
        or int(history_record.get("shot_index", -1)) != int(target_record.get("shot_index", -2))
    ):
        raise ValueError("continuation pair crosses a sequence, split, or shot boundary")
    window_frames = int(target_record["window_frames"])
    overlap_frames = int(target_record["overlap_frames"])
    if int(history_record["window_frames"]) != window_frames or int(history_record["overlap_frames"]) != overlap_frames:
        raise ValueError("continuation pair history and target layouts differ")
    expected_target_start = float(history_record["start_sec"]) + (window_frames - overlap_frames) / VIDEO_FPS
    if not math.isclose(float(target_record["start_sec"]), expected_target_start, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("continuation pair target does not begin at the A/B continuation stride")
    root = Path(manifest_directory)
    history_path = root / str(record["history_cache_path"])
    target_path = root / str(record["target_cache_path"])
    history_video, history_audio, history_metadata = load_latent_cache(
        history_path, expected_sample_id=str(history_record["sample_id"]),
        expected_window_frames=window_frames, expected_overlap_frames=overlap_frames,
    )
    target_video, target_audio, target_metadata = load_latent_cache(
        target_path, expected_sample_id=str(target_record["sample_id"]),
        expected_window_frames=window_frames, expected_overlap_frames=overlap_frames,
    )
    video_overlap_steps = overlap_frames // VIDEO_CLIP_FRAMES * 5
    audio_overlap_steps = round(overlap_frames / VIDEO_FPS * AUDIO_LATENT_RATE)
    if history_video.shape[:2] != target_video.shape[:2] or history_video.shape[-2:] != target_video.shape[-2:]:
        raise ValueError("continuation pair video latent geometries differ")
    if history_video.shape[-3] < video_overlap_steps or target_video.shape[-3] <= video_overlap_steps:
        raise ValueError("continuation pair video latents cannot provide the required A tail and B suffix")
    if (history_audio is None) != (target_audio is None):
        raise ValueError("continuation pair must have audio in both windows or neither")
    if history_audio is not None:
        if history_audio.shape[:-1] != target_audio.shape[:-1] or history_audio.shape[-1] < audio_overlap_steps:
            raise ValueError("continuation pair audio latents cannot provide the required A tail")
    return (
        target_video,
        history_video.narrow(-3, history_video.shape[-3] - video_overlap_steps, video_overlap_steps).clone(),
        target_audio,
        None if history_audio is None else history_audio.narrow(-1, history_audio.shape[-1] - audio_overlap_steps, audio_overlap_steps).clone(),
        history_metadata,
        target_metadata,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source media file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    """Hash a JSONL cache manifest so resumed runs cannot silently mix data."""
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"continuation manifest not found: {manifest}")
    return sha256_file(manifest)


def save_continuation_training_checkpoint(
    path: str | Path,
    *,
    model_state_dict: Mapping[str, torch.Tensor],
    optimizer_state_dict: Any | None = None,
    scheduler_state_dict: Any | None = None,
    global_step: int = 0,
    seed: int = 42,
    manifest_hash: str | None = None,
    region_config: ContinuationRegionConfig | None = None,
    rng_state: Any | None = None,
    random_state: Any | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write a full resumable continuation training checkpoint.

    LoRA-only exports intentionally remain small safetensors files written by
    ``ModelLogger``.  This checkpoint is the heavy resume artifact and includes
    optimizer/scheduler state plus the configuration invariants required to
    reject silent data/config drift.
    """
    payload = {
        "kind": TRAINING_CHECKPOINT_KIND,
        "version": 1,
        "model_state_dict": {
            str(key): value.detach().cpu() for key, value in model_state_dict.items()
        },
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "global_step": int(global_step),
        "seed": int(seed),
        "manifest_hash": manifest_hash,
        "region_config": None if region_config is None else region_config.to_dict(),
        "rng_state": rng_state,
        "random_state": random_state,
        "extra_metadata": dict(extra_metadata or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_continuation_training_checkpoint(
    path: str | Path,
    *,
    expected_manifest_hash: str | None = None,
    expected_region_config: ContinuationRegionConfig | None = None,
) -> dict[str, Any]:
    """Load and validate a full continuation checkpoint on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("kind") != TRAINING_CHECKPOINT_KIND:
        raise ValueError(
            f"{path} is not a continuation training checkpoint "
            f"(expected kind={TRAINING_CHECKPOINT_KIND!r})"
        )
    if expected_manifest_hash is not None:
        stored_hash = payload.get("manifest_hash")
        if stored_hash != expected_manifest_hash:
            raise ValueError(
                "continuation checkpoint manifest hash mismatch; "
                f"checkpoint={stored_hash!r}, current={expected_manifest_hash!r}"
            )
    if expected_region_config is not None:
        stored = payload.get("region_config") or {}
        stored_fingerprint = {
            key: stored.get(key) for key in CONTINUATION_REGION_FINGERPRINT_FIELDS
        }
        expected_fingerprint = continuation_region_fingerprint(expected_region_config)
        if stored_fingerprint != expected_fingerprint:
            raise ValueError(
                "continuation checkpoint conditioning contract mismatch; "
                f"checkpoint={stored_fingerprint}, current={expected_fingerprint}"
            )
    if "model_state_dict" not in payload or not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError("continuation checkpoint is missing model_state_dict")
    return dict(payload)


def save_lora_safetensors(
    state_dict: Mapping[str, torch.Tensor],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save LoRA tensors in the safetensors format used by inference scripts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        str(key): value.detach().float().cpu().contiguous()
        for key, value in state_dict.items()
    }
    file_metadata = {
        "format": LORA_FORMAT_VERSION,
        **{str(key): str(value) for key, value in (metadata or {}).items()},
    }
    try:
        from safetensors.torch import save_file
    except ImportError:  # pragma: no cover - fallback for old environments
        torch.save(tensors, destination)
        return
    save_file(tensors, str(destination), metadata=file_metadata)


def load_lora_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load LoRA safetensors produced by the continuation exporter."""
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except ImportError as exc:  # pragma: no cover - fallback for old environments
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"LoRA payload is not a state dict: {path}") from exc
        return payload


def apply_continuation_lora(pipe: Any, lora_path: str | Path, lora_scale: float = 1.0) -> dict[str, Any]:
    """Load a continuation LoRA with an explicit scale.

    ``lora_scale=0`` deliberately avoids touching the pipeline and therefore
    yields the exact base-model behavior.
    """
    if lora_scale < 0:
        raise ValueError(f"lora_scale must be non-negative, got {lora_scale}")
    if lora_scale == 0:
        return {"applied": False, "scale": 0.0, "lora_path": str(lora_path)}
    pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_scale)
    return {"applied": True, "scale": float(lora_scale), "lora_path": str(lora_path)}


class ContinuationLatentDataset(torch.utils.data.Dataset):
    """Small manifest-backed dataset for the continuation training stage."""

    def __init__(self, manifest: str | Path, *, split: str | None = None,
                 max_items: int | None = None):
        manifest = Path(manifest)
        if manifest.is_dir():
            if split is None:
                raise ValueError("split is required when manifest is a directory")
            manifest = manifest / split / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"continuation manifest not found: {manifest}")
        self.manifest_path = manifest.resolve()
        # The training runner uses this flag to skip media preprocessing and pass
        # cached tensors directly into the model.
        self.load_from_cache = True
        self.records: list[dict[str, Any]] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if split is not None and record.get("split", split) != split:
                    continue
                self.records.append(record)
                if max_items is not None and len(self.records) >= max_items:
                    break
        if not self.records:
            raise ValueError(f"continuation manifest contains no records: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index % len(self.records)]
        if record.get("schema_version") == PAIR_MANIFEST_SCHEMA_VERSION:
            (
                video, history_video, audio, history_audio, _, metadata,
            ) = load_continuation_pair_cache(record, manifest_directory=self.manifest_path.parent)
            if video.ndim != 5:
                raise ValueError(f"cached target video latent must be [B,C,T,H,W], got {list(video.shape)}")
            target_record = record["target"]
            window_frames = int(metadata["window_frames"])
            audio = conform_audio_latent_steps(audio, window_frames)
            history_audio = conform_audio_latent_steps(history_audio, window_frames)
            return {
                "input_latents": video,
                "continuation_history_video_latents": history_video,
                "audio_input_latents": audio,
                "continuation_history_audio_latents": history_audio,
                "prompt": target_record.get("prompt", "The same single-shot scene continues naturally."),
                "negative_prompt": " ",
                "height": int(video.shape[-2] * 16),
                "width": int(video.shape[-1] * 16),
                "num_frames": window_frames,
                "seed": int(record.get("seed", 42)),
                "cache_metadata": metadata,
                "continuation_prefix_mode": "masked",
                "continuation_pair_id": record["pair_id"],
            }
        cache_path = Path(record["cache_path"])
        if not cache_path.is_absolute():
            cache_path = self.manifest_path.parent / cache_path
        cache_metadata = record.get("metadata")
        if not isinstance(cache_metadata, Mapping):
            cache_metadata = {}
        # Reused caches may be exposed under logical aliases such as
        # ``<sample_id>:plain`` while the linked .pt retains the original
        # sample id and encoding geometry.  Validate against the metadata
        # embedded in that cache record rather than the alias fields.
        video, audio, memory, metadata = load_latent_cache_full(
            cache_path,
            expected_sample_id=cache_metadata.get("sample_id", record.get("sample_id")),
            expected_window_frames=cache_metadata.get("window_frames", record.get("window_frames")),
            expected_overlap_frames=cache_metadata.get("overlap_frames", record.get("overlap_frames")),
        )
        if video.ndim != 5:
            raise ValueError(f"cached video latent must be [B,C,T,H,W], got {list(video.shape)}")
        # ``none`` windows are the hard-cut / new-shot case: they train the same
        # cached latents but with no history rows and no zero-weighted prefix.
        prefix_mode = str(record.get("prefix_mode", "masked"))
        use_history = prefix_mode != "none"
        # Caches written by an older builder keep the VAE's ``ceil`` length;
        # conform on read so a 43 GB cache does not have to be re-encoded for
        # the audio row count to match the packed position grid.
        audio = conform_audio_latent_steps(audio, int(metadata["window_frames"]))
        # Long-horizon memory (M1-fast): a clean latent block cut from the shot
        # before the window's own context.  Absent in legacy caches, where the
        # training pipeline simply degrades to the no-memory model.
        result: dict[str, Any] = {
            "input_latents": video,
            "continuation_memory_video_latents": memory,
            "continuation_history_video_latents": video.clone() if use_history else None,
            "audio_input_latents": audio,
            "continuation_history_audio_latents": None if (audio is None or not use_history) else audio.clone(),
            "prompt": record.get("prompt", "The same single-shot scene continues naturally."),
            "negative_prompt": " ",
            "height": int(video.shape[-2] * 16),
            "width": int(video.shape[-1] * 16),
            "num_frames": int(metadata["window_frames"]),
            "seed": int(record.get("seed", 42)),
            "cache_metadata": metadata,
            "continuation_prefix_mode": prefix_mode,
        }
        return result


def resolve_h3_video_frames(requested_frames: int) -> int:
    """Round up to the H3 temporal contract ``17*n + 5``."""
    if requested_frames < VIDEO_LENGTH_REMAINDER:
        return VIDEO_LENGTH_REMAINDER
    n = (requested_frames - VIDEO_LENGTH_REMAINDER + VIDEO_CLIP_FRAMES - 1) // VIDEO_CLIP_FRAMES
    return VIDEO_CLIP_FRAMES * n + VIDEO_LENGTH_REMAINDER


def validate_continuation_layout(window_frames: int, overlap_frames: int, hard_core_frames: int) -> None:
    if resolve_h3_video_frames(window_frames) != window_frames:
        raise ValueError(f"window_frames must satisfy 17n+5, got {window_frames}")
    if overlap_frames <= 0 or overlap_frames >= window_frames:
        raise ValueError("overlap_frames must be positive and smaller than window_frames")
    # Two overlap families are legal.  ``17n`` (legacy taper-refine windows)
    # only needs to contain whole VAE clips.  ``17n+5`` is the H3-aligned
    # family: it is the one the masked-av-v14 prefix uses, and it is the only
    # family whose latent length is ``5n+2``, i.e. the same contract as the
    # window itself.  The native 39 + 51m grid is a subset of it (39, 90, 141,
    # ... = 17*2+5, 17*5+5, 17*8+5, ...), so the two checks collapse into one.
    is_h3_aligned_overlap = is_h3_aligned_frames(overlap_frames)
    if overlap_frames % VIDEO_CLIP_FRAMES and not is_h3_aligned_overlap:
        raise ValueError(
            "overlap_frames must contain complete H3 Video VAE clips, or lie on "
            "the H3-aligned 17n+5 grid (22, 39, 56, 73, ...)"
        )
    if hard_core_frames < 0 or hard_core_frames >= overlap_frames:
        raise ValueError("hard_core_frames must be inside the overlap")
    if hard_core_frames and hard_core_frames % VIDEO_CLIP_FRAMES:
        raise ValueError("hard_core_frames must be clip aligned")


def resolve_timeline(window_frames: int, overlap_frames: int = 34) -> TimelineResolution:
    validate_continuation_layout(window_frames, overlap_frames, VIDEO_CLIP_FRAMES)
    duration = window_frames / VIDEO_FPS
    return TimelineResolution(
        video_frames=window_frames,
        video_fps=VIDEO_FPS,
        audio_samples=round(duration * AUDIO_SAMPLE_RATE),
        audio_latent_steps=round(duration * AUDIO_LATENT_RATE),
        duration_sec=duration,
        overlap_audio_samples=round(overlap_frames / VIDEO_FPS * AUDIO_SAMPLE_RATE),
        overlap_audio_latent_steps=round(overlap_frames / VIDEO_FPS * AUDIO_LATENT_RATE),
    )


_SHOT_RE = re.compile(
    r"\[Shot\s+(\d+)(?:/\d+)?\s*\|\s*(?:(?:start|end)\s*\|\s*)?"
    r"([0-9.]+)s\s*-\s*([0-9.]+)s\s*\]",
    re.I,
)


def parse_shot_ranges(prompt: str) -> list[ShotRange]:
    """Parse the stable shot headers emitted by the source annotation pipeline."""
    ranges = [ShotRange(int(i), float(start), float(end)) for i, start, end in _SHOT_RE.findall(prompt or "")]
    return [r for r in ranges if r.end_sec > r.start_sec]


def extract_shot_prompt(prompt: str, shot_index: int) -> str | None:
    """Extract one annotated shot body without neighboring shots or cut markers."""
    matches = list(_SHOT_RE.finditer(prompt or ""))
    for position, match in enumerate(matches):
        if int(match.group(1)) != int(shot_index):
            continue
        body_start = match.end()
        body_end = matches[position + 1].start() if position + 1 < len(matches) else len(prompt)
        body = prompt[body_start:body_end]
        body = re.sub(r"\s*---\s*cut\s*:\s*[^\n]*---\s*", "\n", body, flags=re.I)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        return body or None
    return None


def clean_prompt(record: Mapping[str, Any], shot_index: int | None = None) -> tuple[str, str]:
    """Return a conservative prompt and its source.

    The generated ``caption`` is often corrupted/template-heavy; the annotated
    prompt is preferred and ASR text is only a fallback when it is usable.
    """
    prompt = str(record.get("prompt") or "").strip()
    if prompt and not _looks_corrupt(prompt):
        if shot_index is not None:
            shot_prompt = extract_shot_prompt(prompt, shot_index)
            if shot_prompt and not _looks_corrupt(shot_prompt):
                return shot_prompt, "prompt_shot"
        return prompt, "prompt"
    asr = record.get("asr_result")
    texts = []
    if isinstance(asr, list):
        for item in asr:
            if isinstance(item, (list, tuple)) and len(item) >= 4 and str(item[3]).strip():
                texts.append(str(item[3]).strip())
    if texts:
        return "The same single-shot scene continues naturally. Spoken dialogue: " + " ".join(texts), "asr_fallback"
    caption = str(record.get("caption") or "").strip()
    if caption and not _looks_corrupt(caption):
        return caption, "caption_fallback"
    return "The same single-shot scene continues naturally.", "generic_fallback"


def _looks_corrupt(text: str) -> bool:
    if len(text) < 12:
        return True
    replacement = text.count("\ufffd")
    repeated = len(re.findall(r"\b(\w+)\s+\1\b", text, flags=re.I))
    template_noise = sum(text.count(token) for token in ("integrated_multimodal_description", "<Picture", "\\n\\n\\n"))
    return replacement > 0 or repeated >= 4 or template_noise >= 2


def _audio_path(record: Mapping[str, Any]) -> str | None:
    value = record.get("audio_path")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("clean_wav_path", "original_wav_path", "path"):
            if value.get(key):
                return str(value[key])
    return None


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    """Yield ``(line_number, record, error)`` without retaining prior records."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                yield line_number, value, None
            except (json.JSONDecodeError, ValueError) as exc:
                yield line_number, None, str(exc)


def split_for_sequence(sequence_id: str, seed: int = 17, ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> str:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must contain three values summing to 1")
    digest = hashlib.sha256(f"{seed}:{sequence_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < ratios[0] else "validation" if value < ratios[0] + ratios[1] else "test"


def iter_continuation_samples(
    source_jsonl: str | Path,
    *,
    window_frames: int = 345,
    overlap_frames: int = 34,
    hard_core_frames: int = 17,
    split_seed: int = 17,
    window_stride_frames: int | None = None,
    require_paths: bool = False,
    conditioning_mode: str = "legacy",
    stats: dict[str, int] | None = None,
) -> Iterator[ContinuationSample]:
    """Yield lightweight sample records without retaining prior source records."""
    if conditioning_mode == "masked-av-v14":
        # Shot-level sampling gives every sample the overlap its own context
        # needs, so the prefix is no longer restricted to the native 39 + 51m
        # grid -- any 17n+5 length is latent-exact with a 17k target.
        if not is_h3_aligned_frames(overlap_frames):
            raise ValueError(
                "masked-av-v14 requires an H3-aligned overlap of 22, 39, 56, 73, ... frames"
            )
        hard_core_frames = 0
    elif conditioning_mode != "legacy":
        raise ValueError(f"unsupported conditioning_mode={conditioning_mode!r}")
    validate_continuation_layout(window_frames, overlap_frames, hard_core_frames)
    stride = window_stride_frames or (window_frames - overlap_frames)
    if stride <= 0:
        raise ValueError("window_stride_frames must be positive")
    if stats is None:
        stats = {}
    for key in ("records", "accepted", "bad_json", "missing_field", "missing_path", "short_shot", "no_shot"):
        stats.setdefault(key, 0)
    for line_number, record, error in iter_jsonl(source_jsonl):
        if error:
            stats["bad_json"] += 1
            continue
        stats["records"] += 1
        assert record is not None
        sequence_id = str(record.get("sequence_id") or "").strip()
        video_path = str(record.get("file_path") or "").strip()
        if not sequence_id or not video_path:
            stats["missing_field"] += 1
            continue
        audio_path = _audio_path(record)
        if require_paths and (not Path(video_path).exists() or (audio_path and not Path(audio_path).exists())):
            stats["missing_path"] += 1
            continue
        shots = parse_shot_ranges(str(record.get("prompt") or ""))
        if not shots:
            stats["no_shot"] += 1
            continue
        split = split_for_sequence(sequence_id, split_seed)
        window_seconds = window_frames / VIDEO_FPS
        stride_seconds = stride / VIDEO_FPS
        for shot in shots:
            prompt, prompt_source = clean_prompt(record, shot_index=shot.shot_index)
            if shot.duration_sec + 1e-6 < window_seconds:
                stats["short_shot"] += 1
                continue
            offset = 0.0
            while offset + window_seconds <= shot.duration_sec + 1e-6:
                start = shot.start_sec + offset
                end = start + window_seconds
                sample_key = f"{sequence_id}:{shot.shot_index}:{start:.6f}:{window_frames}:{overlap_frames}"
                sample = ContinuationSample(
                    sample_id=hashlib.sha1(sample_key.encode()).hexdigest()[:16],
                    sequence_id=sequence_id,
                    video_path=video_path,
                    audio_path=audio_path,
                    shot_index=shot.shot_index,
                    shot_start_sec=shot.start_sec,
                    shot_end_sec=shot.end_sec,
                    start_sec=start,
                    end_sec=end,
                    window_frames=window_frames,
                    overlap_frames=overlap_frames,
                    hard_core_frames=hard_core_frames,
                    transition_frames=0 if conditioning_mode == "masked-av-v14" else overlap_frames - hard_core_frames,
                    prompt=prompt,
                    prompt_source=prompt_source,
                    prompt_cleaning_version="v1",
                    audio_missing=audio_path is None,
                    split=split,
                    conditioning_mode=conditioning_mode,
                )
                stats["accepted"] += 1
                yield sample
                offset += stride_seconds


def build_continuation_samples(*args, **kwargs) -> tuple[list[ContinuationSample], dict[str, int]]:
    """Materialize samples for small tests only; production callers stream."""
    stats: dict[str, int] = {}
    samples = list(iter_continuation_samples(*args, stats=stats, **kwargs))
    return samples, stats


# Shot-level (stage-1) defaults.  ``MAX_CONTEXT_FRAMES`` is the cap on how far
# back the context reaches; the actual context is ``min(previous shot, cap)``
# snapped down onto the overlap grid, so short shots get a short -- but
# complete -- context instead of being padded out to a fixed length.
MAX_SHOT_CONTEXT_FRAMES = 141
MIN_SHOT_TARGET_FRAMES = 2 * VIDEO_CLIP_FRAMES
MAX_SHOT_WINDOW_FRAMES = 345


def _memory_only_shot_sample(
    record: Mapping[str, Any],
    shot: "ShotRange",
    *,
    split: str,
    max_window_frames: int,
    min_target_frames: int,
    include_prompt: bool = True,
) -> ContinuationSample | None:
    """One prefix-free sample per shot: ``p(target shot | memory, text)``.

    The memory-only contract removes the variable-length context entirely, so a
    sample is exactly one shot and the whole window is the prediction target.
    Short-range consistency is expected to come from the memory slots
    (``--memory-anchor``) instead of from a latent prefix.
    """
    start_frame = round(shot.start_sec * VIDEO_FPS)
    end_frame = round(shot.end_sec * VIDEO_FPS)
    window_frames = resolve_h3_overlap_frames(min(end_frame - start_frame, max_window_frames))
    if window_frames < min_target_frames:
        return None
    sequence_id = str(record.get("sequence_id") or "").strip()
    if include_prompt:
        prompt, prompt_source = clean_prompt(record, shot_index=shot.shot_index)
    else:
        prompt, prompt_source = "", "skipped"
    audio_path = _audio_path(record)
    sample_key = f"{sequence_id}:memonly:{shot.shot_index}:{start_frame}:{window_frames}"
    return ContinuationSample(
        sample_id=hashlib.sha1(sample_key.encode()).hexdigest()[:16],
        sequence_id=sequence_id,
        video_path=str(record.get("file_path") or "").strip(),
        audio_path=audio_path,
        shot_index=shot.shot_index,
        shot_start_sec=shot.start_sec,
        shot_end_sec=shot.end_sec,
        start_sec=start_frame / VIDEO_FPS,
        end_sec=(start_frame + window_frames) / VIDEO_FPS,
        window_frames=window_frames,
        # The layout validator still wants a legal overlap; nothing uses it,
        # because ``prefix_mode="none"`` means no prefix rows are ever built.
        overlap_frames=MIN_OVERLAP_FRAMES,
        hard_core_frames=0,
        transition_frames=0,
        prompt=prompt,
        prompt_source=prompt_source,
        prompt_cleaning_version="v1",
        audio_missing=audio_path is None,
        split=split,
        conditioning_mode="masked-av-v14",
        prefix_mode="none",
    )


def _plain_first_shot_sample(
    record: Mapping[str, Any],
    shots: Sequence[ShotRange],
    *,
    split: str,
    max_window_frames: int,
    min_target_frames: int,
) -> ContinuationSample | None:
    """The prefix-free row for one record, or None when its first shot is too short."""
    shot = shots[0]
    start_frame = round(shot.start_sec * VIDEO_FPS)
    end_frame = round(shot.end_sec * VIDEO_FPS)
    # A prefix-free window is a plain 17n+5 window: the whole thing is the
    # prediction target, so there is no 17k target / 17k+5 window split.
    window_frames = resolve_h3_overlap_frames(min(end_frame - start_frame, max_window_frames))
    if window_frames < min_target_frames:
        return None
    prompt, prompt_source = clean_prompt(record, shot_index=shot.shot_index)
    sequence_id = str(record.get("sequence_id") or "").strip()
    sample_key = f"{sequence_id}:plain:{start_frame}:{window_frames}"
    audio_path = _audio_path(record)
    return ContinuationSample(
        sample_id=hashlib.sha1(sample_key.encode()).hexdigest()[:16],
        sequence_id=sequence_id,
        video_path=str(record.get("file_path") or "").strip(),
        audio_path=audio_path,
        shot_index=shot.shot_index,
        shot_start_sec=shot.start_sec,
        shot_end_sec=shot.end_sec,
        start_sec=start_frame / VIDEO_FPS,
        end_sec=(start_frame + window_frames) / VIDEO_FPS,
        window_frames=window_frames,
        # A prefix-free window still has to pass the layout check, so it carries
        # the smallest legal overlap; nothing is ever masked with it.
        overlap_frames=MIN_OVERLAP_FRAMES,
        hard_core_frames=0,
        transition_frames=0,
        prompt=prompt,
        prompt_source=prompt_source,
        prompt_cleaning_version="v1",
        audio_missing=audio_path is None,
        split=split,
        conditioning_mode="masked-av-v14",
        prefix_mode="none",
    )


SHOT_CONTINUATION_STAT_KEYS = (
    "records", "accepted", "accepted_plain", "accepted_memory_only", "bad_json",
    "missing_field", "missing_path", "no_shot", "single_shot", "short_target",
    "short_context", "no_room", "pairs", "duplicate_pairs",
)


def _require_shot_conditioning(conditioning_mode: str) -> None:
    if conditioning_mode != "masked-av-v14":
        raise ValueError("shot-level sampling currently requires masked-av-v14")


def _init_shot_stats(stats: dict[str, int] | None) -> dict[str, int]:
    stats = {} if stats is None else stats
    for key in SHOT_CONTINUATION_STAT_KEYS:
        stats.setdefault(key, 0)
    return stats


def iter_record_shot_continuation_samples(
    record: Mapping[str, Any],
    *,
    max_context_frames: int = MAX_SHOT_CONTEXT_FRAMES,
    min_target_frames: int = MIN_SHOT_TARGET_FRAMES,
    max_window_frames: int = MAX_SHOT_WINDOW_FRAMES,
    split_seed: int = 17,
    require_paths: bool = False,
    conditioning_mode: str = "masked-av-v14",
    include_plain: bool = False,
    include_prompt: bool = True,
    stats: dict[str, int] | None = None,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> Iterator[ContinuationSample]:
    """Plan the shot-level samples of one source record.

    This is the single implementation shared by the cache builder, the supply
    statistics and the dedup audit -- callers that only need counts or a
    duplicate-free subset pass ``stats`` / ``seen_pairs`` and consume the same
    samples the trainer would see.

    ``seen_pairs`` is a caller-owned set of ``(previous_shot_caption_hash,
    target_shot_caption_hash)`` keys.  The training index is built from
    overlapping multi-shot windows of the same episodes, so the same shot pair
    shows up in several records; passing one shared set keeps the first
    occurrence and skips the rest (``stats["duplicate_pairs"]``).
    """
    _require_shot_conditioning(conditioning_mode)
    max_context_frames = resolve_h3_overlap_frames(max_context_frames)
    if max_context_frames < MIN_OVERLAP_FRAMES:
        raise ValueError("max_context_frames is below the smallest usable overlap")
    max_target_frames = (max_window_frames - MIN_OVERLAP_FRAMES) // VIDEO_CLIP_FRAMES * VIDEO_CLIP_FRAMES
    if max_target_frames < min_target_frames:
        raise ValueError("max_window_frames leaves no room for the smallest target")
    if min_target_frames % VIDEO_CLIP_FRAMES:
        raise ValueError("min_target_frames must be a multiple of 17")
    stats = _init_shot_stats(stats)
    sequence_id = str(record.get("sequence_id") or "").strip()
    video_path = str(record.get("file_path") or "").strip()
    if not sequence_id or not video_path:
        stats["missing_field"] += 1
        return
    audio_path = _audio_path(record)
    if require_paths and (not Path(video_path).exists() or (audio_path and not Path(audio_path).exists())):
        stats["missing_path"] += 1
        return
    shots = parse_shot_ranges(str(record.get("prompt") or ""))
    if not shots:
        stats["no_shot"] += 1
        return
    split = split_for_sequence(sequence_id, split_seed)
    if include_plain:
        plain = _plain_first_shot_sample(
            record, shots, split=split,
            max_window_frames=max_window_frames, min_target_frames=min_target_frames,
        )
        if plain is None:
            stats["short_target"] += 1
        else:
            stats["accepted_plain"] += 1
            yield plain
    if len(shots) < 2:
        stats["single_shot"] += 1
        return
    frames = [round(shot.start_sec * VIDEO_FPS) for shot in shots]
    frames.append(round(shots[-1].end_sec * VIDEO_FPS))
    # Captions are expensive (one regex pass over the whole record prompt per
    # shot, so quadratic on a long source), and are resolved lazily: only the
    # emitted sample or the dedup key actually needs them.
    prompts: dict[int, tuple[str, str]] = {}
    need_prompt = include_prompt or seen_pairs is not None

    def shot_prompt(index: int) -> tuple[str, str]:
        if index not in prompts:
            prompts[index] = clean_prompt(record, shot_index=shots[index].shot_index)
        return prompts[index]

    for position in range(1, len(shots)):
        stats["pairs"] += 1
        previous_frames = frames[position] - frames[position - 1]
        shot_frames = frames[position + 1] - frames[position]
        target_frames = min(
            shot_frames // VIDEO_CLIP_FRAMES * VIDEO_CLIP_FRAMES, max_target_frames,
        )
        if target_frames < min_target_frames:
            stats["short_target"] += 1
            continue
        context_frames = resolve_h3_overlap_frames(min(previous_frames, max_context_frames))
        if context_frames < MIN_OVERLAP_FRAMES:
            stats["short_context"] += 1
            continue
        if context_frames > max_window_frames - target_frames:
            context_frames = resolve_h3_overlap_frames(max_window_frames - target_frames)
        if context_frames < MIN_OVERLAP_FRAMES:
            stats["no_room"] += 1
            continue
        shot = shots[position]
        start_frame = frames[position] - context_frames
        if start_frame < 0:
            stats["short_context"] += 1
            continue
        prompt, prompt_source = shot_prompt(position) if need_prompt else ("", "skipped")
        if seen_pairs is not None:
            previous_prompt, _ = shot_prompt(position - 1)
            dedup_key = (
                hashlib.sha1(previous_prompt.encode()).hexdigest(),
                hashlib.sha1(prompt.encode()).hexdigest(),
            )
            if dedup_key in seen_pairs:
                stats["duplicate_pairs"] += 1
                continue
            seen_pairs.add(dedup_key)
        window_frames = context_frames + target_frames
        sample_key = (
            f"{sequence_id}:{shot.shot_index}:{start_frame}:{window_frames}:{context_frames}"
        )
        stats["accepted"] += 1
        yield ContinuationSample(
            sample_id=hashlib.sha1(sample_key.encode()).hexdigest()[:16],
            sequence_id=sequence_id,
            video_path=video_path,
            audio_path=audio_path,
            shot_index=shot.shot_index,
            shot_start_sec=shot.start_sec,
            shot_end_sec=shot.end_sec,
            start_sec=start_frame / VIDEO_FPS,
            end_sec=(start_frame + window_frames) / VIDEO_FPS,
            window_frames=window_frames,
            overlap_frames=context_frames,
            hard_core_frames=0,
            transition_frames=0,
            prompt=prompt,
            prompt_source=prompt_source,
            prompt_cleaning_version="v1",
            audio_missing=audio_path is None,
            split=split,
            conditioning_mode="masked-av-v14",
        )


def iter_record_shot_memory_only_samples(
    record: Mapping[str, Any],
    *,
    max_window_frames: int = MAX_SHOT_WINDOW_FRAMES,
    min_target_frames: int = MIN_SHOT_TARGET_FRAMES,
    split_seed: int = 17,
    require_paths: bool = False,
    include_prompt: bool = True,
    stats: dict[str, int] | None = None,
) -> Iterator[ContinuationSample]:
    """Yield every shot of one record as a context-free (memory-only) sample."""
    stats = _init_shot_stats(stats)
    sequence_id = str(record.get("sequence_id") or "").strip()
    video_path = str(record.get("file_path") or "").strip()
    if not sequence_id or not video_path:
        stats["missing_field"] += 1
        return
    audio_path = _audio_path(record)
    if require_paths and (not Path(video_path).exists() or (audio_path and not Path(audio_path).exists())):
        stats["missing_path"] += 1
        return
    shots = parse_shot_ranges(str(record.get("prompt") or ""))
    if not shots:
        stats["no_shot"] += 1
        return
    split = split_for_sequence(sequence_id, split_seed)
    for shot in shots:
        sample = _memory_only_shot_sample(
            record, shot, split=split, max_window_frames=max_window_frames,
            min_target_frames=min_target_frames, include_prompt=include_prompt,
        )
        if sample is None:
            stats["short_target"] += 1
            continue
        stats["accepted_memory_only"] += 1
        yield sample


def iter_shot_memory_only_samples(
    source_jsonl: str | Path,
    *,
    max_window_frames: int = MAX_SHOT_WINDOW_FRAMES,
    min_target_frames: int = MIN_SHOT_TARGET_FRAMES,
    split_seed: int = 17,
    require_paths: bool = False,
    include_prompt: bool = True,
    stats: dict[str, int] | None = None,
) -> Iterator[ContinuationSample]:
    """Stream ``p(target shot | memory, text)`` samples: one shot, no context.

    This is the memory-only contract of the long-horizon memory plan: the
    variable-length context is removed, the whole window is the target, and the
    only cross-shot conditioning is the memory block the cache builder attaches
    (``build_continuation_cache.py --memory-anchor``).
    """
    stats = _init_shot_stats(stats)
    for _line_number, record, error in iter_jsonl(source_jsonl):
        if error:
            stats["bad_json"] += 1
            continue
        stats["records"] += 1
        assert record is not None
        yield from iter_record_shot_memory_only_samples(
            record,
            max_window_frames=max_window_frames,
            min_target_frames=min_target_frames,
            split_seed=split_seed,
            require_paths=require_paths,
            include_prompt=include_prompt,
            stats=stats,
        )


def iter_shot_continuation_samples(
    source_jsonl: str | Path,
    *,
    max_context_frames: int = MAX_SHOT_CONTEXT_FRAMES,
    min_target_frames: int = MIN_SHOT_TARGET_FRAMES,
    max_window_frames: int = MAX_SHOT_WINDOW_FRAMES,
    split_seed: int = 17,
    require_paths: bool = False,
    conditioning_mode: str = "masked-av-v14",
    include_plain: bool = False,
    include_prompt: bool = True,
    stats: dict[str, int] | None = None,
    seen_pairs: set[tuple[str, str]] | None = None,
) -> Iterator[ContinuationSample]:
    """Stream ``p(target shot | previous shot, text)`` samples with variable context.

    One sample is one *whole* target shot preceded by as much of the previous
    shot as fits, so the sample-level overlap varies per shot instead of being a
    run-level constant:

    * ``target`` = the shot's length rounded **down** to ``17k`` frames.  The
      target must be a multiple of 17 while the window stays on ``17n+5``,
      which forces an H3-aligned (``17n+5``) context -- see
      :func:`validate_continuation_layout`.
    * ``context`` = ``min(previous shot, max_context_frames)`` rounded **down**
      onto the ``17n+5`` grid, then clamped so that
      ``context + target <= max_window_frames``.  The clamp lands on the grid
      automatically, because ``345 - 17k`` is itself ``5 mod 17``.
    * Rounding is always *down*: it keeps the context inside the previous shot
      and the target inside its own shot, so a sample never straddles three
      shots.

    Short shots keep a short context rather than being dropped, which is the
    main recovery relative to a fixed 141-frame context: a shot only has to be
    at least ``MIN_OVERLAP_FRAMES`` long to serve as the context of the next
    one.  Cross-shot pairs come from a single source record (one concatenated
    mp4), so every pair is physically contiguous and lives in one file.

    ``include_plain`` interleaves the prefix-free rows (the record's first shot)
    in the same pass, which is how the anti-forgetting mixture is produced
    without scanning the source twice.  ``seen_pairs`` turns the stream into a
    duplicate-free subset (see :func:`iter_record_shot_continuation_samples`).
    """
    # Validate before opening the source: generators are lazy, and a caller
    # that passes a wrong conditioning mode should hear about it immediately.
    _require_shot_conditioning(conditioning_mode)
    stats = _init_shot_stats(stats)
    for _line_number, record, error in iter_jsonl(source_jsonl):
        if error:
            stats["bad_json"] += 1
            continue
        stats["records"] += 1
        assert record is not None
        yield from iter_record_shot_continuation_samples(
            record,
            max_context_frames=max_context_frames,
            min_target_frames=min_target_frames,
            max_window_frames=max_window_frames,
            split_seed=split_seed,
            require_paths=require_paths,
            conditioning_mode=conditioning_mode,
            include_plain=include_plain,
            include_prompt=include_prompt,
            stats=stats,
            seen_pairs=seen_pairs,
        )


def iter_first_shot_samples(
    source_jsonl: str | Path,
    *,
    min_target_frames: int = MIN_SHOT_TARGET_FRAMES,
    max_window_frames: int = MAX_SHOT_WINDOW_FRAMES,
    split_seed: int = 17,
    require_paths: bool = False,
    stats: dict[str, int] | None = None,
) -> Iterator[ContinuationSample]:
    """Yield the prefix-free rows (each record's first shot) on their own.

    This is the standalone form of ``include_plain=True``; the rows are the
    anti-forgetting mixture of the stage-1 plan (``prefix_mode="none"``), i.e.
    the model keeps its text-to-video ability while the continuation rows teach
    it to infer a shot from its predecessor.
    """
    if stats is None:
        stats = {}
    for key in ("records", "accepted", "bad_json", "missing_field", "missing_path",
                "no_shot", "short_target"):
        stats.setdefault(key, 0)
    for _line_number, record, error in iter_jsonl(source_jsonl):
        if error:
            stats["bad_json"] += 1
            continue
        stats["records"] += 1
        assert record is not None
        sequence_id = str(record.get("sequence_id") or "").strip()
        video_path = str(record.get("file_path") or "").strip()
        if not sequence_id or not video_path:
            stats["missing_field"] += 1
            continue
        audio_path = _audio_path(record)
        if require_paths and (not Path(video_path).exists() or (audio_path and not Path(audio_path).exists())):
            stats["missing_path"] += 1
            continue
        shots = parse_shot_ranges(str(record.get("prompt") or ""))
        if not shots:
            stats["no_shot"] += 1
            continue
        sample = _plain_first_shot_sample(
            record, shots, split=split_for_sequence(sequence_id, split_seed),
            max_window_frames=max_window_frames, min_target_frames=min_target_frames,
        )
        if sample is None:
            stats["short_target"] += 1
            continue
        stats["accepted"] += 1
        yield sample

def iter_continuation_pairs(
    source_jsonl: str | Path,
    *,
    window_frames: int = 345,
    overlap_frames: int = 34,
    hard_core_frames: int = 17,
    split_seed: int = 17,
    pair_stride_frames: int | None = None,
    require_paths: bool = False,
    stats: dict[str, int] | None = None,
) -> Iterator[ContinuationPair]:
    """Yield shot-safe directional pairs where A's tail conditions later B.

    The pair stride is fixed by the physical continuation contract unless a
    larger sampling stride is requested: B begins exactly
    ``window_frames - overlap_frames`` after A.
    """
    validate_continuation_layout(window_frames, overlap_frames, hard_core_frames)
    continuation_stride = window_frames - overlap_frames
    sampling_stride = pair_stride_frames or continuation_stride
    if sampling_stride <= 0:
        raise ValueError("pair_stride_frames must be positive")
    if stats is None:
        stats = {}
    for key in (
        "records", "accepted_pairs", "bad_json", "missing_field", "missing_path",
        "no_shot", "short_shot", "short_pair_shot",
    ):
        stats.setdefault(key, 0)
    window_seconds = window_frames / VIDEO_FPS
    continuation_seconds = continuation_stride / VIDEO_FPS
    sampling_seconds = sampling_stride / VIDEO_FPS
    required_seconds = 2 * window_seconds - overlap_frames / VIDEO_FPS
    for _, record, error in iter_jsonl(source_jsonl):
        if error:
            stats["bad_json"] += 1
            continue
        stats["records"] += 1
        assert record is not None
        sequence_id = str(record.get("sequence_id") or "").strip()
        video_path = str(record.get("file_path") or "").strip()
        if not sequence_id or not video_path:
            stats["missing_field"] += 1
            continue
        audio_path = _audio_path(record)
        if require_paths and (not Path(video_path).exists() or (audio_path and not Path(audio_path).exists())):
            stats["missing_path"] += 1
            continue
        shots = parse_shot_ranges(str(record.get("prompt") or ""))
        if not shots:
            stats["no_shot"] += 1
            continue
        split = split_for_sequence(sequence_id, split_seed)
        for shot in shots:
            if shot.duration_sec + 1e-6 < required_seconds:
                stats["short_pair_shot"] += 1
                continue
            prompt, prompt_source = clean_prompt(record, shot_index=shot.shot_index)
            offset = 0.0
            while offset + required_seconds <= shot.duration_sec + 1e-6:
                history_start = shot.start_sec + offset
                target_start = history_start + continuation_seconds
                history_key = f"{sequence_id}:{shot.shot_index}:{history_start:.6f}:{window_frames}:{overlap_frames}"
                target_key = f"{sequence_id}:{shot.shot_index}:{target_start:.6f}:{window_frames}:{overlap_frames}"
                common = dict(
                    sequence_id=sequence_id, video_path=video_path, audio_path=audio_path,
                    shot_index=shot.shot_index, shot_start_sec=shot.start_sec, shot_end_sec=shot.end_sec,
                    window_frames=window_frames, overlap_frames=overlap_frames,
                    hard_core_frames=hard_core_frames, transition_frames=overlap_frames - hard_core_frames,
                    prompt=prompt, prompt_source=prompt_source, prompt_cleaning_version="v1",
                    audio_missing=audio_path is None, split=split,
                )
                history = ContinuationSample(
                    sample_id=hashlib.sha1(history_key.encode()).hexdigest()[:16],
                    start_sec=history_start, end_sec=history_start + window_seconds, **common,
                )
                target = ContinuationSample(
                    sample_id=hashlib.sha1(target_key.encode()).hexdigest()[:16],
                    start_sec=target_start, end_sec=target_start + window_seconds, **common,
                )
                pair_key = f"{history.sample_id}:{target.sample_id}"
                stats["accepted_pairs"] += 1
                yield ContinuationPair(
                    pair_id=hashlib.sha1(pair_key.encode()).hexdigest()[:16],
                    sequence_id=sequence_id, split=split, history=history, target=target,
                )
                offset += sampling_seconds


def build_continuation_pairs(*args, **kwargs) -> tuple[list[ContinuationPair], dict[str, int]]:
    """Materialize directional pairs for tests and small manifest inspections."""
    stats: dict[str, int] = {}
    pairs = list(iter_continuation_pairs(*args, stats=stats, **kwargs))
    return pairs, stats


def region_weights(length: int, hard_core: int, transition: int, *, transition_weight: float = 0.5, first_suffix_clip: int = 5, first_suffix_weight: float = 3.0, suffix_weight: float = 1.0, device=None, dtype=torch.float32) -> torch.Tensor:
    if length <= 0 or hard_core < 0 or transition < 0 or hard_core + transition > length:
        raise ValueError("invalid region lengths")
    weights = torch.zeros(length, device=device, dtype=dtype)
    if transition:
        weights[hard_core:hard_core + transition] = torch.linspace(transition_weight, 0.0, transition, device=device, dtype=dtype)
    suffix_start = hard_core + transition
    if suffix_start < length:
        first_end = min(length, suffix_start + first_suffix_clip)
        weights[suffix_start:first_end] = first_suffix_weight
        weights[first_end:] = suffix_weight
    return weights


def v3_continuation_inputs(
    target_clean: torch.Tensor,
    history_tail_clean: torch.Tensor,
    shared_noise: torch.Tensor,
    timestep: torch.Tensor | float,
    add_noise: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | float], torch.Tensor],
    *,
    hard_core: int,
    transition: int,
    transition_start_weight: float = 0.8,
    transition_end_weight: float = 0.5,
    time_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct v3 teacher forcing with shared A/B overlap noise.

    ``history_tail_clean`` covers only the overlap prefix.  It is noised with
    the exact prefix slice of B's initial noise, so no clean latent is injected
    at a nonterminal timestep.
    """
    time_dim = time_dim % target_clean.ndim
    prefix_steps = hard_core + transition
    if prefix_steps <= 0 or prefix_steps >= target_clean.shape[time_dim]:
        raise ValueError("v3 prefix must be positive and smaller than the target timeline")
    if target_clean.shape != shared_noise.shape:
        raise ValueError("target and shared noise tensors must have equal shapes")
    if (
        history_tail_clean.ndim != target_clean.ndim
        or history_tail_clean.shape[:time_dim] != target_clean.shape[:time_dim]
        or history_tail_clean.shape[time_dim + 1:] != target_clean.shape[time_dim + 1:]
        or history_tail_clean.shape[time_dim] != prefix_steps
    ):
        raise ValueError("v3 history tail must match the target non-temporal axes and overlap length")
    if not 0 < transition_end_weight <= transition_start_weight <= 1:
        raise ValueError("v3 transition weights must satisfy 0 < end <= start <= 1")
    main_noisy = add_noise(target_clean, shared_noise, timestep)
    history_noise = shared_noise.narrow(time_dim, 0, prefix_steps)
    history_noisy = add_noise(history_tail_clean, history_noise, timestep)
    weights = torch.ones(prefix_steps, dtype=target_clean.dtype, device=target_clean.device)
    if transition:
        weights[hard_core:] = torch.linspace(
            transition_start_weight, transition_end_weight, transition,
            dtype=target_clean.dtype, device=target_clean.device,
        )
    shape = [1] * target_clean.ndim
    shape[time_dim] = prefix_steps
    blend = weights.reshape(shape)
    result = main_noisy.clone()
    main_prefix = main_noisy.narrow(time_dim, 0, prefix_steps)
    result.narrow(time_dim, 0, prefix_steps).copy_(main_prefix * (1 - blend) + history_noisy * blend)
    main_target = shared_noise - target_clean
    history_target = history_noise - history_tail_clean
    target = main_target.clone()
    target.narrow(time_dim, 0, prefix_steps).copy_(
        main_target.narrow(time_dim, 0, prefix_steps) * (1 - blend) + history_target * blend
    )
    return result, target, main_noisy


def weighted_flow_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, *, time_dim: int = -1) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shape")
    time_dim = time_dim % prediction.ndim
    if weights.numel() != prediction.shape[time_dim]:
        raise ValueError("weights must match the latent time axis")
    per_token = (prediction.float() - target.float()).square().movedim(time_dim, -1).flatten(0, -2).mean(dim=0)
    expanded = weights.to(per_token.device, per_token.dtype)
    denominator = expanded.sum().clamp_min(1e-8)
    loss = (per_token * expanded).sum() / denominator
    return loss, {"weighted_tokens": float(denominator.item()), "unweighted_mse": float(per_token.mean().item())}


def continuation_flow_loss(
    video_prediction: torch.Tensor,
    video_target: torch.Tensor,
    video_weights: torch.Tensor,
    *,
    audio_prediction: torch.Tensor | None = None,
    audio_target: torch.Tensor | None = None,
    audio_weights: torch.Tensor | None = None,
    lambda_audio: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine independently normalized video/audio masked flow losses."""
    video_loss, video_info = weighted_flow_loss(video_prediction, video_target, video_weights, time_dim=-3 if video_prediction.ndim >= 4 else -1)
    total = video_loss
    info = {"video_loss": float(video_loss.detach().item()), **{f"video_{k}": v for k, v in video_info.items()}}
    if audio_prediction is not None or audio_target is not None or audio_weights is not None:
        if audio_prediction is None or audio_target is None or audio_weights is None:
            raise ValueError("audio prediction, target, and weights must be provided together")
        if lambda_audio < 0:
            raise ValueError("lambda_audio must be non-negative")
        audio_loss, audio_info = weighted_flow_loss(audio_prediction, audio_target, audio_weights, time_dim=-1)
        total = total + lambda_audio * audio_loss
        info.update({"audio_loss": float(audio_loss.detach().item()), "lambda_audio": float(lambda_audio), **{f"audio_{k}": v for k, v in audio_info.items()}})
    return total, info
