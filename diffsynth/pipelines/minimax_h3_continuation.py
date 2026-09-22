"""CPU-safe planning and assembly primitives for MiniMax-H3 continuation.

This module deliberately does not import the H3 pipeline.  Keeping the time
arithmetic and state contract independent from model loading makes it possible
to validate a long-video plan before allocating a CUDA model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .h3_appearance_memory import (
    H3AppearanceMemoryBank,
    H3AppearanceMemoryConfig,
    appearance_memory_config_from_legacy,
)
from ..utils.h3_memory_rollout import H3MemoryRolloutPlan, H3MemorySlotBuffer


H3_VIDEO_CLIP_LENGTH = 17
H3_VIDEO_FRAME_REMAINDER = 5
H3_VIDEO_FPS = 24
H3_AUDIO_SAMPLE_RATE = 32_000
H3_AUDIO_LATENT_RATE = 40
H3_CONTINUATION_MODES = frozenset(("hard", "masked-av-v14"))


def is_masked_av_context_frames(frame_count: int) -> bool:
    """Return whether a frame count is an exact joint H3 video/audio head."""
    frame_count = int(frame_count)
    return frame_count >= 39 and (frame_count - 39) % 51 == 0


def h3_continuation_video_latent_steps(overlap_frames: int) -> int:
    """Map a continuation head to latent tokens, including native 39f heads.

    Legacy 17-frame groups map to five tokens.  Native Masked AV heads start
    at 39 frames and include the two endpoint tokens, so 39f maps to 12
    tokens rather than the legacy ``39 // 17 * 5`` approximation.
    """
    overlap_frames = int(overlap_frames)
    if overlap_frames == 0:
        return 0
    if is_masked_av_context_frames(overlap_frames):
        return 2 + 5 * ((overlap_frames - 5) // 17)
    if overlap_frames > 0 and overlap_frames % H3_VIDEO_CLIP_LENGTH == 0:
        return (overlap_frames // H3_VIDEO_CLIP_LENGTH) * 5
    raise ValueError(
        "continuation overlap must be a legacy 17-frame multiple or an exact "
        f"Masked AV length (39, 90, ...), got {overlap_frames}"
    )


def resolve_h3_video_frames(
    requested_frames: int,
    *,
    clip_length: int = H3_VIDEO_CLIP_LENGTH,
    remainder: int = H3_VIDEO_FRAME_REMAINDER,
) -> int:
    """Round a requested H3 video length up to the ``17n + 5`` contract."""
    if not isinstance(requested_frames, int) or isinstance(requested_frames, bool):
        raise ValueError(f"requested_frames must be an integer, got {requested_frames!r}")
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be positive, got {requested_frames}")
    if clip_length <= 0 or not 0 <= remainder < clip_length:
        raise ValueError(f"invalid H3 shape rule: clip_length={clip_length}, remainder={remainder}")
    return max(remainder, ((requested_frames - remainder + clip_length - 1) // clip_length) * clip_length + remainder)


def validate_clip_aligned_overlap(
    overlap_frames: int,
    resolved_video_frames: int,
    *,
    clip_length: int = H3_VIDEO_CLIP_LENGTH,
) -> None:
    """Validate the fixed Retake prefix required by a continuation window."""
    if not isinstance(overlap_frames, int) or isinstance(overlap_frames, bool):
        raise ValueError(f"overlap_frames must be an integer, got {overlap_frames!r}")
    if overlap_frames <= 0:
        raise ValueError(f"overlap_frames must be positive, got {overlap_frames}")
    if overlap_frames >= resolved_video_frames:
        raise ValueError(
            "overlap_frames must be smaller than the resolved video window: "
            f"overlap_frames={overlap_frames}, resolved_video_frames={resolved_video_frames}"
        )
    if overlap_frames % clip_length and not is_masked_av_context_frames(overlap_frames):
        raise ValueError(
            "overlap_frames must align to complete H3 Video VAE clips or an exact "
            "joint Masked AV head: "
            f"requested={overlap_frames}, clip_length={clip_length}"
        )


def h3_video_latent_steps(
    resolved_video_frames: int,
    *,
    clip_length: int = H3_VIDEO_CLIP_LENGTH,
    remainder: int = H3_VIDEO_FRAME_REMAINDER,
    tokens_per_clip: int = 5,
) -> int:
    """Return the temporal H3 video-VAE latent length for a resolved window."""
    if resolved_video_frames % clip_length != remainder:
        raise ValueError(
            f"resolved_video_frames must satisfy {clip_length}n+{remainder}, got {resolved_video_frames}"
        )
    return ((resolved_video_frames - remainder) // clip_length) * tokens_per_clip + 2



@dataclass(frozen=True)
class H3ContinuationConfig:
    """Invariant timing configuration shared by every continuation window."""

    requested_window_frames: int = 243
    overlap_frames: int = 34
    video_fps: int = H3_VIDEO_FPS
    audio_sample_rate: int = H3_AUDIO_SAMPLE_RATE
    audio_latent_rate: int = H3_AUDIO_LATENT_RATE
    video_clip_length: int = H3_VIDEO_CLIP_LENGTH
    video_frame_remainder: int = H3_VIDEO_FRAME_REMAINDER
    audio_crossfade_ms: float = 0.0

    def resolved_window_frames(self, requested_frames: int | None = None) -> int:
        return resolve_h3_video_frames(
            self.requested_window_frames if requested_frames is None else requested_frames,
            clip_length=self.video_clip_length,
            remainder=self.video_frame_remainder,
        )

    def validate(self, requested_frames: int | None = None, *, requires_overlap: bool = True) -> int:
        if self.video_fps <= 0 or self.audio_sample_rate <= 0 or self.audio_latent_rate <= 0:
            raise ValueError("video_fps, audio_sample_rate, and audio_latent_rate must all be positive")
        if not 0 <= self.audio_crossfade_ms <= 200:
            raise ValueError(f"audio_crossfade_ms must be between 0 and 200, got {self.audio_crossfade_ms}")
        resolved = self.resolved_window_frames(requested_frames)
        if requires_overlap:
            validate_clip_aligned_overlap(
                self.overlap_frames, resolved, clip_length=self.video_clip_length
            )
        return resolved

@dataclass(frozen=True)
class H3VideoLatentContinuation:
    """A clip-aligned video latent prefix exported by a previous H3 window."""

    latents: torch.Tensor
    overlap_frames: int
    video_fps: int = H3_VIDEO_FPS
    clip_length: int = H3_VIDEO_CLIP_LENGTH
    tokens_per_clip: int = 5
    overlap_mode: str = "hard"
    min_anchor_weight: float = 0.2
    hard_core_frames: int = 0
    max_anchor_weight: float = 0.8
    # Optional boundary motion state.  These tensors are kept out of the
    # manifest by json_safe_metadata just like the latent tail itself.
    motion_velocity: torch.Tensor | None = None
    motion_token_count: int = 0
    motion_start_weight: float = 0.5
    motion_end_weight: float = 0.1

@dataclass(frozen=True)
class H3AudioLatentContinuation:
    """A time-aligned audio latent prefix exported by a previous H3 window."""

    latents: torch.Tensor
    overlap_frames: int
    video_fps: int = H3_VIDEO_FPS
    audio_sample_rate: int = H3_AUDIO_SAMPLE_RATE
    audio_latent_rate: int = H3_AUDIO_LATENT_RATE
    overlap_mode: str = "hard"
    min_anchor_weight: float = 0.2
    hard_core_frames: int = 0
    max_anchor_weight: float = 0.8

@dataclass(frozen=True)
class H3PipelineResult:
    """Opt-in H3 one-window result, including final denoised latents."""

    video: list[Any]
    audio: torch.Tensor
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    resolved_video_frames: int
    video_fps: int
    audio_sample_rate: int
    video_latent_steps: int
    audio_latent_steps: int


def validate_video_latent_continuation(
    continuation: H3VideoLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
    expected_overlap_frames: int,
) -> int:
    """Validate a video tail and return its latent-prefix extent."""
    if continuation.overlap_frames != expected_overlap_frames:
        raise ValueError(
            f"continuation video overlap_frames={continuation.overlap_frames} does not match expected {expected_overlap_frames}"
        )
    if continuation.video_fps != H3_VIDEO_FPS:
        raise ValueError(f"continuation video FPS must be {H3_VIDEO_FPS}, got {continuation.video_fps}")
    if continuation.clip_length != H3_VIDEO_CLIP_LENGTH or continuation.tokens_per_clip != 5:
        raise ValueError("continuation video VAE clip metadata must be clip_length=17 and tokens_per_clip=5")
    _validate_overlap_mode(
        continuation.overlap_mode, continuation.min_anchor_weight,
        continuation.max_anchor_weight, continuation.hard_core_frames,
        expected_overlap_frames, continuation.clip_length,
    )
    validate_clip_aligned_overlap(expected_overlap_frames, resolved_video_frames, clip_length=continuation.clip_length)
    expected_steps = h3_continuation_video_latent_steps(expected_overlap_frames)
    tail = continuation.latents
    if tail.ndim != 5 or target_latents.ndim != 5:
        raise ValueError("continuation video latents must have shape [B, C, T, H, W]")
    if tail.shape[0] != target_latents.shape[0] or tail.shape[1] != target_latents.shape[1]:
        raise ValueError("continuation video latent batch/channels do not match target")
    if tail.shape[-2:] != target_latents.shape[-2:]:
        raise ValueError("continuation video latent spatial shape does not match target")
    if tail.shape[2] != expected_steps:
        raise ValueError(f"continuation video latent time extent must be {expected_steps}, got {tail.shape[2]}")
    return expected_steps


def estimate_video_latent_velocity(
    video_latents: torch.Tensor,
    *,
    history_steps: int = 4,
) -> torch.Tensor:
    """Estimate a robust final-token velocity from a clean video latent.

    A median of recent first differences is less sensitive to one noisy VAE or
    diffusion token than using only ``z[-1] - z[-2]``.  The returned tensor has
    one temporal token and can therefore be extrapolated into the next window.
    """
    if video_latents.ndim != 5:
        raise ValueError("video_latents must have shape [B, C, T, H, W]")
    if video_latents.shape[2] < 2:
        raise ValueError("at least two video latent tokens are required for motion handoff")
    history_steps = max(1, min(int(history_steps), video_latents.shape[2] - 1))
    recent = video_latents[:, :, -(history_steps + 1):]
    differences = recent[:, :, 1:] - recent[:, :, :-1]
    velocity = differences.median(dim=2, keepdim=True).values
    return velocity.detach().clone()


def video_motion_anchor(
    continuation: H3VideoLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Predict and weight the first free suffix tokens from boundary velocity."""
    prefix_steps = validate_video_latent_continuation(
        continuation, target_latents,
        resolved_video_frames=resolved_video_frames,
        expected_overlap_frames=continuation.overlap_frames,
    )
    velocity = continuation.motion_velocity
    count = min(int(continuation.motion_token_count), target_latents.shape[2] - prefix_steps)
    if velocity is None or count <= 0:
        return None
    if velocity.ndim != 5 or velocity.shape[:2] != target_latents.shape[:2] or velocity.shape[-2:] != target_latents.shape[-2:]:
        raise ValueError("motion_velocity must match video latent batch, channels, and spatial shape")
    if velocity.shape[2] != 1:
        raise ValueError("motion_velocity must contain exactly one temporal token")
    last = continuation.latents[:, :, -1:].to(device=target_latents.device, dtype=target_latents.dtype)
    velocity = velocity.to(device=target_latents.device, dtype=target_latents.dtype)
    steps = torch.arange(1, count + 1, device=target_latents.device, dtype=target_latents.dtype)
    shape = [1] * target_latents.ndim
    shape[2] = count
    predicted = last + velocity * steps.reshape(shape)
    weights = torch.linspace(
        float(continuation.motion_start_weight),
        float(continuation.motion_end_weight),
        count,
        device=target_latents.device,
        dtype=target_latents.dtype,
    )
    return predicted, weights


def validate_audio_latent_continuation(
    continuation: H3AudioLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
    expected_overlap_frames: int,
) -> int:
    """Validate an audio tail and return its latent-prefix extent."""
    if continuation.overlap_frames != expected_overlap_frames:
        raise ValueError(
            f"continuation audio overlap_frames={continuation.overlap_frames} does not match expected {expected_overlap_frames}"
        )
    if continuation.video_fps != H3_VIDEO_FPS:
        raise ValueError(f"continuation audio video_fps must be {H3_VIDEO_FPS}, got {continuation.video_fps}")
    if continuation.audio_sample_rate != H3_AUDIO_SAMPLE_RATE:
        raise ValueError(f"continuation audio sample rate must be {H3_AUDIO_SAMPLE_RATE}, got {continuation.audio_sample_rate}")
    if continuation.audio_latent_rate != H3_AUDIO_LATENT_RATE:
        raise ValueError(f"continuation audio latent rate must be {H3_AUDIO_LATENT_RATE}, got {continuation.audio_latent_rate}")
    _validate_overlap_mode(
        continuation.overlap_mode, continuation.min_anchor_weight,
        continuation.max_anchor_weight, continuation.hard_core_frames,
        expected_overlap_frames, H3_VIDEO_CLIP_LENGTH,
    )
    validate_clip_aligned_overlap(expected_overlap_frames, resolved_video_frames)
    expected_steps = round(expected_overlap_frames / continuation.video_fps * continuation.audio_latent_rate)
    tail = continuation.latents
    if tail.ndim != 3 or target_latents.ndim != 3:
        raise ValueError("continuation audio latents must have shape [C, channels, T]")
    if tail.shape[:2] != target_latents.shape[:2]:
        raise ValueError("continuation audio latent batch/channels do not match target")
    if tail.shape[-1] != expected_steps:
        raise ValueError(f"continuation audio latent time extent must be {expected_steps}, got {tail.shape[-1]}")
    return expected_steps


def apply_video_latent_continuation(
    continuation: H3VideoLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return latent-prefix conditioning and a ``1=regenerate`` mask.

    The overlap prefix uses the exact hard history rows shared by Retake and
    the masked-av-v14 latent handoff (no taper ramp); the tail remains a
    regenerated target.
    """
    prefix_steps = validate_video_latent_continuation(
        continuation, target_latents,
        resolved_video_frames=resolved_video_frames,
        expected_overlap_frames=continuation.overlap_frames,
    )
    input_latents = target_latents.clone()
    input_latents[:, :, :prefix_steps] = continuation.latents.to(
        device=target_latents.device, dtype=target_latents.dtype
    )
    mask, _ = _make_overlap_mask(
        target_latents.shape[2], prefix_steps, continuation.overlap_mode,
        continuation.min_anchor_weight, continuation.max_anchor_weight,
        continuation.hard_core_frames // continuation.clip_length * continuation.tokens_per_clip,
        target_latents.device, target_latents.dtype,
    )
    return input_latents, mask


def apply_audio_latent_continuation(
    continuation: H3AudioLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return latent-prefix conditioning and a ``1=regenerate`` mask."""
    prefix_steps = validate_audio_latent_continuation(
        continuation, target_latents,
        resolved_video_frames=resolved_video_frames,
        expected_overlap_frames=continuation.overlap_frames,
    )
    input_latents = target_latents.clone()
    input_latents[..., :prefix_steps] = continuation.latents.to(
        device=target_latents.device, dtype=target_latents.dtype
    )
    hard_core_steps = round(
        continuation.hard_core_frames / continuation.video_fps * continuation.audio_latent_rate
    )
    mask, _ = _make_overlap_mask(
        target_latents.shape[-1], prefix_steps, continuation.overlap_mode,
        continuation.min_anchor_weight, continuation.max_anchor_weight, hard_core_steps,
        target_latents.device, target_latents.dtype,
    )
    return input_latents, mask


def _validate_overlap_mode(
    mode: str,
    min_anchor_weight: float,
    max_anchor_weight: float,
    hard_core_frames: int,
    overlap_frames: int,
    clip_length: int,
) -> None:
    if mode not in H3_CONTINUATION_MODES:
        raise ValueError(f"unsupported continuation overlap_mode={mode!r}")
    if not 0.0 < min_anchor_weight <= max_anchor_weight <= 1.0:
        raise ValueError(
            "anchor weights must satisfy 0 < min <= max <= 1, got "
            f"min={min_anchor_weight}, max={max_anchor_weight}"
        )
    if hard_core_frames < 0 or hard_core_frames > overlap_frames or hard_core_frames % clip_length:
        raise ValueError(
            "hard_core_frames must be clip-aligned and within the overlap: "
            f"hard_core={hard_core_frames}, overlap={overlap_frames}, clip_length={clip_length}"
        )


def _make_overlap_mask(
    total_steps: int,
    prefix_steps: int,
    mode: str,
    min_anchor_weight: float,
    max_anchor_weight: float,
    hard_core_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mask = torch.ones(total_steps, dtype=torch.float32, device=device)
    if mode == "hard":
        mask[:prefix_steps] = 0.0
        return mask.to(dtype=dtype), None
    # Only the hard / masked-av-v14 modes are active in this codebase.  No
    # taper (weighted anchor) branch exists anymore, so a non-hard mode is
    # rejected rather than silently degrading.
    raise ValueError(f"unsupported continuation overlap_mode={mode!r}")


def project_video_motion_anchor(
    latents: torch.Tensor,
    scheduler: Any,
    timestep: torch.Tensor | None,
    anchor_clean: torch.Tensor | None,
    anchor_noise: torch.Tensor | None,
    weights: torch.Tensor | None,
    *,
    start_step: int,
) -> torch.Tensor:
    """Soft-project noisy motion predictions onto the first suffix tokens."""
    if anchor_clean is None or anchor_noise is None or weights is None:
        return latents
    count = min(int(weights.numel()), anchor_clean.shape[2], anchor_noise.shape[2], latents.shape[2] - start_step)
    if count <= 0:
        return latents
    if start_step < 0 or anchor_clean.shape != anchor_noise.shape:
        raise ValueError("invalid motion anchor offset or clean/noise shape")
    if timestep is None:
        noisy_anchor = anchor_clean
    else:
        noisy_anchor = scheduler.add_noise(anchor_clean, anchor_noise, timestep)
    shape = [1] * latents.ndim
    shape[2] = count
    blend = weights[:count].to(device=latents.device, dtype=latents.dtype).reshape(shape)
    result = latents.clone()
    current = latents[:, :, start_step:start_step + count]
    anchor = noisy_anchor[:, :, :count].to(device=latents.device, dtype=latents.dtype)
    result[:, :, start_step:start_step + count].copy_(current * (1 - blend) + anchor * blend)
    return result


@dataclass(frozen=True)
class H3Segment:
    """One requested generation window in an immutable ordered plan."""

    prompt: str | None = None
    requested_frames: int | None = None
    segment_id: str | None = None
    controls: Mapping[str, Any] = field(default_factory=dict)
    semantic_state: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class H3SegmentPlan:
    """Global conditions plus an ordered sequence of requested windows."""

    global_prompt: str
    segments: tuple[H3Segment, ...]
    plan_id: str = "h3-continuation-plan"
    global_controls: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.global_prompt:
            raise ValueError("global_prompt must not be empty")
        if not self.segments:
            raise ValueError("segment plan must contain at least one segment")
        if len({segment.segment_id for segment in self.segments if segment.segment_id is not None}) != len(
            [segment for segment in self.segments if segment.segment_id is not None]
        ):
            raise ValueError("segment_id values must be unique when provided")

    def resolved_prompt(self, segment_index: int) -> str:
        segment = self.segments[segment_index]
        return self.global_prompt if not segment.prompt else f"{self.global_prompt}\n\n{segment.prompt}"

@dataclass(frozen=True)
class ResolvedContinuationWindow:
    """A window after H3 shape snapping and physical-time boundary resolution."""

    segment_index: int
    segment_id: str
    requested_video_frames: int
    resolved_video_frames: int
    overlap_video_frames: int
    timeline_start_video_frame: int
    timeline_end_video_frame: int
    new_video_frames: int
    video_fps: int
    audio_sample_rate: int
    audio_latent_rate: int
    resolved_audio_samples: int
    overlap_audio_samples: int
    resolved_audio_latents: int
    overlap_audio_latents: int
    video_latent_steps: int

    @property
    def window_seconds(self) -> float:
        return self.resolved_video_frames / self.video_fps

    @property
    def overlap_seconds(self) -> float:
        return self.overlap_video_frames / self.video_fps

    @property
    def timeline_start_seconds(self) -> float:
        return self.timeline_start_video_frame / self.video_fps

    @property
    def timeline_end_seconds(self) -> float:
        return self.timeline_end_video_frame / self.video_fps


def resolve_continuation_window(
    config: H3ContinuationConfig,
    *,
    segment_index: int,
    requested_video_frames: int | None = None,
    prior_timeline_video_frames: int = 0,
    segment_id: str | None = None,
) -> ResolvedContinuationWindow:
    """Resolve one window; all audio coordinates come from its video timeline."""
    if segment_index < 0:
        raise ValueError(f"segment_index must be non-negative, got {segment_index}")
    if prior_timeline_video_frames < 0:
        raise ValueError("prior_timeline_video_frames must be non-negative")
    requires_overlap = segment_index > 0
    resolved = config.validate(requested_video_frames, requires_overlap=requires_overlap)
    overlap = config.overlap_frames if requires_overlap else 0
    if requires_overlap and prior_timeline_video_frames < overlap:
        raise ValueError(
            "prior timeline is shorter than the required overlap: "
            f"prior_timeline_video_frames={prior_timeline_video_frames}, overlap_frames={overlap}"
        )
    start = 0 if not requires_overlap else prior_timeline_video_frames - overlap
    new_frames = resolved - overlap
    end = start + resolved
    return ResolvedContinuationWindow(
        segment_index=segment_index,
        segment_id=segment_id or f"segment-{segment_index:04d}",
        requested_video_frames=config.requested_window_frames if requested_video_frames is None else requested_video_frames,
        resolved_video_frames=resolved,
        overlap_video_frames=overlap,
        timeline_start_video_frame=start,
        timeline_end_video_frame=end,
        new_video_frames=new_frames,
        video_fps=config.video_fps,
        audio_sample_rate=config.audio_sample_rate,
        audio_latent_rate=config.audio_latent_rate,
        resolved_audio_samples=round(resolved / config.video_fps * config.audio_sample_rate),
        overlap_audio_samples=round(overlap / config.video_fps * config.audio_sample_rate),
        resolved_audio_latents=round(resolved / config.video_fps * config.audio_latent_rate),
        overlap_audio_latents=round(overlap / config.video_fps * config.audio_latent_rate),
        video_latent_steps=h3_video_latent_steps(
            resolved,
            clip_length=config.video_clip_length,
            remainder=config.video_frame_remainder,
        ),
    )


def json_safe_metadata(value: Any) -> Any:
    """Convert state metadata to JSON without embedding tensor payloads."""
    if isinstance(value, torch.Tensor):
        return {
            "__tensor_metadata__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_safe_metadata(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__repr__": repr(value), "__type__": type(value).__qualname__}

@dataclass(frozen=True)
class H3WindowStateRecord:
    window: ResolvedContinuationWindow
    seed: int
    prompt: str
    scheduler_controls: Mapping[str, Any] = field(default_factory=dict)
    model_identity: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    continuation_mode: str = "retake-hard"
    global_reference_frames: int = 0
    appearance_memory_mode: str = "disabled"
    appearance_memory_frames: int = 0

@dataclass
class ContinuationState:
    """Mutable runtime state. Media and latent tails intentionally stay in memory."""

    plan_id: str
    global_prompt: str
    config: H3ContinuationConfig
    base_seed: int
    model_identity: str | None = None
    next_segment_index: int = 0
    emitted_video_frames: int = 0
    emitted_audio_samples: int = 0
    records: list[H3WindowStateRecord] = field(default_factory=list)
    video_tail: Any = field(default=None, repr=False, compare=False)
    audio_tail: torch.Tensor | None = field(default=None, repr=False, compare=False)
    video_latent_tail: H3VideoLatentContinuation | None = field(default=None, repr=False, compare=False)
    audio_latent_tail: H3AudioLatentContinuation | None = field(default=None, repr=False, compare=False)
    appearance_memory: H3AppearanceMemoryBank | None = field(default=None, repr=False, compare=False)
    memory_buffer: Any = field(default=None, repr=False, compare=False)
    latent_artifacts: Mapping[str, str] = field(default_factory=dict)
    experiment_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def replay_only(self) -> bool:
        return not bool(self.latent_artifacts)

    def memory_metadata(self) -> dict[str, Any]:
        """Provenance of the long-horizon memory the rollout conditioned on."""
        if self.memory_buffer is None:
            return {}
        return {"long_horizon_memory": self.memory_buffer.provenance_metadata()}

    def to_manifest(self) -> "H3ContinuationStateManifest":
        return H3ContinuationStateManifest(
            plan_id=self.plan_id,
            global_prompt=self.global_prompt,
            config=self.config,
            base_seed=self.base_seed,
            model_identity=self.model_identity,
            next_segment_index=self.next_segment_index,
            emitted_video_frames=self.emitted_video_frames,
            emitted_audio_samples=self.emitted_audio_samples,
            records=tuple(self.records),
            latent_artifacts=dict(self.latent_artifacts),
            replay_only=self.replay_only,
            experiment_metadata={**dict(self.experiment_metadata), **self.memory_metadata()},
            appearance_memory_metadata=(
                self.appearance_memory.provenance_metadata()
                if self.appearance_memory is not None else {}
            ),
        )

@dataclass(frozen=True)
class H3ContinuationStateManifest:
    """JSON-safe resume/replay metadata; it never contains tensor contents."""

    plan_id: str
    global_prompt: str
    config: H3ContinuationConfig
    base_seed: int
    model_identity: str | None
    next_segment_index: int
    emitted_video_frames: int
    emitted_audio_samples: int
    records: tuple[H3WindowStateRecord, ...]
    latent_artifacts: Mapping[str, str]
    replay_only: bool
    experiment_metadata: Mapping[str, Any] = field(default_factory=dict)
    appearance_memory_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe_metadata(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

@dataclass(frozen=True)
class H3ContinuationResult:
    """Assembled media plus the state needed to inspect or continue the run."""

    video: list[Any] | None
    audio: torch.Tensor | None
    state: ContinuationState

    @property
    def manifest(self) -> H3ContinuationStateManifest:
        return self.state.to_manifest()


def assemble_video_suffix(
    previous_video: Sequence[Any] | None,
    current_video: Sequence[Any],
    overlap_frames: int,
) -> list[Any]:
    """Append a later window's non-overlap suffix."""
    current = list(current_video)
    if previous_video is None:
        if overlap_frames:
            raise ValueError("the first video window must not declare an overlap")
        return current
    if overlap_frames <= 0:
        raise ValueError("later video windows require a positive overlap_frames")
    if len(previous_video) < overlap_frames or len(current) < overlap_frames:
        raise ValueError("video inputs are shorter than overlap_frames")
    suffix = current[overlap_frames:]
    return list(previous_video) + suffix


def assemble_audio_suffix(
    previous_audio: torch.Tensor | None,
    current_audio: torch.Tensor,
    overlap_samples: int,
    *,
    crossfade_samples: int = 0,
) -> torch.Tensor:
    """Append an audio suffix, optionally blending within the retained overlap.

    The crossfade replaces the last samples of the already-emitted overlap with
    the time-aligned prefix decoded by the current window.  It therefore keeps
    the exact hard-splice sample count while making the final retained samples
    agree with the suffix that follows them.
    """
    if current_audio.ndim < 1:
        raise ValueError("current_audio must have a sample dimension")
    if previous_audio is None:
        if overlap_samples:
            raise ValueError("the first audio window must not declare an overlap")
        return current_audio
    if previous_audio.ndim != current_audio.ndim or previous_audio.shape[:-1] != current_audio.shape[:-1]:
        raise ValueError("previous_audio and current_audio must have matching non-time dimensions")
    if overlap_samples <= 0:
        raise ValueError("later audio windows require a positive overlap_samples")
    if previous_audio.shape[-1] < overlap_samples or current_audio.shape[-1] < overlap_samples:
        raise ValueError("audio inputs are shorter than overlap_samples")
    if not 0 <= crossfade_samples <= overlap_samples:
        raise ValueError("crossfade_samples must be between zero and overlap_samples")
    suffix = current_audio[..., overlap_samples:]
    if crossfade_samples == 0:
        return torch.cat([previous_audio, suffix], dim=-1)

    old_tail = previous_audio[..., -crossfade_samples:]
    new_tail = current_audio[..., overlap_samples - crossfade_samples:overlap_samples]
    phase = torch.linspace(0, torch.pi / 2, crossfade_samples, device=current_audio.device, dtype=torch.float32)
    fade_out = torch.cos(phase).to(dtype=current_audio.dtype)
    fade_in = torch.sin(phase).to(dtype=current_audio.dtype)
    while fade_out.ndim < current_audio.ndim:
        fade_out = fade_out.unsqueeze(0)
        fade_in = fade_in.unsqueeze(0)
    blended_tail = old_tail * fade_out + new_tail * fade_in
    return torch.cat([previous_audio[..., :-crossfade_samples], blended_tail, suffix], dim=-1)


def normalize_audio_to_timeline(audio: torch.Tensor, expected_samples: int) -> torch.Tensor:
    """Crop or edge-pad a VAE waveform to its declared physical timeline.

    H3 samples audio latent lengths at 40 Hz, while a video boundary can fall
    between audio latent ticks.  This small normalization is the only place
    where that quantization is reconciled with the video-owned output clock.
    """
    if audio.ndim < 1 or expected_samples <= 0:
        raise ValueError("audio must have a sample dimension and expected_samples must be positive")
    actual_samples = audio.shape[-1]
    if actual_samples == expected_samples:
        return audio
    if actual_samples > expected_samples:
        return audio[..., :expected_samples]
    if actual_samples == 0:
        raise ValueError("cannot pad an empty audio waveform to the physical timeline")
    return torch.cat([audio, audio[..., -1:].expand(*audio.shape[:-1], expected_samples - actual_samples)], dim=-1)


def load_h3_segment_plan(path: str | Path) -> H3SegmentPlan:
    """Load the deliberately small JSON plan format used by inference examples."""
    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read segment plan {plan_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"segment plan {plan_path} is not valid JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("segment plan JSON must be an object")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("segment plan JSON field 'segments' must be a list")
    segments = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, Mapping):
            raise ValueError(f"segments[{index}] must be an object")
        unknown = set(item) - {"prompt", "requested_frames", "segment_id", "controls", "semantic_state"}
        if unknown:
            raise ValueError(f"segments[{index}] contains unsupported fields: {', '.join(sorted(unknown))}")
        segments.append(H3Segment(
            prompt=item.get("prompt"), requested_frames=item.get("requested_frames"), segment_id=item.get("segment_id"),
            controls=item.get("controls", {}), semantic_state=item.get("semantic_state", {}),
        ))
    return H3SegmentPlan(
        global_prompt=payload.get("global_prompt", ""), segments=tuple(segments),
        plan_id=payload.get("plan_id", plan_path.stem), global_controls=payload.get("global_controls", {}),
    )


def cp_window_metadata(
    window: ResolvedContinuationWindow,
    *,
    seed: int,
    num_inference_steps: int,
    continuation_mode: str,
) -> dict[str, Any]:
    """Return the exact CP-group invariant payload for one continuation window."""
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    return {
        "segment_index": window.segment_index,
        "segment_id": window.segment_id,
        "resolved_video_frames": window.resolved_video_frames,
        "overlap_video_frames": window.overlap_video_frames,
        "timeline_start_video_frame": window.timeline_start_video_frame,
        "timeline_end_video_frame": window.timeline_end_video_frame,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "continuation_mode": continuation_mode,
    }


def validate_cp_window_metadata(local: Mapping[str, Any], peer_payloads: Sequence[Mapping[str, Any]]) -> None:
    """Fail before denoising when a CP rank resolved a different next window."""
    local_safe = json_safe_metadata(local)
    for rank, payload in enumerate(peer_payloads):
        if json_safe_metadata(payload) != local_safe:
            raise ValueError(f"CP rank {rank} resolved different continuation window metadata")


def is_designated_cp_writer(rank: int, *, writer_rank: int = 0) -> bool:
    return rank == writer_rank


class H3ContinuationRunner:
    """Run H3 one window at a time while retaining a fixed Retake prefix.

    ``pipeline`` only needs to be callable with the existing H3 keyword
    interface and return ``(video, audio)``.  This deliberately small protocol
    keeps the Retake-hard MVP usable with both the real pipeline and CPU fakes.
    """

    _RUNNER_OWNED_KEYS = frozenset({
        "prompt", "num_frames", "seed", "retake_video", "frame_regions_to_retake",
        "retake_audio", "retake_audio_sample_rate", "seconds_regions_to_retake",
    })

    def __init__(
        self,
        pipeline: Any,
        config: H3ContinuationConfig = H3ContinuationConfig(),
        *,
        model_identity: str | None = None,
        seed_stride: int = 1,
        continuation_mode: str = "retake-hard",
        prefer_latent_handoff: bool = False,
        manifest_directory: str | Path | None = None,
        artifact_writer: Any = None,
        window_synchronizer: Any = None,
        global_latent_decode: bool = False,
        global_reference_frames: int = 0,
        boundary_reference_frames: int = 0,
        appearance_memory_config: H3AppearanceMemoryConfig | None = None,
        memory_rollout_plan: H3MemoryRolloutPlan | None = None,
        memory_slot_encoder: Any = None,
        memory_slot_observer: Any = None,
    ) -> None:
        if continuation_mode not in ("retake-hard", "latent-handoff", "masked-av-v14"):
            raise ValueError(
                "continuation_mode=" + repr(continuation_mode) +
                " is not available; expected 'retake-hard', 'latent-handoff', or 'masked-av-v14'"
            )
        if continuation_mode == "masked-av-v14" and not prefer_latent_handoff:
            raise ValueError(f"{continuation_mode} requires prefer_latent_handoff=True")
        if continuation_mode == "masked-av-v14" and not is_masked_av_context_frames(config.overlap_frames):
            raise ValueError(
                "masked-av-v14 requires an exact joint H3 video/audio context "
                f"(39, 90, 141, ...), got {config.overlap_frames}"
            )
        if not 0 <= global_reference_frames <= 4:
            raise ValueError("global_reference_frames must be between 0 and 4")
        if boundary_reference_frames not in (0, 1):
            raise ValueError("boundary_reference_frames must be 0 or 1")
        resolved_appearance_config = appearance_memory_config_from_legacy(
            global_reference_frames,
            boundary_reference_frames,
            explicit=appearance_memory_config,
        )
        if resolved_appearance_config.mode != "disabled":
            if global_latent_decode and (appearance_memory_config is not None or global_reference_frames):
                if appearance_memory_config is None:
                    raise ValueError("global reference frames cannot be combined with global latent decode")
                raise ValueError("appearance memory bank cannot be combined with global latent decode")
            if appearance_memory_config is not None and continuation_mode != "masked-av-v14":
                raise ValueError("explicit appearance memory config requires continuation_mode='masked-av-v14'")
            if appearance_memory_config is not None and not prefer_latent_handoff:
                raise ValueError("explicit appearance memory config requires prefer_latent_handoff=True")
        if global_reference_frames and global_latent_decode:
            raise ValueError("global reference frames cannot be combined with global latent decode")
        if seed_stride <= 0:
            raise ValueError(f"seed_stride must be positive, got {seed_stride}")
        self.pipeline = pipeline
        self.config = config
        self.model_identity = model_identity
        self.seed_stride = seed_stride
        self.continuation_mode = continuation_mode
        self.prefer_latent_handoff = prefer_latent_handoff
        self.manifest_directory = Path(manifest_directory) if manifest_directory is not None else None
        self.artifact_writer = artifact_writer
        self.window_synchronizer = window_synchronizer
        self.global_latent_decode = bool(global_latent_decode)
        self.global_reference_frames = int(global_reference_frames)
        self.boundary_reference_frames = int(boundary_reference_frames)
        resolved_memory_plan = (
            memory_rollout_plan if memory_rollout_plan is not None else H3MemoryRolloutPlan()
        )
        if not isinstance(resolved_memory_plan, H3MemoryRolloutPlan):
            raise TypeError(
                "memory_rollout_plan must be H3MemoryRolloutPlan, got "
                f"{type(resolved_memory_plan).__name__}"
            )
        if resolved_memory_plan.enabled:
            # The trained contract is memory-only masked-av-v14: the slots are the
            # only cross-shot conditioning, and every window is a full-length
            # target rather than a continuation of a shared latent timeline.
            if continuation_mode != "masked-av-v14":
                raise ValueError(
                    "long-horizon memory slots require continuation_mode='masked-av-v14', got "
                    f"{continuation_mode!r}"
                )
            if not prefer_latent_handoff:
                raise ValueError("long-horizon memory slots require prefer_latent_handoff=True")
            if global_latent_decode:
                raise ValueError(
                    "long-horizon memory slots cannot be combined with global latent decode: "
                    "the slots are rebuilt from each window's decoded frames"
                )
        self.memory_rollout_plan = resolved_memory_plan
        self.memory_slot_encoder = memory_slot_encoder
        self.memory_slot_observer = memory_slot_observer
        self.appearance_memory_config = resolved_appearance_config

    def seed_for_segment(self, base_seed: int, segment_index: int) -> int:
        return base_seed + segment_index * self.seed_stride

    def _encode_memory_slot_frames(
        self,
        frames: Sequence[Any],
        *,
        height: int | None = None,
        width: int | None = None,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
    ) -> torch.Tensor:
        """Encode one long-horizon memory slot from the rollout's own frames.

        Training encodes a slot as an independent ``17n+5`` VAE clip
        (``diffsynth.utils.continuation_lora.encode_memory_slot``), so inference
        re-encodes the same number of frames the same way.  Slicing the window's
        latent timeline instead would only be equivalent on a grid-aligned cut,
        and a shot boundary is not one.
        """
        prepared = list(frames)
        if not prepared:
            raise ValueError("memory slot cannot be encoded from zero frames")
        encoder = self.memory_slot_encoder
        if encoder is not None:
            tensor = encoder(prepared, height=height, width=width)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("memory_slot_encoder must return a torch.Tensor")
            return tensor.detach()

        pipe = self.pipeline
        video_vae = getattr(pipe, "video_vae", None)
        preprocess_video = getattr(pipe, "preprocess_video", None)
        if (
            video_vae is None
            or not callable(getattr(video_vae, "encode_video", None))
            or not callable(preprocess_video)
        ):
            raise TypeError(
                "long-horizon memory slots need a pipeline exposing preprocess_video and "
                "video_vae.encode_video; pass memory_slot_encoder for a custom pipeline"
            )
        load_models_to_device = getattr(pipe, "load_models_to_device", None)
        if callable(load_models_to_device):
            load_models_to_device(("video_vae",))
        if height is not None and width is not None:
            prepared = [
                frame.convert("RGB").resize((int(width), int(height)), Image.LANCZOS)
                if hasattr(frame, "resize") else frame
                for frame in prepared
            ]
        with torch.no_grad():
            frames_tensor = preprocess_video(
                prepared, torch_dtype=torch.float32, min_value=0, device=pipe.device
            )
            latents = video_vae.encode_video(
                frames_tensor, dtype=pipe.torch_dtype,
                tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap,
            )
        return latents.detach()


    def _validate_controls(self, controls: Mapping[str, Any], *, scope: str) -> None:
        invalid = self._RUNNER_OWNED_KEYS.intersection(controls)
        if invalid:
            keys = ", ".join(sorted(invalid))
            raise ValueError(f"{scope} controls cannot override runner-owned fields: {keys}")

    def _pipeline_result(self, output: Any) -> tuple[list[Any] | None, torch.Tensor | None, Any]:
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
        elif hasattr(output, "video") and hasattr(output, "audio"):
            video, audio = output.video, output.audio
        else:
            raise TypeError("H3 continuation pipeline must return (video, audio) or an object with video/audio")
        if video is not None:
            video = list(video)
        if audio is not None and not isinstance(audio, torch.Tensor):
            raise TypeError("pipeline audio output must be a torch.Tensor or None")
        return video, audio, output

    def _persist_manifest(self, state: ContinuationState) -> None:
        if self.manifest_directory is None:
            return
        self.manifest_directory.mkdir(parents=True, exist_ok=True)
        target = self.manifest_directory / "continuation_state.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(state.to_manifest().to_json() + "\n", encoding="utf-8")
        temporary.replace(target)

    def _write_artifacts(
        self,
        window: ResolvedContinuationWindow,
        video: list[Any] | None,
        audio: torch.Tensor | None,
        state: ContinuationState,
    ) -> Mapping[str, str]:
        if self.artifact_writer is None:
            return {}
        artifacts = self.artifact_writer(window=window, video=video, audio=audio, state=state)
        if artifacts is None:
            return {}
        if not isinstance(artifacts, Mapping) or not all(isinstance(key, str) and isinstance(value, str) for key, value in artifacts.items()):
            raise TypeError("artifact_writer must return a mapping of string names to string paths")
        return dict(artifacts)

    def run(
        self,
        plan: H3SegmentPlan,
        *,
        base_seed: int = 42,
        pipeline_kwargs: Mapping[str, Any] | None = None,
        state: ContinuationState | None = None,
    ) -> H3ContinuationResult:
        """Generate all remaining plan segments in order and assemble suffixes."""
        pipeline_kwargs = dict(pipeline_kwargs or {})
        self._validate_controls(pipeline_kwargs, scope="pipeline")
        self._validate_controls(plan.global_controls, scope="global")
        if state is None:
            state = ContinuationState(
                plan_id=plan.plan_id,
                global_prompt=plan.global_prompt,
                config=self.config,
                base_seed=base_seed,
                model_identity=self.model_identity,
            )
        elif state.plan_id != plan.plan_id or state.config != self.config:
            raise ValueError("resume state must use the same segment plan id and continuation configuration")
        elif state.next_segment_index > len(plan.segments):
            raise ValueError("resume state segment cursor exceeds the segment plan")

        assembled_video: list[Any] | None = None
        assembled_audio: torch.Tensor | None = None
        assembled_video_latents: torch.Tensor | None = None
        assembled_audio_latents: torch.Tensor | None = None
        memory_buffer = state.memory_buffer
        if self.memory_rollout_plan.enabled and memory_buffer is None:
            memory_buffer = H3MemorySlotBuffer(
                self.memory_rollout_plan,
                overlap_frames=(
                    self.config.overlap_frames
                    if is_masked_av_context_frames(self.config.overlap_frames) else 0
                ),
                observer=self.memory_slot_observer,
            )
            state.memory_buffer = memory_buffer
        appearance_bank = state.appearance_memory
        if appearance_bank is None:
            appearance_bank = H3AppearanceMemoryBank(self.appearance_memory_config)
            state.appearance_memory = appearance_bank
        elif appearance_bank.config != self.appearance_memory_config:
            raise ValueError("resume state appearance memory config must match the runner config")
        # Resuming is defined at the context/state level.  Full assembled media is
        # intentionally not reconstructed from an in-memory tail or a JSON file.
        if state.next_segment_index:
            raise ValueError("resuming a partial run requires supplying assembled media through a future artifact loader")

        for segment_index in range(state.next_segment_index, len(plan.segments)):
            segment = plan.segments[segment_index]
            self._validate_controls(segment.controls, scope=f"segment {segment_index}")
            window = resolve_continuation_window(
                self.config,
                segment_index=segment_index,
                requested_video_frames=segment.requested_frames,
                prior_timeline_video_frames=state.emitted_video_frames,
                segment_id=segment.segment_id,
            )
            seed = self.seed_for_segment(state.base_seed, segment_index)
            prompt = plan.resolved_prompt(segment_index)
            if self.window_synchronizer is not None:
                self.window_synchronizer(window=window, seed=seed, controls={**plan.global_controls, **segment.controls})
            request = {**pipeline_kwargs, **plan.global_controls, **segment.controls}
            request.update(prompt=prompt, num_frames=window.resolved_video_frames, seed=seed)
            if segment_index > 0:
                bank_references = appearance_bank.build_references(request.get("references"))
                if bank_references is not None:
                    request["references"] = bank_references
            if memory_buffer is not None:
                memory_block = memory_buffer.slots(
                    window_start_frame=window.timeline_start_video_frame,
                    window_end_frame=window.timeline_end_video_frame,
                    encode=lambda frames: self._encode_memory_slot_frames(
                        frames,
                        height=request.get("height"),
                        width=request.get("width"),
                        tiled=bool(request.get("tiled", True)),
                        tile_size=int(request.get("tile_size", 256)),
                        tile_overlap=int(request.get("tile_overlap", 64)),
                    ),
                )
                if memory_block:
                    request["memory_latents"] = memory_block
            has_video_context = state.video_tail is not None
            has_audio_context = state.audio_tail is not None
            use_video_latents = self.prefer_latent_handoff and state.video_latent_tail is not None
            use_audio_latents = self.prefer_latent_handoff and state.audio_latent_tail is not None
            use_context_decode = (
                not self.global_latent_decode
                and segment_index > 0
                and (use_video_latents or use_audio_latents)
                and callable(getattr(self.pipeline, "decode_continuation_suffix", None))
            )
            if use_video_latents:
                request["continuation_video_latents"] = state.video_latent_tail
            elif has_video_context:
                request.update(
                    retake_video=list(state.video_tail),
                    frame_regions_to_retake=[(window.overlap_video_frames, window.resolved_video_frames)],
                )
            if use_audio_latents:
                request["continuation_audio_latents"] = state.audio_latent_tail
            elif has_audio_context:
                request.update(
                    retake_audio=state.audio_tail,
                    retake_audio_sample_rate=self.config.audio_sample_rate,
                    seconds_regions_to_retake=[(window.overlap_seconds, window.window_seconds)],
                )

            if self.prefer_latent_handoff:
                request["return_latents"] = True
            if self.global_latent_decode:
                request["decode_output"] = False
            if use_context_decode:
                request["decode_output"] = False
            try:
                raw_pipeline_output = self.pipeline(**request)
            except TypeError as error:
                # An older one-window pipeline may not expose ``return_latents``.
                # Retake remains a valid, explicit fallback in that case.
                if not self.prefer_latent_handoff or "return_latents" not in request:
                    raise
                request.pop("return_latents")
                request.pop("decode_output", None)
                use_context_decode = False
                raw_pipeline_output = self.pipeline(**request)
            except ValueError:
                # A pipeline can reject a stale/incompatible latent artifact.
                # Preserve the trained Retake path whenever decoded tails still
                # exist; otherwise keep the validation error instead of hiding
                # that neither continuation path can be used.
                if not (use_video_latents or use_audio_latents) or not (has_video_context or has_audio_context):
                    raise
                request.pop("continuation_video_latents", None)
                request.pop("continuation_audio_latents", None)
                if has_video_context:
                    request.update(
                        retake_video=list(state.video_tail),
                        frame_regions_to_retake=[(window.overlap_video_frames, window.resolved_video_frames)],
                    )
                if has_audio_context:
                    request.update(
                        retake_audio=state.audio_tail,
                        retake_audio_sample_rate=self.config.audio_sample_rate,
                        seconds_regions_to_retake=[(window.overlap_seconds, window.window_seconds)],
                    )
                raw_pipeline_output = self.pipeline(**request)
                use_video_latents = use_audio_latents = False
            current_video, current_audio, raw_output = self._pipeline_result(raw_pipeline_output)
            if self.global_latent_decode:
                if not hasattr(raw_output, "video_latents") or not hasattr(raw_output, "audio_latents"):
                    raise TypeError("global latent decode requires a pipeline result with video/audio latents")
                video_steps = h3_continuation_video_latent_steps(window.overlap_video_frames)
                audio_steps = window.overlap_audio_latents
                assembled_video_latents = (
                    raw_output.video_latents.detach().clone()
                    if assembled_video_latents is None
                    else torch.cat([assembled_video_latents, raw_output.video_latents[:, :, video_steps:]], dim=2)
                )
                assembled_audio_latents = (
                    raw_output.audio_latents.detach().clone()
                    if assembled_audio_latents is None
                    else torch.cat([assembled_audio_latents, raw_output.audio_latents[..., audio_steps:]], dim=-1)
                )
                current_video = current_audio = None


            if use_context_decode:
                if not hasattr(raw_output, "video_latents") or not hasattr(raw_output, "audio_latents"):
                    raise TypeError("context decode requires a pipeline result with video/audio latents")
                current_video, current_audio = self.pipeline.decode_continuation_suffix(
                    previous_video_latents=(
                        state.video_latent_tail.latents if use_video_latents else None
                    ),
                    current_video_latents=raw_output.video_latents if use_video_latents else None,
                    previous_audio_latents=(
                        state.audio_latent_tail.latents if use_audio_latents else None
                    ),
                    current_audio_latents=raw_output.audio_latents if use_audio_latents else None,
                    overlap_video_steps=(
                        h3_continuation_video_latent_steps(window.overlap_video_frames)
                    ),
                    overlap_audio_steps=window.overlap_audio_latents,
                    overlap_video_frames=window.overlap_video_frames,
                    overlap_audio_samples=window.overlap_audio_samples,
                    tiled=bool(request.get("tiled", True)),
                    tile_size=int(request.get("tile_size", 256)),
                    tile_overlap=int(request.get("tile_overlap", 64)),
                )
            expected_video_frames = (
                window.resolved_video_frames - window.overlap_video_frames
                if use_context_decode else window.resolved_video_frames
            )
            if current_video is not None and len(current_video) != expected_video_frames:
                raise ValueError(
                    f"pipeline returned {len(current_video)} video frames; expected {expected_video_frames}"
                )
            if current_audio is not None:
                expected_audio_samples = (
                    window.resolved_audio_samples - window.overlap_audio_samples
                    if use_context_decode else window.resolved_audio_samples
                )
                current_audio = normalize_audio_to_timeline(current_audio, expected_audio_samples)
            if memory_buffer is not None and current_video is not None:
                # The slots must be rebuilt from what the rollout actually
                # produced, which is why they are observed here rather than
                # carried over from the reference media.
                memory_buffer.observe(
                    current_video,
                    start_frame=(
                        window.timeline_start_video_frame
                        + (window.overlap_video_frames if use_context_decode else 0)
                    ),
                )
            has_latent_output = (
                hasattr(raw_output, "video_latents")
                or hasattr(raw_output, "audio_latents")
            )
            if segment_index > 0 and current_video is None and current_audio is None and not has_latent_output:
                raise ValueError("neither latent handoff nor decoded Retake produced a usable continuation modality")

            if current_video is not None:
                if use_context_decode:
                    assembled_video = list(assembled_video or []) + list(current_video)
                else:
                    assembled_video = assemble_video_suffix(
                        assembled_video, current_video,
                        window.overlap_video_frames if assembled_video is not None else 0,
                    )
            crossfade_samples = round(self.config.audio_crossfade_ms / 1000 * self.config.audio_sample_rate)
            if current_audio is not None:
                if use_context_decode:
                    assembled_audio = torch.cat([assembled_audio, current_audio], dim=-1)
                else:
                    assembled_audio = assemble_audio_suffix(
                        assembled_audio, current_audio,
                        window.overlap_audio_samples if assembled_audio is not None else 0,
                        crossfade_samples=crossfade_samples if assembled_audio is not None else 0,
                    )

            # Latent tails are state, not decoded media.  This distinction is
            # important for context-parallel inference where non-writer ranks
            # deliberately skip VAE decoding but must still join the next
            # denoising window with the same latent context.
            if current_video is not None:
                state.video_tail = current_video[-self.config.overlap_frames:]
                if segment_index == 0:
                    appearance_bank.initialize_anchors(current_video, segment_index=segment_index)
                else:
                    appearance_bank.update_from_window(
                        current_video,
                        segment_index=segment_index,
                        overlap_frames=window.overlap_video_frames,
                    )
                    if self.appearance_memory_config.boundary_reference_frames and state.video_tail:
                        appearance_bank.set_boundary(
                            state.video_tail[-1],
                            segment_index=segment_index,
                            frame_index=len(current_video) - 1,
                        )
            if self.prefer_latent_handoff and hasattr(raw_output, "video_latents"):
                expected_steps = h3_continuation_video_latent_steps(self.config.overlap_frames)
                state.video_latent_tail = H3VideoLatentContinuation(
                    latents=raw_output.video_latents[:, :, -expected_steps:].detach().clone(),
                        overlap_frames=self.config.overlap_frames,
                        video_fps=self.config.video_fps,
                        clip_length=self.config.video_clip_length,
                        overlap_mode="hard",
                )
            if current_audio is not None:
                tail_samples = round(self.config.overlap_frames / self.config.video_fps * self.config.audio_sample_rate)
                state.audio_tail = current_audio[..., -tail_samples:].detach().clone()
            if self.prefer_latent_handoff and hasattr(raw_output, "audio_latents"):
                expected_steps = round(self.config.overlap_frames / self.config.video_fps * self.config.audio_latent_rate)
                state.audio_latent_tail = H3AudioLatentContinuation(
                    latents=raw_output.audio_latents[..., -expected_steps:].detach().clone(),
                        overlap_frames=self.config.overlap_frames,
                        video_fps=self.config.video_fps,
                        audio_sample_rate=self.config.audio_sample_rate,
                        audio_latent_rate=self.config.audio_latent_rate,
                        overlap_mode="hard",
                )
            # A single physical timeline remains meaningful for video-only and
            # audio-only Retake runs, so advance it even when a modality is
            # intentionally absent from a pipeline result.
            state.emitted_video_frames = window.timeline_end_video_frame
            state.emitted_audio_samples = round(
                state.emitted_video_frames / self.config.video_fps * self.config.audio_sample_rate
            )
            artifacts = self._write_artifacts(window, current_video, current_audio, state)
            state.records.append(H3WindowStateRecord(
                window=window, seed=seed, prompt=prompt, scheduler_controls={**plan.global_controls, **segment.controls},
                model_identity=self.model_identity, artifacts=artifacts,
                continuation_mode=(
                    self.continuation_mode
                    if self.continuation_mode == "masked-av-v14"
                    else ("latent-handoff" if (use_video_latents or use_audio_latents) else "retake-hard")
                ),
                global_reference_frames=len(appearance_bank.anchors),
                appearance_memory_mode=self.appearance_memory_config.mode,
                appearance_memory_frames=appearance_bank.active_reference_count(),
            ))
            state.next_segment_index = segment_index + 1
            self._persist_manifest(state)
        if self.global_latent_decode:
            decoder = getattr(self.pipeline, "decode_latent_timeline", None)
            if not callable(decoder):
                raise TypeError("global latent decode requires pipeline.decode_latent_timeline")
            assembled_video, assembled_audio = decoder(
                video_latents=assembled_video_latents,
                audio_latents=assembled_audio_latents,
                tiled=bool(pipeline_kwargs.get("tiled", True)),
                tile_size=int(pipeline_kwargs.get("tile_size", 256)),
                tile_overlap=int(pipeline_kwargs.get("tile_overlap", 64)),
            )
            if assembled_video is not None:
                assembled_video = list(assembled_video[: state.emitted_video_frames])
            if assembled_audio is not None:
                target_audio_samples = round(
                    state.emitted_video_frames / self.config.video_fps * self.config.audio_sample_rate
                ) + 1
                assembled_audio = normalize_audio_to_timeline(
                    assembled_audio, target_audio_samples
                )
        return H3ContinuationResult(video=assembled_video, audio=assembled_audio, state=state)


def evaluate_continuation_joins(result: H3ContinuationResult) -> dict[str, Any]:
    """Produce dependency-free boundary diagnostics for an assembled run.

    Measurements are intentionally dependency-light diagnostics, not perceptual
    quality claims.  They cover adjacent-frame change at joins, sampled
    long-range appearance/motion drift, and independent audio energy, spectrum,
    and phase diagnostics.
    """
    def frame_tensor(frame: Any) -> torch.Tensor | None:
        try:
            if isinstance(frame, torch.Tensor):
                return frame.detach().float()
            return torch.from_numpy(np.asarray(frame).copy()).float()
        except (TypeError, ValueError, RuntimeError):
            return None

    def frame_difference(first: Any, second: Any) -> float | None:
        first_tensor, second_tensor = frame_tensor(first), frame_tensor(second)
        if first_tensor is None or second_tensor is None or first_tensor.shape != second_tensor.shape:
            return None
        return float((first_tensor - second_tensor).abs().mean())

    joins: list[dict[str, Any]] = []
    records = result.state.records
    for record in records[1:]:
        window = record.window
        frame_boundary = window.timeline_start_video_frame + window.overlap_video_frames
        sample_boundary = round(frame_boundary / window.video_fps * window.audio_sample_rate)
        item: dict[str, Any] = {
            "segment_index": window.segment_index,
            "video_frame": frame_boundary,
            "audio_sample": sample_boundary,
            "time_seconds": frame_boundary / window.video_fps,
            "video_mean_absolute_difference": None,
            "audio_energy_jump": None,
            "audio_spectral_jump": None,
            "audio_phase_jump": None,
        }
        if result.video is not None and 0 < frame_boundary < len(result.video):
            difference = frame_difference(result.video[frame_boundary - 1], result.video[frame_boundary])
            item["video_mean_absolute_difference"] = difference if difference is not None else "unavailable"
        if result.audio is not None and 0 < sample_boundary < result.audio.shape[-1]:
            radius = min(round(0.05 * window.audio_sample_rate), sample_boundary, result.audio.shape[-1] - sample_boundary)
            if radius > 1:
                before = result.audio[..., sample_boundary - radius:sample_boundary].float()
                after = result.audio[..., sample_boundary:sample_boundary + radius].float()
                item["audio_energy_jump"] = float((after.square().mean() - before.square().mean()).abs())
                before_spectrum = torch.fft.rfft(before, dim=-1).abs().mean(dim=0)
                after_spectrum = torch.fft.rfft(after, dim=-1).abs().mean(dim=0)
                item["audio_spectral_jump"] = float((after_spectrum - before_spectrum).abs().mean())
                before_complex = torch.fft.rfft(before, dim=-1).mean(dim=0)
                after_complex = torch.fft.rfft(after, dim=-1).mean(dim=0)
                phase_delta = torch.angle(after_complex) - torch.angle(before_complex)
                # Weight phase by spectral magnitude so near-silent bins do not
                # dominate the reported discontinuity.
                magnitude = (before_complex.abs() + after_complex.abs()) * 0.5
                item["audio_phase_jump"] = float(
                    (torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)).abs() * magnitude).sum()
                    / magnitude.sum().clamp_min(torch.finfo(magnitude.dtype).eps)
                )
        joins.append(item)

    video_metrics: dict[str, Any] = {
        "sampled_frame_count": 0,
        "long_range_mean_absolute_difference": None,
        "sampled_motion_mean_absolute_difference": None,
        "sampled_luminance_standard_deviation": None,
    }
    if result.video:
        sample_count = min(32, len(result.video))
        indices = torch.linspace(0, len(result.video) - 1, sample_count).round().to(torch.long).tolist()
        frames = [frame_tensor(result.video[index]) for index in indices]
        frames = [frame for frame in frames if frame is not None]
        video_metrics["sampled_frame_count"] = len(frames)
        if len(frames) >= 2:
            video_metrics["long_range_mean_absolute_difference"] = frame_difference(frames[0], frames[-1])
            adjacent = [
                frame_difference(first, second)
                for first, second in zip(frames, frames[1:])
            ]
            adjacent = [value for value in adjacent if value is not None]
            if adjacent:
                video_metrics["sampled_motion_mean_absolute_difference"] = float(sum(adjacent) / len(adjacent))
        if frames:
            luminance = [frame.float().mean() for frame in frames]
            video_metrics["sampled_luminance_standard_deviation"] = float(torch.stack(luminance).std(unbiased=False))

    audio_metrics: dict[str, Any] = {
        "rms": None,
        "peak": None,
        "clipping_fraction": None,
    }
    if result.audio is not None and result.audio.numel():
        audio = result.audio.float()
        audio_metrics = {
            "rms": float(audio.square().mean().sqrt()),
            "peak": float(audio.abs().max()),
            "clipping_fraction": float((audio.abs() >= 0.999).float().mean()),
        }
    return {
        "plan_id": result.state.plan_id,
        "global_prompt": result.state.global_prompt,
        "model_identity": result.state.model_identity,
        "base_seed": result.state.base_seed,
        "configuration": json_safe_metadata(result.state.config),
        "experiment_metadata": json_safe_metadata(result.state.experiment_metadata),
        "windows": [json_safe_metadata(record) for record in records],
        "joins": joins,
        "video_metrics": video_metrics,
        "audio_metrics": audio_metrics,
        "assembled_video_frames": len(result.video) if result.video is not None else None,
        "assembled_audio_samples": result.audio.shape[-1] if result.audio is not None else None,
    }


def write_continuation_evaluation(result: H3ContinuationResult, output_path: str | Path) -> dict[str, Any]:
    """Persist an inspectable JSON report after the runner completes."""
    report = evaluate_continuation_joins(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def compare_continuation_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare ablation reports while enforcing a common planning intent."""
    if not reports:
        raise ValueError("at least one continuation report is required")
    names = list(reports)
    reference = reports[names[0]]
    # Overlap and crossfade are intentional ablation variables, so only the
    # plan and seed policy are invariants at this comparison layer.  Every run
    # still retains its full resolved configuration and window records below.
    invariant_keys = ("plan_id", "global_prompt", "model_identity", "base_seed")
    mismatches = {
        name: [key for key in invariant_keys if report.get(key) != reference.get(key)]
        for name, report in reports.items()
    }
    invalid = {name: keys for name, keys in mismatches.items() if keys}
    if invalid:
        raise ValueError(f"ablation reports do not hold planning intent constant: {invalid}")
    summary = {
        "plan_id": reference.get("plan_id"),
        "base_seed": reference.get("base_seed"),
        "runs": {
            name: {
                "model_identity": report.get("model_identity"),
                "configuration": report.get("configuration"),
                "assembled_video_frames": report.get("assembled_video_frames"),
                "assembled_audio_samples": report.get("assembled_audio_samples"),
                "joins": report.get("joins", []),
            }
            for name, report in reports.items()
        },
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
