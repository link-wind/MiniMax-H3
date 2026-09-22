"""Diff-VF style scheduling primitives for MiniMax-H3 continuation.

The functions in this module are deliberately independent from model loading.
They define the global time-line, noise streams, window weights, temporal
extended sampling permutation, prediction fusion, and decode resource guard.
The H3 runner can use them from CPU contract tests before a checkpoint is
loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class DiffVFConfig:
    """Validated controls for one complete Diff-VF run."""

    hni_weight: float = 0.5
    wws_weighting: str = "cosine-squared"
    tes_enabled: bool = False
    tes_window_steps: int = 0
    tes_stride: int = 1
    tes_timesteps: tuple[int, ...] = ()
    fusion_local_start: float = 0.25
    fusion_local_end: float = 0.85
    audio_tes: str = "disabled"
    decode_strategy: str = "unified"
    max_decode_memory_gib: float | None = None
    sampling_semantics: str = "existing"
    paper_hni_innovation_weight: float = 0.2
    paper_tes_interleave_count: int = 0
    paper_tes_early_fraction: float = 0.9
    paper_fusion_alpha: float = 0.5
    paper_fusion_power: float = 6.0

    def validate(self) -> "DiffVFConfig":
        if not 0.0 <= self.hni_weight <= 1.0:
            raise ValueError(f"hni_weight must be in [0, 1], got {self.hni_weight}")
        if self.wws_weighting not in {"center-distance", "cosine-squared"}:
            raise ValueError(f"unsupported wws_weighting={self.wws_weighting!r}")
        if self.tes_window_steps < 0 or self.tes_stride <= 0:
            raise ValueError("tes_window_steps must be non-negative and tes_stride must be positive")
        if any(int(step) != step or step < 0 for step in self.tes_timesteps):
            raise ValueError("tes_timesteps must contain non-negative integer timestep ids")
        if not 0.0 <= self.fusion_local_start <= 1.0 or not 0.0 <= self.fusion_local_end <= 1.0:
            raise ValueError("fusion local coefficients must be in [0, 1]")
        if self.audio_tes not in {"disabled", "experimental"}:
            raise ValueError("audio_tes must be 'disabled' or 'experimental'")
        if self.audio_tes == "experimental" and not self.tes_enabled:
            raise ValueError("experimental audio_tes requires tes_enabled=True")
        if self.decode_strategy not in {"unified", "temporal-chunk"}:
            raise ValueError("decode_strategy must be 'unified' or 'temporal-chunk'")
        if self.max_decode_memory_gib is not None and self.max_decode_memory_gib <= 0:
            raise ValueError("max_decode_memory_gib must be positive when provided")
        if self.sampling_semantics not in {"existing", "paper-strict"}:
            raise ValueError("sampling_semantics must be 'existing' or 'paper-strict'")
        if not 0.0 <= self.paper_hni_innovation_weight <= 1.0:
            raise ValueError("paper_hni_innovation_weight must be in [0, 1]")
        if self.paper_tes_interleave_count < 0:
            raise ValueError("paper_tes_interleave_count must be non-negative")
        if not 0.0 < self.paper_tes_early_fraction <= 1.0:
            raise ValueError("paper_tes_early_fraction must be in (0, 1]")
        if not 0.0 <= self.paper_fusion_alpha <= 1.0:
            raise ValueError("paper_fusion_alpha must be in [0, 1]")
        if self.paper_fusion_power <= 0.0:
            raise ValueError("paper_fusion_power must be positive")
        if self.sampling_semantics == "paper-strict":
            if self.wws_weighting != "center-distance":
                raise ValueError("paper-strict sampling requires wws_weighting='center-distance'")
            if not self.tes_enabled:
                raise ValueError("paper-strict sampling requires tes_enabled=True")
            if self.audio_tes != "disabled":
                raise ValueError("paper-strict sampling keeps audio_tes disabled")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class DiffVFWindow:
    """A resolved global latent interval."""

    index: int
    start: int
    end: int
    overlap_start: int = 0
    overlap_end: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start

    def validate(self, timeline_length: int) -> "DiffVFWindow":
        if self.index < 0 or not 0 <= self.start < self.end <= timeline_length:
            raise ValueError(f"invalid Diff-VF window {self}")
        if not self.start <= self.overlap_start <= self.overlap_end <= self.end:
            raise ValueError(f"invalid overlap bounds in window {self}")
        return self


@dataclass(frozen=True)
class DiffVFTimeline:
    """Complete, equal-shaped window plan used by WWS/TES."""

    length: int
    windows: tuple[DiffVFWindow, ...]
    audio_length: int | None = None
    audio_latent_length: int | None = None

    def validate(self) -> "DiffVFTimeline":
        if self.length <= 0 or not self.windows:
            raise ValueError("Diff-VF timeline requires a positive length and at least one window")
        for window in self.windows:
            window.validate(self.length)
        lengths = {window.length for window in self.windows}
        if len(lengths) != 1:
            raise ValueError("Diff-VF requires equal-shaped windows")
        coverage = [0] * self.length
        for window in self.windows:
            for position in range(window.start, window.end):
                coverage[position] += 1
        if any(value == 0 for value in coverage):
            raise ValueError("Diff-VF windows leave uncovered timeline tokens")
        if self.audio_length is not None and self.audio_length <= 0:
            raise ValueError("audio_length must be positive")
        if self.audio_latent_length is not None and self.audio_latent_length <= 0:
            raise ValueError("audio_latent_length must be positive")
        return self


@dataclass(frozen=True)
class DiffVFManifest:
    config: DiffVFConfig
    seed: int
    timeline: DiffVFTimeline
    rng_streams: Mapping[str, int]
    tes_permutation: tuple[int, ...] = ()
    tes_windows: tuple[tuple[int, ...], ...] = ()
    tes_active_steps: tuple[int, ...] = ()
    fusion_coefficients: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    experimental: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "timeline": {
                "length": self.timeline.length,
                "audio_length": self.timeline.audio_length,
                "audio_latent_length": self.timeline.audio_latent_length,
                "windows": [asdict(window) for window in self.timeline.windows],
            },
            "rng_streams": dict(self.rng_streams),
            "tes_permutation": list(self.tes_permutation),
            "tes_windows": [list(window) for window in self.tes_windows],
            "tes_active_steps": list(self.tes_active_steps),
            "fusion_coefficients": {
                key: list(value) for key, value in self.fusion_coefficients.items()
            },
            "experimental": self.experimental,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


@dataclass(frozen=True)
class TESWindow:
    """Sparse TES window with explicit validity mask and source coordinates."""

    source_indices: tuple[int, ...]
    valid_length: int

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(index < self.valid_length for index in range(len(self.source_indices)))


def build_tes_windows(length: int, *, window_steps: int, stride: int = 1) -> tuple[TESWindow, ...]:
    """Build deterministic sparse windows without duplicating final tokens."""
    permutation = temporal_permutation(length, stride)
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    windows = []
    for start in range(0, len(permutation), window_steps):
        indices = permutation[start:start + window_steps]
        valid_length = len(indices)
        padded = tuple(indices) + (-1,) * (window_steps - valid_length)
        windows.append(TESWindow(padded, valid_length))
    flattened = [index for window in windows for index in window.source_indices[:window.valid_length]]
    if sorted(flattened) != list(range(length)):
        raise RuntimeError("TES windows must cover every source index exactly once")
    return tuple(windows)


def build_paper_tes_windows(
    length: int, *, interleave_count: int, window_steps: int | None = None,
) -> tuple[TESWindow, ...]:
    """Build the N exact Diff-VF interleaved sequences ``n, n+N, ...``."""
    if length <= 0 or interleave_count <= 0:
        raise ValueError("paper TES length and interleave_count must be positive")
    longest = math.ceil(length / interleave_count)
    if window_steps is None:
        window_steps = longest
    if window_steps < longest:
        raise ValueError("paper TES window_steps cannot truncate an interleaved sequence")
    windows = []
    for offset in range(interleave_count):
        indices = tuple(range(offset, length, interleave_count))
        windows.append(TESWindow(indices + (-1,) * (window_steps - len(indices)), len(indices)))
    flattened = [index for window in windows for index in window.source_indices[:window.valid_length]]
    if sorted(flattened) != list(range(length)):
        raise RuntimeError("paper TES windows must cover every source index exactly once")
    return tuple(windows)


def tes_timestep_active(
    step_index: int,
    total_steps: int,
    *,
    configured_timesteps: Sequence[int] = (),
    early_fraction: float = 1.0 / 3.0,
) -> bool:
    """Return whether TES runs at a denoising step."""
    if total_steps <= 0 or not 0 <= step_index < total_steps:
        raise ValueError("step_index must be inside total_steps")
    if configured_timesteps:
        return step_index in {int(value) for value in configured_timesteps}
    if not 0.0 < early_fraction <= 1.0:
        raise ValueError("early_fraction must be in (0, 1]")
    return step_index < max(1, math.ceil(total_steps * early_fraction))


def validate_audio_tes_alignment(
    video_length: int,
    audio_length: int,
    *,
    video_fps: int = 24,
    audio_latent_rate: int = 40,
) -> int:
    """Validate and return the rounded audio/video latent rate ratio."""
    if video_length <= 0 or audio_length <= 0 or video_fps <= 0 or audio_latent_rate <= 0:
        raise ValueError("audio TES lengths and rates must be positive")
    expected = round(video_length / video_fps * audio_latent_rate)
    if expected != audio_length:
        raise ValueError(
            "audio TES timeline is not aligned with video physical time: "
            f"expected {expected} latent steps for {video_length} video steps, got {audio_length}"
        )
    return expected


def derive_rng_seed(base_seed: int, stream: str, index: int = 0) -> int:
    """Derive a stable 63-bit seed without mutating global RNG state."""
    payload = f"{int(base_seed)}:{stream}:{int(index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _generator(seed: int, device: torch.device | str) -> torch.Generator:
    device = torch.device(device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def hni_initialize(
    shape: Sequence[int],
    windows: Sequence[DiffVFWindow],
    *,
    hni_weight: float,
    seed: int,
    time_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Create one global HNI tensor with shared values in every overlap.

    A single global innovation tensor is used so overlap values are never
    independently re-sampled.  Window-specific weights select which mixture
    applies to a position; the first covering window owns an overlap position.
    """
    if not 0.0 <= hni_weight <= 1.0:
        raise ValueError(f"hni_weight must be in [0, 1], got {hni_weight}")
    shape = tuple(int(value) for value in shape)
    resolved_time_dim = time_dim if time_dim >= 0 else len(shape) + time_dim
    if not 0 <= resolved_time_dim < len(shape) or shape[resolved_time_dim] <= 0:
        raise ValueError("time_dim must identify a positive temporal dimension")
    timeline_length = shape[resolved_time_dim]
    timeline = DiffVFTimeline(timeline_length, tuple(windows)).validate()
    base_seed = derive_rng_seed(seed, "hni-base")
    innovation_seed = derive_rng_seed(seed, "hni-innovation")
    base = torch.randn(shape, generator=_generator(base_seed, device), device=device, dtype=dtype)
    innovation = torch.randn(shape, generator=_generator(innovation_seed, device), device=device, dtype=dtype)
    weights = torch.zeros(timeline_length, device=device, dtype=dtype)
    assigned = torch.zeros(timeline_length, device=device, dtype=torch.bool)
    for window in timeline.windows:
        indices = torch.arange(window.start, window.end, device=device)
        pending = ~assigned.index_select(0, indices)
        owned = indices[pending]
        weights.index_copy_(0, owned, torch.full_like(owned, hni_weight, dtype=dtype))
        assigned.index_fill_(0, owned, True)
    view = [1] * len(shape)
    view[resolved_time_dim] = timeline_length
    result = base * weights.reshape(view).sqrt() + innovation * (1.0 - weights.reshape(view)).sqrt()
    return result, {"base": base_seed, "innovation": innovation_seed}


def paper_hni_permutation(length: int, clip_count: int, clip_index: int) -> tuple[int, ...]:
    """Return Diff-VF's cyclic within-group source mapping for one clip."""
    if length <= 0 or clip_count <= 0 or not 0 <= clip_index < clip_count:
        raise ValueError("paper HNI requires positive length/count and a valid clip index")
    mapping = []
    for position in range(length):
        group_start = position // clip_count * clip_count
        group_offset = position % clip_count
        group_length = min(clip_count, length - group_start)
        mapping.append(group_start + (group_offset + clip_index) % group_length)
    return tuple(mapping)


def paper_hni_initialize(
    local_shape: Sequence[int],
    windows: Sequence[DiffVFWindow],
    *,
    innovation_weight: float,
    seed: int,
    time_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Compose a global timeline using Diff-VF's hierarchical noise formula."""
    if not 0.0 <= innovation_weight <= 1.0:
        raise ValueError("innovation_weight must be in [0, 1]")
    local_shape = tuple(int(value) for value in local_shape)
    resolved_time_dim = time_dim if time_dim >= 0 else len(local_shape) + time_dim
    if not 0 <= resolved_time_dim < len(local_shape) or local_shape[resolved_time_dim] <= 0:
        raise ValueError("time_dim must identify a positive temporal dimension")
    if not windows:
        raise ValueError("paper HNI requires at least one window")
    local_length = local_shape[resolved_time_dim]
    if any(window.length != local_length for window in windows):
        raise ValueError("paper HNI local shape must match every window length")
    timeline = DiffVFTimeline(max(window.end for window in windows), tuple(windows)).validate()
    base_seed = derive_rng_seed(seed, "paper-hni-base")
    base = torch.randn(local_shape, generator=_generator(base_seed, device), device=device, dtype=dtype)
    output_shape = list(local_shape)
    output_shape[resolved_time_dim] = timeline.length
    result = torch.zeros(output_shape, device=device, dtype=dtype)
    assigned = torch.zeros(timeline.length, device=device, dtype=torch.bool)
    streams = {"base": base_seed}
    for window in timeline.windows:
        if window.index == 0:
            mixed = base
        else:
            innovation_seed = derive_rng_seed(seed, "paper-hni-innovation", window.index)
            innovation = torch.randn(local_shape, generator=_generator(innovation_seed, device), device=device, dtype=dtype)
            mixed = math.sqrt(1.0 - innovation_weight) * base + math.sqrt(innovation_weight) * innovation
            streams[f"innovation_{window.index}"] = innovation_seed
        mapping = torch.tensor(
            paper_hni_permutation(local_length, len(timeline.windows), window.index),
            device=device, dtype=torch.long,
        )
        permuted = mixed.index_select(resolved_time_dim, mapping)
        global_indices = torch.arange(window.start, window.end, device=device)
        pending = ~assigned.index_select(0, global_indices)
        if torch.any(pending):
            target_indices = global_indices[pending]
            local_indices = torch.arange(local_length, device=device)[pending]
            result.index_copy_(resolved_time_dim, target_indices, permuted.index_select(resolved_time_dim, local_indices))
            assigned.index_fill_(0, target_indices, True)
    if not bool(torch.all(assigned)):
        raise RuntimeError("paper HNI windows leave global tokens uninitialized")
    return result, streams


def window_weights(
    window_length: int,
    overlap: int,
    *,
    strategy: str = "cosine-squared",
    is_first: bool = False,
    is_last: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return one local window's temporal prediction weights."""
    if window_length <= 0 or overlap < 0 or overlap >= window_length:
        raise ValueError("window_length must be positive and overlap in [0, window_length)")
    if strategy not in {"center-distance", "cosine-squared"}:
        raise ValueError(f"unsupported WWS weighting strategy {strategy!r}")
    weights = torch.ones(window_length, dtype=dtype, device=device)
    if overlap == 0:
        return weights
    if strategy == "cosine-squared":
        ramp = torch.linspace(0.0, math.pi / 2, overlap, dtype=dtype, device=device)
        if not is_first:
            weights[:overlap] = torch.sin(ramp).square()
        if not is_last:
            weights[-overlap:] = torch.cos(ramp).square()
    else:
        positions = torch.arange(overlap, dtype=dtype, device=device)
        left = (positions + 1.0) / overlap
        right = (overlap - positions) / overlap
        if not is_first:
            weights[:overlap] = left
        if not is_last:
            weights[-overlap:] = right
    return weights


def build_wws_weight_map(
    timeline: DiffVFTimeline,
    *,
    strategy: str = "cosine-squared",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[num_windows, timeline_length]`` weights and coverage mass."""
    timeline.validate()
    result = torch.zeros((len(timeline.windows), timeline.length), dtype=dtype, device=device)
    for index, window in enumerate(timeline.windows):
        # bfloat16 cannot represent every position on a long global timeline.
        # Compute temporal weights in fp32 so a valid endpoint does not round
        # into a zero center-distance weight before the final dtype cast.
        local = torch.ones(window.length, dtype=torch.float32, device=device)
        left_overlap = max(0, timeline.windows[index - 1].end - window.start) if index else 0
        right_overlap = max(0, window.end - timeline.windows[index + 1].start) if index + 1 < len(timeline.windows) else 0
        if strategy == "cosine-squared":
            if left_overlap:
                ramp = torch.linspace(0.0, math.pi / 2, left_overlap, dtype=torch.float32, device=device)
                local[:left_overlap] = torch.sin(ramp).square()
            if right_overlap:
                ramp = torch.linspace(0.0, math.pi / 2, right_overlap, dtype=torch.float32, device=device)
                local[-right_overlap:] = torch.cos(ramp).square()
        elif strategy == "center-distance":
            positions = torch.arange(window.start, window.end, dtype=torch.float32, device=device)
            center = (window.start + window.end - 1) / 2.0
            local = (window.length + 1.0) / 2.0 - (positions - center).abs()
        else:
            raise ValueError(f"unsupported WWS weighting strategy {strategy!r}")
        result[index, window.start:window.end] = local.to(dtype=dtype)
    mass = result.sum(dim=0)
    if torch.any(mass <= 0):
        missing = torch.nonzero(mass <= 0, as_tuple=False).flatten().tolist()
        intervals = ", ".join(f"[{window.start}, {window.end})" for window in timeline.windows)
        raise RuntimeError(
            "WWS weight map leaves uncovered tokens "
            f"{missing[:16]} across windows {intervals}"
        )
    return result / mass.unsqueeze(0), mass


def merge_window_predictions(
    predictions: Sequence[torch.Tensor],
    windows: Sequence[DiffVFWindow],
    *,
    timeline_length: int,
    time_dim: int,
    strategy: str = "cosine-squared",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate local predictions into one global prediction timeline."""
    if len(predictions) != len(windows) or not predictions:
        raise ValueError("predictions and windows must be non-empty and have equal length")
    first = predictions[0]
    resolved = time_dim if time_dim >= 0 else first.ndim + time_dim
    if not 0 <= resolved < first.ndim:
        raise ValueError("time_dim outside prediction dimensions")
    if any(pred.shape != first.shape for pred in predictions):
        raise ValueError("all local predictions must have equal shape")
    if first.shape[resolved] != windows[0].length:
        raise ValueError("prediction temporal extent does not match the first window")
    timeline = DiffVFTimeline(timeline_length, tuple(windows)).validate()
    weights, mass = build_wws_weight_map(
        timeline, strategy=strategy, dtype=first.dtype, device=first.device
    )
    output_shape = list(first.shape)
    output_shape[resolved] = timeline_length
    merged = torch.zeros(output_shape, device=first.device, dtype=first.dtype)
    view = [1] * first.ndim
    view[resolved] = first.shape[resolved]
    for index, (prediction, window) in enumerate(zip(predictions, windows)):
        local_weight = weights[index, window.start:window.end].reshape(view)
        merged.narrow(resolved, window.start, window.length).add_(prediction * local_weight)
    mass_shape = [1] * first.ndim
    mass_shape[resolved] = timeline_length
    return merged, mass


def merge_window_states(
    states: Sequence[torch.Tensor],
    windows: Sequence[DiffVFWindow],
    *,
    timeline_length: int,
    time_dim: int,
) -> torch.Tensor:
    """Fuse independently scheduler-stepped WWS states with paper weights."""
    merged, _ = merge_window_predictions(
        states, windows, timeline_length=timeline_length, time_dim=time_dim,
        strategy="center-distance",
    )
    return merged


def temporal_permutation(length: int, stride: int) -> tuple[int, ...]:
    """Interleave distant time positions using a deterministic stride."""
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    order: list[int] = []
    for offset in range(stride):
        order.extend(range(offset, length, stride))
    if sorted(order) != list(range(length)):
        raise RuntimeError("TES permutation is not a bijection")
    return tuple(order)


def sparse_windows(permutation: Sequence[int], window_steps: int) -> tuple[tuple[int, ...], ...]:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    return tuple(
        tuple(permutation[start:start + window_steps])
        for start in range(0, len(permutation), window_steps)
    )


def inverse_scatter(values: torch.Tensor, indices: Sequence[int], *, time_dim: int) -> torch.Tensor:
    """Scatter sparse TES values back to their original time coordinates."""
    resolved = time_dim if time_dim >= 0 else values.ndim + time_dim
    if not 0 <= resolved < values.ndim:
        raise ValueError("time_dim outside tensor dimensions")
    if len(indices) != values.shape[resolved]:
        raise ValueError("indices length must equal sparse tensor time extent")
    output = torch.empty_like(values)
    for source, target in enumerate(indices):
        if target < 0:
            raise ValueError("TES indices must be non-negative")
        output.narrow(resolved, int(target), 1).copy_(values.narrow(resolved, source, 1))
    return output


def fusion_coefficient(step_index: int, total_steps: int, *, local_start: float, local_end: float) -> float:
    if total_steps <= 0 or not 0 <= step_index < total_steps:
        raise ValueError("step_index must be inside total_steps")
    if not 0.0 <= local_start <= 1.0 or not 0.0 <= local_end <= 1.0:
        raise ValueError("fusion coefficients must be in [0, 1]")
    if total_steps == 1:
        return float(local_end)
    alpha = step_index / (total_steps - 1)
    return float(local_start + (local_end - local_start) * alpha)


def paper_fusion_coefficient(
    step_index: int,
    total_steps: int,
    *,
    alpha: float = 0.5,
    power: float = 6.0,
) -> float:
    """Return Diff-VF's early-to-late global-state coefficient ``c(t)``."""
    if total_steps <= 0 or not 0 <= step_index < total_steps:
        raise ValueError("step_index must be inside total_steps")
    if not 0.0 <= alpha <= 1.0 or power <= 0.0:
        raise ValueError("paper fusion alpha must be in [0, 1] and power positive")
    progress = 0.0 if total_steps == 1 else step_index / (total_steps - 1)
    return float(alpha * (0.5 * (1.0 + math.cos(math.pi * progress))) ** power)


def fuse_states(
    local: torch.Tensor,
    global_state: torch.Tensor | None,
    *,
    global_coefficient: float,
) -> torch.Tensor:
    """Fuse Diff-VF's independently stepped local and global state paths."""
    if global_state is None:
        return local
    if local.shape != global_state.shape:
        raise ValueError("local and global states must have identical shapes")
    if not 0.0 <= global_coefficient <= 1.0:
        raise ValueError("global_coefficient must be in [0, 1]")
    return (1.0 - global_coefficient) * local + global_coefficient * global_state


def fuse_predictions(
    wws: torch.Tensor,
    tes: torch.Tensor | None,
    *,
    coefficient: float,
) -> torch.Tensor:
    """Fuse noise predictions; missing TES is never treated as zero."""
    if tes is None:
        return wws
    if wws.shape != tes.shape:
        raise ValueError("WWS and TES predictions must have identical shapes")
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("fusion coefficient must be in [0, 1]")
    return wws * coefficient + tes * (1.0 - coefficient)


def estimate_decode_memory_gib(
    latent_shape: Sequence[int], *, bytes_per_element: int = 2, workspace_multiplier: float = 2.0
) -> float:
    if any(int(value) <= 0 for value in latent_shape):
        raise ValueError("latent_shape must contain positive dimensions")
    if bytes_per_element <= 0 or workspace_multiplier <= 0:
        raise ValueError("bytes_per_element and workspace_multiplier must be positive")
    elements = math.prod(int(value) for value in latent_shape)
    return elements * bytes_per_element * workspace_multiplier / (1024 ** 3)


def check_decode_budget(estimate_gib: float, config: DiffVFConfig) -> str:
    config.validate()
    if estimate_gib < 0:
        raise ValueError("decode memory estimate must be non-negative")
    if config.max_decode_memory_gib is None or estimate_gib <= config.max_decode_memory_gib:
        return config.decode_strategy
    if config.decode_strategy == "temporal-chunk":
        return "temporal-chunk"
    raise MemoryError(
        f"unified Diff-VF decode estimate {estimate_gib:.3f} GiB exceeds "
        f"budget {config.max_decode_memory_gib:.3f} GiB"
    )
