"""ARCHIVED / DEPRECATED: MiniMax-H3 taper (taper-refine) latent-anchor helpers.

These helpers implemented the discontinued `taper-refine` continuation family
(see archive/docs and archive/outputs/discontinued_methods).  The current
production continuation modes are masked-av-v14 / hard / latent-handoff, which
treat the overlap prefix as a fully protected hard history (no taper ramp).  In
those modes `_make_overlap_mask(mode="hard")` returns `(mask, None)`, so the
taper weight/projection helpers below were no-ops and are archived rather than
kept alive in the active pipelines.

Retained (active) equivalents live in
`diffsynth/pipelines/minimax_h3_continuation.py`: ``apply_video/audio_latent_continuation``
and the ``hard`` branch of ``_make_overlap_mask``.  Motion-handoff anchors
(``video_motion_anchor`` / ``project_video_motion_anchor``) are still in use and
are NOT archived.
"""



from __future__ import annotations

from typing import Any

import torch


def video_taper_anchor_weights(
    continuation: H3VideoLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
) -> torch.Tensor | None:
    """Return v2 per-token anchor weights after the clip-aligned hard core."""
    prefix_steps = validate_video_latent_continuation(
        continuation, target_latents,
        resolved_video_frames=resolved_video_frames,
        expected_overlap_frames=continuation.overlap_frames,
    )
    _, weights = _make_overlap_mask(
        target_latents.shape[2], prefix_steps, continuation.overlap_mode,
        continuation.min_anchor_weight, continuation.max_anchor_weight,
        continuation.hard_core_frames // continuation.clip_length * continuation.tokens_per_clip,
        target_latents.device, target_latents.dtype,
    )
    return weights




def audio_taper_anchor_weights(
    continuation: H3AudioLatentContinuation,
    target_latents: torch.Tensor,
    *,
    resolved_video_frames: int,
) -> torch.Tensor | None:
    """Return v2 per-token anchor weights on the audio latent time axis."""
    prefix_steps = validate_audio_latent_continuation(
        continuation, target_latents,
        resolved_video_frames=resolved_video_frames,
        expected_overlap_frames=continuation.overlap_frames,
    )
    _, weights = _make_overlap_mask(
        target_latents.shape[-1], prefix_steps, continuation.overlap_mode,
        continuation.min_anchor_weight, continuation.max_anchor_weight,
        round(continuation.hard_core_frames / continuation.video_fps * continuation.audio_latent_rate),
        target_latents.device, target_latents.dtype,
    )
    return weights




def _project_taper_anchor(
    latents: torch.Tensor,
    scheduler: Any,
    timestep: torch.Tensor | None,
    anchor_clean: torch.Tensor | None,
    anchor_noise: torch.Tensor | None,
    weights: torch.Tensor | None,
    *,
    time_dim: int,
) -> torch.Tensor:
    """Project a noisy, timestep-matched history anchor onto a latent prefix.

    The anchor is intentionally applied only to the overlap prefix.  Keeping
    the suffix untouched lets the current window retain its own stochastic
    continuation.  ``None`` weights are the fast path used by hard and plain
    latent-handoff modes.
    """
    if anchor_clean is None or anchor_noise is None or weights is None:
        return latents
    prefix_steps = min(int(weights.numel()), latents.shape[time_dim], anchor_clean.shape[time_dim])
    if prefix_steps <= 0:
        return latents
    if anchor_clean.shape != anchor_noise.shape:
        raise ValueError("taper anchor clean/noise tensors must have identical shapes")
    if anchor_clean.shape[time_dim] < prefix_steps:
        raise ValueError("taper anchor is shorter than its weight vector")
    if timestep is None:
        noisy_anchor = anchor_clean
    else:
        noisy_anchor = scheduler.add_noise(anchor_clean, anchor_noise, timestep)
    shape = [1] * latents.ndim
    shape[time_dim] = prefix_steps
    blend = weights[:prefix_steps].to(device=latents.device, dtype=latents.dtype).reshape(shape)
    result = latents.clone()
    current = latents.narrow(time_dim, 0, prefix_steps)
    anchor = noisy_anchor.narrow(time_dim, 0, prefix_steps).to(device=latents.device, dtype=latents.dtype)
    result.narrow(time_dim, 0, prefix_steps).copy_(current * (1 - blend) + anchor * blend)
    return result




def project_video_taper_anchor(
    latents: torch.Tensor,
    scheduler: Any,
    timestep: torch.Tensor | None,
    anchor_clean: torch.Tensor | None,
    anchor_noise: torch.Tensor | None,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    return _project_taper_anchor(
        latents, scheduler, timestep, anchor_clean, anchor_noise, weights, time_dim=2
    )




def project_audio_taper_anchor(
    latents: torch.Tensor,
    scheduler: Any,
    timestep: torch.Tensor | None,
    anchor_clean: torch.Tensor | None,
    anchor_noise: torch.Tensor | None,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    return _project_taper_anchor(
        latents, scheduler, timestep, anchor_clean, anchor_noise, weights, time_dim=2
    )



# ---------------------------------------------------------------------------
# derive_rng_seed: seed-derivation helper formerly used by the discontinued
# FreeNoise-style shared-noise timeline (`shared_noise` on H3ContinuationRunner,
# removed from the active pipeline).  It produced a stable 63-bit per-stream
# seed for drawing the deterministic global noise timeline.  Only that caller
# existed, so it is archived here with the shared-noise feature.
# ---------------------------------------------------------------------------
from dataclasses import dataclass  # noqa: F401  (kept for archive readability)


def derive_rng_seed(base_seed: int, stream: str, index: int = 0) -> int:
    """Derive a stable 63-bit seed without mutating global RNG state."""
    import hashlib

    payload = f"{int(base_seed)}:{stream}:{int(index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
