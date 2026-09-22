## Context

`joint_multidiffusion()` builds prompt and H3 `packed` metadata once for each local window.  During denoising it replaces that window's local latent tensors with a view of the global latent timeline, but leaves the precomputed `packed` positions unchanged.  Consequently every regular WWS window has local video/audio coordinates despite referring to a later global interval.  TES already reconstructs `packed` from explicit global indices, exposing the intended mechanism and the WWS/TES inconsistency.

H3's video VAE decodes the final global latent timeline in temporal clips of 17 output frames.  It is not the source of a repeating semantic discontinuity, but it makes an unstable later latent trajectory visibly recur at its decode cadence.

## Goals / Non-Goals

**Goals:**

- Bind every WWS model call to the actual continuous global video and audio source coordinates of the window it reads.
- Rebuild independent positive and negative CFG `packed` metadata without changing prompt, reference, conditioning-anchor or scheduler semantics.
- Provide CPU-level evidence that the second window's positions differ from local-origin positions and advance continuously over the global timeline.
- Preserve TES's existing sparse absolute-index behavior and all non-Diff-VF generation paths.

**Non-Goals:**

- This change does not modify H3 DiT/VAE weights, RoPE scaling, scheduler parameters, HNI, WWS weighting or decode implementation.
- This change does not promise to eliminate all long-horizon drift or replace continuation-specific training.
- This change does not support absolute source indices for ref-block layouts unless that layout already supports the necessary position semantics.

## Decisions

### Rebuild packed metadata at the WWS prediction boundary

Before each WWS CFG forward, derive dense index ranges from `video_start`, `audio_start`, `local_video_steps`, `local_audio_steps` and the total timeline lengths.  Rebuild `packed` for each CFG condition using the same final H3 packing unit used by TES.  This gives the DiT the same position metadata as the global latent view supplied to the call.

Rebuilding once at window initialization is insufficient: the required global offset is only known after the complete timeline and stride are resolved.  Mutating the original positive `packed` for both CFG branches is also incorrect because text lengths and token tags belong to each condition.

### Guard unsupported reference-block packing

Absolute source indices are currently implemented for FL2VA packing, not `ref_blocks`.  The normal WWS path SHALL fail before denoising when a request combines reference blocks with nonzero global offset, rather than silently resetting positions.  The initial GPU control experiment uses text-only/FL2VA requests.  Extending ref-block position construction is a separate change.

### Preserve TES packing as an independent sparse path

TES continues to create source-indexed `packed` metadata from its sparse permutation.  WWS uses dense ranges and does not share or cache TES metadata, because the token sequence shape and coordinates differ.

### Test metadata rather than model outputs on CPU

CPU tests inspect the generated position IDs and source-index arguments rather than instantiate H3 weights.  A GPU fixed-seed two-window control then tests the integrated forward path.  This keeps the correctness check deterministic and isolates the coordinate contract from stochastic visual quality.

## Risks / Trade-offs

- [Reference-block users cannot use nonzero-offset WWS] -> Fail clearly before denoising; retain existing sequential/reference paths.
- [Repacking every window and timestep adds CPU/device-transfer overhead] -> Pack construction is small relative to a DiT forward; consider per-window caching only after correctness and profiling.
- [Absolute H3 RoPE positions outside the short training horizon reduce quality] -> This repair establishes coordinate consistency but does not address long-horizon RoPE extrapolation; keep the first evaluation to two windows.
- [A remaining flicker is attributed solely to VAE] -> Compare the repaired and prior variants using the same seed/settings and 17-frame cadence metrics before changing decode code.
