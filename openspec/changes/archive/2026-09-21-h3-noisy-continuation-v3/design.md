## Context

The existing latent handoff copies the preceding clean latent tail into the
next window. `taper-refine-v2` additionally projects a timestep-aligned noisy
anchor over its transition band, but retains a clean hard core and initializes
the successor window with independent noise. Full 50-step runs show that this
can perturb the second window's whole denoising trajectory rather than only
the seam. The current continuation LoRA pilot mirrors v2 clean teacher forcing
and therefore cannot establish whether a stable noisy-state condition works.

## Goals / Non-Goals

**Goals:**

- Define a v3 mode in which adjacent windows share a deterministic global
  video/audio noise field at their physical overlap.
- Keep every v3 overlap token at the scheduler's current noise level before a
  DiT prediction, including the exact-history portion.
- Build directional A/B cache records so training identifies the preceding
  window and later target window explicitly.
- Make all timing, noise slicing, masks, and A/B ownership CPU-testable.
- Gate GPU promotion on multi-seed stability rather than one successful video.

**Non-Goals:**

- This change does not replace the production `retake-hard` default.
- This change does not alter pretrained H3 weights, VAE weights, or scheduler
  equations.
- This change does not start 16-GPU training or claim quality improvement
  before the base v3 ablation passes.

## Decisions

### 1. Use a global noise coordinate system

For a two-window timeline, create one deterministic noise field covering the
assembled temporal extent. Each window obtains its initial video/audio noise
by slicing that field at its timeline start. The 34-frame overlap therefore
has identical initial noise in both windows. Audio uses the corresponding
physical-time 40 Hz extent. This replaces independent per-window seed use in
v3 only.

Using one shared seed but regenerating a full local tensor is rejected because
the successor overlap would not equal the predecessor's tail coordinates.

### 2. Condition with scheduler-aligned noisy history

At each denoise timestep `t`, v3 forms `q_t(history_clean, history_noise)`.
The hard region is a weight-1 projection of this noisy history, while the
transition band linearly reduces its history projection. The suffix remains
the successor window's ordinary noisy state. The implementation must not use
the inpaint mask to reinsert a clean history latent at high-noise timesteps.

At the terminal clean state, the hard region is exactly restored from the
history tail. This preserves overlap ownership without presenting clean and
high-noise latent distributions to the DiT in one input.

### 3. Make A/B samples directional

For every eligible shot, construct A `[start, start+window)` and B
`[start+window-overlap, start+2*window-overlap)`. B is the target; only A's
last overlap is history. Both records retain source coordinates, prompt,
split, and shared sequence identity. The split remains sequence-level.

The cache stores A and B latent references rather than duplicating an
unbounded full history tensor. Training loads the B target and A tail. A small
later phase may mix generated/perturbed histories, but that is not required for
the first v3 base-contract implementation.

### 4. Isolate inference validation from LoRA validation

The first GPU gate compares base `latent-handoff`, v2, and v3 on the existing
362-frame structured plan and a seed matrix. Only if v3 avoids full-window
flicker at the base level will a v3 LoRA pilot be trained. The LoRA comparison
uses a fixed plan, seed matrix, decode path, and LoRA scale.

### 5. Preserve absolute H3 temporal positions

The shared-noise timeline and the H3 RoPE timeline must use the same latent
coordinates. Every non-initial v3 window therefore passes its video and audio
source indices, together with the complete global latent lengths, to the FL2VA
packed-sequence builder. The overlap prefix in window B must receive the same
temporal position ids as window A's tail; only its suffix advances to new
global positions. This applies to inference and the corresponding
teacher-forced training contract.

The temporal origin is cached from the first window's positive prompt length
and reused for every later window in the same v3 run. This keeps the original
first-window coordinate scale while preventing a longer or shorter continuation
prompt from translating the physical video/audio timeline. A new source index
zero resets the cache; callers may provide an explicit origin when composing a
custom timeline.

## Risks / Trade-offs

- [H3 Retake expects a clean fixed prefix] -> retain Retake-hard as the
  fallback and compare v3 against it before promotion.
- [Global noise allocation increases memory] -> generate/slice per modality
  deterministically and retain only the local window and overlap noise.
- [A/B pairs reduce eligible samples] -> report pair acceptance/rejection
  separately from single-window counts.
- [Ground-truth histories hide rollout error] -> add perturbed or generated
  history augmentation only after the clean v3 contract is stable.
- [One seed appears stable by chance] -> require a predeclared multi-seed
  report and inspect second-window temporal metrics.

## Migration Plan

1. Add pure CPU helpers and tests for global noise slicing and noisy-history
   projection.
2. Add opt-in `taper-refine-v3`; leave existing modes byte-for-byte unchanged.
3. Add A/B index/cache support and manifest validation.
4. Run the base multi-seed GPU matrix, then a bounded v3 LoRA pilot only if it
   passes its stability gate.
5. Retain v2 artifacts and roll back by selecting `retake-hard` or v2 mode.

## Open Questions

- Whether H3 can extrapolate absolute temporal RoPE positions beyond the
  training window; this is an evaluation risk, not a reason to reset overlap
  positions when using the shared-noise v3 contract.
- What fraction of perturbed/generated history is needed after clean v3
  teacher forcing passes its base stability gate.
