## Context

The preceding change implemented the continuation dataset, latent-cache,
taper-refine-v2 loss, LoRA training entrypoint, and evaluation utilities. The
remaining uncertainty is operational: the real source corpus must be encoded
at the same 480x832 and 345-frame geometry used by the planned pilot, and the
result must be reproducible before a longer run is attempted.

## Goals / Non-Goals

**Goals:**

- Produce a deterministic, shot-safe 345-frame pilot manifest.
- Resize video before H3 VAE encoding and verify expected video/audio latent
  shapes and metadata.
- Run a bounded LoRA pilot with resumable checkpoints.
- Compare base and LoRA under identical seeds, plans, and decode paths.

**Non-Goals:**

- No changes to pretrained H3 weights, VAE, scheduler, or continuation
  algorithm.
- No full-corpus cache build or long multi-GPU training in this pilot.
- No new bridge, redraw, or post-processing method is introduced.

## Decisions

### 1. Use 345 frames and one geometry everywhere

The pilot fixes `window_frames=345`, `overlap_frames=39` (12 video latent
tokens / about 65 audio latent steps), `hard_core=12`, `transition=0`,
`fps=24`, audio sample rate 32 kHz, and video resolution
480x832. 345 satisfies H3's `17n+5` temporal constraint and is shot-safe for
the selected corpus. The cache builder SHALL resize decoded frames before VAE
encoding, preserving aspect ratio by center crop/pad according to the H3
pipeline convention.

### 2. Keep cache and manifest deterministic

The source JSONL is streamed and split by `sequence_id`. Pilot selection uses
a fixed seed and stable sample ordering. Cache metadata includes source hash,
window coordinates, geometry, model/VAE identifiers, dtype, and latent shapes;
training MUST reject mismatches.

### 3. Validate before training

At least ten real windows are encoded and round-tripped before the pilot. The
expected shapes at 345 frames and 480x832 are video `[1,24,102,30,52]` and
audio `[2,32,575]` (subject to the installed H3 VAE implementation). A failed
shape, timing, or metadata check blocks training rather than silently falling
back to another configuration.

### 4. Bound the pilot and use paired evaluation

The first run uses 2,000 train, 200 validation, and 200 test items, with a
500-1000 step single-GPU pilot and rank-32 LoRA. The continuation objective is
`masked-av-v14`: the 39-frame overlap is hard-protected and only the suffix is
supervised. Evaluation compares base+mask-v14 and LoRA+mask-v14 using the same 345x2 plan, seed,
overlap, scheduler, and one-time latent decode.

## Risks / Trade-offs

- [Resizing changes crop or face quality] -> record resize policy and inspect
  samples before caching; reject corrupted or extreme-aspect-ratio videos.
- [Cache consumes excessive storage] -> build by split and bounded item count;
  retain manifests and hashes separately from media.
- [Pilot overfits a small subset] -> sequence-level split and fixed held-out
  evaluation; do not promote a checkpoint without overall-quality checks.
- [LoRA improves seam metrics while damaging motion or quality] -> require
  paired seam, motion, visual, and audio gates before expansion.

## Migration Plan

1. Add resize-aware cache arguments and run a 10-item real cache smoke.
2. Build the 2000/200/200 pilot manifests and caches.
3. Run the bounded LoRA pilot and export a checkpoint.
4. Run paired evaluation; retain the base model as fallback if gates fail.
5. Only after passing gates, scale to 5000/500/500 and longer training.

## Open Questions

- Whether center crop or pad is preferable for the corpus's aspect-ratio tails.
- Whether the first pilot needs the planned 75/25 continuation/ordinary-SFT mix;
  the initial run will use continuation-only for attribution.
