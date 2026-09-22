## 1. Shared-Noise and Noisy-History Contracts

- [x] 1.1 Add CPU-only helpers that derive video/audio window noise from one
  deterministic global temporal noise field.
- [x] 1.2 Add a v3 overlap projector that constructs scheduler-aligned noisy
  history for hard and transition regions without clean inpaint reinsertion.
- [x] 1.3 Add CPU tests for 345/362-frame overlap equality, audio alignment,
  nonterminal noisy history, and exact terminal hard-history ownership.

## 2. Opt-in V3 Inference

- [x] 2.1 Add `taper-refine-v3` as an opt-in continuation mode while preserving
  existing Retake, latent handoff, taper-refine, and v2 behavior.
- [x] 2.2 Wire shared global noise coordinates and v3 scheduler projection into
  the H3 continuation runner and audio-video denoising loop.
- [x] 2.3 Record v3 noise-coordinate, conditioning, and decode metadata in the
  continuation state manifest.
- [x] 2.4 Run targeted CPU regression tests proving existing modes are unchanged.
- [x] 2.5 Pass absolute video/audio source indices and global latent lengths to
  ordinary v3 FL2VA packing, keep a prompt-invariant temporal origin across
  windows, and test that both sides of an overlap receive identical H3 temporal
  position ids.

## 3. Directional A/B Data and Training

- [x] 3.1 Extend the index builder with an opt-in shot-safe A/B pair manifest
  whose B start equals A start plus `window_frames - overlap_frames`.
- [x] 3.2 Extend latent caching and validation to resolve A-tail history and B
  target metadata without cross-split leakage.
- [x] 3.3 Adapt the continuation LoRA collator/loss to reproduce the v3 shared
  noise and noisy-history condition during teacher forcing.
- [x] 3.4 Add CPU tests for pair coordinates, cache schema, v3 masks, weighted
  video/audio loss, and checkpoint configuration validation.

## 4. Evaluation and Promotion

- [ ] 4.1 Run a fixed 362-frame structured base matrix across latent handoff,
  v2, and v3 with the predeclared seed matrix.
- [ ] 4.2 Run a bounded 345-frame A/B v3 LoRA pilot only if the base matrix
  avoids persistent second-window flicker.
- [ ] 4.3 Compare base and v3 LoRA using the same plan, seed matrix, unified
  decode path, seam/motion/flicker/audio metrics, and subjective inspection.
- [ ] 4.4 Document promotion or rejection before any multi-GPU scale-up, then
  run strict OpenSpec validation and targeted tests.
