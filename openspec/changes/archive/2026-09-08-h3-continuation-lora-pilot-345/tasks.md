## 1. Resize-aware cache

- [x] 1.1 Add `--height` and `--width` to `build_continuation_cache.py` and
  validate 480x832/345-frame defaults against H3 geometry.
- [x] 1.2 Resize video frames before VAE encoding and record resize policy in
  metadata.
- [x] 1.3 Add CPU tests for geometry arguments, temporal arithmetic, and expected
  345-frame latent shape metadata.
- [x] 1.4 Encode at least 10 real 345-frame samples in MiniMax-H3 and verify
  cache roundtrip, audio alignment, and metadata.

## 2. Pilot manifests and cache

- [x] 2.1 Build deterministic 345-frame train/validation/test manifests with
  2000/200/200 item limits and sequence-level isolation.
- [x] 2.2 Add pilot statistics for audio availability, shot duration, prompt
  quality, and rejection reasons.
- [x] 2.3 Build pilot latent caches by split with bounded item counts and stable
  manifest hashes.
- [x] 2.4 Run CPU preflight over all pilot manifests and reject stale or partial
  cache entries.

## 3. LoRA pilot

- [ ] 3.1 Run a 500-1000 step rank-32 bf16 single-GPU continuation pilot with
  345 frames, 39-frame (12-token) overlap, and mask-v14 suffix-only weights.
- [ ] 3.2 Verify checkpoint save/resume, LoRA export, scale=0 equivalence, and
  scale>0 adapter loading.
- [ ] 3.3 Record training loss (terminal tqdm plus optional CSV/TensorBoard),
  video/audio effective token counts, and random seed/config metadata.

## 4. Paired evaluation and decision

- [ ] 4.1 Run base and LoRA on the same 345-frame x 2 single-shot plan with one
  unified latent decode.
- [ ] 4.2 Generate seam, motion, flicker, audio, and overall-quality reports.
- [ ] 4.3 Apply promotion gates and document whether to scale to 5000/500/500.
- [ ] 4.4 Run `openspec validate h3-continuation-lora-pilot-345 --strict` and
  targeted CPU tests; record any deferred GPU or subjective checks.
