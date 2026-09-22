## Why

The continuation LoRA implementation is complete, but it has not yet been
validated on correctly resized 345-frame MiniMax-H3 latent windows. A small,
reproducible pilot is needed to catch video/audio shape, timing, cache, and
training regressions before spending resources on a larger run.

## What Changes

- Build a shot-safe pilot manifest from the existing LongVideoGen JSONL source.
- Resize and encode a small set of 345-frame windows at 480x832 with H3 Video
  and Audio VAE, and validate cache metadata and roundtrips.
- Add a deterministic 2000/200/200 train/validation/test pilot subset.
- Run a short continuation LoRA pilot and save a resumable checkpoint.
- Evaluate base and LoRA with the same 345-frame x 2 plan and report seam and
  overall-quality metrics.

## Capabilities

### New Capabilities

- `h3-continuation-lora-pilot`: Reproducible 345-frame cache, pilot training,
  and paired evaluation workflow for continuation LoRA.

### Modified Capabilities

## Impact

- Uses `build_continuation_dataset.py`, `build_continuation_cache.py`, the H3
  training entrypoint, and the continuation evaluation runner.
- Adds no changes to the pretrained H3 model, VAE, scheduler, or existing
  training-free continuation behavior.
- Requires the MiniMax-H3 environment and GPU only for real VAE encoding,
  training, and inference; manifest and configuration checks remain CPU-safe.
