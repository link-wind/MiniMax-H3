## Why

`taper-refine-v2` can preserve a local overlap but intermittently produces
persistent flicker in the second window. Its clean hard core, independently
sampled successor noise, and per-step transition projection create a condition
distribution that the base H3 model was not trained to denoise. The existing
v2 LoRA pilot remains a baseline, but it is not suitable for scaling until the
inference and training contracts are aligned.

## What Changes

- Add a v3 latent-continuation contract that uses one shared temporal noise
  field across adjacent windows.
- Replace clean-latent reinsertion during v3 denoising with scheduler-aligned
  noisy history anchors for both video and audio.
- Construct directional A/B training examples: the tail of window A conditions
  the overlapping prefix of its later window B.
- Train and evaluate a v3 LoRA only after CPU contracts and multi-seed base
  inference establish that the new condition is stable.
- Retain `retake-hard`, latent handoff, and `taper-refine-v2` unchanged as
  existing baselines.

## Capabilities

### New Capabilities

- `h3-noisy-continuation-v3`: Scheduler-aligned, shared-noise audio-video
  latent continuation, paired A/B cache construction, and stability-focused
  evaluation for MiniMax-H3.

### Modified Capabilities

- None.

## Impact

- Affects the MiniMax-H3 continuation runner and audio-video pipeline,
  continuation cache builder, continuation LoRA dataset/loss path, and tests.
- Adds a new opt-in continuation mode; existing modes and the pretrained H3
  checkpoint remain unchanged.
- Requires GPU only for actual H3 inference, VAE cache generation, and LoRA
  training. Noise, scheduler, manifest, and temporal-alignment contracts must
  remain CPU-testable.
