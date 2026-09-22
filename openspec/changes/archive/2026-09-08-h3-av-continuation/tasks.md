## 1. Contracts and CPU-safe foundations

- [x] 1.1 Add typed segment-plan, resolved-window, continuation-state, and state-manifest data models with JSON-safe metadata serialization.
- [x] 1.2 Implement pure helpers that resolve `17n+5` video lengths, validate clip-aligned overlap, and derive audio sample/latent boundaries from one physical timeline.
- [x] 1.3 Implement pure video-frame and waveform suffix assembly helpers, including an optional duration-preserving equal-power audio crossfade.
- [x] 1.4 Add CPU-safe unit tests for plan validation, resolved timeline arithmetic, partial-clip rejection, suffix ownership, crossfade duration, and state serialization.

## 2. Retake-hard continuation runner

- [x] 2.1 Add `H3ContinuationRunner` with first-window generation, segment prompt resolution, deterministic per-window seed policy, and structured state updates.
- [x] 2.2 Implement later-window `retake-hard` requests that use prior video/audio tails as fixed prefixes and regenerate only the resolved suffix.
- [x] 2.3 Support valid video-only and audio-only Retake continuation sources without requiring the absent modality.
- [x] 2.4 Persist per-window run manifest metadata and optional intermediate media artifacts without serializing large tensors into JSON.
- [x] 2.5 Add runner tests using a CPU fake pipeline to verify call order, Retake masks/regions, prompt selection, state advancement, and no duplicated overlap output.

## 3. Pipeline latent handoff

- [x] 3.1 Add an opt-in H3 Pipeline result surface that returns final video/audio latents and resolved output metadata while preserving default `(video, audio)` compatibility.
- [x] 3.2 Define continuation video/audio latent input objects and validate latent channels, spatial shape, temporal extent, VAE clip metadata, FPS, sample rate, and overlap extent before denoising.
- [x] 3.3 Implement latent-tail conditioning as a fixed Retake-equivalent prefix without video/audio decode-reencode, and enforce mutual exclusivity with same-modality external Retake inputs.
- [x] 3.4 Implement runner preference for compatible latent handoff and decoded Retake fallback when latent input is disabled or unavailable.
- [x] 3.5 Add CPU-safe tests for default API compatibility, opt-in latent results, compatible/incompatible latent validation, latent preference, and no-fallback failure diagnostics.

## 4. Inference entrypoints and CP coordination

- [x] 4.1 Add a continuation inference example and segment-plan example under `examples/minimax_h3/model_inference/` with 243-frame / 34-frame-overlap defaults.
- [x] 4.2 Add explicit CLI validation for checkpoint/model identity, window size, overlap, FPS, sample rate, segment-plan ordering, crossfade, and continuation mode.
- [x] 4.3 Integrate the runner with the existing 30-second local CP inference path without changing the default single-window script behavior.
- [x] 4.4 Synchronize CP-group segment index, resolved boundaries, seed, inference-step count, and next-window context strategy; restrict output/manifest writing to the designated rank.
- [x] 4.5 Add logical CP tests for identical window-resolution metadata and writer-only artifact behavior; document any required multi-GPU validation command.

## 5. Evaluation and validation

- [x] 5.1 Add an executable continuation evaluation entrypoint that records model identity, seed policy, segment plan, resolved configuration, and per-window artifacts.
- [x] 5.2 Implement join diagnostics for frame/time coordinates, video continuity, audio energy/spectral discontinuity, and synchronized audio/video boundary coordinates.
- [x] 5.3 Implement comparison reporting for Retake-hard, latent handoff, overlap-size, and crossfade ablations while holding plan/checkpoint/prompt/seed policy explicit.
- [x] 5.4 Add test fixtures that run the evaluation planning and report generation without CUDA or a loaded H3 checkpoint.
- [x] 5.5 Run targeted unit tests and `openspec validate h3-av-continuation --strict`; record any unrun GPU, CP, audio-quality, and long-horizon validation as deferred work.

## 6. Deferred follow-up work

- [x] 6.0 Add an opt-in latent-velocity `motion-handoff` experiment that predicts and softly projects the first video suffix tokens during denoising; validate it with CPU tests before GPU comparison.
- [x] 6.1 Out of scope: continuation LoRA/SFT is moved to `h3-continuation-lora-training`; this change remains training-free and does not enable a progressive per-token noise schedule.
- [x] 6.2 Specify local overlap `taper-refine` semantics and GPU cost/quality validation before exposing multi-window latent/noise-prediction blending.
- [x] 6.3 Evaluate long-term visual/audio anchor strategies and define explicit scene-transition behavior before adding compressed feature memory.
- [x] 6.4 Investigate explicit two-sided semantic conditioning only for demonstrated middle-dialogue text-placement or prosody failures.
- [x] 6.5 Add an opt-in global joint MultiDiffusion experiment with shared video/audio noise, overlap-prediction blending, single global VAE decode, CPU contract tests, and a bounded GPU comparison command.
