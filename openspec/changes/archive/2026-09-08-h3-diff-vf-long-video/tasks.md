## 1. Contracts and configuration

- [x] 1.1 Add typed `DiffVFConfig`, resolved-window, global-timeline, RNG-stream, and manifest models with JSON-safe serialization.
- [x] 1.2 Add validation for equal-shaped complete plans, `17n+5` frame lengths, clip-aligned overlap, physical audio/video boundaries, supported modes, and incompatible Retake/resume options.
- [x] 1.3 Add CLI/config plumbing for `diff-vf`, HNI weight, WWS weighting, TES schedule, fusion schedule, audio TES flag, decode strategy, memory budget, and diagnostic output paths.

## 2. Hybrid Noise Initialization

- [x] 2.1 Implement deterministic base/window/TES RNG stream derivation independent of global framework RNG state.
- [x] 2.2 Implement HNI construction for global video and audio noisy latent timelines with configurable constant or per-window `w`.
- [x] 2.3 Add overlap equality, weight-boundary, dtype/shape, and provenance manifest CPU tests.

## 3. Weighted Window Sampling

- [x] 3.1 Extract the existing joint MultiDiffusion loop into a reusable WWS kernel over global video/audio latent timelines.
- [x] 3.2 Implement `center-distance` and `cosine-squared` normalized overlap weight maps with coverage diagnostics.
- [x] 3.3 Ensure one scheduler update per global modality per timestep and preserve H3 prompt/reference encoding semantics for each local window.
- [x] 3.4 Add CPU tests for coverage/normalization and a fixed-seed GPU equivalence smoke against the current joint-multidiffusion prototype.

## 4. Temporal Extended Sampling

- [x] 4.1 Implement deterministic video temporal permutation, sparse-window partitioning, mask handling, and inverse scatter.
- [x] 4.2 Add absolute source-index propagation so H3 position metadata remains tied to original global time coordinates.
- [x] 4.3 Implement explicit TES timestep scheduling with early-noise default and per-step coverage diagnostics.
- [x] 4.4 Keep audio TES disabled by default; add an explicitly labeled experimental audio permutation path with alignment validation.
- [x] 4.5 Add CPU permutation/scatter tests and a short video-only GPU TES smoke.

## 5. Local-global fusion and runner integration

- [x] 5.1 Implement validated timestep coefficient schedules and prediction-space fusion for WWS/TES, separately per modality.
- [x] 5.2 Add the `diff-vf` continuation runner path that initializes HNI, executes WWS/TES/fusion, updates one global latent timeline, and records state.
- [x] 5.3 Preserve existing Retake, latent-handoff, and joint-multidiffusion behavior and add mode-isolation regression tests.
- [x] 5.4 Add failure handling for unsupported heterogeneous windows, missing modality policy, invalid schedules, and incompatible context sources.

## 6. Decode and evaluation

- [x] 6.1 Implement unified global video/audio latent decode and physical timeline cropping without overlap duplication.
- [x] 6.2 Implement decode peak-memory estimation, configured budget guard, and explicit temporal-chunk fallback diagnostics.
- [x] 6.3 Add a reproducible Diff-VF evaluation entrypoint with fixed plan/model/seed variant matrices and per-run manifests.
- [x] 6.4 Add boundary, long-range visual, audio energy/spectral/phase, quality, runtime, and peak-memory metrics with separate modality reports.
- [x] 6.5 Add CPU-safe contract coverage for timelines, fusion, decode decisions, and manifest persistence.

## 7. GPU validation and documentation

- [x] 7.1 Run a fixed 15-second x 2-window GPU ablation for HNI, WWS weighting, TES, fusion, and conservative audio settings.
- [x] 7.2 Compare against Retake-hard, latent-handoff, and current joint-multidiffusion baselines; record seam and quality regressions.
- [x] 7.3 Add user-facing H3 Diff-VF usage documentation, limitations, memory guidance, and experimental-mode warnings.
- [x] 7.4 Run targeted tests, `openspec validate h3-diff-vf-long-video --strict`, and record any deferred multi-GPU or long-horizon validation.
