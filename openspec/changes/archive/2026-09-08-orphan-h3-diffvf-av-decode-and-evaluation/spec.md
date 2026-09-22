## ADDED Requirements

### Requirement: One global timeline decode
The Diff-VF path SHALL decode the completed global video and audio latent timelines once and SHALL assemble output using the same resolved physical boundaries.

#### Scenario: Unified decode has no duplicate overlap
- **WHEN** a complete global latent timeline is decoded
- **THEN** the output frame and audio sample counts SHALL equal the resolved plan duration and SHALL contain no overlap duplication

### Requirement: Decode memory guard
The runner SHALL estimate peak decode memory before decoding and SHALL either execute within the configured budget, use an explicitly selected temporal-chunk strategy, or fail with a diagnostic.

#### Scenario: Budget is insufficient
- **WHEN** the estimated unified decode exceeds the configured memory budget and chunked decode is not selected
- **THEN** the runner SHALL fail before allocating the full decode and report the estimate and budget

### Requirement: Reproducible Diff-VF evaluation
The project SHALL provide an evaluation entrypoint that compares HNI, WWS, TES, fusion, decode strategy, and conservative audio ablations under the same plan, model, scheduler, and seed.

#### Scenario: Evaluation records a run
- **WHEN** an evaluation variant completes
- **THEN** it SHALL write configuration, model identity, RNG policy, elapsed time, peak memory, output paths, and per-join metrics

### Requirement: Boundary and long-range metrics
Evaluation SHALL report video boundary continuity, long-range appearance/motion consistency, audio energy/spectral/phase continuity, output quality, and failure/degradation reasons.

#### Scenario: Audio and video are diagnosed separately
- **WHEN** a variant changes only the video TES or fusion setting
- **THEN** the report SHALL preserve separate video and audio metrics so a visual improvement cannot hide an audio regression

### Requirement: CPU-safe contract coverage
The implementation SHALL include CPU-safe tests for HNI determinism, window coverage, WWS normalization, TES permutation/scatter, fusion coefficients, timeline ownership, decode budget decisions, and manifest serialization.

#### Scenario: Tests run without H3 weights
- **WHEN** the targeted contract suite runs without CUDA or a loaded checkpoint
- **THEN** it SHALL validate pure scheduling and data-contract behavior without invoking model inference
