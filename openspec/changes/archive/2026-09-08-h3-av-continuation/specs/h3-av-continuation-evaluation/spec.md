## ADDED Requirements

### Requirement: Reproducible continuation evaluation
The project SHALL provide an executable continuation evaluation entrypoint that accepts a segment plan, model identity, seed, window configuration, overlap configuration, and output directory and records the resolved run configuration.

#### Scenario: Same configuration is reproducible
- **WHEN** the evaluation is run twice with the same checkpoint, plan, seed, resolved configuration, and deterministic runtime settings
- **THEN** it SHALL record matching planned boundaries and per-window seed assignments for both runs

#### Scenario: Invalid configuration is diagnosed
- **WHEN** an evaluation receives an invalid window shape, overlap, or segment plan
- **THEN** it SHALL fail before model inference with a diagnostic identifying the invalid field

### Requirement: Boundary continuity diagnostics
The evaluation SHALL report video, audio, and joint boundary diagnostics for every assembled window join.

#### Scenario: Video boundary diagnostics are emitted
- **WHEN** an assembled run contains at least one join
- **THEN** the report SHALL include frame/time boundary locations and a video continuity measurement or explicitly marked unavailable metric

#### Scenario: Audio boundary diagnostics are emitted
- **WHEN** an assembled run contains at least one join
- **THEN** the report SHALL include sample/time boundary locations and audio energy or spectral discontinuity measurements or explicitly marked unavailable metrics

#### Scenario: Joint timing is emitted
- **WHEN** an assembled run contains at least one join
- **THEN** the report SHALL include the video-frame and audio-sample coordinates derived from the same physical boundary time

### Requirement: Baseline and ablation comparison
The evaluation SHALL support comparisons across hard Retake continuation, latent handoff when available, overlap sizes, and audio crossfade settings without changing the segment plan.

#### Scenario: Overlap ablation has comparable timeline intent
- **WHEN** an evaluator runs two valid overlap configurations for the same segment plan
- **THEN** the report SHALL record each resolved overlap and hold prompt, checkpoint, seed policy, and requested segment order constant

#### Scenario: Latent handoff comparison names the fallback
- **WHEN** latent handoff is compared with decoded Retake continuation
- **THEN** the report SHALL identify which runs used latent handoff and which used the Retake fallback

### Requirement: Testable CPU-safe planning coverage
The continuation implementation SHALL include CPU-safe tests for shape resolution, overlap validation, suffix ownership, timeline accounting, state serialization, and latent compatibility checks.

#### Scenario: No-GPU tests validate timeline assembly
- **WHEN** the continuation test suite runs without a loaded H3 checkpoint or CUDA device
- **THEN** it SHALL validate the deterministic planning and assembly contracts without attempting model denoising

#### Scenario: GPU quality validation remains explicit
- **WHEN** the change is marked implementation-complete
- **THEN** unrun multi-window GPU quality, CP, and long-horizon evaluations SHALL be recorded as explicit validation work rather than assumed from CPU-safe tests
