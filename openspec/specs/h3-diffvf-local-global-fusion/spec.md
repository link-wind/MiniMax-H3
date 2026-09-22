# h3-diffvf-local-global-fusion Specification

## Purpose
TBD - created by archiving change h3-diff-vf-long-video. Update Purpose after archive.
## Requirements
### Requirement: Local-global prediction fusion
The runner SHALL fuse WWS and available TES noise predictions in prediction space using a timestep-dependent coefficient before applying the scheduler step.

#### Scenario: Both paths are available
- **WHEN** WWS and TES predictions exist for a modality at one timestep
- **THEN** the runner SHALL compute `c(t) * eps_wws + (1-c(t)) * eps_tes` with a validated coefficient in `[0, 1]`

#### Scenario: Only local prediction is available
- **WHEN** TES is disabled or not scheduled for the current timestep
- **THEN** the runner SHALL use WWS as the complete prediction without treating a missing TES tensor as zeros

### Requirement: Modality-specific fusion policy
The runner SHALL allow video and audio to select independent TES enablement and fusion schedules, while preserving the same physical time boundaries.

#### Scenario: Video-only TES
- **WHEN** video TES is enabled and audio TES is disabled
- **THEN** video SHALL use local-global fusion and audio SHALL follow its configured conservative policy without changing the audio timeline

### Requirement: Fusion provenance is auditable
The runner SHALL record per-modality fusion schedules, effective coefficients, prediction coverage, and fallback decisions.

#### Scenario: Fusion manifest is complete
- **WHEN** a Diff-VF run completes or fails after initialization
- **THEN** its manifest SHALL identify whether each timestep used WWS-only or fused prediction for each modality

