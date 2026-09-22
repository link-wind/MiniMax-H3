## ADDED Requirements

### Requirement: Weighted Window Sampling
The runner SHALL perform local window noise prediction from a shared global latent state at every configured diffusion timestep and SHALL merge all predictions covering each global token before one scheduler update per modality.

#### Scenario: Overlap predictions are normalized
- **WHEN** two or more windows cover a global token at a timestep
- **THEN** the merged prediction SHALL be the non-negative normalized weighted average of all covering window predictions

#### Scenario: Global scheduler advances once
- **WHEN** WWS finishes prediction accumulation for a timestep
- **THEN** each enabled modality SHALL execute exactly one scheduler step on the complete global latent timeline

### Requirement: Configurable WWS weighting
The runner SHALL support `center-distance` and `cosine-squared` overlap weighting with explicit normalization and a manifest record of the selected strategy.

#### Scenario: Weight strategy is selected
- **WHEN** a caller selects a supported WWS weighting strategy
- **THEN** the runner SHALL use that strategy for every applicable overlap and expose the effective weight map for diagnostics

#### Scenario: Unsupported strategy is rejected
- **WHEN** a caller supplies an unknown WWS weighting strategy
- **THEN** the runner SHALL fail before denoising

### Requirement: Sequential compatibility is preserved
WWS SHALL be an explicit Diff-VF mode and SHALL NOT change the default behavior of Retake, latent handoff, or existing joint MultiDiffusion requests.

#### Scenario: Existing mode remains unchanged
- **WHEN** a caller selects an existing continuation mode
- **THEN** the runner SHALL bypass Diff-VF HNI/WWS scheduling and retain that mode's current validation and assembly semantics
