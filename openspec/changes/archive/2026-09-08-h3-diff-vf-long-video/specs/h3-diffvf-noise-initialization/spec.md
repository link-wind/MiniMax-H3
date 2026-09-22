## ADDED Requirements

### Requirement: Hybrid noise initialization
The Diff-VF runner SHALL construct reproducible initial video and audio noisy latent timelines from a shared base noise stream and configurable per-window innovation noise.

#### Scenario: Deterministic HNI replay
- **WHEN** two runs use the same model identity, plan, base seed, noise dtype, and HNI weight
- **THEN** they SHALL produce identical initial global latent tensors and record the same RNG stream metadata

#### Scenario: HNI weight is validated
- **WHEN** a caller supplies an HNI weight outside the inclusive range `[0, 1]`
- **THEN** the runner SHALL fail before model inference and report the invalid value

### Requirement: Shared global overlap noise
The runner SHALL derive every overlapping window slice from one global initial-noise timeline and SHALL NOT independently resample an overlap slice.

#### Scenario: Overlap has one initial-noise owner
- **WHEN** two resolved windows cover the same physical latent interval
- **THEN** their initial overlap latent views SHALL be equal before the first scheduler step

#### Scenario: Independent suffix innovation
- **WHEN** HNI weight is less than one
- **THEN** non-overlapping window regions SHALL include independently generated innovation noise without changing the shared overlap values

### Requirement: Noise provenance is recorded
The runner SHALL record HNI mode, weight schedule, base/window RNG stream identifiers, dtype, latent shapes, and resolved window slices in the experiment manifest.

#### Scenario: Manifest contains HNI controls
- **WHEN** HNI initialization completes
- **THEN** the manifest SHALL contain enough metadata to distinguish shared-only, mixed, and independent-noise ablations
