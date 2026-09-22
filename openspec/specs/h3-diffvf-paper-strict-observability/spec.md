# h3-diffvf-paper-strict-observability Specification

## Purpose
TBD - created by archiving change h3-strict-diffvf-reproduction. Update Purpose after archive.
## Requirements
### Requirement: Strict-run manifest
The system SHALL record a strict Diff-VF run manifest containing the selected sampling semantic, strict HNI innovation weight, WWS weighting, TES interleave plan, actual TES-active steps, state-fusion coefficients, seed streams, and global timeline.

#### Scenario: Reproducible strict run
- **WHEN** a strict Diff-VF request resolves its sampling plan
- **THEN** the returned or persisted manifest SHALL contain sufficient configuration and deterministic seed information to recreate that plan

### Requirement: Unified timeline decode
The system SHALL decode the completed strict global video and audio latent timelines once after all denoising steps.

#### Scenario: Strict multi-window output
- **WHEN** a strict run finishes denoising two or more windows
- **THEN** it SHALL not perform independent per-window VAE decodes or splice decoded window outputs

