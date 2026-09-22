# h3-continuation-lora-evaluation Specification

## Purpose
TBD - created by archiving change h3-continuation-lora-training. Update Purpose after archive.
## Requirements
### Requirement: Run paired continuation comparisons
The evaluator SHALL compare base H3 with continuation LoRA under the same checkpoint, prompt, segment plan, overlap, scheduler settings, seed policy, and single-decode assembly path.

#### Scenario: Baseline and LoRA share generation inputs
- **WHEN** an evaluation case is run for both methods
- **THEN** the report records identical resolved window boundaries and all differing adapter settings explicitly

### Requirement: Report synchronized seam diagnostics
The evaluator SHALL report per-join video, audio, and A/V timing diagnostics for at least the first suffix clip and the full continuation chain.

#### Scenario: Video boundary is analyzed
- **WHEN** a generated join is evaluated
- **THEN** the report includes brightness/color difference, person-box displacement when detectable, motion/optical-flow difference, and temporal flicker statistics

#### Scenario: Audio boundary is analyzed
- **WHEN** audio is available at a generated join
- **THEN** the report includes RMS/响度差, spectral discontinuity, crossfade duration, and audio-video boundary offset in milliseconds

### Requirement: Support ablation and regression gates
The evaluator SHALL support overlap, transition weight, suffix weight, LoRA scale, and sample-category ablations and SHALL flag regressions against the base method.

#### Scenario: Ablation is reproducible
- **WHEN** an ablation matrix is launched
- **THEN** each row has a unique configuration identifier and retains the same source cases and seed policy

#### Scenario: Quality regression is detected
- **WHEN** LoRA improves seam metrics but degrades overall quality beyond configured thresholds
- **THEN** the report marks the configuration as failed rather than presenting it as an improvement

