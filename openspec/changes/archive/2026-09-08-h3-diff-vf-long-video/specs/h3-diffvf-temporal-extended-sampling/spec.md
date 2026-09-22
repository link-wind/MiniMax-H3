## ADDED Requirements

### Requirement: Video temporal extended sampling
The Diff-VF runner SHALL support an opt-in video TES pass that applies a deterministic temporal permutation to the global video latent timeline, denoises sparse windows, and inverse-scatters predictions to the original time coordinates.

#### Scenario: TES covers each token once
- **WHEN** a valid TES permutation is constructed
- **THEN** every source time index SHALL occur exactly once in the permutation and inverse scatter SHALL restore the original index set

#### Scenario: TES handles a short final window
- **WHEN** the permutation length is not divisible by the sparse window length
- **THEN** the final sparse window SHALL use an explicit mask and SHALL NOT duplicate or silently drop time tokens

### Requirement: TES schedule is explicit
The runner SHALL accept a timestep schedule for TES and SHALL record the schedule, permutation seed, sparse window length, and absolute source indices.

#### Scenario: TES is disabled by schedule
- **WHEN** the current timestep is outside the configured TES schedule
- **THEN** the runner SHALL produce no TES prediction and SHALL leave the fusion path to WWS alone

### Requirement: Audio TES is experimental
The runner SHALL disable audio TES by default and SHALL require an explicit experimental flag plus aligned audio permutation metadata before enabling it.

#### Scenario: Default audio path is conservative
- **WHEN** Diff-VF mode is selected without the audio TES experimental flag
- **THEN** audio SHALL use the configured WWS or conservative continuation path and the manifest SHALL state that audio TES was disabled

#### Scenario: Audio TES opt-in is labeled
- **WHEN** a caller enables audio TES experimentally
- **THEN** the runner SHALL mark the run experimental and record independent audio continuity metrics
