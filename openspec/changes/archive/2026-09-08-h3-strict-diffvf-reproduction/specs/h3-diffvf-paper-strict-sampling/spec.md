## ADDED Requirements

### Requirement: Explicit strict sampling mode
The system SHALL expose a `paper-strict` Diff-VF sampling semantic that is opt-in and SHALL preserve the existing prediction-level joint MultiDiffusion semantic as the default.

#### Scenario: Existing experiment remains compatible
- **WHEN** a joint MultiDiffusion request omits the strict semantic
- **THEN** the system SHALL use the established prediction-level merge and scheduler update path

#### Scenario: Strict experiment is requested
- **WHEN** a request selects `paper-strict`
- **THEN** the system SHALL validate strict-only controls and execute state-level HNI, WWS, and TES semantics

### Requirement: Paper HNI initialization
The strict mode SHALL initialize each short clip from the first clip initial noise and a clip-specific innovation according to `sqrt(1-w) * Z_T^0 + sqrt(w) * epsilon_n`, then apply the paper's deterministic cyclic group permutation before composing the global timeline.

#### Scenario: Zero innovation weight
- **WHEN** strict HNI uses `innovation_weight=0`
- **THEN** every pre-permutation clip SHALL equal the first clip initial noise

#### Scenario: Cyclic group mapping
- **WHEN** strict HNI creates a run with N clips
- **THEN** each group position `(b + a*N)` in clip n SHALL read from `((b+n) mod N + a*N)` before global composition

#### Scenario: Overlap ownership is deterministic
- **WHEN** two strict HNI clips cover the same global latent token
- **THEN** the earliest clip in timeline order SHALL own that token's initialized value

### Requirement: State-level WWS
The strict mode SHALL obtain every WWS window state from the same `Z_t`, apply one scheduler update to each local state, and fuse those resulting `Z_(t-1)` states using normalized linear center-distance weights.

#### Scenario: Overlapping WWS states
- **WHEN** a global token is predicted by two or more WWS windows
- **THEN** its local-path state SHALL be the normalized weighted sum of the independently stepped window states

#### Scenario: Strict weight selection
- **WHEN** strict mode is active
- **THEN** the system SHALL reject a WWS weighting strategy other than `center-distance`

### Requirement: Early state-level TES
The strict mode SHALL build TES sequences from interleaved global video indices, independently step them from the same `Z_t`, and scatter each valid result to its original source index to form the global state path.

#### Scenario: Interleaved coverage
- **WHEN** a strict TES plan is built with N interleave groups
- **THEN** every valid global video latent token SHALL appear in exactly one TES sequence and scatter back to its original index

#### Scenario: TES phase selection
- **WHEN** a strict run is in its configured early denoising fraction
- **THEN** the system SHALL compute the TES global state path; outside that phase it SHALL use the local WWS state path alone

### Requirement: Paper state fusion schedule
While strict TES is active, the system SHALL fuse local and global `Z_(t-1)` state paths as `(1-c(t))*Z_local + c(t)*Z_global`, where `c(t)` follows the configured paper cosine schedule and decreases from early to late denoising.

#### Scenario: Initial strict fusion step
- **WHEN** strict TES is active at the first inference step
- **THEN** the global-state contribution SHALL equal the configured initial paper coefficient

#### Scenario: Final strict fusion step
- **WHEN** the inference step is outside the TES phase or the cosine coefficient is zero
- **THEN** the output state SHALL equal the local WWS state

### Requirement: Absolute H3 temporal coordinates
Every strict WWS or TES model forward SHALL use source-indexed H3 packed metadata corresponding to the global video and audio latent tokens supplied to that forward.

#### Scenario: Later WWS window
- **WHEN** strict WWS evaluates a non-initial window
- **THEN** its H3 video and audio temporal positions SHALL begin at that window's global offsets rather than zero

#### Scenario: Unsupported reference layout
- **WHEN** strict sampling requires nonzero source indices for reference blocks that cannot encode them
- **THEN** the system SHALL reject the request before denoising
