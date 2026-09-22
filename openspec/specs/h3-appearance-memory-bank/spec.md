# h3-appearance-memory-bank Specification

## Purpose
TBD - created by archiving change h3-appearance-memory-bank. Update Purpose after archive.
## Requirements
### Requirement: Opt-in appearance memory bank
The continuation system SHALL provide an opt-in appearance memory bank for later H3 windows. When the bank is disabled, the runner SHALL preserve the existing continuation behavior without injecting bank-derived references. When enabled, the bank SHALL be composed of trusted anchors, selected memory frames, and an optional boundary reference, and SHALL enforce the configured reference limit before inference.

#### Scenario: Bank is disabled by default
- **WHEN** a continuation run uses the default configuration without enabling the appearance memory bank
- **THEN** later H3 requests SHALL contain no bank-derived references and SHALL otherwise match the existing runner behavior

#### Scenario: Enabled bank conditions later windows
- **WHEN** a continuation run enables the bank and reaches a window after the first window
- **THEN** the runner SHALL pass the configured trusted anchors, selected memory frames, and optional boundary reference through the existing H3 references surface

#### Scenario: Reference budget is validated
- **WHEN** a caller configures more visual reference frames than the supported maximum
- **THEN** the runner SHALL reject the configuration before model inference and report the configured and maximum counts

### Requirement: Trusted anchors are immutable
The bank SHALL initialize trusted anchors from the first completed window using stable middle frames rather than the final overlap tail. Trusted anchors SHALL remain unchanged for the run and SHALL NOT be replaced or mutated by later windows.

#### Scenario: Middle frames become trusted anchors
- **WHEN** the first window completes with the bank enabled
- **THEN** trusted anchors SHALL be sampled from middle frames of that first window and SHALL exclude the boundary tail

#### Scenario: Later windows cannot rewrite trusted anchors
- **WHEN** later windows complete and a candidate frame differs from the trusted anchors
- **THEN** the candidate SHALL NOT replace or modify any trusted anchor for the current run

#### Scenario: New run starts with an empty bank
- **WHEN** a new continuation run starts
- **THEN** the bank SHALL be reinitialized from that run's first window rather than reusing prior run anchors

### Requirement: Memory frames are selected with diversity and trust
The bank SHALL select a bounded set of memory frames from completed history using deterministic temporal diversity and consistency checks against trusted anchors. Near-duplicate candidates SHALL be excluded, and memory frames SHALL NOT be promoted to trusted anchors.

#### Scenario: Diverse candidates are selected
- **WHEN** a candidate pool contains temporally separated frames with distinct appearance
- **THEN** the bank SHALL return a bounded set of memory frames that preserves temporal spread

#### Scenario: Near-duplicate candidates are skipped
- **WHEN** a candidate is visually close to an existing trusted anchor or selected memory frame
- **THEN** the bank SHALL skip that candidate without increasing the memory frame count

#### Scenario: Drifted candidates do not become anchors
- **WHEN** a later candidate differs substantially from the trusted anchors
- **THEN** the candidate SHALL be eligible only as a non-trusted memory or boundary reference and SHALL NOT enter the trusted anchor set

### Requirement: Boundary reference is separate from the bank
When boundary references are enabled, the runner SHALL add the final clean frame from the current overlap as a boundary reference for the next window. The boundary reference SHALL be treated separately from trusted anchors and SHALL NOT be promoted into the trusted anchor set.

#### Scenario: Boundary uses the latest overlap tail
- **WHEN** a later window has a decoded video tail from the prior window
- **THEN** the next request SHALL include that tail's final frame as the boundary reference

#### Scenario: Boundary does not mutate the anchor set
- **WHEN** the next window completes
- **THEN** the boundary reference from the prior window SHALL NOT be added to the trusted anchor list

### Requirement: Local position semantics are preserved
The appearance memory bank SHALL communicate only through the existing H3 references/ref_blocks conditioning path. The runner SHALL NOT add global video/audio source indices, global latent lengths, or a global temporal position origin for bank-enabled masked-av-v14 windows.

#### Scenario: Bank-enabled request uses local reference path
- **WHEN** the runner constructs a later `masked-av-v14` request with bank references
- **THEN** the request SHALL include references and SHALL NOT include `video_source_indices`, `audio_source_indices`, `global_video_latent_length`, `global_audio_latent_length`, or `global_temporal_position_origin`

#### Scenario: Target positions remain local
- **WHEN** a packed sequence is built with bank references through `_build_packed_ref2va`
- **THEN** target video/audio positions SHALL start after the local text/reference cursor and SHALL NOT depend on a cross-window absolute position accumulator

### Requirement: Bank state metadata is serializable
The continuation state SHALL record bank configuration and provenance metadata in the manifest without embedding media tensors. If bank media cannot be restored from persisted artifacts, the state SHALL remain replay-only rather than claim resumable appearance memory.

#### Scenario: Bank provenance is recorded
- **WHEN** a continuation run completes with the bank enabled
- **THEN** the manifest SHALL record the bank policy, trusted anchor source segment/frame locations, selected memory source segment/frame locations, and boundary-reference mode

#### Scenario: No tensor payload is embedded
- **WHEN** the manifest is serialized to JSON
- **THEN** it SHALL NOT contain PIL images or tensor values and SHALL include only metadata or explicit artifact references

