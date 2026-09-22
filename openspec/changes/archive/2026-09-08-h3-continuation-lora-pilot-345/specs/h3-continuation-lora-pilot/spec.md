## ADDED Requirements

### Requirement: Resize-aware 345-frame cache construction
The cache builder MUST accept explicit video height and width, resize frames
before H3 VAE encoding, and record the resulting geometry in cache metadata.

#### Scenario: Valid 345-frame cache
- **WHEN** a shot-safe 345-frame sample is encoded with height 480 and width 832
- **THEN** the cache contains video and audio latents plus metadata for 345
  frames, 39-frame overlap, 24 fps, 32 kHz audio, and the encoded geometry

#### Scenario: Invalid geometry or temporal length
- **WHEN** the requested geometry or frame count is incompatible with H3's
  temporal constraints
- **THEN** the builder fails with an actionable validation error before writing
  a training cache

### Requirement: Deterministic pilot split
The pilot dataset MUST select 2,000 train, 200 validation, and 200 test samples
using stable ordering, a fixed seed, and sequence-level isolation.

#### Scenario: Repeated pilot selection
- **WHEN** the indexer is run twice with the same source, seed, and limits
- **THEN** all split manifests and their hashes are identical

#### Scenario: Sequence isolation
- **WHEN** a sequence contributes one or more windows to a pilot split
- **THEN** no window from that sequence appears in another split

### Requirement: Preflight cache validation
Training MUST provide a CPU-safe preflight that validates manifest schema,
cache metadata, latent shapes, region arithmetic, and LoRA configuration.

#### Scenario: Preflight succeeds
- **WHEN** all pilot records use the expected 345-frame geometry and metadata
- **THEN** preflight exits successfully without loading CUDA model weights

#### Scenario: Preflight rejects stale cache
- **WHEN** a cache has a different VAE identifier, shape, dtype, or window length
- **THEN** preflight fails and identifies the mismatch and rebuild action

### Requirement: Paired pilot evaluation
The evaluation runner MUST compare base and LoRA continuation with identical
plan, seed, scheduler, overlap, and decode strategy, and emit machine-readable
seam and overall-quality metrics.

#### Scenario: Comparable evaluation pair
- **WHEN** base and LoRA runs use the same 345-frame x 2 plan
- **THEN** the report records both variants and their shared evaluation inputs

#### Scenario: Promotion gate fails
- **WHEN** LoRA seam metrics improve but visual, motion, or audio quality exceeds
  configured regression thresholds
- **THEN** the report marks the pilot failed and does not recommend expansion
