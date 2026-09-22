# h3-continuation-dataset Specification

## Purpose
TBD - created by archiving change h3-continuation-lora-training. Update Purpose after archive.
## Requirements
### Requirement: Stream and normalize source annotations
The dataset builder SHALL stream the source JSONL line by line, preserve source identifiers, and emit a JSON-safe sample index without loading the complete annotation file or copying source media.

#### Scenario: Large JSONL is processed incrementally
- **WHEN** the builder processes `data_with_face_and_speech_and_caption.jsonl`
- **THEN** peak annotation memory remains bounded by the configured batch size and each emitted record contains source paths and identifiers

#### Scenario: Media normalization metadata is recorded
- **WHEN** a sample is accepted for training
- **THEN** its index records 24 fps video, 32 kHz audio, resolved frame/sample counts, and the VAE/cache version

### Requirement: Reject cross-shot continuation samples
The dataset builder SHALL create continuation pairs only when the complete history, transition band, and suffix lie inside one original shot with valid boundaries.

#### Scenario: Shot cut intersects a proposed window
- **WHEN** a proposed window overlaps a parsed shot cut or an unknown boundary
- **THEN** the builder rejects the continuation sample and records a machine-readable rejection reason

#### Scenario: Clean single-shot interval is long enough
- **WHEN** one shot contains at least the required window length
- **THEN** the builder may emit one or more deterministic continuation samples from that shot

### Requirement: Preserve sequence-level split isolation
The builder SHALL assign all samples from one `sequence_id` to exactly one of train, validation, or test splits.

#### Scenario: Adjacent windows share a sequence
- **WHEN** multiple windows are generated from the same source sequence
- **THEN** they appear in the same split and cannot leak across split manifests

### Requirement: Produce clip-aligned multimodal latent cache
The cache builder SHALL encode normalized video and audio with the configured H3 VAEs and validate the `17n+5` video contract plus physical audio alignment before writing a cache record.

#### Scenario: Invalid video length is requested
- **WHEN** a window length does not satisfy `17n+5`
- **THEN** cache generation fails with a validation error before VAE encoding

#### Scenario: Video and audio boundaries are resolved
- **WHEN** a valid 24 fps window is cached
- **THEN** audio samples and audio latent steps are derived from the same physical start/end time and stored in the cache metadata

