# h3-diffvf-window-position-continuity Specification

## Purpose
TBD - created by archiving change h3-diffvf-absolute-window-positions. Update Purpose after archive.
## Requirements
### Requirement: WWS uses absolute H3 temporal positions
For every normal WWS prediction, the system SHALL construct the H3 video and audio position metadata from the dense global source indices of the global latent slices supplied to that prediction.  The system SHALL provide the complete global video and audio latent lengths when constructing that metadata.

#### Scenario: Second dense window starts after the first window
- **WHEN** a two-window Diff-VF run predicts the second WWS window at a nonzero video and audio timeline offset
- **THEN** its video and audio position metadata SHALL begin at that global offset rather than at local index zero

#### Scenario: Overlapping dense windows predict a shared token
- **WHEN** two WWS windows include the same global video or audio latent token
- **THEN** both model calls SHALL assign that token the same H3 temporal position metadata

### Requirement: CFG branches retain independent source-indexed packing
The system SHALL rebuild source-indexed H3 packing independently for the positive and negative CFG conditions of every WWS window prediction.

#### Scenario: Positive and negative prompts have different packed text lengths
- **WHEN** a WWS call uses CFG conditions with different text-token metadata
- **THEN** each condition SHALL retain its own text positions and token tags while sharing the same global video and audio source indices

### Requirement: Unsupported reference-block offsets fail explicitly
The system SHALL reject a Diff-VF WWS request that needs a nonzero absolute source offset while using a packing layout that cannot encode those source indices.

#### Scenario: Later WWS reference-block window
- **WHEN** a non-initial WWS window contains reference blocks and absolute source-index packing is unsupported for that layout
- **THEN** the system SHALL raise a descriptive error before entering the denoising loop

