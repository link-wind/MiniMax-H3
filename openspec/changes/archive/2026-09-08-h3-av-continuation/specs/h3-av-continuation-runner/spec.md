## ADDED Requirements

### Requirement: Multi-window joint continuation
The system SHALL provide an H3 continuation runner that generates one or more joint video/audio windows in segment-plan order and assembles one continuous output timeline.

#### Scenario: Text-only run generates a first window
- **WHEN** a caller starts a continuation run with a valid global prompt and one segment
- **THEN** the runner SHALL invoke H3 without prior Retake context and emit that window as the initial timeline content

#### Scenario: Subsequent segment uses prior tail
- **WHEN** a continuation run contains a segment after the first segment
- **THEN** the runner SHALL construct the next request from the previous emitted video/audio tail and generate only the next suffix as new timeline content

### Requirement: Video overlap is VAE-clip aligned
The runner SHALL resolve video window lengths with the H3 `17n+5` shape rule and SHALL require a positive overlap frame count that is a whole multiple of the active Video VAE clip length and smaller than the resolved window length.

#### Scenario: Valid overlap is accepted
- **WHEN** a 243-frame request uses a 34-frame overlap with a 17-frame Video VAE clip length
- **THEN** the runner SHALL accept the configuration and record the resolved frame boundaries in the continuation state

#### Scenario: Partial VAE clip overlap is rejected
- **WHEN** a caller requests an overlap that is not divisible by the active Video VAE clip length
- **THEN** the runner SHALL fail before model denoising with an error that reports the requested overlap and required clip alignment

### Requirement: Audio and video use one physical timeline
The runner SHALL derive audio overlap samples and audio-latent overlap steps from the resolved video-frame overlap and configured FPS, sample rate, and audio-latent rate rather than accepting an independently drifting audio boundary.

#### Scenario: Audio overlap follows a resolved video overlap
- **WHEN** the resolved overlap is 34 frames at 24 FPS with a 32 kHz audio rate and 40 audio latent/s
- **THEN** the runner SHALL derive an overlap duration of 34/24 seconds and record the corresponding sample and latent boundaries

#### Scenario: Rounded boundary does not change timeline ownership
- **WHEN** sample rounding is required for a non-integer overlap duration
- **THEN** the runner SHALL use the same recorded resolved sample boundary for suffix assembly and optional crossfade without duplicating or omitting output duration

### Requirement: Retake-hard MVP conditioning
The baseline continuation mode SHALL use existing H3 Retake semantics: the overlap prefix is fixed and the post-overlap suffix is regenerable for both supplied modalities.

#### Scenario: Later window fixes the overlap prefix
- **WHEN** the runner builds a later window in `retake-hard` mode
- **THEN** it SHALL provide the prior video/audio tail as Retake input and request regeneration only after the resolved overlap boundary

#### Scenario: One-modality continuation remains supported
- **WHEN** a caller configures a video-only or audio-only continuation source supported by H3 Retake
- **THEN** the runner SHALL preserve the supplied modality as Retake context and SHALL not require an unavailable context for the other modality

### Requirement: Suffix ownership and output assembly
The assembled timeline SHALL retain the previously emitted overlap as the authoritative content and SHALL append only the new suffix from each later window.

#### Scenario: Later windows do not duplicate overlap frames
- **WHEN** two valid windows with an overlap are assembled
- **THEN** the final frame count SHALL equal the first resolved frame count plus the later resolved frame count minus the resolved overlap frame count

#### Scenario: Optional audio crossfade preserves duration
- **WHEN** waveform crossfade is enabled at a join
- **THEN** the runner SHALL apply it within the join neighborhood while preserving the assembled audio sample count

### Requirement: Segment state and deterministic replay metadata
The runner SHALL maintain a serializable continuation state containing global conditions, segment cursor, resolved timeline boundaries, per-window seeds, scheduler controls, and short-term context metadata.

#### Scenario: Segment-specific prompt is selected
- **WHEN** the runner begins a segment with a segment prompt and global prompt
- **THEN** it SHALL construct and record the resolved prompt context for that segment without mutating prior segment records

#### Scenario: Completed run can be inspected
- **WHEN** a window finishes successfully
- **THEN** the runner SHALL persist or return state metadata sufficient to identify its model/checkpoint, seed, resolved boundaries, and segment order

### Requirement: Context-parallel window coordination
The continuation runner SHALL treat context parallelism as an intra-window mechanism and SHALL synchronize window-level state in every CP group.

#### Scenario: CP ranks resolve an identical next window
- **WHEN** a continuation window runs with CP world size greater than one
- **THEN** all ranks in the CP group SHALL use the same segment index, resolved shape, overlap boundaries, seed, and inference-step count

#### Scenario: Only the designated writer emits user output
- **WHEN** a CP continuation window finishes
- **THEN** only the designated writer rank SHALL write the assembled media and state manifest while all ranks retain or receive compatible context for the next window

### Requirement: Opt-in joint MultiDiffusion experiment
The system SHALL provide an explicit `joint-multidiffusion` experimental mode
that denoises a complete set of equal-shaped windows on shared global video and
audio noisy latent timelines.  It SHALL remain distinct from the sequential
Retake and latent-handoff modes.

#### Scenario: Overlap predictions are merged before one global step
- **WHEN** a valid two-or-more-window joint MultiDiffusion run advances one diffusion timestep
- **THEN** the system SHALL obtain one video and one audio prediction per window, blend their overlap predictions with complementary normalized weights, and execute exactly one scheduler update for each global modality

#### Scenario: Incompatible sequential context is rejected
- **WHEN** a joint MultiDiffusion request supplies Retake input, continuation latents, unequal resolved window shapes, resume state, bridge decode, or dynamically extracted reference anchors
- **THEN** the system SHALL fail before denoising with a diagnostic naming the unsupported condition

#### Scenario: The result owns one globally decoded timeline
- **WHEN** a joint MultiDiffusion run completes
- **THEN** the system SHALL decode the global video/audio latent timelines once and crop or normalize the result to the plan's resolved physical output duration
