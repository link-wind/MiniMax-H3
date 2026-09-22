## ADDED Requirements

### Requirement: Shared continuation noise timeline
The system SHALL provide an opt-in v3 continuation mode that derives each
window's video and audio initial noise from deterministic global temporal noise
coordinates. Adjacent windows SHALL receive identical initial noise values for
their physical overlap, while existing continuation modes retain their current
noise behavior.

#### Scenario: Adjacent video overlap shares noise
- **WHEN** two 345-frame windows have a 34-frame overlap
- **THEN** their v3 initial video-noise slices are equal for all 10 overlapping
  video latent steps and differ only outside their shared timeline coordinates

#### Scenario: Audio overlap follows physical time
- **WHEN** two v3 windows share 34 video frames at 24 fps
- **THEN** their audio-noise slices are equal for the aligned 40 Hz overlap
  extent

### Requirement: Scheduler-aligned v3 history condition
The v3 continuation mode SHALL present history-conditioned overlap latents at
the active scheduler noise level before each DiT prediction. It SHALL not
reinsert a clean history latent through the denoise mask while the active
timestep is noisy.

#### Scenario: Noisy hard history at a nonterminal timestep
- **WHEN** v3 evaluates an overlap at a nonterminal scheduler timestep
- **THEN** the hard history region equals the scheduler-noised history anchor
  rather than the clean history latent

#### Scenario: Exact terminal history ownership
- **WHEN** v3 reaches the terminal clean state
- **THEN** its hard overlap region equals the exported preceding latent tail

### Requirement: Directional A/B continuation records
The dataset builder SHALL emit an opt-in directional A/B record only when both
windows fit inside one annotated shot. The B target SHALL start exactly one
`window_frames - overlap_frames` stride after A, and A's final overlap SHALL
be the only history supplied to B.

#### Scenario: Valid A/B pair
- **WHEN** a shot has duration at least `2 * window_duration - overlap_duration`
- **THEN** the builder emits a pair with aligned video, audio sample, and audio
  latent boundaries for A and B

#### Scenario: Short shot rejection
- **WHEN** a shot cannot contain both A and B without crossing its boundary
- **THEN** the builder emits no A/B pair and records the rejection reason

### Requirement: Stability-gated v3 evaluation
The evaluation workflow SHALL compare base v3 with existing baselines under a
fixed plan, decode path, and seed matrix before launching a v3 LoRA scale-up.

#### Scenario: Base instability blocks training promotion
- **WHEN** a predeclared seed produces persistent second-window flicker or
  exceeds the configured temporal regression threshold
- **THEN** the workflow records the failure and does not promote the v3 LoRA
  pilot to a larger training run
