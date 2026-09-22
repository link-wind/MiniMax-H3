# h3-continuation-latent-handoff Specification

## Purpose
TBD - created by archiving change h3-av-continuation. Update Purpose after archive.
## Requirements
### Requirement: Opt-in H3 latent result
The H3 Pipeline SHALL provide an opt-in result surface that returns final video and audio latents together with resolved output-shape metadata, without changing the default `(video, audio)` return behavior.

#### Scenario: Default callers remain compatible
- **WHEN** a caller invokes the existing H3 Pipeline without requesting latents
- **THEN** it SHALL receive the existing decoded video/audio return shape and behavior

#### Scenario: Opt-in result exposes resolved latents
- **WHEN** a caller requests final latents from a successful H3 inference
- **THEN** the result SHALL include final video latents, final audio latents, resolved frame count, FPS, audio sample rate, and latent time dimensions

### Requirement: Continuation latent input
The H3 continuation interface SHALL accept clip-aligned video-latent and time-aligned audio-latent history for a subsequent window and SHALL apply it as fixed overlap context.

#### Scenario: Compatible latent tail is consumed
- **WHEN** a runner provides a latent tail whose dimensions, modality channels, and overlap extent match the resolved target request
- **THEN** the pipeline SHALL use that tail as the fixed continuation context without decoding and re-encoding it

#### Scenario: Incompatible latent tail fails early
- **WHEN** a caller provides a latent tail with incompatible channel count, spatial shape, time extent, resolved FPS, sample rate, or VAE metadata
- **THEN** the pipeline SHALL reject the request before denoising and identify the incompatible field

### Requirement: Retake fallback remains available
The continuation runner SHALL fall back to decoded Retake conditioning when latent handoff is unavailable, disabled, or incompatible and a valid decoded tail exists.

#### Scenario: MVP fallback runs without latent support
- **WHEN** latent handoff is disabled and the prior window exposes decoded video/audio tail data
- **THEN** the runner SHALL construct the next window through existing Retake inputs

#### Scenario: No valid fallback is reported
- **WHEN** latent handoff is unavailable or rejected and no valid decoded Retake tail exists
- **THEN** the runner SHALL fail before invoking the next window and report that neither continuation path is usable

### Requirement: Latent state persistence is explicit
The system SHALL persist latent continuation context only through explicit tensor artifacts or explicit in-memory state and SHALL not silently serialize large tensors into JSON manifests.

#### Scenario: Persisted state records tensor artifacts
- **WHEN** a runner is configured for resumable latent continuation
- **THEN** its manifest SHALL store artifact paths and tensor metadata required to validate and restore the latent tails

#### Scenario: Replay-only state does not claim latent resume
- **WHEN** a runner does not persist latent tensors
- **THEN** its manifest SHALL mark the state replay-only and SHALL not claim that a latent handoff resume is available

