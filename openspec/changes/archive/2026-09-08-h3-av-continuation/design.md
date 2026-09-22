## Context

MiniMax-H3 currently denoises one joint video/audio window at a time. The pipeline supports FL2VA, Ref2VA, video Retake, audio Retake, independent video/audio schedulers, and packed non-causal attention. Video Retake operates on video-VAE clips of 17 frames; audio Retake operates on a 32 kHz waveform whose VAE latent rate is 40 steps/second. The current pipeline returns decoded video and waveform only, so a naïve continuation implementation would decode a previous window, read it back, and encode its overlap again for every next window.

The repository also contains a 30-second Ring Context Parallel (CP) path. CP shards a single packed sequence across ranks; it does not define how multiple generation windows retain history, map audio/video time, or construct an output timeline.

The continuation design must preserve existing one-window API behavior and make the least-trained MVP possible. It therefore builds first on Retake and only introduces latent handoff as an optional extension. Progressive per-token noise, external audio inpainting, and continuation SFT remain follow-up work rather than implicit behavior changes.  A bounded global MultiDiffusion inference experiment is allowed only as an explicit opt-in diagnostic path.

## Goals / Non-Goals

**Goals:**

- Generate a long, joint video/audio timeline from a sequence of H3 windows.
- Make the next window condition on a clip-aligned tail of the prior output while preserving one physical time axis for video and audio.
- Preserve the existing Retake behavior as the MVP conditioning mechanism.
- Provide a typed, serializable continuation state and deterministic segment-plan contract.
- Add an optional latent handoff path that avoids history decode/re-encode while providing a safe Retake fallback.
- Make CP behavior explicit: CP handles one window; the runner handles window-to-window state transitions.
- Make quality regressions measurable with deterministic synthetic tests and an executable evaluation report.

**Non-Goals:**

- Alter H3 DiT architecture, its trained weights, or existing FL2VA/Ref2VA checkpoint semantics.
- Train a CAM/APM memory module, a continuation LoRA/SFT, or progressive-noise model in this change. Continuation LoRA/SFT is explicitly deferred to `h3-continuation-lora-training`; this change remains training-free only.
- Make full global Temporal MultiDiffusion the default for long-video denoising.
- Add a third-party audio generation/inpainting dependency. Waveform crossfade is only a boundary safeguard.
- Guarantee indefinite no-drift generation or replace dedicated dialogue-editing systems.

## Decisions

### 1. Use a runner above the existing Pipeline

Add `H3ContinuationRunner` rather than embedding window state in `MiniMaxH3Pipeline.__call__`.

```text
Segment plan + ContinuationState
                |
                v
       H3ContinuationRunner
         |             |
         |             +--> build window request / Retake or latent handoff
         v
  MiniMaxH3Pipeline (one joint denoise window)
         |
         v
 append suffix, update state, emit artifacts
```

The pipeline remains a reusable one-window generator. The runner owns timeline arithmetic, state serialization, seed progression, output assembly, segment prompts, and CP synchronization.

Alternative considered: add `continuation_*` state directly to `MiniMaxH3Pipeline.__call__`. This would mix long-running orchestration with the single-window API, complicate every current inference caller, and make checkpoint/retry behavior unclear.

### 2. Define an explicit window contract in physical time

Each plan segment declares target duration, prompt, and optional generation controls. The runner snaps every video window to the existing `17n+5` contract and derives all other boundaries from its resolved frame count.

```text
video_fps = 24
audio_sample_rate = 32000
audio_latent_rate = 40

window_seconds = resolved_video_frames / video_fps
overlap_seconds = overlap_frames / video_fps
audio_overlap_samples = round(overlap_seconds * audio_sample_rate)
audio_overlap_latents = round(overlap_seconds * audio_latent_rate)
```

`overlap_frames` MUST be a positive multiple of the Video VAE clip length (17) and MUST be strictly smaller than the resolved window frame count. The default MVP configuration is 243 frames with 34-frame overlap. Users may select another valid length, including 719-frame 30-second windows.

Alternative considered: accept unrelated video-frame and audio-second overlaps. It is more flexible but permits persistent audio/video timeline drift and makes cross-modal evaluation ambiguous.

### 3. MVP uses Retake with hard suffix ownership

The first window is a normal H3 generation. Every subsequent window receives the previous tail as a Retake prefix. The runner marks that prefix fixed and marks the remaining suffix regenerable. The assembled timeline owns the prior window's overlap; the new window contributes only frames and samples after the overlap.

```text
previous emitted: [ history ---------------------- | overlap ]
next H3 window:                                [ overlap | suffix ]
assembled:        [ history ---------------------- | overlap | suffix ]
```

This avoids averaging independently decoded frames and ensures each output time has one authoritative owner. The runner may optionally apply an equal-power 50-200 ms waveform crossfade around the join, but it MUST not duplicate audio duration.

Alternative considered: decoded-frame crossfade. It can blur motion and changes the visual identity of the overlap; latent/noise refinement is a better future path.

### 4. State is structured and serializable

`ContinuationState` contains:

- immutable run identity: global prompt, optional references/anchors, output configuration and base seed;
- timeline: resolved frame/sample counts, current segment index and physical-time cursor;
- short-term context: emitted video/audio tail and, when available, their latent tails;
- semantic progress: segment-plan identifier and declared dialogue/lyrics/music state;
- reproducibility: per-window seed, scheduler settings, checkpoint/model identity;
- optional artifact paths for recovery.

The state serializer MUST not silently embed large GPU tensors in JSON. It stores tensor metadata and references to an explicit tensor artifact, or marks the run as replay-only when artifacts are not persisted.

Alternative considered: derive all state from the final MP4. This loses latent context, segment prompt history, original scheduler inputs, and reproducibility metadata.

### 5. Latent handoff is optional and backward compatible

Extend the H3 Pipeline with an opt-in result object or opt-in `return_latents=True` that exposes final video/audio latents plus resolved output metadata. Add continuation latent inputs that are mutually exclusive with external `retake_video`/`retake_audio` for the same modality.

The runner prefers latent handoff when both prior latent state and compatible pipeline support exist. Otherwise it falls back to existing Retake using decoded tail frames/waveform. It MUST reject incompatible dtype, shape, device-independent metadata, frame resolution, audio rate, or overlap extent before denoising.

Alternative considered: replace Retake with latent-only inputs. This would remove existing external editing workflows and prevent a no-training MVP.

### 6. Preserve and expose model-native non-causal context for middle Retake

For a middle-region Retake, the existing model already packs fixed audio/video context on both sides of the regenerated region and uses non-causal attention. The runner does not add a UniCATS-style two-sided semantic module. Such a module is deferred to a future precision dialogue-editing change, triggered only by demonstrated text-placement or prosody failures.

### 7. CP is synchronized at window boundaries

In CP mode all ranks in a CP group run exactly the same segment plan, resolved window shape, seed, and number of denoise steps. Rank 0 is the only writer of user-facing output and state manifest. The generated context required by the next window is either retained identically by every rank or broadcast from rank 0 before the next window; the implementation MUST choose one and test it.

The runner does not alter Ring attention or its packed-shard metadata. A single window may use CP=1 or any supported CP configuration.

### 8. Phase advanced features behind explicit modes

The runner exposes a baseline `retake-hard` mode in this change. It reserves named but disabled-by-default extension points for:

- `latent-handoff` for no-reencode continuation;
- `taper-refine` for local overlap noise/latent blending;
- `progressive-noise` for a continuation-trained per-token schedule.

Only `retake-hard` and `latent-handoff` are production-compatible implementation modes. After the GPU ablation in the validation record, `taper-refine` is exposed as an explicit latent-only experiment using a static overlap mask ramp; it is not a production default and does not claim noise-aligned refinement. `progressive-noise` remains unavailable until separately specified and trained.

### 9. Use a small immutable global reference bank for long-term appearance

The runner may opt in to `global_reference_frames` (1-4). After the first
window has been decoded, it samples uniformly spaced middle frames (never the
boundary tail) and passes copies of those frames to every later H3 request as
image `references`. Caller-supplied references are preserved and remain before
the generated anchors. The default is zero, so existing continuation behavior
is unchanged.

These anchors are appearance conditions, not timeline content: they are never
appended to the output, stored in the JSON manifest, or used to replace the
short-term Retake/latent tail. The bank is immutable for one run so a bad later
window cannot overwrite the stable identity anchor. It is intended for a
single scene and stable cast/lighting; an intentional scene, character, or
time-of-day transition must start a new run (or explicitly disable the bank)
instead of carrying the old anchors into the new scene. This MVP does not yet
compress anchors into features or automatically detect scene transitions.

### 10. Add an opt-in latent-velocity boundary experiment

The overlap prefix carries appearance and short-term content, but it does not
specify the derivative of the latent trajectory at the first free suffix token.
The experimental `motion-handoff` mode estimates a robust final video-latent
velocity from recent first differences and predicts only the first few suffix
tokens. During every denoising step those predictions are converted to the
current noisy timestep and softly projected after the fixed overlap.  It keeps
the existing `taper-refine-v2` overlap behavior so its GPU comparison isolates
the first-free-token state constraint from a hard-prefix regression.

This mode is video-only, requires latent handoff, is disabled by default, and
does not claim to replace continuation training. Audio motion/prosody handoff
and optical-flow/pose controls remain separate follow-up experiments so seam
diagnostics can attribute any quality change.

### 11. Add an opt-in global joint MultiDiffusion experiment

`joint-multidiffusion` is a bounded, GPU-only inference experiment for testing
whether the seam originates in divergent diffusion trajectories rather than in
VAE decode or output assembly.  It requires all planned windows to resolve to
the same spatial shape and frame count.  It does not run Retake, latent
handoff, local taper projection, bridge decoding, resume, CP, or dynamically
extracted reference anchors.

```text
global x_T (video and audio)
       |
       +--> local window A --> epsilon_A
       +--> local window B --> epsilon_B
                         |
             complementary cosine-squared blend on overlap
                         |
                one global scheduler step per modality
```

Every window still runs its native prompt/reference encoding and H3 joint
audio/video noise prediction.  At each diffusion timestep, the runner slices
the current global video/audio latents into local windows, predicts each local
noise field, merges their overlap predictions with complementary weights, and
updates each global modality exactly once.  The final global latents are VAE
decoded once and cropped/normalized to the physical frame/sample timeline.

The first implementation is intentionally restricted to a single GPU and a
complete, non-resumable plan.  It should be evaluated against the fixed plan,
checkpoint, seed, and single-decode baseline; it is not production behavior
until it demonstrates better boundary continuity without a material quality or
cost regression.

## Risks / Trade-offs

- [Independent windows accumulate visual, acoustic, and semantic drift] → Start with bounded-window evaluation, carry short-term overlap and global references, and record long-horizon metrics before training a continuation model.
- [Video overlap misaligns with a coupled 17-frame VAE clip] → Validate the requested overlap against the VAE clip length before invoking H3.
- [Audio and video timelines drift due to rounding] → Derive all audio boundaries from resolved video frames and store resolved frame/sample counts in state.
- [Decode/re-encode degrades retained history] → Use Retake only as the MVP; add latent handoff and compare it directly against the fallback.
- [Waveform crossfade hides but does not fix semantic discontinuity] → Limit crossfade to optional boundary polish and report it separately from latent continuity metrics.
- [Reference anchors reduce motion or block legitimate scene changes] → Treat anchors as optional immutable conditions, sample middle frames, and require a new run at intentional scene transitions; include no-anchor/anchor ablations.
- [CP ranks diverge at a window boundary] → Synchronize plan and random inputs, assert matching resolved state metadata, and test logical CP behavior.
- [Large state artifacts make resume fragile] → Write atomic manifests and tensor artifacts per completed window; retain an explicit replay-only fallback.
- [Joint denoising increases peak activation cost and cannot stream completed windows] → Restrict it to equal-shaped, complete single-GPU experiments and retain sequential Retake as the production baseline.

## Migration Plan

1. Add the new runner, plan schema, baseline example, and tests without changing existing H3 Pipeline call behavior.
2. Add opt-in Pipeline latent result/input surfaces with compatibility checks; preserve all Retake argument behavior and defaults.
3. Add a guarded CP adapter around the existing 30-second example after CP=1 behavior is verified.
4. Publish evaluation fixtures and record baseline metrics before enabling any refinement mode.

Rollback consists of removing the new runner invocation or disabling `return_latents`; existing FL2VA, Ref2VA, Retake, and CP single-window scripts continue to call the unchanged default pipeline interface. Generated artifacts are additive and can be ignored by prior tooling.

## Open Questions

- Should a latent handoff artifact contain the complete prior window or only the clip-aligned tail required by the next request?
- For CP, is all-rank retention or rank-0 broadcast more memory-efficient and robust for the target topology?
- What overlap duration is best for dialogue, music, and environment sound separately: one clip (17 frames), two clips (34 frames), or longer?
- What objective boundary metrics correlate best with human assessment of H3 audio joins?
- Does a global image/video/audio reference improve long-horizon identity without inducing video stagnation?
- When a later segment intentionally changes scene, should the plan switch or clear global reference anchors?
