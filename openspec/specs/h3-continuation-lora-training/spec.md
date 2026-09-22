# h3-continuation-lora-training Specification

## Purpose
TBD - created by archiving change h3-continuation-lora-training. Update Purpose after archive.
## Requirements
### Requirement: Construct timestep-aligned taper-refine-v2 inputs
The continuation training forward SHALL construct hard-core, transition-band, and suffix regions from clean history and target latents using scheduler-consistent noisy anchors at the sampled timestep.

#### Scenario: Transition uses noisy anchor at timestep t
- **WHEN** a training example is sampled at timestep `t`
- **THEN** the transition latent is a weight-ramped mixture of `scheduler.add_noise(history_clean, epsilon_anchor, t)` and the main noisy target, while the hard core remains the clean history context

#### Scenario: Video and audio use one physical boundary
- **WHEN** the video overlap is 34 frames at 24 fps
- **THEN** the audio overlap is derived as 34/24 seconds and converted independently to 32 kHz samples and 40 latent steps

### Requirement: Apply masked and region-weighted flow loss
The trainer SHALL exclude hard-core tokens from loss, apply configurable transition and suffix weights, normalize video and audio losses independently, and combine them with an explicit audio coefficient.

#### Scenario: Hard core is preserved without training pressure
- **WHEN** the loss mask covers the hard-core prefix
- **THEN** its valid loss weight is zero and it contributes no numerator or denominator to the reported loss

#### Scenario: First suffix clip receives higher weight
- **WHEN** a sample has a complete first suffix VAE clip
- **THEN** that clip receives the configured high weight and later suffix tokens receive the normal suffix weight

### Requirement: Train only continuation LoRA parameters
The training entrypoint SHALL freeze the H3 VAEs, text encoders, RoPE, scheduler, and base DiT weights, and SHALL inject LoRA into the configured DiT target modules.

#### Scenario: Default target modules are used
- **WHEN** no target override is supplied
- **THEN** LoRA modules are attached to `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2`, with all non-LoRA parameters frozen

#### Scenario: Checkpoint contains adapter weights
- **WHEN** a training checkpoint or final export is written
- **THEN** it contains LoRA parameters and configuration metadata, and can be loaded on top of the same base checkpoint without changing the default pipeline behavior

### Requirement: Support reproducible and resource-aware training
The trainer SHALL expose deterministic seed policy, bf16/gradient-checkpointing options, optional CP configuration, validation-only mode, and resumable checkpoints.

#### Scenario: CPU validation does not launch training
- **WHEN** the validation-only command is run without CUDA or model weights
- **THEN** it validates dataset schema, window arithmetic, region masks, LoRA target names, and optimizer settings and exits successfully or with actionable errors

#### Scenario: Resume restores continuation configuration
- **WHEN** training resumes from a checkpoint
- **THEN** optimizer state, LoRA weights, global step, seed policy, dataset manifest hash, and region-weight configuration are restored or rejected on mismatch

