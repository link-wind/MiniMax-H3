# h3-cp-aware-sft-training Specification

## Purpose
TBD - created by archiving change h3-30s-sft-ring-cp. Update Purpose after archive.
## Requirements
### Requirement: CP group receives the same training sample
The H3 SFT dataloader SHALL hand the same sample to every rank in a CP group while allowing different samples across DP groups.

#### Scenario: Sampler maps ranks to replicated samples
- **WHEN** a dataloader is configured with CP size 2 and DP size 2
- **THEN** rank pairs `(0,1)` and `(2,3)` each receive the same sample within their pair and independent samples between pairs

#### Scenario: Accelerate preparation remains safe
- **WHEN** `accelerator.prepare` is applied to the CP-aware dataloader
- **THEN** it does not break the CP group replication guarantee

### Requirement: Random sources are synchronized inside a CP group
The H3 SFT loss SHALL use identical timestep, video noise, and audio noise on every rank in a CP group.

#### Scenario: Leader broadcasts timestep and noises
- **WHEN** the CP leader samples `timestep_video`, `timestep_audio`, video noise, and audio noise
- **THEN** every CP rank receives the same tensors before the model forward

#### Scenario: Different DP groups may sample independently
- **WHEN** two different DP groups run the same training step
- **THEN** they are not required to use identical random tensors

### Requirement: Loss reduction is CP-correct
The H3 SFT loss SHALL preserve the current video mean plus audio mean semantics after CP sharding.

#### Scenario: Uneven local shard counts do not bias loss
- **WHEN** video and audio target elements are distributed unevenly across CP shards
- **THEN** the globally reduced loss equals the non-CP loss within floating tolerance

#### Scenario: Video and audio are reduced separately
- **WHEN** the loss is reduced across CP ranks
- **THEN** local loss numerators and valid element counts are reduced separately for video and audio before combining

### Requirement: Parameter gradient integration covers all CP shards
The training integration SHALL ensure optimizer updates use parameter gradients from every CP shard, including all model parameters.

#### Scenario: ZeRO-3 strategy is explicit
- **WHEN** training runs with DeepSpeed ZeRO-3
- **THEN** the process group strategy either uses the global group with CP-correct loss normalization or uses a DP-only parameter state group with explicit CP gradient reduction

#### Scenario: All parameter gradients are reduced
- **WHEN** a CP training step completes
- **THEN** gradients for MLP, QKV, out projection, embedding projections, and other trainable parameters include all CP shard contributions

### Requirement: 30-second SFT launch path is documented
The training entrypoint SHALL expose CP size and provide a runnable command shape for 30-second H3 SFT.

#### Scenario: CPU-safe setup is checked without launching training
- **WHEN** a user validates the 30-second SFT launch configuration
- **THEN** configuration, sampler, loss, and model compatibility checks can run without GPU training

#### Scenario: Actual GPU launch is deferred
- **WHEN** this change is completed
- **THEN** actual 30-second GPU SFT launch and training-curve validation remain explicit deferred tasks
