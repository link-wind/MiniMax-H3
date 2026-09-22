# h3-ring-context-parallel Specification

## Purpose
TBD - created by archiving change h3-30s-sft-ring-cp. Update Purpose after archive.
## Requirements
### Requirement: CP preserves packed attention semantics
The H3 DiT SHALL compute the same packed varlen attention under Ring CP as the non-CP reference, treating global segments and pad segments consistently.

#### Scenario: Logical CP=2 matches CP=1
- **WHEN** a tiny packed sequence with at least two global segments and a pad segment is processed by CP=1 and logical CP=2
- **THEN** the per-token outputs match within BF16-compatible floating tolerance

#### Scenario: Shard boundary crosses a segment boundary
- **WHEN** a local shard starts or ends inside a global segment
- **THEN** query tokens only attend to key/value tokens in the same global segment

#### Scenario: Pad segment semantics are explicit
- **WHEN** the reference uses `cu_seqlens=[0, used, seq_len]`
- **THEN** the CP implementation either preserves the pad segment as a separate attention segment or documents and tests the new reference semantics

### Requirement: Local shard metadata mapping
The CP forward path SHALL retain global token positions and global segment boundaries while remapping all local indexing.

#### Scenario: Per-token metadata is sharded consistently
- **WHEN** a packed sequence is split into local shards
- **THEN** `cu_seqlens`, `img_pos`, `audio_pos`, `text_pos`, `token_tags`, `inverse_indices`, RoPE positions, and local output positions are each mapped to local indices without changing global semantics

#### Scenario: K/V chunk positions are available to the mask
- **WHEN** attention consumes a remote K/V chunk
- **THEN** the chunk's global start/end and segment membership are available for local segment masking

### Requirement: Ring attention gradients
The Ring CP backward path SHALL produce gradients for every parameter that receive contributions from every CP shard.

#### Scenario: Parameter gradients match CP=1
- **WHEN** a tiny model computes backward with CP=1 and logical CP=2
- **THEN** gradients for QKV, out projection, MLP, and other trainable parameters match within floating tolerance

#### Scenario: K/V gradients are not limited to local queries
- **WHEN** a key/value shard is attended to by queries from multiple CP shards
- **THEN** the accumulated key/value gradient includes contributions from all query shards before optimizer integration

### Requirement: Gradient checkpoint compatibility
The Ring CP attention SHALL work with the existing gradient checkpointing wrappers used by H3 training.

#### Scenario: Checkpointed forward/backward matches uncheckpointed path
- **WHEN** the same CP configuration runs with gradient checkpointing disabled and enabled
- **THEN** forward outputs and parameter gradients match within floating tolerance

### Requirement: CPU-runnable verification
The Ring CP math and metadata mapping SHALL be verifiable without a GPU or multi-node training run.

#### Scenario: Logical shards run in a single process
- **WHEN** tests execute a logical CP=2 path with explicit shard metadata
- **THEN** all required output and gradient comparisons can be completed without CUDA

#### Scenario: GPU parity is a deferred validation task
- **WHEN** the change is archived
- **THEN** any unrun multi-GPU parity checks are explicitly recorded as deferred GPU validation tasks rather than silently assumed complete
