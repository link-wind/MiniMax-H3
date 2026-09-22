## 1. Core Metadata and Reference

- [x] 1.1 Add a `diffsynth/core/context_parallel/` package with ring CP helpers and tests
- [x] 1.2 Implement global packed metadata and local shard metadata mapping
- [x] 1.3 Implement exact non-CP attention reference using global `cu_seqlens`
- [x] 1.4 Implement logical Ring CP attention with online softmax and segment mask
- [x] 1.5 Add unit tests for local shard mapping across segment boundaries
- [x] 1.6 Add unit tests for CP=1 vs CP=2 forward parity
- [x] 1.7 Add unit tests for CP=1 vs CP=2 parameter gradient parity
- [x] 1.8 Add unit tests for gradient checkpointing compatibility

## 2. H3 Model Integration

- [x] 2.1 Add optional CP metadata parameters to H3 attention and DiT forward without changing CP=1 behavior
- [x] 2.2 Map H3 packed sequence fields to local shard metadata
- [x] 2.3 Route H3 attention to logical Ring CP path when CP is enabled
- [x] 2.4 Add tiny H3 integration tests for CP=1 vs CP=2 forward and gradients

## 3. Training Integration

- [x] 3.1 Add CP-aware sampler/dataloader helper that replicates samples inside a CP group
- [x] 3.2 Add CP leader broadcast for timestep, video noise, and audio noise in H3 loss
- [x] 3.3 Add local numerator/valid-count loss reduction helper for video and audio
- [x] 3.4 Add tests for CP-aware sampler and uneven local loss counts
- [x] 3.5 Add tests for random source synchronization without GPU

## 4. DeepSpeed and Topology Integration

- [x] 4.1 Add CP size and DP size arguments to the H3 training entrypoint and config validation
- [x] 4.2 Document and validate the DeepSpeed ZeRO-3 global-group strategy
- [x] 4.3 Add the parameter gradient reduction integration point for CP groups
- [x] 4.4 Add CPU-safe topology and setup validation tests
- [x] 4.5 Implement custom autograd Ring attention with real CP collectives, local shard output, and K/V gradient reduction
- [x] 4.6 Add a gloo multi-process CPU parity test for distributed Ring CP
- [x] 4.7 Wire H3 local packed metadata, local logits, and local loss shards into the distributed path

## 5. Pending Distributed and Topology Integration

- [x] 5.1 Add gloo multi-process tests for CP sampler, random broadcast, loss reduction, and the no-`accelerator.prepare` dataloader policy
- [x] 5.2 Add a 24-process CP=8/DP=3 accelerate config and validate the documented 30-second launch shape
- [x] 5.3 Validate DeepSpeed ZeRO-3 global-group integration with CP-correct loss normalization
- [x] 5.4 Implement the DP-only ZeRO/FSDP fallback with explicit CP gradient reduction if the global-group strategy fails
- [x] 5.5 Add a CPU-safe `--validate_cp_setup` mode to the H3 training entrypoint
- [x] 5.6 Decide and test the token refiner CP strategy, defaulting to replicated computation
- [x] 5.7 Decide whether to remove the pad segment as a separate optimization after the real CP path is stable
- [x] 5.8 Implement an optional FlashAttention/blockwise Ring backend with segment-safe online softmax
- [x] 5.9 Add GPU forward and gradient coverage for the FlashAttention/blockwise Ring backend

## 6. Deferred GPU Validation

- [x] 6.1 Run GPU CP=1 vs CP=2 forward parity
- [x] 6.2 Run GPU parameter gradient and single-step loss parity
- [x] 6.3 Run a 30-second H3 SFT launch smoke test
- [ ] 6.4 Measure GPU memory, communication, and throughput at CP=1/2/4/8 (deferred: user decided not to run CP comparison)
- [ ] 6.5 Compare a short training curve against the non-CP baseline (deferred: no multi-step training curve in this pass)
