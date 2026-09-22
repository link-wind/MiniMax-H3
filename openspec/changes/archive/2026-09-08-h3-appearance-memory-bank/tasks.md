## 1. Contracts and configuration

- [x] 1.1 Add `H3AppearanceMemoryConfig` with `mode`, trusted-anchor count, memory-frame budget, boundary-reference flag, and total visual-reference limit, including validation.
- [x] 1.2 Add typed bank data structures for trusted anchors, selected memory frames, boundary reference, and provenance metadata.
- [x] 1.3 Preserve `global_reference_frames` / `boundary_reference_frames` backward compatibility and map legacy values to the new bank config when the explicit bank config is absent.
- [x] 1.4 Add CPU-safe tests for default-disabled behavior, invalid bank configuration, reference limits, and legacy parameter compatibility.

## 2. Bank core behavior

- [x] 2.1 Implement trusted-anchor initialization from the first completed window's middle frames, excluding the final overlap tail.
- [x] 2.2 Implement deterministic memory-frame selection using temporal buckets and visual-diversity/consistency checks, with near-duplicate exclusion and bounded memory count.
- [x] 2.3 Implement boundary-reference extraction from the prior window's final decoded tail frame.
- [x] 2.4 Implement reference-list construction that preserves user references before bank references and returns a flat H3 `references` list.
- [x] 2.5 Add CPU-safe bank tests for anchor immutability, memory diversity, near-duplicate skipping, boundary separation, and provenance tracking.

## 3. Runner integration and state

- [x] 3.1 Add bank configuration arguments to `H3ContinuationRunner` and reject bank use with unsupported paths such as `global_latent_decode`.
- [x] 3.2 Add in-memory bank runtime state to `ContinuationState` and bank provenance metadata to `H3ContinuationStateManifest` without embedding media tensors.
- [x] 3.3 Update the runner loop to initialize/update the bank after decoded windows and inject bank references before later windows.
- [x] 3.4 Assert that bank-enabled `masked-av-v14` requests pass references without `video_source_indices`, `audio_source_indices`, or global latent length/origin parameters.
- [x] 3.5 Add runner integration tests using a CPU fake pipeline for reference injection, state advancement, manifest metadata, and default no-injection behavior.

## 4. Inference entrypoint

- [x] 4.1 Extend `examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py` with opt-in appearance-bank CLI flags for mode, anchor count, memory count, boundary reference, and max visual references.
- [x] 4.2 Add CLI validation for bank mode compatibility with `masked-av-v14`, latent handoff, overlap, and reference limits.
- [x] 4.3 Add an example plan/configuration for a 60s `masked-av-v14` appearance-bank run and document the expected comparison setup.

## 5. Appearance-drift evaluation

- [x] 5.1 Add an appearance-drift evaluation entrypoint that runs no-bank, static global-reference, and dynamic-bank modes for the same segment plan and writes a JSON report.
- [x] 5.2 Implement appearance-consistency reporting with optional CLIP/LPIPS-style metrics and explicit unavailable-metric records when dependencies are missing.
- [x] 5.3 Record per-window reference composition, selected anchor/memory source segment and frame locations, seeds, and resolved run configuration in the report.
- [x] 5.4 Add CPU-safe evaluation tests for reproducible report metadata, invalid configuration diagnostics, and report generation without model inference.

## 6. Validation and docs

- [x] 6.1 Run targeted unit tests for the bank, runner integration, and evaluation entrypoint.
- [x] 6.2 Run `openspec validate h3-appearance-memory-bank --strict`.
- [x] 6.3 Record the exact deferred 60s GPU A/B command and expected artifacts for `masked-av-v14` no-bank, static, and dynamic bank; GPU execution was not available in this environment.
- [x] 6.4 Update `docs/zh/minimax_h3_continuation_design.md` or a new section to describe bank configuration, local-position constraints, and scene-change reset behavior.
