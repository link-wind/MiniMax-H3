## 1. Strict Sampling Primitives

- [x] 1.1 Extend `DiffVFConfig` and manifest serialization with explicit strict semantic, paper HNI, TES, and fusion controls plus validation.
- [x] 1.2 Implement deterministic paper HNI clip initialization, cyclic group permutation, and global timeline composition.
- [x] 1.3 Implement paper center-distance state weight maps and generic local-state fusion helpers.
- [x] 1.4 Implement exact interleaved TES sequence planning, validity-aware scatter, and paper cosine state-fusion coefficient helpers.

## 2. H3 Strict Runner

- [x] 2.1 Add an opt-in strict branch to `joint_multidiffusion()` that preserves existing behavior by default and validates the H3-supported input scope.
- [x] 2.2 Run independent per-window scheduler updates from each common `Z_t`, then form video and audio continuous WWS state paths with absolute source-index packing.
- [x] 2.3 Run early video TES as an independent state path with absolute source indices, fuse it with WWS using the paper schedule, and retain continuous audio WWS semantics.
- [x] 2.4 Record resolved strict scheduling details in the Diff-VF manifest and retain unified global timeline decode.

## 3. Verification

- [x] 3.1 Add CPU tests for paper HNI coefficient limits, cyclic mapping, deterministic overlap ownership, WWS state weights, TES coverage, and cosine state fusion.
- [x] 3.2 Add focused runner contract tests for strict-mode configuration, strict-branch source-index propagation, and existing-mode compatibility.
- [x] 3.3 Run the focused Diff-VF test suite and OpenSpec validation; document any unavailable GPU smoke or quality comparison separately.
