## ADDED Requirements

### Requirement: Reproducible appearance-drift evaluation
The project SHALL provide an executable appearance-drift evaluation entrypoint that accepts a segment plan, checkpoint identity, seed, window configuration, overlap configuration, bank configuration, and output directory. The entrypoint SHALL record the resolved run configuration for reproducibility.

#### Scenario: Same configuration is reproducible
- **WHEN** the evaluation runs twice with the same plan, checkpoint, seed, window, overlap, and bank configuration
- **THEN** the two reports SHALL record matching planned boundaries, per-window seeds, and bank settings

#### Scenario: Invalid configuration fails early
- **WHEN** the evaluation receives an invalid overlap or bank configuration
- **THEN** it SHALL fail before model inference and identify the invalid field

### Requirement: Appearance consistency reporting
The evaluation SHALL report appearance consistency between trusted appearance references and each later window or final timeline, or SHALL explicitly mark the metric as unavailable.

#### Scenario: Later-window appearance metric is emitted
- **WHEN** an evaluated run contains at least one window after the first
- **THEN** the report SHALL include an early-versus-later appearance consistency measurement or an explicit unavailable marker

#### Scenario: Unavailable metrics are identified
- **WHEN** an optional appearance metric cannot be computed because its dependency is missing
- **THEN** the report SHALL name the missing dependency and SHALL NOT present a synthetic score as measured output

### Requirement: Baseline and memory-bank ablation comparison
The evaluation SHALL support `masked-av-v14` comparisons across no bank, static global reference frames, and the dynamic appearance memory bank while holding prompt, checkpoint, seed, and segment plan constant.

#### Scenario: Comparable modes are recorded
- **WHEN** an evaluator runs no-bank, static-reference, and dynamic-bank modes for the same plan
- **THEN** the report SHALL include each mode, the reference count/type per later window, and the appearance consistency metric

#### Scenario: Selected bank provenance is recorded
- **WHEN** dynamic-bank mode selects anchors and memory frames
- **THEN** the report SHALL record the source segment and frame locations used for those references

### Requirement: CPU-safe memory bank coverage
The implementation SHALL include CPU-safe tests for bank configuration validation, trusted-anchor selection, memory-frame diversity selection, reference injection, state serialization, and local position preservation.

#### Scenario: No-GPU tests validate bank behavior
- **WHEN** the test suite runs without an H3 checkpoint or CUDA device
- **THEN** it SHALL validate bank selection, reference injection, and serialization without model denoising

#### Scenario: Local reference positions are asserted
- **WHEN** a CPU test builds a packed ref2va sequence with bank references
- **THEN** the test SHALL assert that target positions start from the local origin and that no global position parameters are present

#### Scenario: GPU quality validation is explicit
- **WHEN** implementation-complete validation has not yet run a long-horizon GPU comparison
- **THEN** the change SHALL record that GPU quality validation is deferred rather than treating CPU-safe tests as proof of visual quality
