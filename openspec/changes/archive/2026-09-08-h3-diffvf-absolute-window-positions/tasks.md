## 1. Absolute-positioned WWS packing

- [x] 1.1 Add a focused helper that rebuilds positive and negative H3 `packed` metadata for a dense WWS window from its global video/audio source ranges.
- [x] 1.2 Call the helper for every WWS window prediction and reject nonzero-offset reference-block layouts before denoising.

## 2. Regression coverage

- [x] 2.1 Add CPU tests for dense second-window video/audio offsets, overlap coordinate equality, CFG branch isolation, and unsupported reference-block failure.
- [x] 2.2 Run targeted tests and a fixed-configuration two-window GPU control; record the command, settings, and any visual/metric result in the implementation notes.
