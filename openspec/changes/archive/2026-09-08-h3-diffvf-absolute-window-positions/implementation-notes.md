# Implementation Notes

## Verification

### Targeted regression tests

```bash
PYTHONPATH=. /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  -m pytest -q tests/test_minimax_h3_diffvf.py \
  -k 'dense_wws_packed or dense_wws_reference'
```

Result: `2 passed, 14 deselected`.

The broader CPU test file was also run with the repository Python environment:

```bash
PYTHONPATH=. pytest -q tests/test_minimax_h3_diffvf.py
```

Result: `14 passed, 2 skipped`.

### Fixed two-window GPU control

The control used the H3 virtual environment and the continuation entrypoint,
with the FL2VA base and transformer checkpoint used by the recorded output:

```bash
PYTHONPATH=. /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA \
  --checkpoint /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/transformer \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --height 480 --width 832 --window-frames 362 --overlap-frames 34 \
  --seed 7 --num-inference-steps 50 --mode diff-vf --hni-weight 0.5 \
  --wws-weighting cosine-squared --output outputs/h3_diffvf_15s_hni-mixed-absolute-positions.mp4 \
  --report outputs/h3_diffvf_15s_hni-mixed-absolute-positions.json \
  --evaluation-variant hni-mixed-absolute-positions
```

The 690-frame (28.75 s) output used two 362-frame windows with a 34-frame
overlap, 107 video latent steps per window, a 10-step video-latent overlap,
and TES disabled. It completed in 1452.50 s with 68.19 GiB CUDA peak memory.

The comparable pre-fix `hni-mixed` control used the same seed and Diff-VF
settings. Its output is `outputs/h3_diffvf_15s_hni-mixed.mp4` and its report
is `outputs/h3_diffvf_15s_hni-mixed.json`.

## Result

Both videos were decoded by `ffmpeg` as RGB24 and measured with mean absolute
pixel difference (MAD) between consecutive display frames. The emitted second
span begins at display frame 362. The comparison intentionally includes both
the one-frame seam and the remaining second-span transitions.

| Metric | Pre-fix hni-mixed | Absolute-positioned WWS |
| --- | ---: | ---: |
| Join MAD at frame 362 | 12.7271 | 11.3238 |
| Post-join mean MAD (328 transitions) | 12.8855 | 3.8098 |
| Post-join median MAD | 11.6678 | 0.9702 |
| Post-join P95 MAD | 31.3101 | 11.9670 |
| Post-join maximum MAD | 44.0207 | 13.4444 |

The pre-fix output has large jumps at frames 374, 391, 408, 425, and later
frame indices spaced by 17. Its mean MAD for `frame_index mod 17 == 0` is
36.3288, compared with 3.1645 to 24.2811 for the other phases. The repaired
output has no such phase spike: all 17 phase means are in the 3.4760 to 4.1214
range. This confirms that the persistent 17-frame flashing came from the WWS
global-latent/local-position mismatch, which the absolute source-index packing
removes.

The first transition at the 15.08 s join remains visibly larger than the
typical second-span transition. This change fixes the repeated post-join
artifact, not all semantic or motion discontinuity between independently
conditioned windows.
