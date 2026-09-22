# MiniMax-H3 归档：已停用 / 不推荐 / 排除的接续方案

本目录归档已停用或已不进入当前主线的 MiniMax-H3 长视频接续方案及其验证记录。
仍在用的方法（masked-av-v14、Retake-hard、Latent handoff、Segment State、Continuation LoRA）不受影响，
统一参考 [主方法总览](../../docs/zh/minimax_h3_continuation_methods_used.md)。

## 归档文档

| 文档 | 内容 |
| --- | --- |
| `minimax_h3_three_continuation_methods_comparison.md` | Retake-hard / Latent-handoff / Taper-refine-v2 三种接续的解码过程对比 |
| `minimax_h3_continuation_validation.md` | 早期接续验证记录，含 bidirectional / Joint MultiDiffusion 等已停用实验 |
| `minimax_h3_continuation_lora_plan.md` | 待确认版 continuation LoRA 训练方案，已被 v15 方案取代 |

> `docs/zh/minimax_h3_continuation_methods_used.md` 的 §7「当前结论与下一步边界」是仍在用的方法状态总表，归档方法的状态仍可在此查阅。

## 归档的代码与测试（Diff-VF / Joint MultiDiffusion）

Diff-VF（含 Joint MultiDiffusion 联合扩散）已从生产路径完全解耦并归档：

- `archive/code/diffsynth_pipelines_minimax_h3_diffvf.py` — Diff-VF 联合扩散 pipeline 模块
- `archive/examples/MiniMax-H3-DiffVF-Eval.py` — Diff-VF 消融评测脚本
- `archive/tests/test_minimax_h3_diffvf.py` — Diff-VF 单元测试
- 对应文档 `archive/docs/minimax_h3_diffvf.md`；主方法文档 §6 已标记为「已归档」

runner 中不再存在 `from ...minimax_h3_diffvf import ...` 引用；shared_noise 所需的 `derive_rng_seed` 已内联到 runner。如需恢复该方向，从 `archive/code/` 取回模块即可。

## 训练侧 Taper-refine 归档

Taper-refine-v2/v3 作为 continuation LoRA 的 conditioning 模式已从训练路径移除：

- `diffsynth/diffusion/loss.py`：删除了 `taper-refine-v2/v3` 的 loss 分支，仅保留 masked-av-v14
  分支；删除了 `taper_refine_inputs` import。
- `diffsynth/utils/continuation_lora.py`：`ContinuationRegionConfig` 的 `conditioning_mode` 仅接受
  `masked-av-v14`；删除 `v3_transition_start/end_weight` 字段与校验；删除 `taper_refine_inputs` 函数；
  collator 的 taper 三元分支简化为 masked-av-v14 路径。
- `examples/minimax_h3/model_training/train.py`：`--continuation_conditioning` choices 仅剩
  `masked-av-v14`；删除 `--continuation_v3_anchor_start/end` 参数与 region config 传参。
- `examples/minimax_h3/model_training/eval_memory_ablation.py`：删除 v3 anchor 属性赋值。
- `examples/minimax_h3/model_evaluation/continuation_lora_evaluation.py`：评测模式改为
  `masked-av-v14`（此前用已失效的 `taper-refine-v2`，会被 runner 白名单拒绝）。

推理侧 runner 仍保留 `project_video/audio_taper_anchor`、`taper_hard_core_frames` 等
锚定/投影基础设施，它们是 masked-av-v14 与 latent handoff 的共享依赖，不属于 taper-refine
方法本身。

对应测试：`tests/test_h3_continuation_lora.py` 已删除 taper-v3 collator 测试与
`taper_refine_inputs` 引用，taper 仅作为历史 contract 字符串保留在 fingerprint mismatch 用例中。


## 归档的实验产物

停用方法的推理/对比产物（视频、plan JSON、state）已从 `outputs/structured_15s/` 移至：

```
archive/outputs/discontinued_methods/structured_15s/
```

包含方法：Taper-refine / Taper-refine-v2 / Motion handoff / Bridge / 双侧 Retake / Joint MultiDiffusion
以及对应的 boundary reference、global decode 等辅助变体。

保留在 `outputs/structured_15s/` 的仅有在用基线：`retake_full50`、`latent_full50`。

## 说明

- `outputs/` 目录已被 `.gitignore` 忽略，产物归档仅涉及磁盘整理，不影响 git。
- appearance memory（`h3_appearance_memory.py`）仍保留为 masked-av-v14 的推理侧实验，代码与产物未归档。
- 已停用方法已从 `minimax_h3_continuation.py`（runner）与 `MiniMax-H3-Continuation.py` （推理脚本）等生产路径中移除 mode 分支/参数，不再进入当前推理或训练流程。
- 代码模块 `minimax_h3_diffvf.py` 及其推理脚本、测试均已移至 `archive/code/`、`archive/examples/`、`archive/tests/`。runner 中为 shared-noise 自带的 `derive_rng_seed` 已内联保留，归档模块可安全移除。
