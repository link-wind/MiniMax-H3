## Why

MiniMax-H3 当前可生成联合视频和音频，并能通过 Retake 固定和重绘指定区域，但其常规推理仍以单个短片段为单位。直接将多个独立片段拼接会在画面运动、人物身份、对白、音乐和环境声上产生边界断裂与长期漂移；而现有 30 秒 Ring CP 能力解决的是单窗口长度，不负责多窗口状态和接续语义。本 change 只验证 training-free 接续路径，continuation LoRA/SFT 已拆分到独立的 `h3-continuation-lora-training` change。

本 change 建立可评测的 H3 视听接续生成基础：先以现有 Retake 实现安全的多窗口 MVP，再为无重编码的 latent 交接、局部 overlap refinement 和后续 continuation 训练提供稳定接口与评测基线。

## What Changes

- 新增 `H3ContinuationRunner`，使用多个时间对齐的 H3 窗口顺序生成长视频，并以历史尾部作为下一窗口的视听条件。
- 新增结构化 `ContinuationState` 和 segment plan，分别保存全局 prompt/reference、最近的视听上下文、时间游标、种子和窗口级提示词，防止对白或场景在窗口边界重复或跳过。
- 为视频使用符合 H3 `17n+5` 和完整 17 帧 VAE clip 的 overlap；为音频使用同一物理时长、但按 32 kHz / 40 latent/s 独立映射的 overlap。
- 新增可选的 `return_latents` 与 continuation latent 输入接口，避免后续窗口将历史 overlap 解码后再次编码；保留当前 Retake 接口以兼容 MVP 和外部视频/音频输入。
- 定义硬 suffix 拼接、可选 waveform equal-power crossfade，以及后续局部 latent/noise-prediction taper refinement 的行为边界。
- 新增接续生成评测和消融入口，至少报告窗口边界的视听连续性、时长/对齐、长期一致性和基础 Retake baseline 对比。
- 定义与 Ring Context Parallel 的协作约束：CP 只负责单个超长窗口，Continuation Runner 负责窗口级时间推进；CP 组内必须运行一致的窗口、随机源和状态转换。

## Capabilities

### New Capabilities

- `h3-av-continuation-runner`: 基于 Retake 和 segment state 的多窗口联合视频音频接续生成、时间对齐与输出拼接。
- `h3-continuation-latent-handoff`: 面向接续窗口的 video/audio latent 导入、导出、校验与 fallback 行为。
- `h3-av-continuation-evaluation`: 接续生成的可复现评测、边界诊断和关键消融报告。

### Modified Capabilities

- 无。现有 `h3-ring-context-parallel` 和 `h3-cp-aware-sft-training` 的需求不改变；本 change 仅消费其单窗口并行能力。

## Impact

- 主要影响 `diffsynth/pipelines/minimax_h3_audio_video.py` 的可选 latent 输入/输出表面，以及新增的接续运行器和示例推理入口。
- 新增 `examples/minimax_h3/model_inference/` 下的接续生成示例、segment plan 示例和测试。
- 新增对视频/音频读取、waveform crossfade、状态序列化及输出拼接的轻量依赖代码；不引入新的模型权重或第三方生成模型。
- 后续可选地接入现有 `examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-30s-local-cp.py`，但不改变其单窗口默认行为。
