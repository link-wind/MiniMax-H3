## Why

`masked-av-v14` 通过本地 canonical 位置避免跨窗口 RoPE 外推，并用 `continuation_*_latents` 保证窗口边界短程接续；但 39 帧左右的 overlap 只携带非常有限的视觉 token，不能提供稳定的人物/物体外观记忆，分钟级生成会逐步出现外观漂移。现有 `global_reference_frames` 只从第一窗口采样固定中间帧，缺少动态选择、信任控制和长程记忆维护。

本 change 以训练无关的方式，在 H3 已有 `ref_blocks` / `references` 接口上增加一个外观记忆银行，用“可信锚点 + 选中历史帧 + 当前边界帧”补上 latent handoff 缺失的长期外观条件，同时保持 `masked-av-v14` 的本地时间位置语义不变。

## What Changes

- 新增 `H3AppearanceReferenceBank` 或等价的内存银行组件，由 `H3ContinuationRunner` 在窗口间维护，并负责把银行内容转换为 H3 `references` 输入。
- 将现有静态 `global_reference_frames` 采样逻辑扩展为三种来源：不可变 trusted anchors、按时间多样性和可用相关性选中的 memory frames、当前窗口 boundary frame。
- 保持现有默认行为不变：未启用外观记忆银行时，后续窗口不注入额外 references；`global_reference_frames` 旧参数继续可用的同时可被新配置替代。
- 只使用 H3 训练过的 reference conditioning 路径，不修改 DiT attention，不引入跨窗口 KV cache，不传全局 source indices / global lengths。
- 在状态 manifest 中记录银行配置、锚点来源、选中帧来源和帧级元数据；不在 JSON 中内嵌媒体 tensor。
- 新增长时外观评估入口，支持 `masked-av-v14` 的 baseline、静态 global references、动态 memory bank 三组对照，并输出可复现配置与外观漂移指标或明确标记的 unavailable metric。
- 增加 CPU 可运行的规划、选择、序列化和兼容性测试；GPU 质量验证作为独立验证任务记录。

## Capabilities

### New Capabilities

- `h3-appearance-memory-bank`: 在 H3 continuation runner 中维护可信外观锚点、选中历史记忆帧和当前边界参考，并安全地映射到现有 `ref_blocks` / `references` 条件。
- `h3-appearance-drift-evaluation`: 对长时 `masked-av-v14` 生成进行可复现的外观一致性评估、baseline 对照和 memory bank 消融。

### Modified Capabilities

无。本 change 不改变已归档 spec 的需求；只扩展现有 continuation runner 的可选推理能力。

## Impact

- 主要影响 `diffsynth/pipelines/minimax_h3_continuation.py`：新增 bank 配置、状态字段、参考帧选择与请求注入逻辑。
- 复用 `diffsynth/pipelines/minimax_h3_audio_video.py` 中已有的 reference preprocessing 和 `_build_packed_ref2va`，不改本地位置打包路径。
- 更新 `examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py`，增加 memory bank 的 opt-in 参数和示例。
- 新增或扩展 `tests/test_minimax_h3_continuation.py` 与评估入口测试；不引入新的模型权重、训练任务或第三方生成模型。
