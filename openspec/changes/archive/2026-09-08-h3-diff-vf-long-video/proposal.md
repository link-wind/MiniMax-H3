## Why

现有 H3 多窗口接续主要由 Retake、latent handoff 和边界后处理组成：两个窗口先独立完成去噪，再尝试在接缝处修补。该路径会在约 15 秒窗口边界产生内容错位、短暂闪烁或后续时间段的质量退化，且统一解码只能部分缓解 VAE 边界问题。

已验证的 `joint-multidiffusion` 原型证明了在每个去噪步共享 overlap 轨迹的可行性，但它仅覆盖 Diff-VF 风格框架的 Weighted Window Sampling（WWS）部分。需要将混合噪声初始化、时间扩展采样和局部-全局融合纳入可配置、可测试、可评测的 H3 长视频生成框架，才能以推理时协同生成替代接缝后补救。

## What Changes

- 将当前受限的 `joint-multidiffusion` 实验模式演进为 Diff-VF 风格的全局时间线采样框架，保持既有 Retake 和 latent-handoff 路径的兼容性。
- 新增 Hybrid Noise Initialization（HNI）：以共享基准噪声和新增高斯噪声构造多窗口初始 latent，并记录可复现实验所需的混合权重、种子和噪声来源。
- 规范并扩展 Weighted Window Sampling（WWS）：在每个 diffusion timestep 对重叠窗口共同预测、按可选权重融合并对统一全局 latent 执行 scheduler step。
- 新增 Temporal Extended Sampling（TES）：在指定 timestep 上对全局时间轴构造稀疏、远距离的时间窗口，执行去噪并 scatter 回原时间线，以加强长距离外观和场景一致性。
- 新增按 timestep 调度的 local-global fusion，将 WWS 本地预测与 TES 全局预测以可审计的融合策略组合；支持视频优先接入，以及保守的音频解耦策略。
- 在单一物理时间轴上管理视频与音频 latent、窗口所有权和最终解码；对内存预算、降级行为和不支持的组合给出明确失败或 fallback。
- 扩展评测与消融工具，产出边界连续性、长程一致性、音频连续性、质量和显存/耗时指标；新增 CPU 合约测试与小规模 GPU 固定种子回归测试。

## Capabilities

### New Capabilities

- `h3-diffvf-noise-initialization`: 为长视频窗口生成可复现、可配置且时间对齐的混合初始噪声。
- `h3-diffvf-joint-window-sampling`: 在统一全局 latent 时间线上执行逐步的加权重叠窗口去噪。
- `h3-diffvf-temporal-extended-sampling`: 构造并执行稀疏远距离时间窗口去噪，并安全回写到全局时间线。
- `h3-diffvf-local-global-fusion`: 按时间步融合局部 WWS 与全局 TES 预测，并定义视听模态策略和降级条件。
- `h3-diffvf-av-decode-and-evaluation`: 对齐音画的全局 latent 解码、资源控制、实验记录和 Diff-VF 消融评测。

### Modified Capabilities

无。现有 `h3-av-continuation` 仍是未归档的独立 change；本 change 以新的 Diff-VF 能力规格定义其可选扩展，不修改尚未进入主规格的需求。

## Impact

- 主要影响 `diffsynth/pipelines/minimax_h3.py`、H3 continuation runner、`examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py`、相关配置和 `tests/test_minimax_h3_continuation.py`。
- 需要增加全局 latent 时间线、噪声/窗口调度和预测融合的内部接口；现有单窗口推理 API、Retake 与 latent-handoff 默认行为不得改变。
- 需要 CUDA GPU 进行端到端验证。全局 VAE decode 可能接近或超过单卡可用显存，因此需要显式的显存估算、分块 decode 或拒绝执行策略。
- TES 对 H3 的联合视频/音频位置编码与音频时序具有较高风险；首个可用版本应允许 video-only TES，音频沿用保守接续并在评测中单独报告。
