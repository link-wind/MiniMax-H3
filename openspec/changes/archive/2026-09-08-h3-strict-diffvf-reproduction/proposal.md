## Why

现有 H3 Diff-VF 实验已修复 WWS 窗口的绝对时间位置编码，消除了后段周期性闪烁；但它仍按预测量融合并使用近似 HNI/TES，和 Diff-VF 论文的状态级采样语义不一致。要判断论文方法是否能改善 15 秒接缝处的构图与运动状态错位，必须提供可审计、可复现的严格实现。

## What Changes

- 新增严格 Diff-VF 采样模式，保留既有预测级 joint MultiDiffusion 作为默认兼容路径。
- 以论文定义实现 HNI：共享第一短片段初始噪声、按创新权重混合，并在短片段组内作循环重排。
- 以论文的线性中心距离权重执行 WWS，并在每个窗口从同一 `Z_t` 独立执行调度器一步后融合 `Z_(t-1)`。
- 以交错索引构造 TES，在早期去噪阶段生成全局状态路径，并按论文余弦系数与局部状态融合。
- 为 H3 的 WWS/TES 保持全局绝对视频/音频 source indices，并明确论文视频 TES 与 H3 音频时间线的适配边界。
- 增加严格模式的配置、运行元数据和 CPU 契约测试；维持整条 latent 时间线一次性 VAE 解码。

## Capabilities

### New Capabilities
- `h3-diffvf-paper-strict-sampling`: 为 MiniMax-H3 提供论文语义的 Diff-VF HNI、WWS、TES 与状态级路径融合。
- `h3-diffvf-paper-strict-observability`: 记录严格采样配置、时间线、实际 TES 计划和融合系数，使实验可复现和比较。

### Modified Capabilities
- 无。

## Impact

- 影响 `diffsynth/pipelines/minimax_h3_diffvf.py` 的纯调度原语及 `diffsynth/pipelines/minimax_h3_audio_video.py` 的 `joint_multidiffusion()`。
- 影响接续运行器暴露给实验脚本的 Diff-VF 配置和 manifest。
- 增加 `tests/test_minimax_h3_diffvf.py` 的 CPU 数值契约测试；不改变模型权重、单窗口推理、Retake 或 latent handoff。
