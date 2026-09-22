## Why

Diff-VF 的 Weighted Window Sampling（WWS）从同一全局 latent 时间线收集各窗口的状态，但普通窗口仍使用从零开始构造的 H3 时间位置编码。第二窗口因此将全局后段 token 当作局部起点去噪，形成不连续的 latent 轨迹，并在 H3 VAE 的 17 帧 temporal decode 网格上表现为持续闪烁。

TES 已正确传递绝对 source indices；WWS 必须采用同一坐标语义，才能让统一全局 latent、预测融合与 H3 DiT 的时序条件一致。

## What Changes

- 为每个 WWS 窗口在每个去噪步建立连续的全局视频与音频 source indices。
- 使用该窗口实际使用的 video/audio latent 及绝对 indices，为正、负 CFG 条件重建 H3 `packed` 位置元数据。
- 保持提示词、参考块、CFG、HNI、WWS 融合、TES 和全局 VAE decode 的既有语义不变。
- 增加 CPU 回归测试，验证第二窗口的 H3 视频和音频 position ids 从其全局偏移处连续开始，而非重置为零。
- 为固定设置的两窗口 GPU 对照增加可重复运行入口和闪烁诊断记录。

## Capabilities

### New Capabilities

- `h3-diffvf-window-position-continuity`: 在 Diff-VF WWS 中把每个局部窗口的 H3 视频与音频位置元数据绑定到全局 latent 时间线。

### Modified Capabilities

无。

## Impact

- 影响 `diffsynth/pipelines/minimax_h3_audio_video.py` 中 `joint_multidiffusion()` 的窗口预测路径和相关测试。
- 不改变单窗口推理、顺序 Retake、latent handoff，或现有 TES 稀疏窗口的绝对位置编码语义。
- 需要一项短 GPU 对照确认修正不会引入位置编码 shape、参考条件或 CFG 回归。
