## Context

MiniMax-H3 的现有 continuation 推理可以将上一窗口尾部作为 latent handoff，并在 overlap 内使用 `taper-refine-v2` 的 timestep-aligned noisy anchor。该策略能保护历史内容，但第一个自由 suffix clip 仍由基础模型按普通生成分布采样，因此窗口边界处可能出现人物位置漂移、光照变化、动作状态跳变和闪烁。

训练数据来自约 8.6GB 的 `data_with_face_and_speech_and_caption.jsonl`。记录包含视频路径、音频路径、`sequence_id`、shot/cut 信息、ASR 和 caption。数据准备必须流式处理，不能把 JSONL 全量载入内存；训练样本只能来自同一 shot 内的真实连续片段。

## Goals / Non-Goals

**Goals:**

- 让 H3 DiT 学会从真实历史 latent 进入第一个自由 suffix clip 的视频和音频联合分布。
- 复现推理时 `taper-refine-v2` 的 hard core/transition/suffix 区域语义，并使用与推理一致的 overlap 对齐。
- 复用现有 H3 SFT/LoRA 基础设施，冻结基础模型和所有编码器，仅保存小型 LoRA 权重。
- 提供可重跑的数据索引、latent cache、训练配置和接缝评测报告。

**Non-Goals:**

- 不修改 H3 VAE、DiT 主干、RoPE 或 scheduler 的结构和预训练权重。
- 不在本变更中实现新的 bridge、joint MultiDiffusion、motion-handoff 或 progressive-noise 推理算法。
- 不跨原始 shot cut 构造正样本，也不把 caption 中未经清洗的模板/乱码直接作为训练 prompt。
- 不承诺仅凭 LoRA 消除所有长链漂移；先验证单镜头 15 秒窗口接续，再扩大到更长序列。

## Decisions

### 1. 以 shot-safe 流式索引作为数据源

索引器逐行读取 JSONL，解析 `sequence_id`、视频/音频路径、prompt、caption、ASR 和 shot 边界，输出轻量 JSONL 索引而不是复制媒体。没有可靠 shot 边界的记录只允许生成单窗口样本，不能生成跨边界 continuation 样本。按 `sequence_id` 做 train/val/test 划分，防止同一视频的相邻窗口泄漏到不同集合。

首版窗口使用 124 帧 smoke、243 帧低成本调试、345 帧正式训练与主要验证；长度必须满足 `17n+5`。345 帧对应 14.375 秒（按 24 fps），默认 overlap 为 34 帧，其中 hard core 17 帧、transition band 17 帧。历史 362 帧结果仅作为旧实验记录，不纳入本变更的训练数据或验收基线。音频边界由视频物理时间推导：24 fps、32 kHz、40 latent steps/s。

### 2. 使用 teacher-forced taper-refine-v2 前向

对每个长度为 `L` 的目标窗口，取前 `K=34` 帧作为真实历史 anchor，构造区域 `[0,17)` hard core、`[17,34)` transition、`[34,L)` suffix。视频和音频共享物理起点，但分别按各自 latent 率切片。

在随机 timestep `t` 下使用两套噪声：

```text
x_t_main = scheduler.add_noise(full_target_latent, epsilon_main, t)
anchor_t = scheduler.add_noise(history_clean, epsilon_anchor, t)

hard core  = history_clean                     (不计 loss)
transition = (1 - w) * x_t_main + w * anchor_t
suffix     = x_t_main
```

`w` 在 transition 中从 1 平滑下降到 0；训练 target 同样使用对应噪声的 flow target，避免把 clean latent 错误注入高噪声层级。hard core 仅作为上下文，不通过梯度学习复制任务。

### 3. 采用区域加权的联合视频/音频 flow-matching loss

默认权重为 hard core `0`、transition `0.5`、第一个完整 suffix video VAE clip `3.0`、剩余 suffix `1.0`；音频总损失乘以 `lambda_audio=0.5` 起步。视频和音频先分别按有效 token 数归一化，再组合，避免音频 token 数量主导梯度。首版只启用 masked flow loss，appearance/identity/motion 辅助损失作为配置关闭的扩展点。

### 4. LoRA 只注入 H3 DiT 的高影响模块

冻结 VAE、文本编码器、RoPE、scheduler 和 DiT 基础参数；默认在 `attn.qkv_proj`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2` 注入 rank 32 LoRA，使用现有 LoRA 保存/加载接口。训练脚本必须支持 bf16、gradient checkpointing、可选 CP 和断点恢复，并保留基础 H3 训练的 `training_cfg_scale` 兼容参数。

### 5. 使用 latent cache，分离昂贵的编码阶段

数据准备阶段统一视频到 24 fps、音频到 32 kHz，执行 H3 Video/Audio VAE 编码并保存 latent、shape、dtype、fps、sample rate、shot id 和源文件校验信息。训练阶段优先读取 cache；cache 元数据不匹配时拒绝训练并要求重新生成，不能静默混用不同 VAE 或采样率。

### 6. 评测以同一计划和种子做配对消融

固定 checkpoint、全局 prompt、segment plan、overlap、scheduler、随机种子和解码路径，至少比较：base+taper-refine-v2、LoRA+taper-refine-v2、不同 LoRA scale、transition 权重和 overlap。报告边界前后的视频亮度/颜色差、人物检测框位移、光流/运动方向差、闪烁频率，以及音频 RMS、频谱差和 A/V 边界时间误差。所有窗口 latent 统一时间线后一次 decode，避免把 VAE 边界伪影误判为 LoRA 效果。

## Risks / Trade-offs

- [标注中的 shot cut 或 caption 噪声污染训练] -> 解析 cut 标记，默认只采样单 shot，并清洗/回退 prompt；保留过滤统计和人工抽样清单。
- [transition 权重过高导致模型过度冻结，运动变慢] -> 首版使用 0.5 并加入无 LoRA、不同权重消融，监控 suffix motion 指标。
- [第一个 suffix clip 的高权重放大局部伪影] -> 要求完整 17 帧 clip 对齐，剔除冻结、严重曝光和人脸检测失败样本。
- [latent cache 占用大量磁盘] -> 索引与 cache 分离，支持按 split/分辨率分批生成和校验，不把媒体复制到索引目录。
- [LoRA 学到数据集特定人物而非接续规律] -> sequence-level split、跨人物/场景采样、保留 base quality 回归测试。
- [联合音视频 loss 梯度失衡] -> 模态独立归一化并暴露 `lambda_audio`，记录每步 video/audio loss。

## Migration Plan

1. 先实现 CPU-only 索引、过滤、窗口解析和 loss 数学单元测试。
2. 在少量缓存样本上运行单卡 smoke LoRA，验证 checkpoint、导出和推理加载。
3. 用固定验证集完成 base 与 LoRA 的配对接续评测，再决定是否扩大数据和训练步数。
4. LoRA 通过显式路径和 scale 加载；未加载 LoRA 时现有 H3 单窗口及 training-free continuation 行为完全不变。

## Open Questions

- 首版数据中 speech_continuation 与无对白动作样本的比例是否需要动态重采样？
- 人物检测框/光流辅助损失是否在 masked flow loss 稳定后再加入？
- LoRA rank 16、32、64 中哪一个在保持基础画质方面最合适？
- 是否需要为有意的场景切换单独训练一个 transition 类别，还是继续将其排除？
