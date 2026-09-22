# MiniMax-H3 核心技术解析

## 1. 文档说明

本文档结合 DiffSynth-Studio 中的 MiniMax-H3 实现，介绍其主要模型结构、数据表示、条件控制和生成流程，为后续研究视听联合接续生成提供背景知识。

需要区分两类能力：

1. MiniMax-H3 模型本身

   - 统一多模态理解和生成；
   - 原生视频与立体声音频联合生成；
   - FL2VA、Ref2VA 和 Retake；
   - 双 VAE、统一多模态 DiT 和 Flow Matching。

2. 当前仓库的扩展能力

   - 30 秒视频音频 SFT；
   - Ring Context Parallel；
   - DeepSpeed ZeRO-3；
   - 多卡长序列训练和推理。

第二类能力属于当前 DiffSynth-Studio 分支的训练和推理工程扩展，不应与 H3 原始模型架构混为一谈。

## 2. 模型整体结构

MiniMax-H3 的核心特点是将文本、图像、视频和音频条件组织为统一序列，再通过同一个大规模 Diffusion Transformer 联合预测目标视频和音频。

```text
文本条件 ─────────────┐
首尾帧/参考图像 ──────┤
参考视频和音频 ───────┤
目标视频 latent ──────┼──> Unified Multimodal DiT
目标音频 latent ──────┘               |
                                      +──> Video prediction
                                      └──> Audio prediction
```

传统级联方案通常先生成视频，再从视频生成音频。H3 的联合方案让视频和音频 token 在同一 self-attention 中交互，有利于建模：

- 口型和对白同步；
- 动作和音效同步；
- 画面节奏和音乐节奏同步；
- 文本语义在视频和音频中的共同呈现。

## 3. 多模态 Packed Sequence

H3 先将不同模态映射到相同 hidden size，再放入一个 packed sequence。

### 3.1 FL2VA 布局

```text
[text | keyframe conditions | target audio | target video | padding]
```

### 3.2 Ref2VA 布局

```text
[text | reference 0 | reference 1 | ... | target audio | target video | padding]
```

Reference block 可以是：

- 图像；
- 无声视频；
- 音频；
- 带声音的视频。

### 3.3 模态标签

每个 packed token 都有对应的模态标签：

```text
0：图像或视频 token
1：文本 token
2：音频 token
-1：padding token
```

不同输入首先经过各自的投影层：

```text
视频 patch：96 -> 5376
音频 latent：32 -> 5376
文本 embedding：5120 -> 5376
```

投影后，所有有效 token 进入共享 Transformer 主干。

这种设计有两个主要优势：

1. 文本、视频和音频能够通过全局 attention 直接交换信息。
2. 首尾帧、参考图像、参考视频、参考音频和 Retake 区域可以统一组织成条件 token。

对应实现：

- `MiniMaxH3Unit_PackedSequenceBuilder`
- `_build_packed_fl2va`
- `_build_packed_ref2va`

## 4. 大规模 Diffusion Transformer

当前实现中的默认 DiT 配置为：

```text
Transformer blocks：50
Hidden size：5376
Attention heads：56
Attention head dimension：128
FFN hidden size：14336
Text dimension：5120
Video latent channels：24
Audio latent dimension：32
Video patch size：(1, 2, 2)
```

模型主干大致为：

```text
Video / Audio / Text projections
               |
               v
       Text Token Refiner
               |
               v
      50 shared DiT blocks
               |
               v
      Modality-specific outputs
          |             |
          v             v
      Video output   Audio output
```

### 4.1 Text Token Refiner

文本 embedding 在进入共享 DiT 前会经过两层 Token Refiner。该模块进一步加工 Qwen3-VL 输出的条件特征，再与视频和音频 token 合并。

### 4.2 输出头

共享 Transformer 输出后，模型使用不同输出投影恢复两种模态：

- 视频输出恢复为每个 `1 x 2 x 2` patch 的 latent 预测；
- 音频输出恢复为 32 维 audio latent 预测。

从当前工程记录估算，H3 DiT 约为 33B 参数量级。

## 5. 模态感知 AdaLN

文本、视频和音频共享 Transformer，但三种模态的数据分布并不相同。H3 使用模态标签和 AdaLN 调制共享计算。

```text
Diffusion timestep
       +
Modality tag
       |
       v
生成 scale / shift / gate
       |
       v
调制 Transformer attention 和 MLP
```

模型定义了三种 AdaLN 模态：

```text
Video / Image
Text
Audio
```

这样，同一个 DiT block 在处理视频和音频时，可以使用不同的归一化与残差门控参数。模型既能共享跨模态语义，又不需要假设不同模态服从相同特征分布。

## 6. 多维 RoPE 位置编码

H3 为 packed sequence 构造三维位置：

```text
(time, height, width)
```

### 6.1 视频位置

- `time` 表示视频 latent 的时间位置；
- `height` 和 `width` 表示视频 patch 的空间坐标。

### 6.2 图像和参考视频位置

- 每个参考图像或视频具有自己的空间网格；
- 参考 block 和目标 block 在 packed 时间轴上依次排列；
- 首尾帧条件根据其首帧或尾帧语义设置时间坐标。

### 6.3 音频位置

- 时间轴对应 audio latent time；
- 额外使用位置轴区分立体声声道并关联目标视频尺度。

这些位置经过三轴 RoPE 后应用于 attention 的 Q 和 K，使模型同时理解：

- 视频先后顺序；
- 画面空间结构；
- 音频事件时间；
- Reference、Keyframe 与目标内容的相对关系。

## 7. Video VAE

H3 在视频 latent 空间中执行扩散生成。当前 Video VAE 的主要配置为：

```text
Video latent channels：24
Spatial compression ratio：16
Base temporal compression ratio：4
VAE clip length：17 frames
Latent tokens per clip：5
```

### 7.1 视频 Patchify

Video VAE 输出的 latent 形状为：

```text
[1, 24, T, H, W]
```

DiT 使用 `1 x 2 x 2` patch：

```text
[1, 24, T, H, W]
        |
        v
[T * (H/2) * (W/2), 96]
```

其中：

```text
96 = 24 * 1 * 2 * 2
```

### 7.2 17 帧 Clip 约束

Video VAE 按 17 帧 clip 处理，clip 内的帧在 latent 空间中耦合。因此：

- 重绘 clip 中任意一帧，实际会影响整个 clip；
- Retake 视频 mask 会向外扩展到完整 17 帧边界；
- Pipeline 输出帧数需要满足特定的时间对齐关系。

当前 Pipeline 将请求帧数向上调整到：

```text
17n + 5
```

常见合法帧数包括：

```text
124
175
719
```

## 8. Audio VAE

Audio VAE 的主要配置为：

```text
Sample rate：32000 Hz
Hop length：800 samples
Latent rate：40 latent/s
Latent dimension：32
Channels：2
```

音频 latent 形状大致为：

```text
[2, 32, audio_time]
```

进入 DiT 前，音频 latent 被整理为：

```text
[2 * audio_time, 32]
```

其中两个声道按 channel-major 方式排列。

与 Video VAE 不同，Audio VAE 的时间压缩是均匀的，不存在 17 帧 clip 结构。其 Retake 时间分辨率为：

```text
1 / 40 s = 25 ms
```

因此视频和音频 Retake 使用不同单位：

- 视频使用帧号并受 17 帧 clip 对齐约束；
- 音频使用秒，并映射到 40 Hz audio latent。

## 9. Qwen3-VL 多模态条件编码器

H3 使用基于 Qwen3-VL 的多模态条件编码器。它不仅编码普通文本，还可以编码与 prompt 共同输入的图像和参考视频。

当前实现中的主要配置为：

```text
Text hidden size：5120
Text layers：50
Text attention heads：64
Text KV heads：8
Vision encoder depth：27
Vision hidden size：1152
Vision output size：5120
```

根据任务不同，条件被组织为不同 presentation：

```text
presentation_t2va
presentation_fl2va
presentation_ref2va
```

Reference 视频会进行采样，并将时间戳信息加入多模态 presentation，使条件编码器能够理解参考帧的时间位置。

## 10. FL2VA

FL2VA 可理解为：

```text
First/Last Frame + Text -> Video + Audio
```

支持以下任务：

- 纯文本生成视频和音频；
- 文本加首帧生成；
- 文本加首尾帧生成。

当前 Keyframe 接口支持：

```text
0：首帧
-1：尾帧
```

关键帧先由 Video VAE 编码，再作为额外视觉条件 token 放入 packed sequence。首帧和尾帧使用不同时间坐标，从而分别约束生成序列的开始和结束。

## 11. Ref2VA

Ref2VA 可理解为：

```text
References + Text -> Video + Audio
```

支持的 Reference 类型包括：

```python
{"type": "image", "image": image}
{"type": "video", "video": frames}
{"type": "audio", "audio": waveform, "sample_rate": sample_rate}
{
    "type": "video_audio",
    "video": frames,
    "audio": waveform,
    "sample_rate": sample_rate,
}
```

Ref2VA 适合：

- 保持人物或物体身份；
- 模仿视觉风格；
- 参考动作和镜头；
- 参考说话人、声音或音乐；
- 使用多个多模态参考条件。

FL2VA 和 Ref2VA 使用不同的 DiT、文本编码器和 processor 权重，必须按照任务加载对应分区，不能只切换 Pipeline 参数。

## 12. Retake 局部重绘

Retake 允许输入原视频或音频，并指定需要重新生成的时间区域。

### 12.1 视频 Retake

```python
retake_video=source_video
frame_regions_to_retake=[(start_frame, end_frame)]
```

### 12.2 音频 Retake

```python
retake_audio=source_audio
retake_audio_sample_rate=sample_rate
seconds_regions_to_retake=[(start_second, end_second)]
```

Mask 的语义为：

```text
0：保持输入 latent
1：重新生成
```

在每个 denoise step 中，Pipeline 会：

1. 将固定区域替换回输入 latent；
2. 为固定 token 设置对应的条件 timestep；
3. 在 scheduler 更新时再次执行 inpainting blend；
4. 只允许 mask 区域发生生成性变化。

Retake 支持：

- 保持视频，仅重新生成音频；
- 保持音频，仅重新生成视频；
- 局部重绘视频和音频；
- 固定历史前缀，生成新的时间后缀。

最后一种用法是 H3 接续生成方案的基础。

## 13. 双 Flow Matching 调度

H3 的视频和音频使用两套独立的 Flow Matching scheduler：

```text
Video flow shift：12.0
Audio flow shift：3.0
```

每个采样 iteration 同时推进两种模态：

```text
Video timestep -> video tokens
Audio timestep -> audio tokens
```

两条轨迹同步迭代，但使用不同 sigma 调度。这是因为视频和音频 latent 的数据分布、时间密度和生成难度不同。

### 13.1 Per-token Timestep

H3 在进入 DiT 前为 packed sequence 构造逐 token timestep：

- 视频目标 token 使用视频 timestep；
- 音频目标 token 使用音频 timestep；
- 固定 Retake token 使用条件 timestep；
- Reference token 使用对应条件强度；
- 不同 timestep 先去重，再通过索引映射到各 token。

这一设计为后续研究按时间递增的 progressive noise 提供了基础。

## 14. CFG 蒸馏

H3 是 CFG 蒸馏模型，推荐配置为：

```text
cfg_scale = 1.0
```

负向 prompt 在默认配置下基本不参与生成。与某些需要较大 CFG 的扩散模型不同，随意提高 H3 的 CFG scale 可能使推理分布偏离蒸馏训练分布。

基础模型通常使用约 50 个推理步。Turbo 权重可以使用更少步数，但属于单独的蒸馏或加速模型版本。

## 15. Pipeline 推理流程

H3 Pipeline 的主要流程如下：

```text
1. Shape alignment
2. Initialize video/audio noise
3. Encode optional training inputs
4. Encode video/audio Retake conditions
5. Encode first/last keyframes
6. Encode multimodal references
7. Encode prompt with Qwen3-VL
8. Build packed multimodal sequence
9. Joint video/audio denoising loop
10. Decode video with Video VAE
11. Decode audio with Audio VAE
12. Return video frames and stereo waveform
```

在去噪循环中：

```text
video_latents_t, audio_latents_t
                |
                v
     model_fn_minimax_h3
                |
        +-------+-------+
        |               |
        v               v
video prediction   audio prediction
        |               |
        v               v
video scheduler    audio scheduler
```

## 16. 显存管理与量化

由于 H3 DiT 参数规模很大，DiffSynth-Studio 提供：

- CPU offload；
- 按剩余显存自动调度模型；
- Video/Audio VAE tiled encode/decode；
- NF4 量化权重；
- Pruned 权重；
- Turbo 少步数权重。

这些能力主要解决部署和显存问题，不改变 H3 统一多模态生成的核心建模方式。

## 17. 当前仓库的 30 秒扩展

H3 原始公开能力主要面向最高约 15 秒视频。当前仓库进一步实现了实验性的 30 秒训练和推理链路。

### 17.1 长序列规模

30 秒、719 帧、480 x 832 配置会形成约 85K token 的 packed sequence，其中包括：

- 视频 token；
- 音频 token；
- 文本 token；
- padding 和条件 token。

### 17.2 Ring Context Parallel

Ring CP 将 packed token 序列分片到多个 GPU：

```text
Global packed sequence
       |
       +──> Rank 0 local tokens
       +──> Rank 1 local tokens
       +──> ...
       └──> Rank N local tokens
```

各 rank 通过 Ring Attention 轮转 K/V，并使用在线 softmax 合并全局 attention 结果，从而避免每张 GPU 保存完整长序列 attention 激活。

### 17.3 ZeRO-3 与训练优化

当前分支还组合了：

- DeepSpeed ZeRO-3 参数和优化器状态分片；
- CPU model initialization；
- CP-aware dataloader；
- CP 组内 timestep 和 noise 同步；
- local video/audio loss；
- gradient checkpointing；
- 多节点训练和 checkpoint 保存。

这些属于长序列训练和推理工程，不是原始 H3 DiT 的必需组成部分。

## 18. 对接续生成有利的基础

H3 适合研究视听联合接续生成，主要因为：

- 视频和音频在同一个 DiT 中联合建模；
- 已支持完整视频和音频 Retake；
- Video/Audio Retake mask 可以独立控制；
- 非因果全局 attention 能利用整个固定上下文；
- 支持 per-token timestep；
- Reference 可以近似作为长期身份和风格 anchor；
- 视频和音频 latent 都具有明确物理时间映射；
- 当前仓库已有 30 秒和 Ring CP 实验基础。

## 19. 对接续生成的限制

现有能力仍有以下限制：

- 原始模型主要在较短视频分布上训练；
- Video VAE 的 17 帧 clip 限制精细视频时间 mask；
- Pipeline 默认只返回解码结果，不直接暴露最终 latent；
- Retake 当前使用二值 mask，没有渐进式 transition band；
- Reference 长期固定可能抑制动作，导致视频停滞；
- 多窗口 decode/re-encode 会累积 VAE 误差；
- 音频语义状态、对白进度和音乐结构需要额外管理；
- 多窗口生成会累积视觉、声音和同步误差。

## 20. 总结

MiniMax-H3 可以概括为：

```text
Qwen3-VL multimodal condition encoder
                +
       Video VAE / Audio VAE
                +
 Unified Packed Multimodal DiT
                +
 Dual Flow Matching schedulers
                +
      FL2VA / Ref2VA / Retake
```

其最核心的技术点不是单独提高视频或音频质量，而是在统一 token 序列和共享 DiT 中联合建模文本、视觉和音频。这也是 H3 区别于“视频模型生成画面，再由后置音频模型配音”方案的关键。

对于后续接续生成研究，最值得利用的能力是：

1. Video/Audio 联合 packed sequence。
2. Retake 的局部固定和重绘机制。
3. Per-token timestep。
4. Reference 多模态条件。
5. 30 秒 Ring CP 长序列计算基础。
