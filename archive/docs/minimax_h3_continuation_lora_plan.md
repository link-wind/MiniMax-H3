# MiniMax H3 Continuation LoRA 训练方案（待确认）

## 1. 目标

在现有 **Masked AV Latent Continuation / latent handoff** 推理路径上训练一个 DiT LoRA，使 H3 在看到上一窗口被保护的 Video + Audio latent 尾部后，能生成与其**位置、姿态、相机运动、光照和语音节奏**连续的后缀。

首要解决的是第一个自由生成 token 之后的状态跳变：人物位置偏移、姿态突变，以及由此诱发的接缝闪烁。它不是训练一个新的长视频模型，也不是让 LoRA 学会复制 overlap。

## 2. 核心判断

当前 latent handoff 已经把历史内容精确带到下一窗口：

```text
窗口 A 的 clean AV latent 尾部
        |
        | 复制到窗口 B 的 overlap 前缀，并由 mask 保护
        v
窗口 B: [固定历史 prefix | 随机初始化的 suffix]
```

问题在于原始 H3 没有专门以这种条件训练。它能读取固定 prefix，但未必会将其中的运动趋势解释成未来 suffix 的初始位置、速度和镜头轨迹。

因此 LoRA 的学习目标应是：

```text
真实连续历史 latent + preserve mask + 当前 diffusion timestep
    -> 与真实未来连续的 suffix flow/velocity prediction
```

而不是：

```text
调大 overlap、调 taper 权重、重建被保护的历史区域
```

## 3. 首版范围

### 做

- 仅训练 H3 DiT LoRA；冻结视频 VAE、音频 VAE、文本编码器、RoPE 和 scheduler。
- 使用真实连续的单镜头音视频作为训练样本。
- 在同一条真实视频中构造“历史 overlap + 未来 suffix”的窗口对。
- 将历史 video/audio latent 作为 clean、masked context 输入。
- 只对可生成的 suffix 计算 flow-matching 损失，并提高接缝后的短区间权重。
- 推理复用现有 `latent-handoff`，首轮不叠加 `taper-refine-v2` 或 `motion-handoff`。
- 继续使用全局 latent timeline 一次 VAE decode 作为独立输出策略，隔离 VAE 边界闪烁。

### 不做

- 不训练 VAE、音频后处理器或独立的 motion/optical-flow 网络。
- 不把双侧 bridge retake、joint MultiDiffusion、全局 reference memory 混入首版训练。
- 不使用 decoded-frame loss、帧插值或视频 crossfade 作为主训练机制。
- 不宣称 LoRA 可以消除独立 VAE decode 导致的所有闪烁。

## 4. 训练样本与窗口构造

训练集必须包含**原始时间连续、单镜头或缓慢连续运动**的视频，且保留同步原始音频。不能把两个独立片段拼接成正样本。

对于一段长视频，在一个随机边界 `c` 构造：

```text
原始连续视频:  ... [ A 的尾部 / B 的 prefix ] [ B 的未来 suffix ] ...
                             <--- K --->

训练给模型:              clean context      +     noisy future
训练监督:                    不计算损失              计算损失
```

其中：

- `K` 为 overlap，首版固定为当前生产基线的 **34 decoded frames**，即两个 17-frame Video VAE clip。
- 视频 overlap 对应 `34 / 17 * 5 = 10` 个视频 temporal latent token。
- 音频 overlap 从同一个物理时间边界推导：`round(34 / 24 * 40) = 57` 个音频 latent steps。
- suffix 是同一条原始视频在边界之后的真实内容，不能重新随机裁到另一时刻。
- 每一条样本随机化边界位置、镜头内运动方向、人物尺度、说话位置和环境声位置，避免 LoRA 记住固定构图。

首轮可用 `124` 帧（约 5.17 秒）的目标窗口训练，作为低显存 smoke 和机制验证；其可生成 suffix 为 90 帧。正式训练和主要验证统一使用 `345` 帧（约 14.375 秒）窗口。训练窗口长度可以略短于最终 15 秒输出，但 overlap、边界位置与推理机制必须一致；历史 362 帧实验不作为本 LoRA 的验收基线。

## 5. 与推理严格对齐的训练前向

对每一个窗口 B，先通过冻结 VAE 编码得到完整真实目标 latent：

```text
x0_video = VAE_video(B 的真实视频)
x0_audio = VAE_audio(B 的真实音频)
```

随后采样同一训练步的 video/audio timestep，并只给 future/suffix 加常规扩散噪声：

```text
history_video = x0_video[:K_video]
history_audio = x0_audio[:K_audio]

x_t_video = [history_video | q_t(x0_video[K_video:], epsilon_video)]
x_t_audio = [history_audio | q_t(x0_audio[K_audio:], epsilon_audio)]

mask_video = [0 ... 0 | 1 ... 1]
mask_audio = [0 ... 0 | 1 ... 1]
```

模型输入的语义必须和线上 handoff 一致：

```text
input_latents_* = history clean latent
denoise_mask_*  = preserve prefix / generate suffix
```

模型看到的 prefix 应保留为 clean context，并在 H3 token timestep 中标记为固定条件；suffix 使用当前随机 timestep 的 noisy latent。这样 LoRA 学到的是“读取历史状态后预测未来”，而不是在全噪声序列上做普通 SFT。

```text
                 Video history clean ─┐
                                       ├─> H3 DiT + LoRA ─> future flow prediction
Audio history clean ───────────────────┤
Text / current timestep / AV masks ────┘
```

## 6. 损失设计

### 6.1 首版损失：masked flow-matching

沿用 H3 的 video/audio flow-matching target，但仅计算 suffix：

```text
L_video = mean_{i in video suffix}( w_i * ||v_hat_i - v_i||^2 )
L_audio = mean_{j in audio suffix}( w_j * ||v_hat_j - v_j||^2 )
L = L_video + lambda_audio * L_audio
```

权重建议：

```text
history prefix                    w = 0
接缝后的第一个 video VAE clip     w = 3
接缝后的其余 suffix               w = 1

接缝后的约一个 video clip 时长音频  w = 2
其余 audio suffix                 w = 1
```

接缝带采用“一个完整 Video VAE clip”而不是任意几帧，保证时序分块对齐。视频和音频分别归一化后再求和，避免音频 token 数量主导训练。

**不对 preserve prefix 计算重建损失。** 该区域在推理中由 latent handoff 提供，其正确性不应占用 LoRA 容量；把它纳入普通 MSE 会鼓励模型学习复制历史，而不是学习过渡后的未来。

### 6.2 第二阶段候选，不进入首版

当首版已经稳定降低边界误差后，才评估下列附加项：

- 将模型预测转换到可比较的 `x0` 或 flow 轨迹后，在接缝带加入 latent temporal-difference loss，显式约束末尾历史速度与首个 suffix 速度。
- 以小比例使用模型生成的历史 tail 或受控 latent 扰动，减小“真实历史”与“模型生成历史”的训练-推理分布差。
- 对长对白加入音频韵律或语义边界采样，但仍以同步 AV 时间轴为准。

这些项不应先于 masked flow-matching。否则无法判断收益来自真正的 continuation 学习，还是来自损失项之间的偶然抵消。

## 7. LoRA 注入位置与初始超参数

LoRA 只注入 DiT：

```text
attn.qkv_proj
attn.out_proj
mlp.fc1
mlp.fc2
```

其中 attention LoRA 是首要部分，因为 continuation 的关键是让历史 AV tokens 影响未来 AV tokens；MLP LoRA 用于补充人物边缘、局部纹理、颜色和声学细节稳定性。

首轮推荐：

```text
rank = 32
alpha = 32
learning rate = 1e-4 起步，配验证集早停
训练 CFG = 与目标 checkpoint 的 CFG-distillation 设定一致
```

数据配比建议：

```text
75% masked continuation 样本
25% 普通完整 AV SFT 样本
```

普通 SFT 样本是正则项，降低 LoRA 仅在有 prefix 时工作正常、无 prefix 时画质或音频质量退化的风险。验证中必须分别测 continuation 与普通单窗口生成。

## 8. 两阶段训练计划

### 阶段 A：机制 smoke

目的：确认训练前向、mask 时间轴、AV 对齐、梯度和 checkpoint 导出正确。

- 使用少量高质量单镜头素材。
- `124` 帧窗口、34 帧 overlap、很小训练步数。
- 仅比较 base 与 LoRA 在固定 prompt、seed、边界上的差异。
- 通过条件：接缝后第一个 VAE clip 的人物/镜头跳变不恶化，普通单窗口画质不明显下降。

### 阶段 B：正式 continuation LoRA

目的：让模型覆盖真实的接续分布。

- 以 345 帧（约 14.375 秒）作为正式训练和主要验证窗口；124/243 帧仅用于 smoke 或低成本调试。
- 数据覆盖人物走动、转身、手势、推拉摇移、口播、环境声、音乐节拍和静态场景。
- 75/25 continuation-SFT 混合。
- 在每个固定训练步保存 LoRA，并用固定的 345 帧 x 2 窗口单镜头套件评估。

### 阶段 C：长窗口确认

目的：排除只适用于短 suffix 的假提升。

- 固定最佳 checkpoint。
- 以 `345` 帧 x 2 窗口生成约 27.3 秒结果（扣除 34 帧 overlap 后），作为正式接续验收样例。
- 只对 base / latent-handoff / latent-handoff+LoRA 做对照；先不加入 taper、motion handoff 或 bridge。

## 9. 推理路径

首版推理必须保持简单，便于归因：

```text
窗口 A 正常生成并保留 final AV latents
             |
             v
窗口 B 使用 latent-handoff：
  [A 的 clean AV overlap latent | B 的新噪声 suffix]
  [mask = 0                   | mask = 1]
             |
             v
H3 + Continuation LoRA 采样 B
             |
             v
各窗口 latent 拼为一条时间线，再统一 VAE decode
```

首版不使用 `taper-refine-v2`。它改变 overlap 内的采样轨迹，会混淆“LoRA 是否学会 suffix continuation”的结果。若 LoRA 证明有效，再以消融方式测试它与 v2 是否互补。

## 10. 验收标准

每个候选 LoRA 都在同一 checkpoint、prompt、seed、窗口长度、overlap 和统一 decode 条件下与 base latent-handoff 对比。

### 自动指标

- 接缝相邻帧像素 MAD 与感知特征差异。
- 接缝前后人物检测框中心、人体 pose 或光流的位移/速度差。
- 接缝前后亮度、色温和局部高频纹理差异。
- 音频能量、频谱、响度与语音 VAD 边界跳变。
- 30 秒后段相对于前段的闪烁率与画质退化。

### 人工检查

按盲测顺序检查：

1. 人物是否在第一个 suffix clip 中出现横跳、转向突变或比例变化。
2. 相机是否在接缝后突然改变运动方向或速度。
3. 脸、服装、光照、背景和阴影是否连续。
4. 口型、对话、环境声和音乐节拍是否跨接缝连续。
5. 普通单窗口生成的画质、运动和音频是否退化。

通过条件不是单个分数下降，而是：多数固定样例的视觉接缝不恶化，并在人物/相机连续性上可重复优于 base；同时普通生成不出现明显质量回退。

## 11. 已知边界与风险

| 风险 | 处理方式 |
| --- | --- |
| 训练只见过真实历史，推理历史是模型生成的 | 第二阶段再引入生成 tail 或受控 latent 扰动；首版先保留可归因的 teacher-forced 训练。 |
| LoRA 学会复制而不学会后续运动 | prefix loss 严格置零，提升第一个自由 clip 权重。 |
| 音频 token 多、主导联合目标 | 视频与音频独立求均值，再用显式 `lambda_audio` 合并。 |
| 数据有剪辑切点，模型学到跳变 | 只采样连续单镜头区间；数据预处理筛掉镜头切换和严重冻结尾部。 |
| 独立 VAE decode 仍闪烁 | 训练评估默认统一 latent timeline decode；该问题不归因于 LoRA。 |
| LoRA 改善接缝却损害普通生成 | 25% 普通 SFT 正则样本，并单独做单窗口回归验证。 |

## 12. 需要确认的决策

1. 首版是否以 **34 帧 overlap** 固定训练，后续再做 17/51 帧消融？建议是。
2. 训练数据是否优先限定为单镜头、人物连续动作和同步原声？建议是，先不要混大量剪辑素材。
3. 是否接受第一轮以 345 帧训练？当前决定是接受；243 帧保留为低成本调试，历史 362 帧结果不纳入本 LoRA 验收。
4. 是否接受首版只使用 masked flow-matching，先不引入显式 motion loss？建议是，先证明基础条件学习有效。
5. 是否接受以 `latent-handoff + LoRA + global latent decode` 作为唯一主对照，而非同时叠加 taper、motion 和 bridge？建议是，避免无法归因。

确认后，下一步应新建一个独立 OpenSpec change，例如 `h3-continuation-lora-training`，将本文拆成 proposal、design、spec 和可执行任务；之后才修改数据处理、训练前向、masked loss 与验证脚本。
