## Context

H3 的单次推理生成一个联合视频/音频窗口。现有 `H3ContinuationRunner` 能用 Retake 或 latent handoff 依次生成窗口，`joint-multidiffusion` 则已经实现一个受限的全局 noisy-latent 时间线：每个时间步从各窗口预测噪声，融合 overlap 后只执行一次全局 scheduler step，并最终统一 VAE decode。

顺序接续的根本问题不在拼接本身，而在两个窗口在 diffusion 期间形成了独立轨迹。后处理、bridge decode 和 overlap crossfade 都只能修饰已分叉的结果。Diff-VF 的核心是让局部重叠轨迹从采样过程开始即共享，并用远距离时间采样持续提供全局约束。

H3 同时具有视频 VAE 的 clip 对齐约束、32 kHz/40 latent-per-second 音频时间轴、联合 DiT 预测和较高显存占用。设计必须保持单窗口/Retake API 不变，且不得将未经验证的 TES 音频行为当作生产默认。

## Goals / Non-Goals

**Goals:**

- 实现可复现、可消融的 Diff-VF 风格长视频采样路径：HNI、WWS、TES 与 local-global fusion。
- 让所有视频窗口在一条全局 latent 时间线上采样，保证每个重叠 latent token 在每一步只被统一更新一次。
- 保持现有 sequential Retake、latent handoff 和 `joint-multidiffusion` 调用语义可用；新路径显式 opt-in。
- 以视频为首个完整验证模态，保持视频/音频时间边界一致，并为音频提供保守、可报告的策略。
- 建立从纯 CPU 合约测试到固定 seed GPU 消融的验证闭环，包含质量、连续性、时延和显存记录。

**Non-Goals:**

- 不修改 H3 DiT、VAE、文本编码器权重，也不声称推理期方法可替代 continuation LoRA/SFT。
- 不在此 change 中支持无限长度、异形窗口、窗口级动态分辨率或断点续跑的 Diff-VF 联合采样。
- 不将 TES 直接应用到音频，除非后续 GPU 消融证明它不会恶化语音韵律、相位和边界质量。
- 不以帧混合、插帧或二次 diffusion repair 作为解决 diffusion-trajectory 分叉的主方案。

## Decisions

### 1. 在 continuation runner 中新增独立 `diff-vf` 模式

`H3ContinuationRunner` 继续负责 segment plan、物理时间线、提示词、seed、输出和评测元数据；H3 Pipeline 继续只处理一次形状固定的联合 denoise 请求。新模式由 runner 编排全局 latent，复用 Pipeline/DiT 的 prompt 编码、noise prediction、scheduler 和 VAE 接口。

```text
plan + DiffVFConfig
        |
        v
H3ContinuationRunner (diff-vf)
  | HNI -> global video/audio x_T
  | for each timestep:
  |   WWS local predictions
  |   TES video predictions (configured steps only)
  |   local-global fusion -> one global step
  v
global latent decode -> timeline/evaluation artifacts
```

替代方案是把窗口循环塞进 `MiniMaxH3Pipeline.__call__`。这会改变单窗口 API，并混淆 window orchestration、state persistence 和模型推理；不采用。

### 2. 使用显式全局时间坐标和窗口计划

初始化阶段先解析所有 segment 为相同空间分辨率、相同窗口帧数、相同 overlap 帧数的 `ResolvedWindow`。时间线由 `start_frame`、`end_frame`、`owned_start_frame` 表示；视频可为 `17n+5`，overlap 必须满足视频 VAE clip 对齐。音频 sample/latent 边界从同一视频物理时间推导。

全局视频 latent 和音频 latent 分别使用完整、连续时间轴。每个 local window 仅保存全局张量的 view，任何预测回写都按全局 index scatter。这样重叠 token 不会同时存在两个独立的 latent 副本。

替代方案是每个窗口保留自己的 latent 再同步 overlap。它更省峰值显存，但会重新引入双轨迹和 scheduler 顺序依赖；不采用。

### 3. HNI 以共享基准噪声建立长程相关性

HNI 为第一个窗口采样基准高斯噪声 `z_base`。第 `i` 个窗口的初始噪声使用其在全局时间线上的基准部分及独立新增噪声 `z_i`：

```text
x_T[i] = sqrt(w_i) * z_base[global_slice(i)]
       + sqrt(1 - w_i) * z_i
```

`w_i` 为 `[0, 1]` 的已记录配置，可设为常数或窗口索引调度；重叠位置必须从同一全局 `x_T` 读取，不能再单独重采样。实现必须使用独立、可推导的 RNG stream（base、window noise、TES permutation），使改变一个消融项不意外改变其他随机源。

HNI 保持人物、场景、光照等全局统计相关性，但不保证接缝局部运动连续，因此不将它作为 WWS 的替代。

替代方案是所有窗口共享完全相同噪声，或全部独立采样。前者会过度复制构图并冻结动态，后者没有长程相关性；不采用。

### 4. WWS 在每一步做预测域加权融合

每个 scheduler timestep，系统从全局 latent 收集每个 local window，调用模型获得对应 video（以及配置允许时 audio）噪声预测，再将预测散射到全局预测累加器。每个 token 的最终预测为所有覆盖该 token 的归一化加权平均；每个模态随后只调用一次 scheduler step。

权重实现两种可选策略：`center-distance`（论文的中心距离权重）与当前原型使用的 `cosine-squared`。边缘非 overlap token 权重为一；overlap 权重必须非负且归一化和为一。诊断输出要保存有效权重图和每个 token 覆盖次数。

替代方案是对去噪后的 latent 做加权。它会在不同 scheduler 轨迹之间混合状态，通常产生色噪声或细节塌陷；不采用。

### 5. TES 只从视频分支开始，并以固定置换窗口采样

TES 根据可复现的时间重排 `P` 将全局视频时间索引交错成若干 sparse windows；每一个 sparse window 覆盖多个原本远离的时间位置。在指定 timestep 上，系统 gather `x_t[P]`、预测噪声、再 inverse-scatter 至原时间线，产生与 WWS 同 shape 的全局预测 `eps_tes`。

`P` 和 sparse-window 划分在一个运行内不可变，并满足：所有原时间 token 恰好覆盖一次、没有越界、最终不足的窗口使用 mask 而非复制 token。TES 的 timestep schedule 必须显式给出，默认只在高噪声早期阶段运行，避免后期破坏局部运动细节。

视频先行是刻意的风险控制。音频在 Diff-VF 模式中默认使用 WWS（若启用）或保守的连续 latent 路径；只有 `audio_tes=experimental` 才允许对齐地运行音频 TES，并必须在 manifest 中标明实验性。

替代方案是直接按视频置换重排音频 latent。视频和音频 token rate、语义局部性和位置编码均不同，容易损害音节/相位连续；不作为默认。

### 6. Local-global fusion 作用在噪声预测而不是 latent

当 timestep 同时存在 `eps_wws` 与 `eps_tes` 时，按配置曲线 `c(t)` 融合：

```text
eps = c(t) * eps_wws + (1 - c(t)) * eps_tes
```

默认使用随去噪推进递增的局部权重：高噪声早期让 TES 提供全局结构，低噪声后期让 WWS 保护短期运动和细节。未执行 TES 的步直接使用 `eps_wws`，不引入零预测。视频和音频的融合策略、覆盖范围和系数必须分别记录。

替代方案是将 TES 结果作为一次额外 scheduler step 或在生成结束时混合 decode。两者都会改变噪声调度且无法保证一致的 state update；不采用。

### 7. 统一 decode 受资源预算控制

Diff-VF 结束后只从全局视频/音频 latent decode 一次，消除跨窗口 VAE decoder 上下文不同产生的边界闪烁。运行前必须估算 global latent、模型 activations 和 decode 工作区；超出 `max_decode_memory_gib` 时：

1. 默认拒绝运行并报告估算和可用预算；
2. 若调用者显式选择 `decode_strategy=temporal-chunk`，使用重叠 VAE decode 分块，保留并报告这不是统一 decode；
3. 禁止静默回退为逐窗口 decode。

替代方案是始终逐窗口解码并拼接。其显存最低，但正是此前接缝闪烁的来源之一；不作为 Diff-VF 默认。

### 8. 实现以阶段化门禁推进

代码按 HNI、WWS 规范化、TES、融合、decode/evaluation 五个阶段提交，每阶段先有 CPU 测试再有小窗口 GPU 固定 seed 消融。Diff-VF 模式保持 `experimental`，只有在既定 prompt/seed、相同 scheduler steps 下满足连续性改善且没有显著质量或资源回归时，才可建议作为用户可选路径。

## Risks / Trade-offs

- [全局 latent 和统一 decode 超出 H100 80 GiB] → 运行前估算并拒绝，提供显式 temporal-chunk decode 诊断路径，不静默改变算法。
- [TES 远距离 gather 破坏局部运动、引入色噪声] → 只在视频高噪声早期启用，独立保存 TES-only、WWS-only、fusion 消融结果。
- [音频 TES 破坏语音和相位] → 默认禁用；用独立音频边界指标决定是否继续。
- [非同形窗口和动态 prompt 无法批量化] → 首版只接受同形完整计划；检测后在启动前失败。
- [HNI 权重提高一致性但降低动态] → 将 `w` 暴露为记录参数，比较 `w=0`、中等值和 `w=1`。
- [position/RoPE 外推使过长时间线退化] → 限制首版支持长度、记录绝对位置，TES 不替代后续 FLEX 类位置编码扩展研究。
- [联合预测显著增加时延] → 报告每步 window/TES 调用数、峰值显存和总耗时，并保留既有顺序模式。

## Migration Plan

1. 引入类型化 `DiffVFConfig`、解析后全局窗口时间线和不执行模型的合约测试；默认不改变。
2. 将当前 joint MultiDiffusion 代码提取为 WWS 内核，验证与现有固定 seed 原型的输出/时间线等价。
3. 接入 HNI，分别验证随机流、`w` 边界和 overlap 单一所有权。
4. 接入视频 TES 与 fusion，先做极短 GPU smoke，再做 15 秒 x 2 窗口固定 prompt/seed 消融。
5. 接入全局 decode 预算门禁、逐模态评测和可复现实验报告；未通过质量门禁时保留模式但不推荐为默认。

回滚只需不选择 `--mode diff-vf`。既有 Retake、latent handoff、joint-multidiffusion 和单窗口调用保持各自当前语义，输出媒体/manifest 均为附加产物。

## Open Questions

- H3 DiT 是否能在不改变其 position-id 构造的前提下接受 TES 稀疏时间序列，还是需要显式向模型传递原始绝对时间索引？
- WWS 与 TES 是否应共享同一 prompt，还是允许 TES 使用仅包含全局 scene anchor 的压缩条件？
- 全局统一 decode 在目标卡上的实际峰值显存是多少，temporal-chunk decode 需要多大 halo 才能无可见边界？
- 哪组人工盲测与自动指标可作为将 `diff-vf` 从实验模式提升为稳定可选项的门槛？
