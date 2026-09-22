# MiniMax-H3 接续生成已使用方法总览

## 1. 目的与现状

本文记录本项目已经实现或实际试跑过的接续方法，作为后续实验的共同基线。它不是最终推荐方案，也不把一次 smoke 的可运行性等同于视觉质量提升。

当前主要问题是 15 秒边界处的单次构图或运动错位，以及部分实验中后段出现的闪烁。根因更接近**相邻窗口从不同初始运动/构图状态出发**，而不是单纯的 MP4 拼接、音频 crossfade 或 VAE 后处理问题。

## 2. 通用基础设施

所有方法共享以下约束：

- 视频以 24 fps 生成；H3 视频时间轴按 17 帧 clip 对齐。
- 音频以 32 kHz 输出，audio latent rate 为 40 steps/s。视频和音频先共享物理时间边界，再映射到各自的 latent 下标。
- 典型两窗口实验使用 362 帧窗口和 34 帧 overlap；视频 overlap 为 10 个 latent steps，音频 overlap 约为 57 个 latent steps。
- 使用全局 prompt 固定人物、场景、光照、声学环境和镜头风格，使用每段 prompt 指定动作和台词。
- 输出中记录 continuation manifest、视频帧边界、音频样本边界和实验配置；比较时固定 plan、seed、分辨率与步数。

## 3. 顺序接续方法

### 3.1 Retake-hard

**机制**：将上一窗口末尾的解码视频和音频作为当前窗口的 Retake 输入。overlap 区域通过 mask 固定，只生成后缀。

**优点**：直接使用 H3 已训练的 Retake 条件路径，最接近官方能力，是稳定的生产基线。

**问题**：历史先经过 VAE decode，再作为 Retake 上下文重新编码；接缝仍可能发生姿态、构图和相机运动跳变。

**状态**：保留为默认基线。

### 3.2 Latent handoff

**机制**：缓存上一窗口的 clean video/audio latent，下一窗口直接将其作为 overlap 前缀条件，避免 decode-reencode 循环。

**优点**：历史状态保存得更完整，减少 VAE 往返误差；在部分 15 秒测试中比 Retake-hard 有更小的边界误差。

**问题**：H3 并未专门训练为“从精确 latent 状态继续”，因此 hard prefix 只能保证重叠内容相同，不能保证后缀第一段运动轨迹自然延续。

**状态**：有效的对照路径，但不替代 Retake-hard 作为默认。

### 3.3 Taper-refine（已归档）

**机制**：对 overlap 使用从强到弱的静态 soft overlap，将历史 clean latent 与当前窗口 latent 线性混合。

**结论**：不推荐。实验中历史 anchor 被过早放开，人物或镜头会在真正的后缀边界前变化，音频边界指标也恶化。

### 3.4 Taper-refine-v2：timestep-aligned noisy anchor（已归档）

**机制**：不直接混合 clean latent。每个 denoise timestep 都将历史 clean latent 按当前 scheduler 加噪，得到 noisy anchor；overlap 前半段保留 hard core，后半段以逐渐减弱的权重投影该 anchor。

**优点**：anchor 与当前噪声层级一致，避免将 clean 状态错误注入高噪声采样过程；能比静态 soft overlap 更好地保持动作轨迹。

**问题**：仍可能在接缝处闪一下，因为它约束的是 overlap 内生成轨迹，不能消除两个窗口的全局状态差。

**状态**：已归档。当前已从训练/推理路径移除，未显示稳定优于 latent handoff 的收益。

### 3.5 Motion-state handoff

**机制**：从上一窗口尾部 latent 估计短时 velocity，并投影为当前窗口后缀开头的 noisy motion anchor，试图显式传递运动方向。

**问题**：H3 latent 中的简单差分不等同于可控的物理运动状态；试验中出现彩色噪声和不稳定闪烁。

**状态**：当前不作为主线。若继续研究，应改用训练过的光流、姿态、相机轨迹或 motion token，而不是直接 latent 差分。

## 4. 接缝局部修复方法

### 4.1 Context decode / 局部 bridge decode

**机制**：当前窗口只扩散一次；解码阶段额外带入上一段 latent 尾部上下文，生成边界附近的 bridge decode，再仅替换边界后的少量帧。

**作用**：诊断或缓解 VAE 时间感受野造成的边界不一致。

**结论**：不是根治方案。它没有改变两窗口的扩散初始状态，过大范围替换还可能将闪烁传播到后段。

### 4.2 中间插帧

**机制**：在两段输出之间额外插入少量过渡帧，不覆盖已有帧。

**作用**：改善播放时长上的突变感。

**结论**：属于展示层补偿，不解决 latent 轨迹分叉；不应作为接续质量的主要指标。

### 4.3 双侧接缝 Retake

**机制**：先得到右窗口完整结果，再以左窗口尾部和右窗口开头为 decoded 双侧上下文，对接缝附近一个 clip 执行第二次 Retake 重绘。

**优点**：对中间编辑而言，H3 的非因果注意力可以同时使用左右信息。

**问题**：试验中出现接缝区域姿态跳变。固定的 decoded 左右帧没有提供连续的 noisy latent 轨迹，第二次生成仍可能选到不同动作解。

**状态**：不作为单镜头接续的主线。

## 5. 长期外观与高层锚点

### 5.1 全局/边界 reference

**机制**：将首帧或上一段末帧作为 keyframe/reference，同时保持全局人物、场景和光照描述。

**作用**：增强人物身份、服装、场景布局与色调的一致性。

**限制**：reference 更擅长约束外观，不能表达速度、姿态变化率或镜头轨迹；过强的 reference 还可能压制新动作。

### 5.2 结构化 H3 prompt 与 segment state

**机制**：全局 prompt 固定不可变事实；segment prompt 只写该窗口新增动作、镜头和对白。台词采用 H3 格式，例如 `<d>[Chinese] ...</d>`。

**作用**：降低文本条件本身在窗口间改变人物、镜头或声学场景的概率。

**限制**：prompt 是语义约束，不是连续运动状态；不能单独消除接缝闪烁。

## 6. 联合扩散方法（已归档：Diff-VF / Joint MultiDiffusion）

### 6.1 Joint MultiDiffusion（已归档）

**机制**：所有窗口共享一条全局 video/audio noisy latent 时间线。每个 denoise step：

1. 每个 WWS 窗口从同一 `Z_t` 读取自己的局部 latent；
2. 分别进行模型前向；
3. 在 overlap 融合各窗口 prediction；
4. 对全局 timeline 各执行一次 scheduler step；
5. 所有步完成后，只做一次全局 VAE decode。

**优点**：从扩散过程中直接消除“第一个窗口已经完成、第二个窗口重新随机起步”的差异。观察上可保持跨接缝的俯身、低头等连续运动轨迹。

**问题**：预测级融合不等同于论文的状态级融合；局部画质和音频边界指标未稳定优于较简单的基线，显存与计算成本也显著增加。

**状态**：已归档。诊断基线已停止作为生产路径，相关代码与产物已移入 `archive/`。

### 6.2 Diff-VF existing 语义（已归档）

**机制**：在 Joint MultiDiffusion 框架上增加：

- HNI：共享/混合初始噪声，增强窗口初始状态相关性；
- WWS：使用 cosine-squared 或中心距离权重融合窗口 prediction；
- TES：视频 sparse temporal sampling，并在早期与 WWS prediction 融合；
- 全局 VAE decode：消除逐窗口 decode 的边界变量。

**已有结果**：15 秒、50-step 固定条件下，mixed HNI 相对独立 HNI 改善了接缝和音频边界指标；TES 可改善长程稳定性，但可能恶化局部接缝。

**限制**：该路径的 WWS/TES 是工程化 prediction-level 近似，不是论文 Diff-VF 的严格状态级采样。

**状态**：已归档。

### 6.3 Diff-VF paper-strict 语义（已归档）

**机制**：为复现论文而新加的 opt-in 路径：

1. **论文 HNI**：首 clip 使用 `Z_T^0`；后续 clip 使用 `sqrt(1-w) * Z_T^0 + sqrt(w) * epsilon_n`，再在 clip 组内循环重排。
2. **状态级 WWS**：每个窗口都从同一个 `Z_t` 独立进行模型前向和 scheduler step；随后用归一化中心距离权重融合 `Z_(t-1)`，而不是融合 prediction。
3. **状态级 TES**：早期步把全局视频时间轴拆成 `n, n+N, ...` 交错序列，各自从同一 `Z_t` 独立 step，再 scatter 回全局时间线。
4. **论文融合日程**：用余弦系数融合 local WWS state 与 global TES state。H3 音频继续走连续 WWS state path，未伪造未验证的音频 TES。
5. **绝对时间位置**：所有 WWS/TES forward 保留其在全局 timeline 中的 source indices，避免 3D RoPE 将后段误当作时间零点。

**已验证**：256x448、两窗口、1-step GPU smoke 成功得到 690 帧统一解码视频，manifest 包含 TES windows、active step、融合系数和 HNI seeds。

**数值修复**：中心距离权重必须以 float32 计算后再转 bfloat16；否则长音频时间轴末端因 bfloat16 位置量化可能被错误赋零权重。

**状态**：已归档。调度语义和端到端可运行性当时已验证；因方向不再继续，未完成最终质量对比。

## 7. 当前结论与下一步边界

| 方法 | 主要处理层级 | 当前定位 |
|---|---|---|
| Retake-hard | 已解码历史条件 | 生产基线 |
| Latent handoff | 历史 latent | 重要对照 |
| Taper-refine | 静态 overlap 混合 | 已归档 |
| Taper-refine-v2 | timestep-aligned anchor | 已归档 |
| Motion handoff | 粗略 latent 运动差分 | 暂停 |
| Bridge / 插帧 / 双侧 Retake | 解码或二次修复 | 非根治辅助 |
| Reference / 结构化 prompt | 外观与语义条件 | 必要但不足 |
| Joint MultiDiffusion | prediction-level 联合扩散 | 已归档 |
| Diff-VF existing | 工程化 HNI/WWS/TES | 已归档（曾有消融结果） |
| Diff-VF paper-strict | 论文状态级联合扩散 | 已归档 |

当前生产/对比基线为 Retake-hard、Latent handoff 与 masked-av-v14。Diff-VF / Joint MultiDiffusion 已归档：其 WWS/TES 属 prediction-level 近似，未优于更简单基线，且显存与成本更高，已停止进一步比较。若后续需要状态级联合扩散，可到 `archive/code/` 恢复。

## 8. 训练型 continuation LoRA（与 training-free 方法区分）

`h3-av-continuation` 中的 `masked-av-v14`（含 Retake-hard / latent handoff）是
training-free 推理方法：它用上一窗口的 hard prefix 约束 overlap，但不改变模型对
第一个自由 suffix clip 的生成分布。早期的 `taper-refine-v2/v3` training-free 推理
路径已归档，见 `archive/`。

`h3-continuation-lora-training` 是独立的新变更：它在同一真实单镜头数据上构造
`[hard core | transition band | suffix]` teacher-forced 前向，训练一个只注入
`attn.qkv_proj/attn.out_proj/mlp.fc1/mlp.fc2` 的 continuation LoRA。训练数据
必须满足：

- 完整窗口落在一个原始 shot 内；
- 视频/音频来自同一物理时间区间；
- latent cache 的 VAE/hash/shape/dtype/采样率匹配；
- 按 `sequence_id` 划分，禁止相邻窗口跨 split 泄漏。

推理时新增 `--lora` 和 `--lora-scale`；不加载 LoRA 时，原有 training-free
方法与单窗口行为保持不变。配对接续评测固定 checkpoint、plan、prompt、
overlap、scheduler、seed 和统一 VAE decode，用来区分训练型 LoRA 与
training-free 方法的实际收益。

详见 [MiniMax-H3 continuation LoRA 数据与训练指南](./minimax_h3_continuation_lora_training.md)。
