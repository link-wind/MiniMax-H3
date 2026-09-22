## Why

MiniMax-H3 的 latent handoff 能保留重叠区域的内容，但没有学习“从历史运动状态进入第一个自由 suffix clip”的条件分布，因而会出现人物偏移、光照突变和边界闪烁。`taper-refine-v2` 能通过 timestep-aligned noisy anchor 改善接续，却仍属于推理时的近似约束；需要使用真实连续单镜头数据训练一个 continuation LoRA，使模型直接学习 `[hard core | transition band | suffix]` 的状态交接。

## What Changes

- 新增面向 H3 continuation 的流式数据索引构建流程，消费现有 `data_with_face_and_speech_and_caption.jsonl`，只在同一原始 shot 内生成训练样本，并按 `sequence_id` 做数据划分。
- 新增视频/音频时间对齐、24 fps / 32 kHz 规范化、H3 VAE latent 缓存和样本质量过滤，避免跨 shot、部分 VAE clip、音画边界不一致的样本进入训练。
- 新增 `taper-refine-v2` teacher-forced 训练前向：使用 clean history、timestep-aligned noisy anchor 和主噪声构造 hard core、transition band、suffix 三个区域。
- 新增区域加权 masked flow-matching loss：hard core 不计损失，transition band 低权重，第一个完整 suffix video clip 高权重，后续 suffix 使用普通权重，并支持音频损失权重配置。
- 新增 H3 DiT LoRA 训练入口和配置，冻结 VAE、文本编码器、RoPE、scheduler 及基础模型权重，仅训练 attention/MLP 的 LoRA 参数。
- 新增 continuation LoRA 的 CPU 配置检查、GPU 训练启动入口、断点恢复、LoRA 导出和推理加载兼容性检查。
- 新增接缝评测与消融：对比 base+taper-refine-v2、LoRA+taper-refine-v2、不同 transition 权重/overlap/LoRA 强度，并报告人物位置、亮度、运动状态、视频闪烁和音频边界指标。
- 将旧变更 `h3-av-continuation` 的 LoRA/SFT 任务移出范围；旧变更只负责 training-free continuation 方法验证。

## Capabilities

### New Capabilities

- `h3-continuation-dataset`: 从标注 JSONL 构建 shot-safe、音画对齐、可缓存 H3 latent 的 continuation 训练数据集。
- `h3-continuation-lora-training`: 基于 `taper-refine-v2` 区域构造和加权 flow-matching 目标训练 H3 continuation LoRA，并提供配置、恢复和导出能力。
- `h3-continuation-lora-evaluation`: 评测 continuation LoRA 对接缝闪烁、人物/构图错位、光照突变、运动连续性及音画同步的影响。

### Modified Capabilities

- 无。旧的 `h3-av-continuation` 变更仅调整未完成任务的范围说明，不修改已归档 capability 的运行时需求。

## Impact

- 新增数据准备、latent cache、训练和评测模块，预计位于 `examples/minimax_h3/model_training/`、`examples/minimax_h3/model_inference/` 及对应 `diffsynth` 工具模块。
- 复用 H3 现有 VAE、scheduler、DiT、LoRA/adapter 和 CP-aware training 基础，不新增生成模型或外部音频生成依赖。
- 训练需要可访问 MiniMax-H3 checkpoint 和 CUDA；索引构建、配置检查、数据过滤和损失单元测试必须支持 CPU/无模型环境。
- 该变更默认不改变现有 H3 单窗口或 training-free continuation 的行为；LoRA 仅通过显式配置加载。
