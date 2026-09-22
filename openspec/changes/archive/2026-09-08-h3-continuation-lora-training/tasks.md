## 1. 数据索引与样本划分

- [x] 1.1 定义 continuation 样本索引 schema，包含 `sequence_id`、源视频/音频路径、shot 区间、窗口起止时间、prompt、ASR/caption 清洗结果和过滤状态。
- [x] 1.2 实现流式 JSONL 扫描器，逐行解析 `data_with_face_and_speech_and_caption.jsonl`，统计坏行、缺失路径、无效 JSON 和字段缺失，不将全量记录载入内存。
- [x] 1.3 实现 `--- cut: frame_gap=... ---` 及可用 shot 元数据解析，无法确认 shot 边界时禁止生成跨边界 continuation 样本。
- [x] 1.4 实现窗口候选生成和 `17n+5` 长度/34 帧 overlap/17+17 区域校验，输出 124/243/345 帧配置的候选索引。
- [x] 1.5 实现 prompt 清洗与回退策略，优先使用可靠 prompt/ASR 摘要，过滤 caption 模板残留、乱码和空文本，并保留清洗版本号。
- [x] 1.6 按 `sequence_id` 实现确定性的 train/validation/test 划分，输出 split manifest、源序列计数和过滤统计。
- [x] 1.7 添加 CPU 单元测试：流式内存边界、shot cut 拒绝、合法窗口生成、sequence-level split 隔离和 prompt 清洗。

## 2. 视频音频裁剪与 latent cache

- [x] 2.1 实现视频裁剪/重采样到 24 fps、音频裁剪/重采样到 32 kHz，并从同一物理时间区间计算视频帧、音频 sample 和 audio latent step 边界。
- [x] 2.2 实现 H3 Video/Audio VAE 批量 cache 入口，支持按 split、分辨率和窗口长度分批运行，避免一次性占满磁盘或显存。
- [x] 2.3 定义 cache 文件布局和 metadata，保存 VAE/checkpoint 标识、dtype、shape、fps、sample rate、shot id、源文件 hash 和窗口坐标。
- [x] 2.4 实现 cache 读取前的 schema/hash/shape/dtype/采样率校验；不匹配时给出 actionable error 并拒绝静默回退。
- [x] 2.5 为无音频样本实现显式静音策略，记录 `audio_missing` 标记，不把缺失音频误当作真实对白。
- [x] 2.6 添加 CPU fake-VAE 测试：latent 时间边界、视频/音频物理对齐、非法 `17n+5` 长度和 metadata mismatch 拒绝。

## 3. Taper-refine-v2 训练前向与损失

- [x] 3.1 新增 continuation batch/collator，将目标窗口拆成 history clean latent、transition band、suffix 及对应 video/audio mask。
- [x] 3.2 实现双噪声构造：按 timestep 生成 `epsilon_main` 和 `epsilon_anchor`，分别计算主 noisy target 与 timestep-aligned noisy anchor。
- [x] 3.3 实现区域输入组合：hard core 使用 clean history，transition 使用从 1 到 0 的 ramp 混合，suffix 使用主 noisy target，并保证 video/audio 边界来自同一物理时间。
- [x] 3.4 实现与噪声来源一致的 flow target，确保 transition target 对应 anchor/main 的混合噪声，不把 clean latent 直接作为高噪声 target。
- [x] 3.5 实现 masked/weighted flow loss：hard core=0、transition=0.5、first suffix clip=3.0、remaining suffix=1.0，且权重可配置。
- [x] 3.6 实现 video/audio 独立有效 token 归一化及 `lambda_audio` 组合，记录模态独立 loss 和有效计数。
- [x] 3.7 保持现有 H3 `training_cfg_scale`、CP rank/world size、gradient checkpointing 和 scheduler 接口兼容。
- [x] 3.8 添加 CPU 数学测试：ramp 单调性、hard core 零梯度、首个 suffix clip 高权重、双噪声 target 对齐、模态归一化和边界 shape。

## 4. Continuation LoRA 训练入口

- [x] 4.1 在现有 H3 training module 基础上新增 continuation 任务配置和 CLI，支持 dataset manifest/cache、overlap、区域权重、audio lambda、seed 和输出目录。
- [x] 4.2 将 LoRA 默认注入 `attn.qkv_proj`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2`，冻结 DiT base、VAE、文本编码器、RoPE 和 scheduler，并检查实际可训练参数集合。
- [x] 4.3 支持 bf16、gradient checkpointing、可选 CP-aware dataloader/随机源同步和单卡 smoke 配置；不改变现有 SFT/Lora 脚本默认行为。
- [x] 4.4 实现 checkpoint 保存/恢复，包含 LoRA 权重、optimizer/scheduler、global step、manifest hash、区域权重和随机种子策略。
- [x] 4.5 实现 validation-only 模式，在无 CUDA/模型权重时检查 manifest、cache、窗口算术、LoRA target、loss 配置和 CP 拓扑。
- [x] 4.6 实现 LoRA 导出和推理加载 smoke test，验证显式 scale=0 时与 base 行为一致、scale>0 时 adapter 正确生效。
- [x] 4.7 添加训练脚本/配置示例：124 帧 CPU 配置检查、345 帧单卡 smoke、345 帧正式训练和断点恢复命令；不在本任务中自动启动 GPU 训练。

## 5. 接续推理与质量评测

- [x] 5.1 为评测固定 checkpoint、segment plan、prompt、overlap、scheduler、seed 和一次性 VAE decode 的组装流程。
- [x] 5.2 实现 base+taper-refine-v2 与 LoRA+taper-refine-v2 的配对 continuation runner，支持 LoRA scale 和 transition 配置矩阵。
- [x] 5.3 实现视频接缝指标：亮度/颜色差、人物检测框位移、光流/latent velocity 差、首个 suffix clip 的闪烁统计和全链趋势。
- [x] 5.4 实现音频接缝指标：RMS/响度差、频谱 discontinuity、相位/波形突变近似、A/V 边界偏移和可选 crossfade 时长。
- [x] 5.5 实现 overlap、transition weight、suffix weight、LoRA rank/scale、样本类别的消融报告，输出机器可读 JSON 和可读 Markdown。
- [x] 5.6 添加回归门槛：LoRA 必须改善边界指标且整体画质/运动指标不超过配置阈值，否则标记实验失败。
- [x] 5.7 添加无 CUDA 的评测规划和报告生成测试，并记录未执行的 GPU、长视频、CP 和主观质量验证。

## 6. 文档、兼容性与交付

- [x] 6.1 编写 continuation 数据集构建文档，说明源 JSONL 字段、shot-safe 规则、窗口长度、音画对齐和 cache 命名。
- [x] 6.2 编写 LoRA 训练与推理文档，说明 `taper-refine-v2` 区域语义、权重含义、LoRA target、加载方式和已知限制。
- [x] 6.3 更新 H3 示例索引，区分旧变更的 training-free 方法与本变更的训练型 LoRA；明确禁止跨 shot 正样本。
- [x] 6.4 运行 targeted CPU tests、静态检查和 `openspec validate h3-continuation-lora-training --strict`。
- [x] 6.5 运行 `openspec validate h3-av-continuation --strict`，确认旧变更的 LoRA 任务已标记为 out of scope 且 training-free 任务描述一致。
- [x] 6.6 汇总 smoke 结果、数据过滤统计、LoRA checkpoint 元数据和评测报告，列出尚未执行的 GPU/长时训练作为 deferred validation。
