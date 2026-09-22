# MiniMax-H3 30s 长视频音频 SFT 训练与推理优化

## 1. 项目概述

本项目基于 DiffSynth-Studio 对 MiniMax-H3 进行 30s 长视频 + 音频 SFT 训练和推理优化。MiniMax-H3 的 30s packed 序列长度达到 85568 token，原始单卡方案无法承载完整权重、attention 和训练状态，项目通过 Ring Context Parallel、DeepSpeed ZeRO-3、CP-aware 数据与 loss、梯度 checkpointing 以及多节点训练脚本，将 24 卡 CP=8 的峰值显存压到约 45GB，并完成 30s 推理链路验证。

## 2. 技术栈

- 语言与框架：Python、PyTorch、DiffSynth-Studio、DeepSpeed、Accelerate
- 分布式训练：torchrun、NCCL、DeepSpeed ZeRO-3
- 长序列并行：Ring Context Parallel、Ring Attention、flash-attn
- 模型：MiniMax-H3 FL2VA DiT
- 硬件：H100 80GB、8 卡/16 卡/24 卡多节点集群
- 数据：30s 视频 + 音频 packed SFT 数据，719 帧、480x832

## 3. 个人职责与核心工作

### 3.1 Ring Context Parallel

- 新增 `diffsynth/core/context_parallel/` 模块，包含 packed sequence 分片元数据、Ring attention、CP process group、CP-aware dataloader 和训练辅助函数。
- 实现 packed 序列的 CP 分片，将 85568 token 按 CP=8 切成每卡约 10696 token。
- 实现 Ring attention，前向通过 KV 环轮转和 online softmax 合并，避免完整 85568x85568 attention 矩阵。
- 接入 flash-attn varlen blockwise 计算，无 flash-attn 或 CPU 环境时回退到 dense 参考实现。
- 反向传播复用同一套 KV 环，K/V 梯度通过 reduce-scatter 归约回各 chunk owner，避免每卡保存全局 K/V 梯度。

### 3.2 DeepSpeed ZeRO-3 显存优化

- 为训练入口增加 `--initialize_model_on_cpu`，让模型在 CPU 上初始化后直接交给 DeepSpeed ZeRO-3 分片。
- 提供 `deepspeed_zero3_cp8.json` 和 `deepspeed_zero3_cp8_no_cpu_checkpoint.json` 两套 ZeRO-3 配置。
- 修复 runner 在 CPU 初始化分支先执行 `model.to(GPU)` 的问题，避免 66GB BF16 权重临时占满显存。
- 将 `train_micro_batch_size_per_gpu` 固定为 1，适配 packed 长序列训练。

### 3.3 CP-aware 数据与 Loss

- 实现 CP-aware dataloader，CP 组内各 rank 消费同一批数据，DP 组之间消费不同样本。
- 通过 `--cp_seed` 固定 CP 内数据顺序，保证多卡数据一致性。
- 实现 timestep、噪声的 CP broadcast，保证同一 CP 组使用相同采样条件。
- 将 loss 改为本地 video/audio target 分片计算，再做分布式加权归约，避免把全局 token 拉到单卡。

### 3.4 梯度 checkpointing 修复

- 修复 DeepSpeed CPU checkpointing 将共享 tensor 清空的问题，解决 `both arguments to matmul need to be at least 1D` 报错。
- 为所有 DiT block 共享的 `t_emb`、`rope_freqs` 设置 `no_checkpointing=True`，避免第一个 block 后共享输入被释放。

### 3.5 多节点训练与保存

- 新增 24 卡 3 节点启动脚本，覆盖 CP=8、CP=4、CP=2，以及 smoke/full 和 no-cpu-checkpoint 变体。
- 支持 `--dataset_repeat`、`--save_steps` 等训练参数，可中途保存 66GB 级别的 checkpoint。
- 记录 8 卡、16 卡、24 卡不同拓扑下的显存和速度数据，沉淀到 `ring_cp_30s_sft.md`。

### 3.6 30s 推理验证

- 新增 30s local CP 推理脚本，支持 4 卡分布式 denoise，并通过 CP group 聚合本地 video/audio rows。
- 完成 VAE CPU offload，先释放 DiT 再执行视频和音频解码，适配 719 帧长视频显存。
- 使用训练产生的 checkpoint 输出 30s 视频 + 音频文件，验证训练链路与推理链路打通。

## 4. 项目成果与关键数据

- 30s packed 序列：85568 token，其中 video 82680、text 447、audio 2398。
- 8 x H100、CP=8 smoke 实测：约 27.03s/it，峰值显存约 73GiB / 80GiB。
- 24 卡、CP=8 实测：峰值显存约 45GB，速度约 37s/step。
- MiniMax-H3 DiT 约 33.1B 参数，BF16 checkpoint 约 66.2GB。
- 通过 ZeRO-3 24 卡分片，模型/优化器态从约 66GB/卡降到约 22GB/卡。
- 生成 30s 推理产物：`models/inference/latest_30s.mp4`。

## 5. 项目经历（简历版）

### MiniMax-H3 30s SFT｜基于 DiffSynth-Studio 的长视频音频生成模型长序列训练与推理优化框架

*技术栈：Python、PyTorch、DiffSynth-Studio、DeepSpeed、Accelerate、Ring Attention、flash-attn、NCCL*

- 基于 DiffSynth-Studio 框架，面向 MiniMax-H3 长视频音频生成模型设计并实现长序列 SFT 训练与推理链路，解决 30s 多模态 packed 序列带来的上下文长度、注意力复杂度与训练显存瓶颈，支撑 33B 级 DiT 在 80GB H100 上稳定训练。
- 提出基于 Ring Context Parallel 的多模态长序列并行方案，将 packed video/text/audio token 统一切分，结合分布式 Ring Attention 与流式归一化机制，降低 attention 显存峰值并随 CP 并行度线性摊薄，兼顾多模态序列切分一致性与通信开销。
- 融合 DeepSpeed ZeRO-3 参数分片、CPU 初始化与梯度重计算，设计“模型状态、序列激活、损失计算”三路并行的显存优化策略，在降低训练峰值显存的同时提升多卡、多节点拓扑下的可扩展性。
- 构建 CP-aware 数据加载与损失计算体系，统一 CP 组内随机状态与目标分片，使长序列训练在数据一致性、数值稳定性和分布式归约开销之间取得平衡。
- 打通训练到推理的全链路，面向 30s 长视频音频生成场景完成多卡上下文并行推理与模型验证，形成可复现的多节点训练、评测和推理方案；在 24 卡 H100 上将峰值训练显存降至约 45GB。

## 6. 当前产出

- `diffsynth/core/context_parallel/`：Ring attention、metadata、training helpers。
- `examples/minimax_h3/model_training/`：30s SFT 训练入口、24 卡 3 节点启动脚本、DeepSpeed 配置。
- `examples/minimax_h3/model_inference/`：30s local CP 推理脚本。
- `examples/minimax_h3/model_training/ring_cp_30s_sft.md`：训练方案、显存、速度和问题记录。
- `models/train/MiniMax-H3-30s-smoke/epoch-0.safetensors`：smoke 训练 checkpoint。
- `models/inference/latest_30s.mp4`：30s 视频 + 音频推理产物。

## 7. 后续待完善

- 完成更完整的模型训练曲线验证，而不只是 smoke/短 step 验证。
- 支持 optimizer、scheduler、global step 和 CP dataloader 游标的完整断点续训。
- 进一步优化 Ring attention 通信开销，降低 CP 多节点下的 P2P 等待。
- 增加 CP=1/2/4/8 的系统显存、吞吐和通信对比实验。
