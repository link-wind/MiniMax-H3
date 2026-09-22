## Why

当前 H3 SFT 入口只能按当前示例分辨率/长度运行，30 秒视频会形成超长 packed sequence，单卡序列激活显存成为瓶颈。需要引入 Ring Context Parallel（CP）以支持单样本长序列全量 SFT，同时保持现有 packed multimodal attention 的语义。

## What Changes

- 为 MiniMax-H3 DiT 增加 Ring CP attention，按 token shard 执行精确全局 attention，保留全局 RoPE position、全局 segment 边界和 pad segment 语义。
- 引入 local shard metadata，使 `cu_seqlens`、`img_pos`、`audio_pos`、`token_tags`、`inverse_indices`、RoPE position 等 per-token 数据可正确映射到本地索引。
- 增加 CP-aware dataloader，使 CP group 内各 rank 拿到同一个训练样本，DP group 之间仍取不同样本。
- 在 loss 入口统一 timestep、video/audio noise 和随机源，由 CP leader 生成并 broadcast。
- 将全局 MSE loss 改为 local numerator + valid count 的 CP 归约语义，分别处理 video 和 audio。
- 明确 DeepSpeed ZeRO-3 / FSDP 进程组策略，保证 MLP、QKV、投影层参数梯度包含所有 CP shard 的贡献。
- 先交付 CPU 可跑的 reference 与单元测试；GPU 训练启动、实际显存和训练曲线验证任务暂缓。

## Capabilities

### New Capabilities
- `h3-ring-context-parallel`: H3 DiT 在 packed sequence 下的精确 Ring CP attention、local shard 映射、custom autograd 和梯度归约。
- `h3-cp-aware-sft-training`: H3 SFT 训练入口的 CP-aware 数据分发、随机源同步、loss 归约和 DeepSpeed/ZeRO 集成。

### Modified Capabilities

## Impact

- `diffsynth/models/minimax_h3_dit.py`：attention 路径、per-token metadata 处理、forward 输入输出。
- `diffsynth/pipelines/minimax_h3_audio_video.py`：packed sequence 构造和模型入口兼容性。
- `diffsynth/diffusion/loss.py`：H3 SFT loss 的随机源和 CP 归约。
- `diffsynth/diffusion/runner.py` 与 H3 训练脚本/配置：CP-aware sampler、进程组和 ZeRO 集成。
- 新增测试覆盖 CP=1/CP=2 的 forward、gradient、loss 对齐。
