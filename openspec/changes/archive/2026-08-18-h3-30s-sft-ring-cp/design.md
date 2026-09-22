## Context

当前 H3 训练路径已经具备 8 卡 DeepSpeed ZeRO-3 的 full SFT 示例，但 30 秒视频会产生很长的 packed sequence，单卡需要保存完整序列的 hidden、QKV、attention 和 MLP 激活。当前 [minimax_h3_dit.py](/gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:70) 的 attention 直接按全局 `cu_seqlens` 切片，`cu_seqlens` 同时承担 segment 边界和索引两个职责。因此 CP 不能只做 kernel 替换，必须同时处理 segment 语义和 per-token metadata。

目标是以 Ring Context Parallel 为主路径：按 token shard 分配 sequence 激活，让每个 rank 只持有 `S/CP` 的 hidden/QKV/MLP 激活，同时保持全局 packed attention 的数学语义。

## Goals / Non-Goals

**Goals:**
- 提供可配置 CP size 的 H3 Ring CP 路径，默认朝 CP=8、DP=3 的 24 卡拓扑设计，但不硬编码。
- 在 CPU 可运行的单进程逻辑 shard 测试中，验证 CP=1/CP=2 的 forward、参数梯度和单步 loss 对齐。
- 保留当前 `cu_seqlens=[0, used, seq_len]` 的 pad segment 语义，除非测试明确改为新参考。
- 实现 CP-aware dataloader、随机源同步和 loss 归约。
- 明确 DeepSpeed ZeRO-3 与 CP 的集成策略。

**Non-Goals:**
- 本变更不要求在真实 GPU 上完成 30 秒 SFT 训练。
- 本变更不实现 Ulysses、2D sequence/tensor parallel 等替代方案。
- 本变更不优化 pad segment 的显存占用；可以先保持与参考一致，后续单独优化。
- 本变更不重写 token refiner 的并行方案，除非主路径测试暴露必要性问题。

## Decisions

### 1. 使用 Ring CP，而不是 Ulysses
Ring CP 按 token 分片，能同时降低 hidden、QKV、attention 和 MLP 的 sequence 激活；Ulysses 按 head 分片，虽然 56 heads 可以被 8 整除，但每个 rank 仍持有全序列 hidden/MLP 激活，显存画像不同。Ring 也更贴合当前按 `cu_seqlens` 做 packed attention 的实现。

### 2. 先建立“逻辑 shard” reference，再接 distributed
在 CPU 阶段实现一个不依赖真实 NCCL collectives 的逻辑 CP 路径：显式传入 rank/world_size 和 global metadata，按 shard 计算 local Q/K/V、local segment mask、online softmax 结果。这样可以在单进程内对比 CP=1/CP=2，避免 GPU 成为第一个阻塞点。之后再在 custom autograd 中接入真实 collectives 和 DeepSpeed。

### 3. 保留 pad segment 的参考语义
当前 `cu_seqlens=[0, used, seq_len]` 使 `[used, seq_len]` 成为一个独立 segment。训练最终输出只取 used 位置，因此后续可以论证去掉 pad，但本变更先保留该语义，并把“是否去掉 pad”留作独立优化。

### 4. 新增独立 CP metadata 抽象
新增一个轻量模块维护：
- global sequence length、global `cu_seqlens`、local shard `[local_start, local_end)`；
- global RoPE position 到 local token 的映射；
- local query 的 segment id；
- 当前 K/V chunk 的 global start/end 和 segment membership；
- 用于输出选取的 `img_pos`、`audio_pos` 等 local index。

该抽象避免把 global `cu_seqlens` 原样传入 local attention，也为后续 `_sdpa_varlen_attention` 替换提供统一边界。

### 5. Ring attention 使用 online softmax + custom autograd
数学语义上等价于完整 attention，但不承诺 bitwise identical。BF16 下只承诺合理 tolerance 的 allclose。实现先使用 PyTorch 算子保证可读性和 CPU 可测，再按需要接入 flash/online kernel。

### 6. 参数梯度归约放在模型/训练层
Ring 的 K/V gradient 需要在不同 query shard 之间归约，但只做 K/V 归约不够。MLP、QKV、out projection、embedding projection 等参数梯度也必须包含所有 CP shard。实现顺序上先保证逻辑 CP 的梯度正确，再接 DeepSpeed/FSDP 归约。

### 7. 训练入口采用 CP-aware sampler + leader broadcast + 全局 loss 归约
- CP group 内复制同一个样本，DP group 之间保持不同样本；
- `timestep_video`、`timestep_audio`、video noise、audio noise 由 CP leader 生成后 broadcast；
- loss 按 video/audio 分别计算 local numerator 和 valid count，再做全局归约，最后按当前 “video mean + audio mean” 组合。

### 8. DeepSpeed ZeRO-3 先使用 global process group
仅把 `num_processes` 改成 24 并手动建 CP/DP group，不会自动让 ZeRO-3 变成 CP=8/DP=3。本变更优先采用 global group + CP-correct loss normalization，让 DeepSpeed 统一归约所有参数梯度；如果实测需要 DP-only 参数状态，再显式在 CP group 内归约参数梯度。

### 9. Ring 内使用 FlashAttention/blockwise 后端
当前 custom autograd Ring 先用 dense `torch.einsum` 保证 CPU/FP64 reference 可读可测。GPU/BF16 训练路径改为可选的官方 `flash_attn` blockwise 后端：每个 K/V chunk 按 segment 调用 `_flash_attn_varlen_forward`，用 `softmax_lse` 做 online merge；backward 传入完整全局 `softmax_lse`。没有 `flash_attn`、CPU 或 FP64 时自动回退 dense，保证现有测试不依赖 GPU 包。该后端不会把 `score` 展开成 `local_q * local_k * heads`，因此是 30s packed sequence 显存可行性的实现基础。

## Risks / Trade-offs

- [Ring online softmax 在不同 chunk 顺序下产生浮点差异] → CP=1 与 CP=2 使用 allclose 而非 bitwise 比较；测试覆盖 forward、gradient、loss。
- [local segment mask 错误导致跨 segment 泄漏] → 用跨 segment 边界 shard 的 CPU 测试覆盖，并保留 pad segment 参考语义。
- [gradient checkpointing 与 custom autograd/collectives 组合导致 deadlock 或重复通信] → 在 CPU 逻辑 CP 测试中先启用现有 checkpoint wrapper，再上真实 distributed。
- [loss 归约因 video/audio 元素数不均而偏置] → 分别归约 video/audio 的 numerator 和 count，不做 local mean 平均。
- [DeepSpeed ZeRO-3 进程组集成复杂] → 先用 global group；自定义 DP group 仅在全局方案被验证不可行后引入。
- [实际训练吞吐未必随 CP 线性提升] → 本变更只保证可训练路径；吞吐和曲线验证列为 GPU deferred task。
- [flash_attn 版本接口不稳定或未安装] → 仅在 GPU/BF16 且可导入时启用，其余路径回退 dense；数值验收使用 allclose。

## Migration Plan

1. 新增 CPU 可运行的 Ring CP reference、local shard metadata 和测试。
2. 将 H3 attention 的 local path 接入新 metadata，保持 CP=1 与现有实现一致。
3. 接入 H3 训练入口：CP-aware sampler、随机源同步、loss 归约。
4. 接入 DeepSpeed ZeRO-3 集成和 30 秒 SFT 启动配置。
5. 在真实 GPU 上验证 CP=1/2/4/8 forward、梯度、单步 loss 和训练曲线；该步骤可按现状推迟。
6. 接入 blockwise FlashAttention Ring 后端，使 30 秒 85k packed token 不再分配完整 dense score。

## Open Questions

- 30 秒 SFT 的最终 CP/DP 拓扑是 24 卡还是更多卡；设计按 CP=8/DP=3 推导，但参数保持可配置。
- token refiner 后续是复制计算还是也走 CP；当前倾向复制，因为文本 token 数量相对很小。
- 是否在 CP 稳定后去掉 pad segment，以省掉对齐带来的额外激活；需要单独变更。
- DeepSpeed 全局 group 是否足以满足训练稳定性；若不足，再评估 DP-only ZeRO 加 CP 内 allreduce。
