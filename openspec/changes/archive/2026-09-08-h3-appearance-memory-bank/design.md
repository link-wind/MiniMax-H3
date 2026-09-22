## Context

`H3ContinuationRunner` 已经支持 `masked-av-v14` 的 latent handoff：后续窗口使用 `continuation_video_latents` / `continuation_audio_latents` 保留短程内容，同时不传全局位置参数，使 H3 每次以本地 canonical 时间网格运行。这个路径已经绕开 RoPE 外推，但一分钟级生成的主要问题转向外观漂移：局部 overlap 只携带约 12 个 video latent token，缺少稳定的人物/物体外观记忆。

现有 runner 已有一个最小静态参考机制 `global_reference_frames`：从第一窗口均匀采样 1-4 张中间帧，作为 image references 传给后续窗口。它接近 TetherCache 中的 Sink，但没有 Memory 动态选择、没有 boundary 与 anchor 分离、没有信任控制和评估入口。

H3 pipeline 的 `_build_packed_ref2va` 已经支持 image/video/audio reference blocks，并在参考块之后以本地 cursor 构造 target 位置。当前实现明确拒绝 `ref_blocks` 与 `video_source_indices` / `audio_source_indices` 同时使用，因此本设计继续走本地位置路径，不进入全局时间轴。

## Goals / Non-Goals

**Goals:**

- 为 `masked-av-v14` 增加一个 opt-in 的长期外观记忆银行，缓解分钟级外观漂移。
- 将 TetherCache 的 Sink/Memory/Recent 思想映射到 H3 可用的 reference conditioning：trusted anchors、selected memory frames、boundary reference。
- 保持本地 canonical 位置语义，不引入跨窗口绝对 RoPE 位置，不修改 H3 DiT 或模型权重。
- 保持默认行为兼容：不启用 bank 时，后续窗口不注入额外 references。
- 提供可复现的评估入口，比较 no-bank、static global references、dynamic bank 三组结果。

**Non-Goals:**

- 不做 KV cache memory，不在 attention 内注入历史 K/V。
- 不切换到 v3 全局时间轴，不为参考帧构造全局 source indices。
- 不训练 memory module、不做 LoRA/SFT、不改 H3 checkpoint。
- 不自动检测场景、角色或光照切换；有意切换场景应开始新 run 或显式禁用 bank。
- 不替换 `continuation_*_latents` / Retake；bank 只补外观条件，局部短程仍由原机制负责。

## Decisions

### 1. Bank 由 runner 管理，Pipeline 只消费 references

新增 `H3AppearanceReferenceBank` 作为 runner 级组件，维护 trusted anchors、memory frames、boundary reference，并输出 H3 `references` 请求参数。Pipeline 的 FL2VA/Ref2VA 行为不改变。

理由：长时记忆是窗口级状态，应该和 `ContinuationState`、segment plan、窗口同步放在同一层。Pipeline 保持单窗口可复用 API。

替代方案：把 bank 塞进 pipeline 单次调用。这会污染单窗口 API，也无法在窗口之间自然维护状态。

### 2. 使用 decoded reference frames，不使用 KV cache

Bank 保存少量 decoded 帧作为 image references，通过现有 `references` / `ref_blocks` 条件进入 H3。不做跨窗口 KV。

理由：H3 DiT 是 diffusion transformer，每个 denoising step 重新计算完整 packed sequence 的 K/V；K/V 依赖当前 timestep、噪声状态和窗口内容，没有可复用的因果 KV cache。强行注入历史 KV 是模型训练分布外的行为，需要架构改动和训练。

替代方案：TetherCache 原样迁移到 KV 空间。本仓库当前 H3 路径没有对应基础设施，且 TetherCache 的 KV 编辑无法直接用于 H3 的 `ref_blocks`。

### 3. Bank 分为 trusted anchors、memory frames、boundary reference 三部分

对应关系：

```text
TetherCache Sink    -> trusted anchors，来自第一窗口中间帧，不可变
TetherCache Memory  -> selected memory frames，来自后续高置信/多样性历史帧
TetherCache Recent  -> boundary reference + continuation latents，负责短程接续
TetherCache TAME    -> 锚点不可变 + 多样性/一致性过滤，避免漂移历史反向污染
```

trusted anchors 只从第一窗口初始化，永不被后续窗口覆盖。memory frames 从后续窗口候选帧中选择，但不得提升为 anchors。boundary reference 每次取当前 overlap 的最后一帧，独立于 bank。

理由：外观漂移的主因是“把越来越脏的生成历史当成外观真值”。保持 anchors 不可变，让 memory 只作为短期补充，可以避免误差累积。

### 4. Memory selection 先做训练无关的 diversity + consistency

默认 selector 使用时间分桶和降采样后的视觉差异进行选择：候选帧按窗口/时间位置分桶，优先保留与现有 anchors/memory 距离较大的帧，并排除近重复帧。同时保留一个 scorer 接口，后续可插入 attention relevance 或可选 CLIP/VLM scorer，而不改变 bank 调用方式。

理由：第一版不需要额外模型依赖，CPU 测试可覆盖；先验证“长期参考是否有效”，再决定是否需要更复杂的相关性选择。

替代方案：直接接 CLIP/VLM。会增加推理依赖和成本，且未证明 bank 本身有效前会把评估变量混在一起。

### 5. 保留旧参数，新增显式 bank 配置

继续保留 `global_reference_frames` 和 `boundary_reference_frames` 参数作为兼容层。新增 `H3AppearanceMemoryConfig` 或等价配置对象，显式启用 `static` / `dynamic` 模式，并限制总 visual references 数量。

```text
disabled          -> 完全保持现有行为
static            -> 等价于现有 global_reference_frames 的中间帧 bank
dynamic           -> trusted anchors + selected memory + optional boundary
```

理由：现有脚本和测试不受影响，同时新的 bank 语义不与旧整数参数混淆。

### 6. 只支持 `masked-av-v14` 本地位置路径

第一版 bank 只在 `continuation_mode == "masked-av-v14"` 且 `prefer_latent_handoff=True` 时启用。其他 continuation mode 继续使用现有 `global_reference_frames` 兼容行为。若启用 bank 的同时配置 `global_latent_decode=True`，应在窗口开始前拒绝，因为 bank 依赖已 decode 的当前窗口帧。

理由：本 change 的目标是验证本地位置 + 长期外观记忆；不与全局时间轴、Diff-VF、joint MultiDiffusion 等复杂路径混在一起。

### 7. State 只持久化 provenance，不内嵌媒体

`ContinuationState` 增加 bank 的 in-memory 运行时对象，manifest 增加 bank metadata：模式、anchors 来源 segment/frame、memory 来源 segment/frame、boundary 开关。媒体帧默认只存在于内存；除非显式持久化 bank artifact，否则 resume 状态标记 replay-only。

理由：与现有 `latent_artifacts` / replay-only 约定一致，避免把大 tensor 写进 JSON。

### 8. 评估入口与 runner 分离

新增 `h3-appearance-drift-evaluation` 入口，复用 runner 和 bank，但输出标准化报告。报告包含 resolved config、per-window seed、reference composition、早期/后期外观一致性指标。CLIP/LPIPS 等指标如果依赖缺失，则记录 unavailable，不伪造分数。

理由：质量评估应独立于实现细节，并且允许后续增加 metric 而不改 runner。

## Risks / Trade-offs

- [References 增加 attention token，可能限制运动或让画面过于贴锚点] → 限制总 reference 数，默认不超过 4 个 visual blocks；anchors 取中间帧而非边界，boundary 单独处理。
- [无模型 relevance scorer 可能选不到最关键记忆帧] → 默认 diversity/consistency selector 先验证机制；保留 scorer 接口，后续可接 attention/CLIP/VLM。
- [decode 帧 re-encode 引入额外成本] → 只传少量帧，且 H3 原生参考路径本就为此设计；不做整窗口 decoded context。
- [memory 帧仍可能携带轻微漂移] → memory 永远低于 trusted anchors，且只保留与 anchors 一致或多样性明显的候选。
- [场景/角色切换时旧 anchors 会拖住新内容] → 不自动检测场景；显式新 run、显式清空 bank，或后续增加 VLM scene-change 检测。
- [manifest 只记录 metadata，无法恢复 bank 帧] → 保持 replay-only 语义；需要恢复时再增加显式 bank tensor/image artifact。
- [与全局位置或 KV 方案混淆] → bank 配置在 `masked-av-v14` 本地路径中才允许，且设计层面禁止同时传全局 source indices。

## Migration Plan

1. 新增 bank 数据结构和配置，默认 disabled，不改现有 runner 调用语义。
2. 在 runner 循环中接入 bank：第一窗口后初始化 anchors，每窗口后更新候选/memory，后续窗口前注入 references。
3. 扩展 state metadata 和 manifest 序列化，保持旧 JSON 字段向后兼容。
4. 新增 CPU 测试与 evaluation entrypoint，先跑 no-GPU 契约测试。
5. 执行 GPU 对照：`masked-av-v14` no-bank、static、dynamic 各跑同一 60s plan，记录外观漂移指标。

Rollback：关闭 bank 配置即可恢复现有行为；bank 相关参数、状态字段和脚本均为新增，不改变现有默认调用路径。

## Open Questions

- Trusted anchors 与 memory frames 的最佳数量是多少：2+1、2+2、还是 3+1？
- Memory 选择是否需要 attention relevance：先做 diversity baseline，还是直接接 CLIP/VLM scorer？
- 后续是否应该支持 video reference block 而不是只支持 image reference，以携带更短的运动/外观上下文？
- 是否需要一个显式的 scene-change detector，还是继续由用户在新 run 时重置 bank？
