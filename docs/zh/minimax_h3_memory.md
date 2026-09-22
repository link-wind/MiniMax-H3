# MiniMax-H3 记忆机制（Memory）文档

> 本文单独介绍 MiniMax-H3 接续生成中的**记忆（memory）** 机制，包括训练侧的记忆条件构造、
> 模型侧的注入方式、以及推理侧的自生成记忆回滚与外观记忆库。模型/推理/训练的整体改动
> 见 `minimax_h3_model_changes.md`。


## 0. 为什么需要记忆：靠"记忆"而不是"整段重放"

单窗口的 latent continuation（见主文档 §2.1）只能把 **紧邻的上一窗尾部** 传下去，它有两个
局限：

- **只有端到端几秒**：上一窗的尾巴只是「刚刚发生的那一下」，无法表达更早时刻建立的角色、
  场景、光照等长期信息；
- **没有全局锚点**：随着 rollout 越来越长，某个「很久以前的基准内容」（比如开头定义的镜头）
  会不断被稀释。

记忆机制在 continuation 之外再添加**更宽、更久**的条件，把两件事解耦：

```
 上一窗尾部 latent        →  微观连续性（动作、运动方向） —— continuation
 记忆槽 STM / LTM         →  宏观一致性（角色、场景、开头） —— memory
```

类比：continuation 像「上句接着下句」，memory 像「写长文时随时回看人物设定表」。两者叠加才是
完整的长视频接续条件。


## 1. 概述

长视频接续需要一个跨窗口的“记忆”信号来提供上下文，避免模型在每一步重新起跳时丢失前面内容。
本项目的记忆机制包含三层：

- **训练侧记忆条件**：每个训练窗口以两个记忆槽（STM/LTM）作为条件，编码后拼入窗口输入。
- **模型侧注入**：DiT 在 embedding 阶段把这些记忆行作为纯条件注入（非预测目标）。
- **推理侧回滚与外观**：rollout 需要用模型自生成的帧重建 STM/LTM；并维护一个外观记忆库用于
  监督/稳定性。

记忆槽顺序是条件契约的一部分：`MEMORY_SLOT_ORDER = ("stm", "ltm")`，训练打包按
“STM 在前、LTM 在后”排列。

## 2. 训练侧记忆条件构造 — `diffsynth/utils/continuation_lora.py`

### 2.1 两个记忆槽的时序结构

训练时，每个窗口以两个记忆槽为条件（打包顺序固定为 `MEMORY_SLOT_ORDER = ("stm", "ltm")`，
即 **STM 在 LTM 之前**）：

```
 序列时间轴 ───────────────────────────────────────────────────────────────►
   │◄· LTM 槽 ·►│                    │◄· STM 槽 ·►│◄── 当前窗口 ──►│
   │ (序列开头)   │            ...    │ (紧贴窗口前)  │               │
   │            │                    │            │               │
   │ lead_steps │                    │ lead_steps │               │
   │  = 36 固定  │                    │  = None     │               │
   │ (时间距离   │                    │ (贴合窗口    │               │
   │  不随长度增长)│                    │  放置)       │               │
   └────────────┘                    └────────────┘               │

```

- **STM（short-term memory，短期）**：`anchor="window-start"`，取窗口紧邻之前的 `stm_frames`
  帧，编码为独立 `17n+5` VAE clip 后**贴合窗口放置**（`lead_steps=None`），提供「刚才发生什么」。
- **LTM（long-term memory，长期）**：`anchor="sequence-head"`，取序列起始的 `ltm_frames` 帧，
  编码后固定到恒定 `lead_steps = 36` 前置。因为它是相对序列开头的**固定基准点**，其时间距离
  不会随 rollout 变长而漂移，始终代表「起点定义的那个世界」。

为什么两者都要：STM 保证局部运动连贯，LTM 保证长程一致性；只有 LTM 这端够「远」且够「稳」，
才能给越来越长的序列一个不会跑的锚。

### 2.2 槽位不存在时：丢弃，不伪造

记忆槽源帧不存在、或与当前窗口重叠时，**丢弃该槽位而不是伪造**：

- 序列头部样本没有上文，故无 STM；
- 窗口仍覆盖序列开头时，LTM 与窗口重叠，故无 LTM。

这是为了让「条件契约」干净——宁可少一个条件，也不喂模型一个「编造出来」的信号。

### 2.3 相关实现


相关工具：
- `MemorySlotSpec` / `memory_slot_specs` / `dual_memory_slot_specs`：定义、生成单槽与双槽规格。
- `check_memory_only_sample` / `audit_memory_slots` / `select_memory_slots`：样本校验、
  已索引 manifest 的记忆槽审计、以及从源帧中选择记忆槽。
- `encode_memory_slot`：把选中的记忆槽帧编码为 VAE latent。
- 槽位不存在或与窗口重叠时**丢弃该槽位而非伪造**：序列头部无 STM；窗口仍覆盖序列头部时无 LTM。

## 3. 模型侧注入 — `diffsynth/models/minimax_h3_dit.py`

`MiniMaxH3DiT` 通过新增的 `mem_pos` / `mem_pos_info` 参数在 `_embed` 阶段注入记忆行：

- **复用视频投影**：记忆行是真实视频 latent，直接复用 `video_patch_proj`，不新增参数，使记忆块
  与它所条件的窗口处于同一 embedding 空间。
- **非预测目标**：记忆行不进入 `img_pos`，因此损失永远看不到它们，属于纯条件。
- **投影调用无条件执行**：`self.video_patch_proj(mem_rows)` **无条件调用**（即使当前 rank 无记忆行
  或样本无记忆槽也调用），只有 `index_add_` 保持条件性，从而避免不同 rank 调用序列不一致。
- **空槽位语义**：某样本没有记忆槽时仍表示“该样本不获得记忆条件”，而非错误。
- 记忆行在 `forward` 中通过 `_embed` 与图片/音频/文本行一同写入 `embeddings`，参与后续注意力，
  但不贡献 loss。

## 4. 推理侧打包 — `diffsynth/pipelines/minimax_h3_audio_video.py`

`MiniMaxH3Pipeline` 在推理时通过 `memory_latents` 接收一个紧凑的干净视频 latent 块，紧贴当前窗口
之前放置：

- `normalize_memory_slots` / `memory_slot_rows`：把传入的记忆 latent 规范化为多槽结构并提取各行。
- `MiniMaxH3Unit_PackedSequenceBuilder._resolve_memory_slots`：在打包 packed 序列时按槽位顺序解析
  记忆行，并与窗口行一起进入模型。

## 5. 推理侧长时程记忆回滚 — `diffsynth/utils/h3_memory_rollout.py`

**核心问题（暴露偏差）**：rollout 必须从**模型自生成的帧**重建 STM/LTM 两个槽；任何其它来源
（参考视频、上一窗口解码尾部）都与训练条件信号不同，这正是 rollout 要度量的暴露偏差。

`H3MemorySlotBuffer` 保留解码时间线的足够帧数，逐帧复现训练切分，再用窗口自身的 VAE 重新编码：

- `H3MemoryRolloutPlan`：plan 化配置，控制是否启用、slot 命名。
- `H3MemorySlotBuffer`：跨窗口持续维护记忆帧，支持 `observe` 观察解码帧、前/尾修剪
  （`_prune`）、按窗口起点取 STM/LTM 槽（`_stm_slot` / `_ltm_slot`）、VAE 调用点回注
  （`_notify`）、以及来源元数据（`provenance_metadata`）。
- 在 `H3ContinuationRunner.run` 中按 `memory_rollout_plan.enabled` 创建缓冲，并用 VAE 回注
  每窗自生成结果以滚动更新记忆。

## 6. 外观记忆库与评估 — `diffsynth/pipelines/h3_appearance_memory.py` / `h3_appearance_evaluation.py`

- `H3AppearanceMemoryBank`：外观记忆库，维护已解码帧的出现特征，支持锚点初始化
  （`initialize_anchors`）、按窗口更新（`update_from_window`）、边界设置（`set_boundary`）、
  相似帧近似去重（`_is_near_duplicate`）与参考构建（`build_references`）；通过特征距离
  （`_feature_distance`）挑选候选帧。
- `h3_appearance_evaluation.py`：外观漂移评估，含直方图相似度（`histogram_appearance_score`）、
  CLIP 打分（`clip_appearance_score`）、LPIPS（`lpips_appearance_score`）与 JSON/Markdown 报告
  （`write_appearance_drift_report`）。

## 7. 涉及文件清单

| 文件 | 归属 | 说明 |
|---|---|---|
| `diffsynth/models/minimax_h3_dit.py` | 模型 | 记忆行 `mem_pos` 注入（纯条件、非目标） |
| `diffsynth/utils/continuation_lora.py` | 训练 | 记忆槽规格/选择/编码（STM/LTM、`MemorySlotSpec`） |
| `diffsynth/pipelines/minimax_h3_audio_video.py` | 推理 | `memory_latents` 参数、记忆槽标准化与打包解析 |
| `diffsynth/utils/h3_memory_rollout.py` | 推理 | `H3MemoryRolloutPlan` / `H3MemorySlotBuffer` 自生成回滚 |
| `diffsynth/pipelines/h3_appearance_memory.py` | 推理 | `H3AppearanceMemoryBank` 外观记忆库 |
| `diffsynth/pipelines/h3_appearance_evaluation.py` | 推理 | 外观漂移评估与报告 |

相关测试：`tests/test_h3_memory_rollout.py`、`tests/test_h3_appearance_memory.py`、
`tests/test_h3_appearance_evaluation.py`、`tests/test_eval_memory_rollout.py`。
