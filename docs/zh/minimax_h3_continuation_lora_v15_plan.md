# MiniMax-H3 接续 LoRA 训练方案（v4.4，纯训练范围）

> 状态：方案整理稿 v4.4。**本轮执行范围：只做训练侧（数据、loss、训练入口、smoke 与短训），不做推理 eval**；训练效果的判定（E1）明确推迟到下一轮推理阶段。v4.2：解除“训练循环内不跑采样器”约束。v4.3：v4.1 clean-prefix mixed 全量运行中；布局沿用 A（345/39），C（243/141）与 A+C 并行暂缓。v4.4（用户拍板）：S1 档 2（真 SF 在线 rollout）实现搁置；S1 档位（档 1 离线误差库 / 是否重启档 2）待 v4.1 全量结果后再定。
> 范围前提：推理路径 masked-av-v14 不变、训练沿用 v14 loss 管道、只训练 DiT LoRA。本文不讨论任何推理侧改动。

## 0. 版本修订记录

| 版本 | 变化 |
| --- | --- |
| v1 | 初始整理稿：Stage 0 布局/pair -> Stage 1 pair SFT -> Stage 2 clean-prefix 混合 -> Stage 3 rollout |
| v2 | grill-me 修订：修正 pair loader bug 诊断、Stage 1 定位、E1 入口冲突；纳入 SVI 优先级；提出 E0 纯推理基线先行 |
| v3 | 按用户决定收窄执行范围：**只考虑训练，不做推理验证**；删除 E0/E1 推理门；验收改为训练侧机制验证；效果判定整体移交下一轮 |
| v4（本版） | 补 S1 范式定位：DF≠SF 是正交维度，v14 loss 已内置 DF 形态，DF 前置已完成、不需要另开阶段；S1 真正缺的是 SF 层（上下文来源从真值窗换成自生成）。端到端目标、三类前缀形态与 SF 档位的完整展开仍待 v4 后续小节补入 |
| v4.1（本版之后） | 落地 masked-av-v14 的 clean-prefix 选项（`prefix_present_mode=noised/clean/mixed`，config+loss+train.py+单测）：clean 形态按推理 denoise_mask=0/timestep=1 注入，mixed=50/50。它是误差注入（S1 档 1）的“误差=0”特例，先单独验证“干净前缀端点进训练分布”是否有用 |
| v4.2（本版） | 解除“训练循环内不跑采样器”约束：S1 档 2（真 SF：训练内 few-step rollout 自生成上下文回注）与 DMD2/轨迹蒸馏（`DirectDistillLoss`/`TrajectoryImitationLoss` 已在 `diffsynth/diffusion/loss.py:517/529`）从“预留/冲突”改为“可执行”；档 1 离线误差库降级为可选对照 |
| v4.3（本版） | 用户拍板：① v4.1 clean-prefix mixed 已进入训练（全量运行中）；② 布局暂不切换——沿用 A（345/39），C（243/141 大上下文）与 A+C 并行暂缓。G1 的布局项据此关闭 |
| v4.4（本版） | 用户拍板：S1 档 2（真 SF 在线 rollout）实现搁置，设计保留存档；本轮训练侧主改动回到 v4.1 clean-prefix mixed。S1 档位（档 1 离线误差库 / 是否重启档 2）待 v4.1 全量结果后再定 |
| v4.5（本版） | 切镜退化修复进入实现：混合前缀数据集（有/无前缀 × 延续/切镜四类）+ 训练侧 `prefix_mode=none` 通路。索引已产出并 CPU 验证，尚未开始 GPU 编码与训练 |

## 1. 一句话思路

把“接续生成”当作 single-shot 续写任务来训练：让模型以上一窗口状态为条件续出同一镜头内容。本轮只交付“训练管线正确跑通并产出一个 LoRA adapter”，不交付“LoRA 有效”的结论。

## 2. 决策日志（已锁定）

| 主题 | 决定 |
| --- | --- |
| 推理 | 锁定 masked-av-v14（不改、不叠 turbo LoRA）；但推理**不在本轮执行范围** |
| 训练 | 沿用 v14 loss 管道；只训练 DiT LoRA；条件形态 masked-av-v14 |
| 布局 | 沿用现状 A（345/39）；C（243/141 大上下文）与 A+C 并行暂缓（v4.3 用户拍板） |
| 排除 | taper-refine / dispatch / FreeNoise / appearance memory 不进入训练改动 |
| 训练侧主改动 | SVI 式噪声水平感知误差注入（取代原 clean-prefix 开关的优先级） |
| 判据 | 本轮：训练侧机制验证（单测/断言/smoke/EMA loss）；效果判据（E1）下一轮 |

## 3. 训练侧要解决的问题（为什么 v3 轮不够）

1. 训练样本是 345 帧首窗切片 + 39 帧 overlap：模型只在“单窗内部”见过真实前缀->真实后缀，条件上下文只有 12 个 video token（~1.6s），训练分布与“长链续写”不匹配。
2. 训练前缀按当前 t 加噪注入（`v3_continuation_inputs`），而推理侧前缀经 `input_latents_*`+`denoise_mask_*` 干净注入（timestep=1）：形态差会让模型把推理时的注入当成陌生分布（HiAR 噪声水平论点）。
3. 每步 DP=4、样本难度差异大：单步 loss 震荡，不能作为训练是否生效的信号。

## 4. 本轮执行范围（只做训练）

```text
S0  数据与 pair 入口修复（CPU 可验证）
S1  SVI 最小实现（loss + dataset + model_fn 注入，训练侧）
S2  smoke 与短训（单测 / 前向-梯度-导出 / 500 步 + EMA loss）
     |—— 本轮到此为止；效果判定（E1 推理验证）留给下一轮
```

### S0 —— 数据与 pair 入口修复

**已核实的代码事实（grill 确认，修复点在 loader）**：

1. **pair loader 切片长度 bug（真 blocker）**：`load_continuation_pair_cache`（`diffsynth/utils/continuation_lora.py:797`）切 A 尾用 `video_overlap_steps = overlap_frames//17*5`：39 帧 -> 10 步；而 masked-av-v14 全链路要求 39 -> 12（`ContinuationRegionConfig.overlap_video_steps=12`，`h3_continuation_video_latent_steps` 语义 `2 + 5*((frames-5)//17)`）。pair 首样本会在 `loss.py` masked-av-v14 分支的 shape 校验处抛错。修复：loader 改用 masked-av 语义（39->12 / 90->27 / 141->42）。
2. **音频前缀长度公式统一**：`loss.py` 的 `audio_hard = round(overlap_video_steps*65/12)` 只在 39/12 时正确（90->146≠150、141->228≠235）；统一为物理时间公式 `round(overlap_frames/24*40)`（loader 的 audio 侧已用此公式）。
3. **一致性断言缺失**：补 `A.tail == B.head` 数值断言（pair 语义下 A 尾部与 B 前缀是同一物理内容）与 masked-av-v14 pair 单测（现有测试只覆盖 legacy 34 帧）。
4. **pair 入口与版本策略**：`ContinuationLatentDataset(manifest=目录)` 只读 `<split>/manifest.jsonl`；pair 训练须把 manifest 指向 `pair_manifest.jsonl` 文件。布局切换要 bump cache schema 版本并在读取侧拒绝混用。

**布局选择（训练语义权衡，不需要推理）**：

| 选项 | 窗口/overlap | 条件上下文 | 每窗新内容 | 训练语义 |
| --- | --- | --- | --- | --- |
| A. 现状 | 345/39 | 12 tokens（~1.6s） | 306 帧 | 与 v3 相同；只用于对照 |
| B. 5s 级 | 243/39 | 12 tokens | 204 帧 | 更贴近 AR 循环节奏，但上下文不变 |
| C. 大上下文 | 243/141 | 42 tokens（~5.9s） | 102 帧 | 条件上下文显著变长，是“持久锚”的训练侧近似 |

- 训练目标如果是要让模型“看到更长的过去再续写”，布局 C 是唯一真正改变条件上下文长度的选项；A/B 只改变新内容节奏。
- C 的代价：每窗新内容少（102 帧），梯度/步效率下降，需要更多步或更长训练；缓存需按 243/141 重建。
- 决策依据：已拍板（v4.3）——沿用 A（345/39）；C/A+C 暂缓。本节选项仅存档，不再执行；C 的缓存无需重建。

**数据可得性检查（CPU 可跑，先于任何训练）**：在源 JSONL 上跑 pair index，量化 `accepted_pairs / short_pair_shot`、start 偏移分布、sequence 覆盖数；若 345/39 pair 产量不足（源视频多 <27.1s），因 C 已暂缓不能自动切 243/141，需回到用户重新评估。

### S1 —— SVI 最小实现（本轮训练侧主改动）

目标：训练时把历史前缀以“与 timestep 匹配的含误差形态”注入，一条改动覆盖暴露偏差 + HiAR 噪声水平论点 + 第二窗数据。

**范式定位：v14 已完成 DF 前置，S1 缺的是 SF 层，不是“再补一档 DF”**。TF / DF / SF 是三个正交维度，不是同一思路的高配/低配：

| 范式 | 上下文来源 | 上下文噪声形态 | 解决的问题 |
| --- | --- | --- | --- |
| TF | 真值（干净） | 全帧共享同一个 t | 学单步去噪 |
| DF | 真值 | 每帧独立噪声水平（把“任意噪声水平上下文”放进训练分布） | 让模型认识混合噪声水平输入；exposure bias 只是缓解，输入来源仍是真值 |
| SF | 模型自己 rollout 的输出 | 与部署时一致 | 训练分布 = 部署分布，根除 exposure bias |

**为什么 DF 是 SF 的必要前置（四点）**：
1. SF 回注的上下文是“过去干净帧 + 当前带噪帧”，这恰是 DF 教过的输入形态；只训 TF 的话，随机 rollout 一两步就 OOD，SF 无法收敛。
2. SF 用 holistic 分布匹配 loss 且梯度穿过多步展开，天然不稳、mode-seeking（HiAR 明确报告低运动塌缩）；论文都把它定位成小步数 post-training，底层单步去噪映射需先由 TF/DF 训稳。
3. 成本结构决定配方顺序：TF/DF 可并行、每步一次 forward，适合大预算主训练；SF 本质串行、需 few-step + 梯度截断，只适合后段精修。
4. 从零直接 SF，误差会在展开中被当真值全强度前传——正是 HiAR 的 `tc*=t_{j+1}` 论点要修正的错误。

**文献实证链**：
- Self-Forcing（arXiv 2506.08009）：明确定位 post-training，先有 TF/DF few-step AR 底座。
- HiAR（arXiv 2603.08703）：先在 hierarchical 调度下训稳 teacher（同噪声水平上下文 = DF 形态），再做 self-rollout 蒸馏 + forward-KL 正则防低运动塌缩。
- Rolling Forcing（arXiv 2509.25161）：joint denoising 渐进噪声水平为底座，再在 self-generated histories 上做 few-step 蒸馏。
- CausVid / Causal-rCM（arXiv 2606.25473）：暴露“光有 DF + DMD 会匹配错分布”，需用 SF 对齐。

**落到 v14/v15 的含义**：
- v14 loss 已内置 DF 形态：前缀按当前 t 加噪注入（`v3_continuation_inputs` 路径）+ `denoise_mask==0` 区域干净注入并置 timestep=1。**DF 前置已完成，不需要另开一个“先补 DF”的阶段**。
- 缺的是 SF 层：把上下文来源从“真值窗”换成“自生成”。下方档位表就是这条线的递进：档 1（对照）离线误差库 = Self-Forcing/SVI 的离线近似；档 2（v4.4 起搁置，设计存档）严格真 SF——训练内 few-step rollout，需 stochastic gradient truncation（最小版 = 回注路径 stop-gradient）+ EMA/λ 混合比例退火 + HiAR 式 forward-KL 护栏，只适合小步数 post-training；档 3（收尾预留）DMD2/轨迹模仿蒸馏。

**约束更新（v4.2）：训练循环内允许跑采样器**。误差来源分三档（S1 内递进，选档由放行门拍板）：

- 档 1（对照，成本最低）：离线误差库——真实窗 + 预测残差，用现有 LoRA/base 对训练窗做单次前向得到残差，按 timestep 存库；训练时在 masked-av-v14 分支按概率采样注入，替代现在“同 t 加噪”的单一路径。
- 档 2（v4.4 搁置，设计存档）：真 SF 在线 rollout——训练内用当前 LoRA 以 few-step（建议 4–8 步）对“上一窗/上下文”自回归采样，把自生成结果（含其误差）回注为条件；回注路径 stop-gradient（SGT 最小版），λ 混合比例从 0 起步退火，配 HiAR 式 forward-KL 护栏防 mode-seeking；few-step 时间表复用推理 scheduler 的 `set_timesteps(n)`。前置：v3/v4.1 先把单步去噪（TF/DF 形态）训稳，SF 只做后段小步数精修。
- 档 3（收尾预留）：DMD2/轨迹模仿蒸馏——把接续能力与 8 步加速并进同一 adapter：dense teacher 轨迹 → 8 步 student，替代推理时“continuation LoRA × turbo LoRA”叠加。`DirectDistillLoss`/`TrajectoryImitationLoss`（`diffsynth/diffusion/loss.py:517/529`）已是现成实现，可直接搬到 continuation 路径。
- clean-prefix 形态（v1 Stage 2 的开关）是上述注入的“误差=0”特例，已由 v4.1 单独落地。

**改动面（四处联动，此前被低估）**：
- dataset / rollout 编排：档 1 误差库加载与样本配对；档 2 的上一窗（或历史多窗）上下文获取与回注键构造；
- loss 分支：masked-av-v14 里按概率选择注入形态；
- model_fn 注入键：`input_latents_*` / `denoise_mask_*` 与 timestep=1 语义对齐（训练侧构造，不碰推理）；
- CP 权重映射：新增键要同步到 CP 的 local rows 映射。

**前置依赖**：S0 的 pair 修复与布局确定。档 1 需误差库 schema 与过期策略（限制混合比例，避免旧库主导）；档 2/3 不需要误差库，依赖的是——单步去噪已由 v3/v4.1 训稳、few-step 时间表可离散化（复用推理 scheduler）、rollout 参考权重（当前 LoRA EMA）与 λ 退火起点。

### S2 —— 训练侧验收（代替推理 eval）

本轮“验收”只证明训练管线正确，不证明 LoRA 有效：

1. 单测：pair loader masked-av 切片（39->12）、`A.tail==B.head`、audio 前缀公式、SVI 注入的噪声水平匹配；
2. smoke：1–2 步前向 + backward + checkpoint/LoRA 导出（沿用 `--validate_continuation_config` 与 smoke 脚本）；
3. 短训：500 步 + EMA loss 曲线 + 无 NaN/无崩溃 + 导出单个 `.safetensors` adapter；
4. 数据报告：pair index 产量统计落盘。

**明确不做（本轮）**：任何 masked-av-v14 推理生成、joins/DINO 漂移评测、base vs LoRA 画质对照。这些归入下一轮 E1，方案另行编写。注意：档 2/3 训练内的 rollout / teacher 轨迹属于训练机制，不算 E1 推理 eval，不违反上述边界。

### 4.6 混合前缀数据集与切镜退化修复（v4.5 落地）

**问题**：v3/v5 的训练样本 100% 带 masked 39 帧前缀，模型从未见过“无前缀”输入；推理侧硬切镜（`continuation=None` 的新镜头窗）因此是 OOD，base 的生成能力被 adapter 带偏，表现为切镜后脸糊。

**做法**：不改推理，只在训练分布里补齐“无前缀”形态，靠**输入有没有 39 帧前缀**区分，不引入 mode token。

| 类别 | 配额 | 前缀 | 取样方式（复用 / 新编码） |
| --- | --- | --- | --- |
| 有前缀 + 延续 | 45,000 | masked | 复用 41,898（40,691 条 + 1,207 条重复填充）/ 长镜头补充 3,102 |
| 有前缀 + 窗内切镜 | 28,000 | masked | 新编码 28,000（stride 窗口 + 切点对齐窗口） |
| 无前缀 + 新镜头 | 15,000 | none | 复用 4,204 / 新编码 10,796（单镜头窗 + caption 身份提示） |
| 无前缀 + 普通窗口 | 12,000 | none | 复用 3,364 / 新编码 8,636（单镜头窗 + 原 caption） |

产物：`outputs/continuation_mix_index/train.jsonl`（100,000 条；49,466 条复用已有 latent，50,534 条待编码）。

**训练侧通路**：

- `ContinuationLatentDataset` 透传 `continuation_prefix_mode`；`none` 时把 `continuation_history_{video,audio}_latents` 置 `None`（省一次 clone）。
- `ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss` 在 `prefix_mode="none"` 时走**纯 flow-matching**：整窗 `add_noise`、`target = noise - clean`、全窗均一权重；不做前缀清零、不做接缝 3× 加权、不做 clean 注入。`masked` 分支与 v3/v5 逐位一致。
- 缺省值仍为 `masked`，旧 manifest 行为不变。

**数据口径修正**（编码链路验证时发现并修复）：

- 窗口是 **345 输出帧 @24fps = 14.375s**（模型 `video_fps=24` 契约），不是 345 个 25fps 源帧（13.8s）；stride 12.75s。
- 650k jsonl 的 `clips[].frame_range` 是**闭区间**，按半开处理会在每个片段边界产生 1 帧空洞。
- 窗口起点向上对齐到输出帧网格，避免 `resolve_media_window` 向下取整到源区间之前。
- 片段音频用 `<dataset>/audio/raw/<stem>_origin.wav`（44.1k→32k）：它与 v3/v5 使用的 `htdemucs/<aug>/vocals.wav` 相关性 0.945，内容等同、时间对齐更准。

**验证（CPU，已通过）**：`verify_multi_segment_windows.py`（跨 2–6 片段抽帧 + 音频拼接）、`verify_continuation_loader.py`（库级加载器逐像素一致）、`build_continuation_cache.py --fake` 端到端、`tests/test_h3_continuation_lora.py` 31 passed。

**编码链路修复（首次真机编码时暴露）**：复用条目带两种残缺 `segments`（40,691 条字段缺失、8,775 条只有占位符）。`cpu_prefetch` 会为**所有**样本预取媒体，而复用短路写在主循环里，导致 ① 占位符条目 `KeyError` 直接崩，② 字段缺失条目退回单文件分支、白解码整个 345 帧窗口（每条 ~10s）。现在预取阶段直接跳过带 `reused_cache_path` 的样本；`load_video_window_segments` 也加了段字段校验以给出可读报错。全量 40,691 个复用 `.pt` 已逐个 `exists()` 预检通过。

## 5. 风险表（训练侧）

| 风险 | 对策 |
| --- | --- |
| pair 与“偏移采样的单窗”内容近等价，S0 后直接跑 pair SFT 收益可能无法归因 | S0 只修入口；S1 的误差注入才是分布改动；若只做 pair SFT 需加“单窗 stride 中段采样”对照组 |
| 误差库随训练推进过期（仅档 1） | 限制混合比例 + 定期刷新（S1 设计时定阈值）；选档 2/3 无此问题 |
| 档 2 真 SF 训练不稳 / mode-seeking（HiAR 报告低运动塌缩） | 只做小步数 post-training；单步去噪先由 v3/v4.1 训稳；λ 从 0 起步退火；回注路径 stop-gradient；加 forward-KL 护栏 |
| 布局 C 每窗新内容少，步效率低 | 接受更小有效步数或增大梯度累积；以 EMA loss 趋势而非绝对值为准 |
| 不做推理验证导致“训练正确但效果未知”被误读为“有效” | 文档与交付物明确标注本轮结论边界 |
| 缓存/版本混用 | schema bump + 读取侧拒绝 + manifest hash 校验（已有机制，补端到端测试） |

## 6. 放行门（训练侧，带 owner）

| # | 门 | 通过标准 |
| --- | --- | --- |
| G0 | 数据可得性 | pair index 产量报告落盘（CPU） |
| G1 | 范围决策 | 布局已拍板（v4.3）：沿用 A（345/39），C/A+C 暂缓；待拍板：是否含 pair SFT 对照组 |
| G2 | S0 修复 | pair loader 10->12 + 单测 + `A.tail==B.head` 断言通过 |
| G3 | S1 档位实现 | 档 2 已搁置（v4.4）；若重启档 1：注入单测 + smoke 前向/梯度通过 |
| G4 | 本轮收尾 | 500 步短训产出 adapter + EMA loss 报告，注明“效果未验证” |

## 7. 决策清单（待用户拍板）

1. 布局档位：已拍板（v4.3）——沿用 A（345/39）；C（243/141）与 A+C 并行暂缓。
2. S1 是否本轮实现，还是先只做 S0 + 传统 pair SFT 对照组（需额外样本设计）？
3. S1 档位：档 2 真 SF 已搁置（v4.4）；档 1 离线误差库是否做、是否重启档 2，等 v4.1 全量结果出来再定？
4. 训练预算（GPU 数、总步数、梯度累积倍数）与 checkpoint 目录。
5. 是否新建 OpenSpec change（`h3-continuation-lora-v15`），旧 `h3-continuation-lora-training` 内容过时待归档。

## 8. 代码锚点清单（实施用）

- pair 数据：`diffsynth/utils/continuation_lora.py`（`load_continuation_pair_cache:760-830` 切片 bug；`iter_continuation_pairs:1274`；`ContinuationLatentDataset:1000`；`PAIR_MANIFEST_SCHEMA_VERSION:29`）
- 布局语义：`diffsynth/pipelines/minimax_h3_continuation.py:46-70`（`is_masked_av_context_frames` / `h3_continuation_video_latent_steps`）
- v14 损失：`diffsynth/diffusion/loss.py` `ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss`（masked-av-v14 分支 ~245-320；`audio_hard` ~299；区域权重 ~395-420）
- 注入键分支（训练侧复用语义，不碰推理）：`diffsynth/pipelines/minimax_h3_audio_video.py:1846-1880`
- 训练入口：`examples/minimax_h3/model_training/train.py`（task `continuation_sft`）；smoke/缓存脚本同目录
- 数据构建：`examples/minimax_h3/model_training/build_continuation_dataset.py` / `build_continuation_cache.py`（`--directional_pairs`）
