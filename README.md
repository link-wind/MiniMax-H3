# MiniMax-H3 代码库使用指南

本仓库内的 **MiniMax-H3 全模态生成**（文本 / 图像 / 视频 / 音频）推理与训练代码库，重点扩展了
**长视频接续生成（Continuation）**与 **AI2V（音频驱动视频）**。

> 本文介绍「如何用这个代码库」。原理与设计细节见下方[相关文档](#相关文档)。

---

## 目录结构

```
examples/minimax_h3/
├── model_inference/          推理脚本
│   ├── MiniMax-H3-FL2VA.py           文→音视频 (T2VA)
│   ├── MiniMax-H3-Ref2VA.py          文+参考帧/参考视频→音视频 (Ref2VA)
│   ├── MiniMax-H3-Retake.py          局部重拍（Retake）
│   ├── MiniMax-H3-Text-Embeddings.py 文本嵌入调试
│   ├── MiniMax-H3-Continuation.py    多窗口接续生成（长视频）
│   ├── MiniMax-H3-AI2V-Masked-1min.py 音频驱动视频 (AI2V) + masked 接续
│   ├── MiniMax-H3-FL2VA-30s-local-cp.py  30s 长窗 + Ring CP
│   ├── run_*.sh                      各任务的批处理入口
│   └── h3_continuation_plan*.json    接续生成的 segment plan
├── model_inference_low_vram/  低显存（disk offload）推理变体
├── model_training/            训练脚本
│   ├── train.py               统一训练入口（sft / continuation_sft）
│   ├── build_continuation_dataset.py   接续训练数据构建
│   ├── build_continuation_cache.py     H3 latent 缓存
│   └── plan_shot_*.py         分镜/窗口切分工具
└── model_evaluation/          评测脚本
```

核心模块位于仓库根 `diffsynth/`：

| 文件 | 作用 |
| --- | --- |
| `diffsynth/pipelines/minimax_h3_audio_video.py` | H3 pipeline：T2VA / Ref2VA / Retake / 接续 / AI2V |
| `diffsynth/pipelines/minimax_h3_continuation.py` | 多窗口接续 runner、latent handoff |
| `diffsynth/diffusion/loss.py` | 接续 flow-matching 损失与区域加权 |
| `diffsynth/utils/continuation_lora.py` | 接续 LoRA 训练子模块 |

---

## 环境准备

```bash
cd DiffSynth-Studio
pip install -e .            # 安装框架（含 H3 依赖）
```

需要显存管理（低显存推理）时，框架会自动按剩余显存分块加载权重，最低约 7G 显存可跑 NF4 量化
模型。详细安装见 `docs/zh/Pipeline_Usage/Setup.md`。

### 本项目实际运行环境

本机实验使用专用的 venv 与 GPU：

| 项 | 值 |
| --- | --- |
| Python 解释器 | `/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python` |
| 运行方式 | `env PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<gpu> <上述python> -u <脚本>` |
| 推进优化 | `PYTORCH_ALLOC_CONF=expandable_segments:True`（大窗/AI2V 常用） |
| 推理 GPU | 默认 `CUDA_VISIBLE_DEVICES=0`，多卡/CP 见训练与 30s 脚本 |

> 下面的示例命令可直接用 `python`；若走本机 venv，把 `python` 换成上面的绝对路径，并加
> `PYTHONPATH="$PWD"`。

---

## MiniMax-H3 权重

### 基础权重

推理/训练需要 H3 基础权重。`$H3ROOT` 指向 FL2VA 根目录，需含
`text_encoder/ transformer/ video_vae/ audio_vae/ processor/`（以及常用基座
`transformer_lora600_v4/`）：

| 项 | 路径 |
| --- | --- |
| FL2VA 根目录 `$H3ROOT` | `/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA` |
| 常用 DiT 基座 | `$H3ROOT/transformer_lora600_v4` |
| AI2V 起始图 | `data/diffsynth_example_dataset/化风行万里/4.jpg` |
| AI2V 驱动音频 | `/tmp/h3_song_100s.wav`（由源音频裁切而来） |

### 已训练 LoRA 一览

训练好的 LoRA 都在 `outputs/continuation_lora/` 下，按实验目录组织、每目录含 `step-*.safetensors`
与 `training_args.json`。下表为已完成训练的主要 LoRA（路径均可直接用于推理）：

| # | 实验目录 | 最终 checkpoint | 前缀模式 | λ_audio | CP / DP | 样本 | 说明 |
|---|---|---|---|---|---|---|---|
| 1 | `h3_continuation_lora_mask_v14_2n16g_cp2_v9_mixedprefix_100k_lambda0p5` | `step-3123.safetensors` | `mixed` | 0.5 | 2 / 8 | ~99936 | 接续主链路推荐；`run_ai2v_50step_60s.sh` 默认用它 |
| 2 | `h3_continuation_lora_mask_v14_2n16g_cp4_v5_gas8` | `step-1271.safetensors` | 默认(noised) | 0.5 | 4 / 4 | ~40691 | 接续；`run_full_with_lora.sh` 默认用它 |
| 3 | `h3_continuation_lora_mask_v14_2n16g_cp4_v3` | `step-10173.safetensors` | 默认(noised) | 0.5 | 4 / 4 | ~40691 | 接续，早期完整长训 |
| 4 | `h3_continuation_lora_mask_v14_2n16g_cp4_v7_lambda1p0_fix` | `step-1271.safetensors` | `mixed` | **1.0** | 4 / 4 | ~40691 | 接续变体（音频权重加大） |
| 5 | `h3_continuation_lora_mask_v14_2n16g_cp2_v10_alignedbase` | `step-1000.safetensors` | `mixed` | **1.0** | 2 / 8 | ~99936 | 接续变体（aligned-base 基座） |
| 6 | `h3_memory_only_stage2_2n16g_v2` | `step-375.safetensors` | — | — | — | — | memory-only stage2（记忆机制） |

> 完整路径前缀均为 `outputs/continuation_lora/`，例如 #1 的最终权重为
> `outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp2_v9_mixedprefix_100k_lambda0p5/step-3123.safetensors`。
> v5 / v9 经过 LoRA 对比评测（见 `outputs/continuation_lora/lora_compare/lora_compare_table.md`），
> 是主要使用的一对；`smoke_memory_only/step-2` 仅流程验证、非成品。

> 接续 LoRA 用 `--lora <路径> --lora-scale 1.0` 加载；memory LoRA 用法见
> `docs/zh/minimax_h3_memory.md`。

---

## 推理

### 1. 文 → 音视频（T2VA）

```bash
python examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py
```

默认生成 480×832、124 帧（约 5s）、50 步去噪的音视频，输出 `t2va.mp4`。改参数直接编辑脚本里的
`pipe(...)` 调用（`height / width / num_frames / num_inference_steps / seed`）。

**现成 bash 批处理**：
- `run_t2va_60s_batch.sh`：一次性跑 5 条 T2VA prompt、各自逐窗接续合成为 60s 视频。
- `run_30s_local_cp_infer.sh`：30s 长窗 + Ring CP 推理。

```bash
# 5 prompts 合成 60s（默认 8 步，768x1344）
bash examples/minimax_h3/model_inference/run_t2va_60s_batch.sh
```

### 2. 文 + 参考 → 音视频（Ref2VA）

`MiniMax-H3-Ref2VA.py`：除 prompt 外还提供参考图 / 参考视频帧，约束人物形象与镜头。

### 3. 局部重拍（Retake）

`MiniMax-H3-Retake.py`：给定一段已有视频的某些帧（region），让模型在这些帧约束下重新生成其余
部分，用于局部修正。

### 4. 长视频接续生成（Continuation）⭐

把长视频切成多个窗口，逐窗以上一窗尾部为条件续写，避免窗口边界跳变。支持三种接续模式：

```bash
python examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base <FL2VA根目录> \
  --checkpoint <transformer权重> \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --output outputs/h3_continuation.mp4 \
  --mode masked-av-v14 \
  --lora <接续LoRA> --lora-scale 1.0 \
  --height 480 --width 832 \
  --window-frames 345 --overlap-frames 39 \
  --num-inference-steps 50
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--mode` | `retake-hard`（默认）/ `latent-handoff` / `masked-av-v14` |
| `--overlap-frames` | 窗口重叠帧数；masked-av 用 39/90/141… |
| `--window-frames` | 每窗帧数，需满足 H3 对齐（如 345） |
| `--lora` / `--lora-scale` | 接续 LoRA 及缩放（`0`=纯基座） |
| `--turbo-lora` | 8 步蒸馏 LoRA（配合 `--flow-shift ~6.0`） |
| `--segment-plan` | 长视频的分镜/窗口切分计划 |

**现成 bash**：`run_single_shot_batch.sh`（单镜头连续接续批处理，逐窗推进并组装输出）。

```bash
bash examples/minimax_h3/model_inference/run_single_shot_batch.sh
```

### 5. AI2V：音频驱动视频

```bash
python examples/minimax_h3/model_inference/MiniMax-H3-AI2V-Masked-1min.py \
  --h3-root <FL2VA根目录> \
  --checkpoint <transformer权重> \
  --image <起始图> --audio <音频wav> \
  --lora <接续LoRA> --lora-scale 1.0 \
  --height 960 --width 544 \
  --window-frames 345 --overlap-frames 39 \
  --windows 5 --steps 50 --seed 0 \
  --output outputs/h3_ai2v.mp4
```

**现成 bash**：
- `run_ai2v_only.sh`：只跑 AI2V 多镜头任务（每窗注图）。
- `run_ai2v_50step_60s.sh`：AI2V 60s，50 步去噪（用 step-1000 LoRA）。

```bash
bash examples/minimax_h3/model_inference/run_ai2v_only.sh
bash examples/minimax_h3/model_inference/run_ai2v_50step_60s.sh
```

> 一次性跑 T2VA + AI2V 的入口见下节「综合批处理」。

### 6. 低显存 / 量化变体

`examples/minimax_h3/model_inference_low_vram/` 提供同名脚本的 disk-offload 变体；`MiniMax-H3-NF4-*.py`、
`MiniMax-H3-Int8-*.py`、`MiniMax-H3-FP8-*.py` 提供量化精度，适合显存受限环境。

### 7. 综合批处理：一次跑 T2VA + AI2V

想一次性跑完 T2VA（5 prompts）与 AI2V 多镜头，用 `run_h3_60s_batch.sh`：

```bash
# 默认：跑 T2VA + AI2V
bash examples/minimax_h3/model_inference/run_h3_60s_batch.sh

# 只跑 AI2V
RUN_T2VA=0 bash examples/minimax_h3/model_inference/run_h3_60s_batch.sh

# 只跑 T2VA
RUN_AI2V=0 bash examples/minimax_h3/model_inference/run_h3_60s_batch.sh
```

脚本内部已配好 venv、权重路径与 GPU，可用环境变量覆盖。下表是 `run_h3_60s_batch.sh` 支持的主要项
（默认值在括号中）：

| 变量 | 作用（默认） |
| --- | --- |
| `RUN_T2VA` / `RUN_AI2V` | 开关各任务（均 `1`） |
| `H3_HEIGHT` / `H3_WIDTH` | T2VA 分辨率（`768` / `1344`） |
| `H3_AI2V_HEIGHT` / `H3_AI2V_WIDTH` | AI2V 分辨率（`960` / `544`） |
| `H3_NUM_INFERENCE_STEPS` | T2VA 去噪步数（`8`） |
| `H3_CHECKPOINT` / `H3_MERGED_CHECKPOINT` | 基座 checkpoint |
| `H3_T2VA_LORA` / `H3_T2VA_LORA_SCALE` | T2VA 接续 LoRA 与缩放（默认空 / `0.7`） |
| `H3_AI2V_IMG` / `H3_AI2V_WAV` | AI2V 起始图 / 驱动音频 |
| `H3_AI2V_LORA` / `H3_AI2V_LORA_SCALE` | AI2V LoRA 与缩放 |
| `H3_AI2V_STEPS` / `H3_AI2V_WINDOWS` | AI2V 去噪步数 / 窗口数（`8` / `8`） |
| `H3_AI2V_GPU` | AI2V 使用哪张卡（`0`） |
| `H3_AI2V_INJECT_IMAGE` | 每窗注入首帧图（`0`；`1`=开） |
| `H3_AI2V_SHOT_WINDOWS` | 窗口边界硬切位置，如 `"0,3,6"`（默认空=不切） |
| `H3_AI2V_OUT` | AI2V 输出路径 |

每个脚本还支持 `-h/--help` 打印完整环境变量说明。

---

## 训练

统一入口是 `examples/minimax_h3/model_training/train.py`。

### 普通 SFT（`sft`）

```bash
python examples/minimax_h3/model_training/train.py \
  --task sft \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/..."   # 或用 --model_paths 指定本地权重 \
  --dataset_metadata_path <数据 metadata> \
  --output_path <输出目录> \
  ...
```

**30s SFT 现成 bash**（`examples/minimax_h3/model_training/` 下，已配好 venv、DeepSpeed、多卡/CP）：

```bash
# 单节点 smoke / full
bash examples/minimax_h3/model_training/gemini_30s_sft_smoke.sh
bash examples/minimax_h3/model_training/gemini_30s_sft_full.sh

# 3 节点 24 卡，配 CP（smoke / full，可切换 CP2 / CP4，以及是否 CPU checkpoint）
bash examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_cp4_full_no_cpu_checkpoint.sh
```

### 接续 LoRA 训练（`continuation_sft`）⭐

训练"以上一窗为条件续写"的 LoRA，配合推理主链路 `masked-av-v14`。

```bash
# 1) 构建接续训练数据
python examples/minimax_h3/model_training/build_continuation_dataset.py ...

# 2) 缓存窗口 latent
python examples/minimax_h3/model_training/build_continuation_cache.py ...

# 3) 训练
python examples/minimax_h3/model_training/train.py \
  --task continuation_sft \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/..." \
  --continuation_manifest <上一步生成的 latent cache manifest> \
  --continuation_prefix_present_mode mixed \
  --continuation_conditioning masked-av-v14 \
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 \
  --output_path <输出目录> \
  ...
```

训练目标：`ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss`，只训 DiT 的 LoRA，overlap 前缀受保护
（权重 0）、首段 suffix 高权重（3.0）、视频/音频按 `λ_audio=0.5` 合并。

**现成 bash**（`examples/minimax_h3/model_training/` 下，已配好 venv、accelerate/DeepSpeed、路径与 GPU）：

```bash
# 8 卡（纯 DP，CP=1，每卡完整 345 帧样本）
bash examples/minimax_h3/model_training/run_continuation_lora_mask_v14_8gpu.sh

# 2 节点 16 卡（CP=4，DP=4）——作者正式配置
# 在两个节点各跑一次
bash examples/minimax_h3/model_training/run_continuation_lora_mask_v14_2node16gpu.sh

# aligned-base 变体（基座权重要先对齐）
bash examples/minimax_h3/model_training/run_continuation_lora_mask_v14_2node16gpu_alignedbase.sh
```

这些脚本内的关键默认：

| 项 | 默认 |
| --- | --- |
| venv | `/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv` |
| 数据缓存 manifest | `outputs/caches/h3_masked_av_v14_cache_8gpu`（`$MANIFEST`） |
| LoRA 输出 | `outputs/continuation_lora/h3_continuation_lora_mask_v14_*`（`$OUTPUT_PATH`） |
| 训练 Adapter | `models/MiniMaxH3/TrainingAdapter/model.safetensors`（`$PRESET_LORA_PATH`） |
| 权重基座 | `--model_id_with_origin_paths "MiniMaxH3:FL2VA/..."`（本地 `$DIFFSYNTH_MODEL_BASE_PATH`） |
| DeepSpeed | `full/deepspeed_zero3_cp8.json`（ZeRO-3，`$ACCELERATE_DEEPSPEED_CONFIG_FILE`） |

缓存构建脚本：`build_continuation_cache.py`（生成上面 `$MANIFEST` 指向的 latent cache）。

---

## 相关文档

- `docs/zh/Model_Details/MiniMax-H3.md` — H3 模型总览与基础推理
- `docs/zh/minimax_h3_model_changes.md` — 三种接续模式（masked-av-v14 / retake-hard / latent-handoff）原理
- `docs/zh/minimax_h3_memory.md` — 记忆（memory）机制
- `docs/zh/minimax_h3_continuation_lora_training.md` — 接续 LoRA 数据与训练
- `docs/zh/minimax_h3_continuation_methods_used.md` — 已使用方法总览

---

## 备注

- 接续 vs Retake 语义不同：Retake 是**局部重拍**某一 region，接续是**整窗延续**；两者在
  pipeline 中互斥（`continuation_*_latents` 与 `retake_*` 不能同时给）。
- 接续训练与推理的**前缀注入分布**要一致（`prefix_present_mode = noised / clean / mixed`），否则
  训练看到的 clean 前缀在推理时会是陌生分布。
