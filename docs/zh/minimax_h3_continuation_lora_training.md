# MiniMax-H3 continuation LoRA 数据与训练指南

本文对应 OpenSpec change `h3-continuation-lora-training`，说明从
`data_with_face_and_speech_and_caption.jsonl` 构建 continuation 训练数据、
缓存 H3 latent、训练 `mask-v14` 区域加权 LoRA，以及加载到
`MiniMax-H3-Continuation.py` 推理的完整流程。

## 1. 数据索引

索引构建由以下脚本完成：

- `examples/minimax_h3/model_training/build_continuation_dataset.py`
- `diffsynth/utils/continuation_lora.py` 中的 `iter_continuation_samples`

源 JSONL 的关键字段：

| 字段 | 用途 |
|---|---|
| `sequence_id` | sequence-level train/validation/test 划分，禁止跨 split 泄漏 |
| `file_path` | 源视频路径 |
| `audio_path` | 音频路径，缺失时显式标记 `audio_missing` |
| `prompt` | 同时承载 shot 边界和 prompt/ASR/caption 清洗候选 |
| `[Shot N \| start-end]` | 被解析为可用 shot 区间 |

索引器逐行流式读取 JSONL，不把全量标注载入内存。只有完整窗口落在同一
shot 内才生成正样本；出现 `--- cut: frame_gap=... ---`、未知 shot 边界或
窗口跨 cut 时记录为拒绝原因。`prompt` 优先于 ASR 摘要，再回退到 caption，
过滤 caption 模板残留、乱码和空文本。

## 2. 窗口与 latent cache

默认正式训练与验证窗口为 `345` 帧（14.375 秒），长度必须满足 `17n+5`：

- 124 帧：CPU/config smoke
- 243 帧：低成本训练调试
- 345 帧：正式训练与主要验证
- 362 帧：仅保留为历史实验记录，不用于本 LoRA 训练验收

默认 overlap 为 39 帧，对应 12 个视频 latent token 和约 65 个音频 latent
step；mask-v14 将 overlap 全部硬保护，不使用 transition band。视频
统一到 24 fps，音频统一到 32 kHz，audio latent rate 固定为 40 steps/s。
视频和音频先由同一物理时间区间导出帧/sample/latent 边界，避免相邻窗口
累积漂移。

`build_continuation_cache.py` 使用 H3 Video/Audio VAE 编码窗口并写入：

```text
<cache_dir>/<split>/<sample_id>.pt
<cache_dir>/<split>/manifest.jsonl
```

每个 cache 记录保存 schema version、shape、dtype、fps、sample rate、shot
区间、源文件 hash 和窗口坐标。读取前校验 schema、hash、shape、dtype 和
采样率；不匹配时拒绝训练并提示重建 cache，不能静默回退。

## 3. `mask-v14` 区域语义

对每个目标窗口，latent 时间轴被分为三个区域：

```text
[0, overlap)               history noisy anchor，硬保护，不参与 loss
[overlap, L)               suffix noisy target，计算 flow loss
```

训练时在同一 timestep 下构造 `epsilon_main` 和共享前缀噪声：

- 主 noisy target：`scheduler.add_noise(full_target, epsilon_main, t)`
- timestep-aligned noisy anchor：`scheduler.add_noise(history_clean, epsilon_main[:overlap], t)`
- overlap loss weight 为 0，suffix 首 5 个 latent token 权重为 3，其余为 1

### 查看训练 loss

- 终端：训练进度条会显示 `loss=...`。
- CSV：增加 `--enable_csv_log`，查看 `<output_path>/loss.csv`。
- TensorBoard：增加 `--enable_tensorboard_log`，运行
  `tensorboard --logdir <output_path>/tensorboard_log`。

区域权重默认：

| 区域 | 权重 |
|---|---|
| hard core | 0 |
| transition | 0.5 |
| 第一个完整 suffix video clip | 3.0 |
| 后续 suffix | 1.0 |
| audio 总 loss | `lambda_audio=0.5` |

视频和音频分别按有效 token 归一化后再组合，避免音频 token 数量主导梯度。

## 4. LoRA 训练

入口为 `examples/minimax_h3/model_training/train.py`，任务名
`continuation_sft`。LoRA 默认注入：

```text
attn.qkv_proj
attn.out_proj
mlp.fc1
mlp.fc2
```

DiT 基础权重、VAE、文本编码器、RoPE 和 scheduler 冻结，只训练上述 LoRA
参数。训练入口支持：

- `--bf16`
- `--use_gradient_checkpointing` / `--use_gradient_checkpointing_offload`
- `--cp_world_size` 与 `--cp_seed`
- `--training_cfg_scale`
- `--validate_continuation_config`
- 完整 checkpoint 保存/恢复

完整 checkpoint 是 `.pt` 文件，包含 LoRA state dict、optimizer/scheduler
state、global step、seed/RNG state、manifest hash 和 region config。恢复时
manifest hash 或 region config 不匹配会直接报错。普通 LoRA 导出仍是
`ModelLogger` 写出的 `.safetensors`。

## 5. 推理加载

`examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py` 新增：

```bash
--lora /path/to/continuation_lora.safetensors \
--lora-scale 1.0
```

`--lora-scale 0` 不修改 pipeline，行为与 base 完全一致；`--lora-scale > 0`
通过 `pipe.load_lora(pipe.dit, lora_path, alpha=scale)` 融合 adapter。

## 6. 评测与限制

配对接续评测入口：

```bash
python examples/minimax_h3/model_evaluation/continuation_lora_evaluation.py \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --checkpoint /path/to/base.safetensors \
  --h3-base /path/to/FL2VA \
  --lora /path/to/continuation_lora.safetensors \
  --validate-only
```

评测固定 checkpoint、plan、prompt、overlap、scheduler、seed 和统一 VAE
decode，报告亮度/颜色差、人物框位移、光流或 latent velocity 差、闪烁、
音频 RMS/频谱/相位、A/V 边界偏移，并支持 transition/suffix/scale/overlap
消融和回归门槛。

已知限制：

- 首版只针对单镜头连续片段；跨 shot 的意图性场景切换不构造正样本。
- LoRA 只承诺改善边界条件分布，不保证完全消除长链漂移。
- GPU 正式训练、长视频、CP 和主观质量验证仍需独立执行并记录。
