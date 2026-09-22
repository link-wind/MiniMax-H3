# MiniMax-H3 continuation LoRA 验证汇总

## 1. 已执行的 CPU smoke

- `pytest -q tests/test_h3_continuation_lora.py tests/test_h3_continuation_lora_evaluation.py tests/test_minimax_h3_continuation.py`
  - 结果：61 passed
  - 覆盖：流式索引、shot-safe 窗口、媒体/latent 对齐、taper-refine-v2 数学、
    区域权重 loss、CP/CFG/gradient checkpointing 接口、完整 checkpoint
    roundtrip、LoRA scale=0/scale>0、配对接续评测、回归门槛、JSON/Markdown
    消融报告
- 124 帧 CPU 配置检查：
  - `train.py --task continuation_sft --validate_continuation_config`
  - 结果：输出 `continuation=true`、region config 和默认 LoRA targets
- 评测规划生成：
  - `continuation_lora_evaluation.py --validate-only`
  - 结果：`outputs/h3_continuation_lora_evaluation/deferred_validation.json`
    列出 base/scale 0/scale 1 的固定矩阵，标记为 planned_not_executed

## 2. 静态与 OpenSpec 校验

- `python -m py_compile`：通过
- `openspec validate h3-continuation-lora-training --strict`：valid
- `openspec validate h3-av-continuation --strict`：valid
- `git diff --check`：无空白错误

## 3. 数据与 checkpoint 状态

已对真实 JSONL 的前 5,000 条记录执行流式索引扫描（未启动 VAE/cache/GPU）。
首次扫描发现存在 `[Shot n/m | end | start-end]` 头部变体；扩展解析器后重新扫描。
以下为修复后的结果，使用 243 帧窗口、34 帧 overlap、17 帧 hard core，并开启
`--require-paths`：

| 指标 | 结果 |
| --- | ---: |
| 源记录 | 5,000 |
| 坏 JSON / 缺失字段 / 无 shot | 0 / 0 / 0 |
| 缺失媒体路径 | 0 |
| 被 short shot 过滤的 shot | 22,436 |
| 接受样本窗口 | 1,125 |
| 接受样本的 sequence 数 | 974 |
| train / validation / test | 1,024 / 58 / 43 |
| 接受样本音频缺失 | 0 |
| prompt source | 1,086 条均为 `prompt` |
| prompt 长度 | 993-3,431 字符，平均 1,909.8 |

样本 shot 时长为 10.20-15.00 秒，平均 13.53 秒；修复后的索引中 `prompt_source`
全部为 `prompt_shot`，且 1,125 条样本均不再包含 `[Shot ...]` 头部或 `--- cut`
标记。prompt 已按 `shot_index` 截取，仅保留当前镜头的 SUBJECT/Scene/Event 描述。

本次 smoke 产物已保存到：

- `outputs/h3_continuation_index_smoke_5000/index.jsonl`
- `outputs/h3_continuation_index_smoke_5000/stats.json`
- `outputs/h3_continuation_index_smoke_5000/report.json`

随机抽查 30 条样本时，SUBJECT/Scene/Event 均存在，且没有 shot header、cut marker
残留；这只是自动化结构抽查，仍建议在正式训练前进行人工语义抽查。

以下完整数据阶段统计仍待补录：

- 完整 JSONL 的过滤统计和 train/validation/test 数量
- 实际 cache 文件数、磁盘占用和 metadata hash
- 真实 LoRA checkpoint 的训练步数、loss 曲线和导出文件 hash

当前 checkpoint 元数据通过单元测试验证：包含 LoRA state dict、
optimizer/scheduler state、global step、seed/RNG、manifest hash 和 region
config，恢复时对 manifest hash 和 region config mismatch 拒绝加载。

## 4. 评测报告状态

评测 CLI 已能生成固定矩阵和 deferred validation manifest；真实 GPU 评测尚未
执行，因此没有可发布的 base/LoRA 亮度差、人物框位移、光流差、闪烁统计、
音频频谱/相位差和回归门槛结果。接入人物检测器或光流函数后，评测 CLI 会把
这些指标写入每个 join 的 JSON 报告。

## 5. Deferred validation

以下项目必须在真实模型环境中执行并单独记录，不作为本次 CPU 验证结论：

- 243/345 帧单卡与多卡 GPU 训练 smoke
- 完整数据集的 latent cache 构建和磁盘/显存验收
- base+taper-refine-v2 与 LoRA+taper-refine-v2 的固定 seed 完整步数配对
- overlap/transition/suffix/LoRA scale/样本类别消融
- 长视频、CP、人物检测框、光流和主观质量评估
- 断点恢复后的优化器状态一致性核对

## 6. 下一步

1. 抽样人工检查 20-50 个清洗后的样本，确认 SUBJECT/Scene/Event 与 shot 时间一致。
2. 通过后对完整 JSONL 生成正式 index；之后进入少量 124 帧真实 H3 VAE cache smoke。
