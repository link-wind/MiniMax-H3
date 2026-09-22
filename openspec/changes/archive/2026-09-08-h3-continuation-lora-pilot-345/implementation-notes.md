# 345 帧 pilot 当前结果

## 已完成

- 使用 `window_frames=345`、`overlap_frames=34`、`hard_core_frames=17`、
  `split_seed=17` 扫描源 JSONL。
- 输出 `outputs/h3_continuation_pilot_345/index.jsonl`，包含 2400 条样本：
  train 2000、validation 200、test 200。
- sequence-level 划分通过：train/validation/test 分别包含 1698/170/174 个
  sequence，未发现跨 split sequence。
- 扫描统计：`records=40487`、`accepted=4511`、`short_shot=186495`、
  `bad_json=0`、`missing_field=0`、`missing_path=0`、`no_shot=0`。
- 10 条真实样本已在 MiniMax-H3 环境完成 480x832 VAE 编码，缓存位于
  `outputs/h3_continuation_latent_cache_smoke_345_resized/train/`。
- 10 条缓存均通过 roundtrip 和 metadata 校验：video
  `[1,24,102,30,52]`、audio `[2,32,575]`、音频窗口 460000 samples、
  resize 策略 `center_crop_bicubic`。

## 全量 cache 结果

- 2400 条 pilot 已在两张 H100 上完成 480x832 VAE 编码。GPU0 分片写入
  1191 条并跳过 2 条已存在项，GPU1 分片写入 1205 条并跳过 2 条；两分片
  `failed=0`，最终文件数为 2400。
- 已合并为 `outputs/h3_continuation_pilot_cache_345/`，按 split 提供统一
  manifest，并通过路径存在性、schema、345 帧布局、latent shape metadata
  和全局 `sample_id` 去重检查。manifest SHA-256 保存在
  `manifest_hashes.json`。

## 尚未执行

- 500-1000 step 单卡 LoRA 训练及 checkpoint 恢复验证。
- base/LoRA 固定计划配对推理和接缝质量门槛。
