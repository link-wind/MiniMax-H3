# 实施与验证记录

## 已实现

- `paper-strict` 是显式 opt-in 采样语义；既有 `existing` 路径保持预测级融合行为。
- 严格路径实现论文定义的 HNI、状态级 WWS、早期视频 TES 和余弦状态融合。
- WWS 与 TES 均使用 H3 全局绝对 video/audio source indices；音频保留连续 WWS 状态路径。
- 完整全局 latent 时间线只执行一次视频/音频 VAE decode。

## 数值修复

真实 H3 smoke 暴露出一个 bfloat16 精度问题：在长度为 1149 的音频全局时间线直接用 bfloat16 计算中心距离权重时，末尾有效 token `1148` 被量化为零权重。`build_wws_weight_map()` 现以 float32 构造全部时间权重，再转换为 latent dtype；新增回归测试覆盖该末尾 token。

## 验证

CPU 验证：

```bash
PYTHONPATH=. pytest -q tests/test_minimax_h3_diffvf.py tests/test_minimax_h3_continuation.py
```

结果：`61 passed, 2 skipped`。

静态验证：

```bash
python -m compileall -q diffsynth/pipelines/minimax_h3_diffvf.py
git diff --check
openspec validate h3-strict-diffvf-reproduction --strict
```

GPU 最小端到端 smoke：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA \
  --checkpoint /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/transformer \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan_15s.json \
  --height 256 --width 448 --window-frames 362 --overlap-frames 34 \
  --seed 7 --num-inference-steps 1 --mode diff-vf --tes \
  --wws-weighting center-distance \
  --diffvf-sampling-semantics paper-strict \
  --output outputs/h3_diffvf_paper_strict_smoke_256x448_1step.mp4 \
  --report outputs/h3_diffvf_paper_strict_smoke_256x448_1step.json
```

结果：成功输出 448x256、24fps、690 帧（28.75 秒）视频、报告和 manifest。manifest 记录了 `sampling_semantics=paper-strict`、两个 TES interleave 窗口、`tes_active_steps=[0]`、视频融合系数 `0.5` 与所有 HNI seed streams。

该 smoke 只验证严格调度、绝对位置和统一解码的可运行性；1 个去噪 step 不具备视觉质量比较意义。后续质量实验应在固定种子、固定 prompt 下对 `existing` 与 `paper-strict` 使用相同完整 step 数对比接缝和后段运动稳定性。
