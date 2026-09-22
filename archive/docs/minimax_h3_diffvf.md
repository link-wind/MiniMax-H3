# MiniMax-H3 Diff-VF 长视频采样

`diff-vf` 是 MiniMax-H3 接续生成的实验模式。它不在生成完成后修补接缝，而是在扩散的每一个时间步，把所有窗口看成一条共享的全局 latent 时间线：局部窗口预测在 overlap 中融合后，全局视频和音频 latent 各自只执行一次 scheduler 更新。最终视频/音频从完整时间线统一解码，因此不会重复输出 overlap。

它仍是推理期实验，不是 continuation 专项训练的替代品。遇到明显构图、动作或叙事漂移时，优先考虑更短的镜头计划、明确的全局提示词和关键帧/角色约束，而不是把 TES 强度继续调高。

## 组件

- HNI（Hybrid Noise Initialization）：`--hni-weight` 控制共享基准噪声比例。`0` 是独立创新噪声，`1` 是完全共享噪声，建议首先测试 `0.5`。
- WWS（Weighted Window Sampling）：每一步融合 overlap 预测。`cosine-squared` 是默认策略；`center-distance` 用于消融对比。
- TES（Temporal Extended Sampling）：只对视频 latent 做稀疏远距离时间采样，并在高噪声早期与 WWS 预测融合。它提升长程外观锚定的可能性，也可能破坏局部运动，因此默认关闭。
- 统一 decode：完成全局采样后只 decode 一次视频/音频 latent。该路径避免逐窗口 VAE 上下文差异，但峰值显存显著更高。

## 单次运行

以下命令以两窗口、15 秒 x 2 的常用形状运行视频 TES。请在 H3 专用环境中执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA \
  --checkpoint /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/transformer \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --output outputs/h3_diffvf_tes.mp4 --mode diff-vf \
  --window-frames 243 --overlap-frames 34 --height 480 --width 832 \
  --seed 7 --num-inference-steps 50 --hni-weight 0.5 \
  --tes --tes-window-steps 12 --tes-stride 2 \
  --fusion-local-start 0.25 --fusion-local-end 0.85
```

报告文件与输出视频同名 `.json`，同时记录 resolved timeline、HNI RNG streams、TES permutation、每步融合系数、边界/长程/音频指标，以及 GPU 峰值显存。窗口状态写在输出旁的 `.state/continuation_state.json`。

## 固定种子消融

`MiniMax-H3-DiffVF-Eval.py` 为每个变体启动独立进程，避免上一轮的模型或 VAE 缓存干扰下一轮。它固定 plan、分辨率、步数与 seed，只改变指定算法变量：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  examples/minimax_h3/model_inference/MiniMax-H3-DiffVF-Eval.py \
  --h3-base /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA \
  --checkpoint /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/transformer \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --output-dir outputs/h3_diffvf_matrix --seed 7
```

可用变体为 `joint-wws`、`hni-independent`、`hni-mixed`、`hni-shared`、`center-distance`、`video-tes`。先用 `--dry-run` 检查写入的 `matrix.json`；实际运行时，按从便宜到昂贵的顺序显式选择，例如 `--variants joint-wws,hni-mixed,video-tes`。

比较时优先检查每个报告的 `joins`、`video_metrics`、`audio_metrics` 以及 `experiment_metadata.runtime`。这些是自动诊断，不应单独作为感知质量结论；尤其应人工检查接缝是否闪烁、运动是否错位、人物/光照是否保持，以及对白是否有音节切断或相位异常。

## 限制与安全边界

- `audio_tes` 默认是 `disabled`。`--audio-tes experimental` 仅用于研究性验证，会改变音频稀疏采样，必须单独审听语音韵律、响度和相位。
- TES 不支持带 `reference blocks` 的 Ref2VA 请求；该组合会在推理前明确拒绝。FL2VA 的 prompt 和 keyframe 条件使用绝对全局时间坐标。
- 所有窗口必须同形，帧数满足 H3 的 `17n+5`，overlap 必须按完整的 17 帧 VAE clip 对齐。
- 统一 decode 的实际峰值可能接近 80 GiB H100 单卡上限。可用 `--max-decode-memory-gib` 设置启动前预算门禁。当前 `--decode-strategy temporal-chunk` 只会给出明确的未实现诊断，不会静默退化为逐窗口 decode。
- 在未完成固定 seed GPU 消融前，不应把 `diff-vf` 作为默认接续模式或声称其一定优于 Retake、latent-handoff、joint-multidiffusion。
