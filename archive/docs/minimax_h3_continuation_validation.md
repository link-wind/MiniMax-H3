# MiniMax-H3 接续生成验证记录

## 已执行的 CPU 验证

```bash
PYTHONPATH="$PWD" pytest -q tests/test_minimax_h3_continuation.py
openspec validate h3-av-continuation --strict
```

当前 CPU 测试覆盖 17n+5 形状解析、17 帧 clip overlap、视听同一物理时间轴、suffix 所有权、等功率音频 crossfade、状态 manifest、Retake-hard、单模态 Retake、latent handoff 与 fallback、CP 元数据一致性、writer-rank 规则和评测报告。

## 已执行的 GPU smoke 验证

使用 MiniMax-H3 专用环境
`/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python`，其依赖为 `transformers 5.14.0`、`torch 2.9.1+cu128`，并在一张 H100 80GB 上加载本地 FL2VA checkpoint。

运行了同一个两窗口 plan（243 帧窗口、34 帧 overlap、480x832、seed=7、仅 1 个 denoise step 的链路 smoke）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA \
  --checkpoint /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/transformer \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --output outputs/h3_continuation_gpu_smoke.mp4 \
  --num-inference-steps 1 --mode retake-hard --height 480 --width 832 --seed 7

# 同一命令，仅改为 --mode latent-handoff，输出 h3_continuation_latent_gpu_smoke.mp4
```

结果：两次均生成 452 帧（18.833333 秒）和 602667 个 32 kHz 音频样本；MP4 stream 为 24 FPS 视频和 32 kHz 音频。Retake-hard 的第二窗口 manifest 为 `retake-hard`；latent 测试的第二窗口为 `latent-handoff`。两个报告均记录边界为第 243 帧 / 第 324000 个音频样本（10.125 秒）。生成文件和 manifest 位于：

- `outputs/h3_continuation_gpu_smoke.mp4`、`outputs/h3_continuation_gpu_smoke.json`
- `outputs/h3_continuation_latent_gpu_smoke.mp4`、`outputs/h3_continuation_latent_gpu_smoke.json`

这是功能与时间轴 smoke，不是质量评测：1 step 的画面和音频不应用于评价效果。

## 已执行的结构化 Prompt 对比

使用 `h3_continuation_plan_15s.json` 的 H3 风格 `integrated_multimodal_description`，在一张 H100 80GB 上执行了两窗口、每窗口 362 帧（约 15.08 秒）、34 帧 overlap、480x832、seed=7、50 steps 的完整对比。两个窗口拼接后为 690 帧（28.75 秒），接缝为第 362 帧 / 482667 个音频样本（15.083333 秒）。

| 方法 | 输出 | 接缝音频能量跳变 | 接缝音频频谱跳变 | 视觉检查 |
| --- | --- | ---: | ---: | --- |
| Retake-hard | `outputs/h3_structured_15s_retake_full50.mp4` | 0.000002579 | 0.034871 | 人物、服装、咖啡馆和接缝前后镜头连续，无全局模糊 |
| Latent handoff | `outputs/h3_structured_15s_latent_full50.mp4` | 0.000002232 | 0.030387 | 人物、服装和雨景连续，无全局模糊；仍需更多 seed 验证场景稳定性 |
| Taper-refine（静态 soft overlap） | `outputs/h3_structured_15s_taper_full50.mp4` | 0.000012163 | 0.047197 | overlap 内过早放开历史 anchor，接缝前人物/镜头发生明显变化；不建议作为默认 |
| Taper-refine-v2（timestep-aligned noisy anchor） | `outputs/h3_structured_15s_taper_v2_full50.mp4` | 0.000001965 | 0.030551 | 画面可生成且接缝稳定；略差于 latent handoff，暂不作为默认 |

本次 `taper-refine` 使用同一 plan、seed、窗口和 50 steps，仅将 latent handoff 的 overlap mask 从二值固定改为从强 anchor 到弱 anchor 的线性渐变。结果比 hard prefix 更差：音频能量跳变约为 latent handoff 的 5.4 倍，频谱跳变约为 1.55 倍；视觉检查也显示人物在真正的 suffix 边界之前已经改变朝向。

因此当前结论是：**不能直接把 clean latent 与新窗口 latent 使用静态线性权重混合**。后续若继续研究 soft overlap，应改为 timestep-aligned noisy anchor，并在接缝前保留一段 hard core；在此之前生产默认仍为 `retake-hard`，latent handoff 仍为第二选择。

随后实现并验证了 `taper-refine-v2`：34 帧 overlap 中前 17 帧保持 hard core，后 17 帧使用随 denoise timestep 对齐的 noisy anchor（视频和音频分别按各自 scheduler）。在相同 plan、seed、分辨率和 50 steps 下，v2 的音频能量跳变优于 Retake-hard，但频谱跳变略高于 latent handoff，未显示出稳定的额外收益。因此 v2 保留为显式实验模式，不改变默认路径。

针对 15 秒边界的闪烁，又完成了一次 v2 修正版对比：删除 scheduler step 后对下一个 timestep 的重复 anchor projection，并将 anchor noise 改为当前窗口初始 noise 的 overlap 前缀（shared-noise）。结果如下：

| 方法 | 音频能量跳变 | 音频频谱跳变 | 接缝相邻帧像素 MAD |
| --- | ---: | ---: | ---: |
| v2（独立 anchor noise + 重复 projection） | 0.000001965 | 0.030551 | 8.50 |
| v2 shared-noise（单次 projection） | 0.000001407 | 0.028657 | 10.54 |

shared-noise 明显改善了音频边界指标，但像素 MAD 没有改善，说明当前残留的视觉闪烁主要来自两个窗口独立 VAE 解码的时间感受野/边界上下文，而不只是 diffusion noise 相位。时间轴仍严格对齐在第 362 帧（15.083333 秒），因此下一步应优先验证 latent 序列的联合 VAE 解码或接缝帧 ownership，而不是继续调 noise 权重。

这组结果说明 1-step 的模糊来自不足的去噪步数，不是接续链路或 VAE 解码错误。两种模式在该单一 seed/prompt 上的音频边界基线都较小，但这不是感知质量评分；仍需要人工听辨对白切分、音色和环境声连续性。当前 production 默认仍建议使用 Retake-hard，因为它走的是 H3 既有 Retake 条件路径；latent handoff 是无 decode/re-encode 的实验路径，尚未经过专门接续训练。

## 待执行的质量与规模验证

## 双侧接缝重绘实验

新增了显式 `bidirectional-seam-retake` 模式，用当前窗口的完整解码结果作为右侧
上下文、上一窗口尾部作为左侧上下文，只对接缝后的 17 帧执行第二次 Retake diffusion；
最终输出只替换这 17 帧，右侧其余帧保持第一次生成结果。

使用与 `taper-refine-v2` 相同的结构化 15 秒 plan、362 帧窗口、34 帧 overlap、
480x832、seed 7、50 steps，输出为：

`outputs/h3_structured_15s_bidirectional_seam17_full50.mp4`

结果：视频仍为 690 帧、28.75 秒，音频/视频时间轴没有变化；接缝音频 energy jump
为 `0.000002312`，spectral jump 为 `0.032886`。接缝相邻视频帧 MAD 为 `0.17005`，
高于 `boundaryref1+taper-refine-v2` 的 `0.05792`。视觉检查显示重绘区域第一帧出现
正面到侧面的姿态跳变，说明当前 H3 checkpoint 对“固定双侧 decoded context + 只重绘
17 帧”的局部条件尚未形成稳定生成分布。

因此该方案已完成工程验证，但当前不建议用于生产，也不应简单扩大重绘范围。下一步
若继续研究，应改为 latent/noisy 双侧 anchor，并让重绘区域至少覆盖一个完整运动过渡，
而不是直接使用 decoded frame Retake。

## Joint MultiDiffusion 实验记录

已实现显式 `--mode joint-multidiffusion`，其行为与顺序 Retake/latent-handoff
完全分离：所有窗口先完成条件编码，视频和音频共用一条全局初始噪声时间线；每个
denoise timestep 分别运行各窗口的 H3 联合 DiT 预测，在 overlap 上以互补的
cosine-squared 权重融合，随后每个模态只做一次 scheduler step，最终执行一次全局
VAE decode。该模式禁止 Retake、latent handoff、bridge decode、resume 和动态
reference bank，只用于定位扩散轨迹分叉是否是接缝根因。

CPU 契约测试已通过：

```bash
PYTHONPATH="$PWD" pytest -q tests/test_minimax_h3_continuation.py
# 33 passed
openspec validate h3-av-continuation --strict
```

使用原始 H3 13 分片、结构化 15 秒 plan、362 帧窗口、34 帧 overlap、832x480、
seed 7 的 1-step GPU smoke 已完成：

- 输出：`outputs/h3_structured_15s_joint_multidiffusion_smoke.mp4`
- 实际输出：690 帧、24 FPS、28.75 秒；音频 32 kHz、28.75 秒
- 接缝坐标：第 362 帧 / 第 482667 个采样点
- 接缝音频 energy jump `1.12e-7`，spectral jump `0.00397`
- 单次全局 VAE decode 峰值约 77.8 GiB（H100 80GB），说明该模式不适合直接作为默认流式生产路径

随后完成了同配置的完整 50-step 对照（约 20 分 21 秒），包括 690 帧的单次全局
video-VAE decode；这次 decode 在一张 H100 80GB 上成功完成，输出为：

`outputs/h3_structured_15s_joint_multidiffusion_full50.mp4`

结果为 690 帧、24 FPS、28.75 秒，音频为 32 kHz 且同样为 28.75 秒。接缝仍在
第 362 帧 / 第 482667 个采样点。自动指标为：

| 方法 | 接缝相邻帧 MAD | 接缝音频 energy jump | 接缝音频 spectral jump |
| --- | ---: | ---: | ---: |
| Latent handoff | 0.24182 | 0.000002232 | 0.030387 |
| boundaryref1 + taper-refine-v2 | 0.05792 | 0.000001407 | 0.028657 |
| Joint MultiDiffusion | 0.07403 | 0.000006738 | 0.045570 |

边界附近的逐帧视觉检查表明，Joint MultiDiffusion 保持了同一人物低头、俯身的
连续运动轨迹，没有出现 latent-handoff 的正面到背面跳变。因此它支持“两个独立
denoising 轨迹是接缝错位来源”的判断。但它的局部像素连续性仍略逊于
`boundaryref1 + taper-refine-v2`，且音频边界指标更差；当前不能作为生产默认。

该模式的代价也很高：两个窗口在每个 timestep 都要运行 H3 联合 DiT，且全局 decode
峰值接近 78 GiB。下一轮应只在视频侧保留 WWS，并为音频恢复已验证较稳定的
latent-handoff/Retake 路径，之后再评估 HNI 对长程人物和场景一致性的增益。

推荐的后续命令（完成分块 decode 后执行）：

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /path/to/FL2VA \
  --checkpoint /path/to/transformer \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan_15s.json \
  --output outputs/h3_structured_15s_joint_multidiffusion_full50.mp4 \
  --mode joint-multidiffusion --window-frames 362 --overlap-frames 34 \
  --height 480 --width 832 --seed 7 --num-inference-steps 50
```

在更多 prompt、seed、镜头运动和 CP 规模下至少使用 50 steps 执行：

```bash
# 单卡两窗口 Retake-hard 质量基线
python examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /path/to/FL2VA \
  --checkpoint /path/to/transformer.safetensors \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --output outputs/h3_continuation_retake.mp4 \
  --mode retake-hard

# 同一 plan 的 latent handoff 消融
python examples/minimax_h3/model_inference/MiniMax-H3-Continuation.py \
  --h3-base /path/to/FL2VA \
  --checkpoint /path/to/transformer.safetensors \
  --segment-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --output outputs/h3_continuation_latent.mp4 \
  --mode latent-handoff

# CP=4 多窗口；默认单窗口路径不受这一参数影响
torchrun --nproc_per_node=4 examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-30s-local-cp.py \
  --cp_world_size 4 \
  --continuation-plan examples/minimax_h3/model_inference/h3_continuation_plan.json \
  --continuation-window-frames 243 \
  --continuation-overlap-frames 34
```

在 GPU 质量验证中需要人工复核边界人物/场景一致性、对白是否截断、音乐节拍与环境声连续性，以及 0/50/100/200 ms crossfade 的听感。长时程漂移、CP 多机稳定性和音质感知指标仍是后续验证工作，不应从 1-step smoke 或 CPU 测试推断为已通过。
## Diff-VF GPU smoke（2026-08-26）

在 MiniMax-H3 专用环境使用 GPU 0、`h3_continuation_plan.json`、seed 7、480x832、243 帧窗口、34 帧 overlap、1 个 diffusion step 完成了 `diff-vf + video TES` smoke：

- 输出：`outputs/h3_diffvf_video_tes_smoke.mp4`，452 帧（18.833 秒），音频 602667 samples（32 kHz）
- 统一 decode：成功；没有重复 overlap，物理边界为第 243 帧/324000 音频采样
- 运行时间：392.64 秒；PyTorch CUDA 峰值分配：66.02 GiB
- 配置：HNI 0.5、cosine-squared WWS、TES stride 2/window 12、音频 TES disabled
- 报告：`outputs/h3_diffvf_video_tes_smoke.json`；状态 manifest：`outputs/h3_diffvf_video_tes_smoke.state/continuation_state.json`

这只是接口、shape、绝对时间坐标、TES scatter、融合与统一解码的门禁，不代表最终画质或长视频收益。随后完成了 50-step 的 HNI/WWS/TES/fusion 固定矩阵和 Retake/latent-handoff/joint-multidiffusion 基线对比；多 GPU、更长 horizon 与人工盲测仍保持为 deferred validation。

## Diff-VF WWS 等价性 smoke（2026-08-26）

新增了固定 seed 的 CUDA kernel-level smoke：对同一组 `[B,C,T,H,W]` 噪声预测，分别调用
Diff-VF 的通用 `merge_window_predictions` 和现有
`merge_joint_multidiffusion_predictions`，在 cosine-squared、两窗口、2-token overlap
配置下逐元素比较。CUDA 环境中测试通过（`rtol=1e-5, atol=1e-6`）。这只证明 WWS
合并核保持原型数值语义，不等同于完整 H3 模型的 50-step 画质等价性。

## Diff-VF 15 秒 HNI 对比（2026-08-26）

在同一个 `h3_continuation_plan_15s.json`、362 帧窗口、34 帧 overlap、480x832、seed=7、
50 steps 配置下，完成了两窗口 Diff-VF HNI 对比。两次均使用 cosine-squared WWS、
TES disabled、audio TES disabled，并执行一次全局 VAE decode。由于旧基线报告生成时
尚未保存视频指标，下面的基线视频/音频数值由同一 MP4 后处理脚本重新提取，边界统一
定义为第 362 帧和第 482667 个 32 kHz sample。

| 方法 | 接缝视频 MAD | 音频 energy jump | 音频 spectral jump | sampled motion MAD | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retake-hard | 42.4863 | 2.45e-6 | 0.03410 | 18.0920 | 未记录 |
| Latent handoff | 61.6630 | 2.15e-6 | 0.03010 | 16.6959 | 未记录 |
| Joint MultiDiffusion | 18.8782 | 6.63e-6 | 0.04556 | 28.9327 | 未记录 |
| Diff-VF, HNI independent (w=0) | 13.9882 | 4.51e-8 | 0.02425 | 26.4076 | 68.18 GiB |
| Diff-VF, HNI mixed (w=0.5) | **12.6289** | **8.20e-9** | **0.01561** | 29.2417 | 68.18 GiB |

`hni-mixed` 是目前这组固定 seed/prompt 上的最佳自动边界结果：相对 HNI independent，
视频 MAD 下降约 9.7%，audio energy jump 下降约 82%；相对 Joint MultiDiffusion，
视频 MAD 下降约 33%。这仍然是单 seed、单场景的工程验证，不能替代多 seed 的感知
质量评测，也不能据此断言模型已经学会 continuation。

## Diff-VF 15 秒完整消融矩阵（2026-08-26）

在上述固定配置上补齐了 HNI、WWS 权重和视频 TES/fusion 消融。所有 Diff-VF 输出均为
690 帧、28.75 秒、24 FPS 视频及同一物理时间轴的 32 kHz 音频，并以一次全局 VAE decode
输出。为消除运行期内存张量与 MP4 编码之间的差异，下表的视频/音频边界指标均从最终 MP4
用同一套 OpenCV + PyAV 后处理提取；边界固定为第 362 帧和第 482667 个音频采样点。

| 变体 | HNI / WWS / TES | 接缝视频 MAD | 音频 energy jump | 音频 spectral jump | sampled motion MAD | 长程 MAD | 时间 / 峰值显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HNI independent | `w=0` / cosine-squared / off | 13.9882 | 4.51e-8 | 0.02425 | 27.2224 | 83.6043 | 1435.37 s / 68.18 GiB |
| HNI mixed | `w=0.5` / cosine-squared / off | 12.6289 | **8.20e-9** | **0.01561** | 29.6936 | 67.8736 | 1666.88 s / 68.18 GiB |
| HNI shared | `w=1` / cosine-squared / off | **11.6857** | 1.18e-7 | 0.02006 | 31.4913 | 83.5623 | 1435.65 s / 68.18 GiB |
| Center-distance | `w=0.5` / center-distance / off | 12.8986 | 5.38e-8 | 0.01665 | 29.4417 | 68.7188 | 1434.32 s / 68.18 GiB |
| Video TES + fusion | `w=0.5` / cosine-squared / on | 14.9284 | 6.02e-8 | 0.02047 | **20.8276** | **47.4195** | 1587.80 s / 68.20 GiB |

`video-tes` 使用 `--tes-window-steps 12 --tes-stride 2`，默认高噪声前 17/50 个 step
启用视频稀疏时间窗口；其 prediction-space fusion 的局部系数从 `0.25` 线性升至约
`0.446`，随后 TES 关闭并只采用 WWS。音频 TES 始终为 `disabled`，音频仅走保守的 WWS
时间线，因此该变体的音频指标仍可与其他项直接比较。

本轮结果的含义如下：

- 对接缝自动指标而言，`w=1` 得到最低视频 MAD，但音频能量跳变明显变差；`w=0.5` 是音画
  边界最平衡的选项，仍是当前实验路径的推荐默认。
- `center-distance` 没有在这个 prompt/seed 上超过 cosine-squared 的 `w=0.5` 视频接缝，
  但频谱跳变接近，说明该权重策略应保留为可复现实验项，而非默认切换。
- TES 显著降低了采样长程 MAD 和 motion MAD，却使局部接缝 MAD 增大约 18%。这符合 TES
  在早期为全局结构提供约束、但可能侵蚀局部运动细节的风险；当前参数不能作为接缝优化方案。
- 所有完整 Diff-VF 运行的统一 decode 峰值约 68.2 GiB，低于 H100 80 GiB 的容量但余量有限。
  这些是单卡、两窗口、单 seed 结果；多 seed、不同镜头运动、人工视频/音频盲测和更长 horizon
  验证仍为后续工作。
