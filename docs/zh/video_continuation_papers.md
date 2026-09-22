# 接续视频生成文献整理

> 整理日期：2026-08-24
>
> 范围：2023-2026 年 arXiv 论文，编号已通过 arXiv API 核对
> 口径：将“接续视频生成”拆成前序片段续写/长视频延展、中间补帧/任意时空补全、故事/多镜头续写、视频外扩四类

## 1. 总览

| 方向 | 篇数 | 代表性方法 | 目标 |
| --- | ---: | --- | --- |
| 前序片段续写 / 长视频延展 | 11 | StreamingT2V、TokensGen、LongCat-Video、FlowC2S | 从已有视频继续生成，或把短视频扩展到长视频 |
| 中间补帧 / 任意时空补全 | 3 | TI2V-Zero、MAVIN、VideoCanvas | 在视频中间或任意时空位置补全内容 |
| 故事 / 多镜头续写 | 3 | MovieDreamer、VideoGen-of-Thought、DreamRunner | 保持角色、风格和叙事，连续生成多镜头内容 |
| 视频外扩 | 2 | Be-Your-Outpainter、OutDreamer | 在原始画面四周扩展视频内容 |

## 2. 综述

| 论文 | 年份 | arXiv | 说明 |
| --- | --- | --- | --- |
| A Survey on Long Video Generation: Challenges, Methods, and Prospects | 2024 | [2403.16407](https://arxiv.org/abs/2403.16407) | 长视频生成的问题定义、方法分类和发展展望，适合先看整体地图。 |
| Video Is Worth a Thousand Images: Exploring the Latest Trends in Long Video Generation | 2024 | [2412.18688](https://arxiv.org/abs/2412.18688) | 长视频生成最新趋势，重点覆盖一致性、长度扩展和评测。 |

## 3. 前序片段续写 / 长视频延展

| 论文 | 年份 | arXiv | 方法类型 | 核心思路 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling | 2023 | [2310.15169](https://arxiv.org/abs/2310.15169) | 免训练 | 通过噪声重排让短视频扩散模型生成更长视频 | 不需要重新训练的快速长度扩展 |
| StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text | 2024 | [2403.14773](https://arxiv.org/abs/2403.14773) | 训练式 / 自回归 | 分块自回归生成，结合短期记忆和长期外观记忆 | 从文本生成 80 到 1200+ 帧的长视频 |
| CoNo: Consistency Noise Injection for Tuning-free Long Video Diffusion | 2024 | [2406.05082](https://arxiv.org/abs/2406.05082) | 免训练 | look-back 机制和一致性噪声注入，改善片段间场景过渡 | 多 prompt 连续生成长视频 |
| FreeLong: Training-Free Long Video Generation with SpectralBlend Temporal Attention | 2024 | [2407.19918](https://arxiv.org/abs/2407.19918) | 免训练 | 用频域混合注意力平衡全局一致性和局部细节 | 把 16 帧短视频模型扩展到 128 帧或更长 |
| Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion | 2024 | [2407.01392](https://arxiv.org/abs/2407.01392) | 训练式 / 自回归 | 把下一 token 预测和全序列扩散结合 | 长时序自回归生成和视频续写 |
| HumanDiT: Pose-Guided Diffusion Transformer for Long-form Human Motion Video Generation | 2025 | [2502.04847](https://arxiv.org/abs/2502.04847) | 训练式 / 姿态引导 | 姿态引导的扩散 Transformer | 长人体动作视频生成 |
| TokensGen: Harnessing Condensed Tokens for Long Video Generation | 2025 | [2507.15728](https://arxiv.org/abs/2507.15728) | 训练式 / 两级框架 | 用压缩视频 token 做全局一致性，再通过 FIFO-Diffusion 平滑连接片段 | 跨片段保持内容一致的长视频 |
| LongCat-Video Technical Report | 2025 | [2510.22200](https://arxiv.org/abs/2510.22200) | 训练式 / 统一模型 | 13.6B DiT，支持 T2V、I2V、Video-Continuation | 分钟级长视频和工程化视频续接 |
| FlowC2S: Flowing from Current to Succeeding Frames for Fast and Memory-Efficient Video Continuation | 2026 | [2604.17625](https://arxiv.org/abs/2604.17625) | 训练式 / flow 模型 | 从当前片段直接流向下一片段，减少输入维度和采样步数 | 快速、省显存的视频续接 |
| Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation（FLEX） | 2026 | [2602.14027](https://arxiv.org/abs/2602.14027) | 免训练 | 频域 RoPE 调制、Antiphase Noise Sampling、Inference-only Attention Sink | 自回归视频模型在推理期做 6x 到 12x 长度外推 |
| PhyWorld: Physics-Faithful World Model for Video Generation | 2026 | [2605.19242](https://arxiv.org/abs/2605.19242) | 世界模型 | 物理一致的 world model | 需要物理稳定性的长续接和预测 |

## 4. 中间补帧 / 任意时空补全

| 论文 | 年份 | arXiv | 方法类型 | 核心思路 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| TI2V-Zero: Zero-Shot Image Conditioning for Text-to-Video Diffusion Models | 2024 | [2404.16306](https://arxiv.org/abs/2404.16306) | 免训练 | repeat-and-slide 加 DDPM inversion，从给定图像逐帧生成 | 首帧或中间帧续写、视频补全、长视频生成 |
| MAVIN: Multi-Action Video Generation with Diffusion Models via Transition Video Infilling | 2024 | [2405.18003](https://arxiv.org/abs/2405.18003) | 训练式 | 生成两个片段之间的 transition video | 多动作片段拼接和自然过渡 |
| VideoCanvas: Unified Video Completion from Arbitrary Spatiotemporal Patches via In-Context Conditioning | 2025 | [2510.08555](https://arxiv.org/abs/2510.08555) | 统一补全框架 | 用 In-Context Conditioning 和 Temporal RoPE 对齐任意时空条件 | 任意位置、任意时刻的视频补全 |

## 5. 故事 / 多镜头续写

| 论文 | 年份 | arXiv | 方法类型 | 核心思路 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequence | 2024 | [2407.16655](https://arxiv.org/abs/2407.16655) | 训练式 / 层级框架 | 自回归剧情规划生成视觉 token，扩散模型负责渲染 | 电影式长叙事和跨场景角色一致性 |
| VideoGen-of-Thought: Step-by-step generating multi-shot video with minimal manual intervention | 2024 | [2412.02259](https://arxiv.org/abs/2412.02259) | 免训练 | 动态剧本规划、身份保持 token、相邻镜头 latent transition | 从一句话自动生成多镜头故事 |
| DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation | 2024 | [2411.16657](https://arxiv.org/abs/2411.16657) | 训练式 / 检索增强 | 检索运动先验，配合时空区域 3D attention 控制 | 复杂多角色、多镜头故事生成 |

## 6. 视频外扩

| 论文 | 年份 | arXiv | 方法类型 | 核心思路 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| Be-Your-Outpainter: Mastering Video Outpainting through Input-Specific Adaptation | 2024 | [2403.13745](https://arxiv.org/abs/2403.13745) | 输入特定适配 | 在单视频上做伪外扩学习，再模式感知外扩 | 视频四周扩展画面并保持时空一致 |
| OutDreamer: Video Outpainting with a Diffusion Transformer | 2025 | [2506.22298](https://arxiv.org/abs/2506.22298) | 训练式 / DiT | 条件外扩分支、mask 自注意力、latent alignment loss | 长视频外扩和跨片段一致外扩 |

## 7. 快速筛选

| 需求 | 优先阅读 |
| --- | --- |
| 先建立整体认知 | 2403.16407、2412.18688 |
| 免训练续接 | FreeNoise、CoNo、FreeLong、TI2V-Zero、VideoGen-of-Thought、FLEX |
| 训练式续接 | StreamingT2V、TokensGen、LongCat-Video、FlowC2S、HumanDiT、MovieDreamer、DreamRunner |
| 中间补全 | MAVIN、VideoCanvas |
| 视频外扩 | Be-Your-Outpainter、OutDreamer |
| 追求长视频工程落地 | LongCat-Video、StreamingT2V、TokensGen、FlowC2S |

> 说明：论文页标注的代码、项目页或权重未逐项验证，使用时以对应仓库当前状态为准。
