# MiniMax-H3 三种接续方法的解码过程

以下都假设窗口 `A` 先生成，窗口 `B` 使用 `A` 尾部的 34 帧作为 overlap；最终成片只保留一份 overlap，`B` 只贡献 overlap 之后的新帧。

## 1. Retake-hard

```text
A 的 latent
  -> VAE 解码
  -> 得到 A 尾部视频/音频
  -> VAE 重新编码为 Retake 条件
  -> B 使用 Retake mask 去噪
  -> VAE 解码 B
  -> 丢弃 B 的 overlap，只拼接 B 的后缀
```

1. 窗口 `A` 完成扩散采样后，先将其 latent 解码为视频帧和音频波形。
2. 从解码结果中截取尾部 overlap，作为 `B` 的 `retake_video` 和 `retake_audio`。
3. H3 的 Retake 编码器再把这段视频和音频编码回 latent，并生成对应的固定区域 mask。
4. `B` 的 overlap 区域在每个去噪步骤中保持为 Retake 条件，只有后缀区域被重新生成。
5. `B` 完成后再次经过 VAE 解码；拼接时删除 `B` 的前 34 帧及对应音频，只追加后缀。

## 2. Latent-handoff

```text
A 的最终 clean latent
  -> 直接截取尾部 latent
  -> 放入 B 的 overlap 前缀
  -> B 使用固定 latent 前缀去噪
  -> VAE 解码 B
  -> 丢弃 B 的 overlap，只拼接 B 的后缀
```

1. 窗口 `A` 完成采样后，直接保留最终的 clean video/audio latent，不先解码。
2. 从 `A` 的 latent 中截取 overlap：34 帧视频对应 10 个视频 temporal latent，音频对应约 57 个 audio latent。
3. 将这些 latent 直接放到 `B` 的 overlap 前缀，并用 hard mask 固定；`B` 只对后缀进行扩散生成。
4. `B` 完成后将其 latent 送入 VAE 解码，得到视频和音频。
5. 拼接时删除 `B` 解码结果中的 overlap，只保留 `B` 的新后缀；与 Retake-hard 相比，历史部分没有经历 `decode -> encode`。

## 3. Taper-refine-v2

```text
A 的最终 clean overlap latent
  -> 按 B 当前 timestep 加噪
  -> 得到 noisy anchor
  -> 在 B 的 overlap 内逐步约束去噪
  -> VAE 解码 B
  -> 丢弃 B 的 overlap，只拼接 B 的后缀
```

1. 窗口 `A` 完成采样后，保留其 clean overlap latent。
2. 在 `B` 的每一个去噪 timestep，根据当前 scheduler 给这段 clean latent 加上同等级噪声，得到 `noisy anchor`。
3. 将 `B` 当前的 noisy latent 向该 anchor 约束：overlap 前部保持较强约束，靠近 overlap 末端时逐渐减弱约束。
4. `B` 完成扩散采样后，将最终 latent 经过 VAE 解码为视频和音频。
5. 拼接时同样删除 `B` 的 overlap，只追加后缀；taper 只影响生成阶段的 latent 轨迹，不改变最终的拼接规则。
