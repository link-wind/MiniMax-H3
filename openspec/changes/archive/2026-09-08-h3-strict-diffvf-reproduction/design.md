## Context

`joint_multidiffusion()` 已有全局 latent 时间线、统一解码和绝对 WWS source positions。它目前为每个窗口预测 velocity，再合并 prediction 并对全局 latent 调度一步；HNI 使用全局 base/innovation 噪声，TES 按通用稀疏窗口拆分。这些选择方便 H3 实验，但不等同于 Diff-VF 的 HNI、WWS、TES 与路径融合公式。

H3 的 DiT 使用打包序列和 3D RoPE。局部窗口与交错 TES 序列都必须保留其真实的全局视频/音频 source indices；否则后段 token 会被错误地编码为局部零时间。论文只定义视频 latent 采样，而 H3 同时生成音频，且二者的 latent rate 不同。

## Goals / Non-Goals

**Goals:**

- 在显式 `paper-strict` 语义下实现论文定义的 HNI、线性中心加权 WWS、早期 TES 和状态级融合。
- 确保局部路径与全局路径均从同一时刻 `Z_t` 出发，并让每个调用使用绝对 H3 时间坐标。
- 以 CPU 张量测试锁定索引、权重、系数和状态融合的数值语义，同时输出完整实验 manifest。
- 保持一次全时间线 VAE decode，排除分窗 VAE 边界作为变量。

**Non-Goals:**

- 不训练或修改 H3/音频 VAE 权重，不承诺严格模式一定提升视觉质量。
- 不在本变更中支持非零位置的 `ref_blocks`，不把 Retake、latent handoff 或二次 bridge 接入严格模式。
- 不将论文的 1000 DDIM timestep 编号逐字迁移到 H3 50-step 采样；采用明确的早期步数比例映射。
- 不将论文的视频 TES 伪装为音频 TES；音频默认走连续 WWS 状态路径。

## Decisions

### 使用显式采样语义开关

`DiffVFConfig` 新增 `sampling_semantics`，值为 `existing`（默认）或 `paper-strict`。严格逻辑仅在后者运行，并要求中心距离权重。这样可以复现论文，同时保留已经验证的预测级路径和既有结果作为对照；直接替换默认行为会让历史实验无法比较。

### 以短片段噪声定义 HNI

严格 HNI 先以独立 RNG 得到第一 clip 噪声 `Z_T^0`。第 n 个 clip 以 `sqrt(1-w) * Z_T^0 + sqrt(w) * epsilon_n` 初始化，再根据论文的 `(b+n) mod N + a*N` 映射在短片段组内循环重排。结果 scatter 到全局时间线；重叠 token 采用最早窗口 owner，保证每个全局 token 只有一个确定的初始值。`innovation_weight=w` 是新字段，避免旧 `hni_weight` 的系数语义含糊。

第一 clip 保持 `Z_T^0` 本身，创新噪声只用于后续 clip。若局部时间长度不是 clip 数 N 的整数倍，最后一个不足 N 的组按其实际长度循环，避免生成越界 source index；这是 H3 长度适配，不改变完整组的论文映射。

备选方案是复用当前全局混合噪声；它保留 overlap 相同值，却无法表达论文“同一首段噪声在 clips 间共享”的相关性，因此不用于严格模式。

### WWS 融合更新后的状态

每一步对每个 WWS 窗口从 `Z_t` 取 view、运行模型并各自调用调度器得到局部 `Z_local_window^(t-1)`；随后按论文的窗口中心距离 `m(i,j)=(U+1)/2-|C_j-i|` 归一化并写回全局 `Z_local^(t-1)`。所有窗口计算完成后才替换全局状态，杜绝顺序污染。

中心距离权重按完整窗口中心直接计算，而不是使用当前 overlap ramp 近似。当边缘/重叠布局的所有 token 都覆盖时，融合必须归一化为 1。备选的 prediction-level merge 继续保留在 `existing` 模式，因其与旧实验数值兼容。

### TES 是同一步的独立全局状态路径

严格 TES 以 interleave count `N` 构造 N 条序列：第 n 条为 `n, n+N, ...`，必要时使用确定的末尾 padding 和 validity mask。每条从同一份 `Z_t` 取 token、独立去噪一步后 scatter 回原全局索引，形成 `Z_global^(t-1)`，而不是生成关键帧或融合预测量。

H3 需要符合模型窗口的固定局部长度。严格视频 TES 的 `N` 和 window length 由显式配置验证，默认由总长度和本地窗口长度推导。视频 TES 调用使用该序列中每个 token 的绝对 video indices；音频默认不做 TES，而保留同一轮的连续 WWS 更新，以免用不等长度的声画稀疏序列引入未验证的时序关系。

### 按论文调度融合两条状态路径

当 TES 活跃时，最终状态为 `(1-c(t))*Z_local + c(t)*Z_global`，其中 `c(t)=alpha*[0.5*(1+cos(pi*(T-t)/T))]^c_s`。内部以 `step_index=0` 表示最大噪声的第一个推理步，因此初期 `c(t)` 高、后期归零。H3 采用 `tes_early_fraction` 映射论文的 `t<900`：默认前 90% 推理步运行 TES，manifest 记录实际活跃步。

### H3 位置与解码保持既有正确约束

所有普通 WWS 和 TES forward 均以全局 video/audio source indices 重建 `packed`。`ref_blocks` 在非零 source offset 下明确拒绝。video/audio timeline 完成后一次性 decode；严格模式不引入新的逐窗 decode 或后处理混合。

## Risks / Trade-offs

- [H3 调度器的局部 step 与全局 batch shape 不同] → 通过无随机 flow step 的 CPU/短 GPU smoke 对照验证，并从同一 `Z_t` clone 读取所有路径。
- [TES 序列长度不能整除总时间线] → 使用 validity mask，padding token 不参与模型输出 scatter 或融合；测试覆盖双射与尾部情况。
- [论文视频 TES 与 H3 音频联合模型的条件长度不同] → 首版将音频保持连续 WWS 路径，manifest 标记该 H3 适配；后续单独验证同步音频 TES。
- [严格 HNI 的 overlap owner 可能减弱相邻窗口局部相关性] → owner 规则、种子和每段噪声全部写入 manifest，并以 `w=0/1` 与循环映射测试锁定行为。
- [严格采样显著增加单步 forward/step 次数] → 仅 opt-in，先使用少步 smoke，再执行固定基线比较；不在未量化前宣称质量提升。
