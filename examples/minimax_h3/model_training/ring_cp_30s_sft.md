# H3 30-second SFT with Ring Context Parallel

This file records the runnable launch shape for the Ring CP path. The code
provides CPU-safe metadata, logical Ring CP parity, CP-aware training helpers,
H3 training entrypoint wiring, and the currently completed GPU validation
results described below.

## Launch shape

The H3 frame count must satisfy `17n + 5`. At 24 fps, `719` frames is the
closest valid count below 30 seconds.

## Gemini 2-node launch

On Gemini, run one `torchrun` command on each node. If you use
`accelerate launch --num_machines 2`, add `--deepspeed_multinode_launcher standard`; without it Accelerate delegates DeepSpeed to its own
multi-node launcher, which needs SSH/hostfile, appends a stale
`.deepspeed_env` in the repo root, and on Gemini only starts local 8 ranks. `torchrun` plus the Accelerate DeepSpeed
environment variables gives the same ZeRO-3 setup while respecting the platform
`NODE_RANK`/`MASTER_ADDR` variables.

The command below is a 2-step smoke that should be used first. It explicitly
removes any stale generated `.deepspeed_env`, then launches through
`torchrun`. If it passes two steps, remove `--dataset_repeat 1`/`--max_steps 2`
or raise the limits.

The same command is saved as
`examples/minimax_h3/model_training/gemini_30s_sft_smoke.sh`, so each Gemini
node only needs:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_smoke.sh
```

After the smoke run passes, the full 10x-repeat run is:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_full.sh
```

For a 24-GPU, 3-node smoke run (8 GPUs per node, `CP=8`, `DP=3`), run the
same script on all three nodes:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_smoke.sh
```

For the 3-node full run with `--dataset_repeat 3` and no step cap:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_full.sh
```

## Performance notes

The measured CP=8 smoke step on 8 x H100 80GB was about `27s/it`. On 24 GPUs
use `CP=8, DP=3` (`--cp_world_size 8`) rather than a smaller CP size: every
CP=4 rank handles twice the sequence tokens, and CP=2 handles four times, so
the step becomes noticeably slower and the per-rank activation/memory/ring
communication grows.

GPU utilization will not stay flat with this stack because each step contains
DeepSpeed ZeRO-3 parameter gathers, Ring attention P2P rotations, and CPU
checkpointing. If you want to try reducing the CPU-checkpoint synchronization
cost, first run the no-CPU-checkpoint smoke wrapper:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_smoke_no_cpu_checkpoint.sh
```

If those two steps fit in memory, run the full 3-node run with the same config:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_full_no_cpu_checkpoint.sh
```

If it fits, this config can be faster than the CPU-checkpoint default because
activation memory no longer round-trips through the host.

To try 24 GPUs with `CP=4, DP=6` instead, use the CP=4 smoke wrapper first:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_cp4_smoke_no_cpu_checkpoint.sh
```

If it fits, the full `CP=4` run is:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_cp4_full_no_cpu_checkpoint.sh
```

To try `CP=2, DP=12` as a worst-case memory/throughput check, use:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_cp2_smoke_no_cpu_checkpoint.sh
```

and if it fits, the full `CP=2` run is:

```bash
bash /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_cp2_full_no_cpu_checkpoint.sh
```

## 2026-08-19 fix

The two-node run was reaching the real model forward but then failed with
`both arguments to matmul need to be at least 1D, but they are 0D and 2D` inside
`MiniMaxH3AdalnProj`. The launch shape was already correct; the failure was
caused by DeepSpeed CPU checkpointing emptying the shared `t_emb` and
`rope_freqs` tensors in place after the first DiT block. `minimax_h3_dit.py`
now marks those shared tensors with `no_checkpointing=True`, so the remaining
blocks and the final layer still receive their full shapes.

```bash
set -euo pipefail

export REPO_ROOT=/gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio
cd "$REPO_ROOT"

export H3_VENV=/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv
export DIFFSYNTH_MODEL_BASE_PATH=/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PATH="$H3_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export H3_CP_TRACE=1

export MASTER_ADDR="$GEMINI_IP_taskrole1_0"
export MASTER_PORT="${GEMINI_taskrole1_0_http_PORT:-29500}"
export NODE_RANK="$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"
export NNODES="${GEMINI_TASK_ROLE_TASK_COUNT_taskrole1:-$GEMINI_TASKS_NUM}"

rm -f "$REPO_ROOT/.deepspeed_env"

# Use the standalone DeepSpeed config with torchrun.
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="$REPO_ROOT/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8.json"
export ACCELERATE_MIXED_PRECISION=bf16
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export ACCELERATE_DEEPSPEED_ZERO3_INIT=true
export ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true
export ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE=none
export ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=none

torchrun \
  --nnodes "$NNODES" \
  --nproc_per_node=8 \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$REPO_ROOT/examples/minimax_h3/model_training/train.py" \
  --dataset_base_path "$REPO_ROOT/models/train/MiniMax-H3-30s-multi-cache" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 719 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMaxH3:FL2VA/transformer/model*.safetensors" \
  --processor_path "/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/processor" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${GEMINI_DATA_OUT:-$REPO_ROOT/models/train/MiniMax-H3-30s-multi-2node}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --initialize_model_on_cpu \
  --cp_world_size 8 \
  --task "sft:train" \
  --max_steps 2
```

The trace prints `[H3CP][rank N] step X forward start`, `forward done`,
`backward done`, and `optimizer step done`. If it hangs on the second step, the
last trace line tells us whether the stall is in data loading, model forward,
backward, or the optimizer/gradient reduction. With `H3_CP_TRACE=1`, a two-node
job should show all 16 ranks entering step 1 forward and then all 16 ranks
leaving forward/backward. If only node 0 or node 1 ranks appear, the other node
did not join with the same `NODE_RANK`/`MASTER_ADDR`, or it is stuck in NCCL.

## Current CP decisions

- Token refiner: replicated computation. Every CP rank runs the token refiner
  over the full text prompt with the original `refiner_packed_seq_params`;
  only the local text rows are selected for the packed decoder input.
- Pad segment: preserved as a separate attention segment
  (`cu_seqlens=[0, used, seq_len]`). Removing it is left for a later
  optimization after the real CP path is stable.
- Parameter gradients: DeepSpeed ZeRO-3 global group is the default. If a
  DP-only ZeRO/FSDP parameter state is used, pass
  `--cp_gradient_reduction dp-only` so the training loop explicitly all-reduces
  parameter gradients inside each CP group.
- Attention backend: the distributed Ring path uses official `flash_attn`
  blockwise attention when the inputs are BF16/FP16 on CUDA and `flash_attn` is
  installed. Forward calls `_flash_attn_varlen_forward` per segment/KV chunk,
  merges via `softmax_lse`, and backward passes the global `softmax_lse`.
  CPU, FP64, or environments without `flash_attn` fall back to the dense
  reference implementation.

## FlashAttention dependency

The 30-second GPU path expects `flash-attn` built for PyTorch 2.9 and CUDA 12.8.
The workspace uses a prebuilt wheel for Python 3.10:

```bash
uv pip install --python /gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
```

## GPU validation status (2026-08-18)

Completed:

- `tests/context_parallel/test_gpu_parity.py` passes on one H100: tiny H3
  CP=1 vs CP=2 forward, parameter gradients, and loss parity.
- `tests/context_parallel/test_gpu_distributed_ring_attention.py` passes on two
  H100s with NCCL. The ring K/V rotation uses `batch_isend_irecv`; the backward
  path all-reduces contiguous K/V gradients.
- A real 33.1B H3 DiT checkpoint loads successfully into the modified
  `MiniMaxH3DiT` in BF16. A one-layer real-weight parity check shows CP=1 vs
  CP=2 forward and representative parameter gradients within BF16-compatible
  tolerance (`max_abs` relative error roughly `0.4%` to `1.4%`).

### 30-second SFT smoke test (CP=8, 8 x H100 80GB)

The 30-second SFT launch smoke test passed end to end with the real H3 DiT
checkpoint and a single cached 719-frame `480x832` sample plus audio. The
packed sequence length is `85568` (`82680` video rows, `447` text rows, and
`2398` audio rows).

The run used the 8-process DeepSpeed ZeRO-3 configuration:

```bash
accelerate launch --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3_cp8_dp1.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-30s-smoke-cache-one \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio \
  --height 480 --width 832 --num_frames 719 --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMaxH3:FL2VA/transformer/model*.safetensors" \
  --processor_path /gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/processor \
  --learning_rate 1e-5 --num_epochs 1 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/MiniMax-H3-30s-smoke \
  --trainable_models dit --use_gradient_checkpointing --find_unused_parameters \
  --initialize_model_on_cpu --cp_world_size 8 --task sft:train --max_steps 2
```

Measured smoke results:

- One training step completed in `27.03s/it`.
- Peak observed GPU memory was about `73 GiB` per rank on `80 GiB` H100s.
- DeepSpeed reported one PyTorch allocator cache flush during the step, which
  indicates high memory pressure but the step still completed.
- The epoch checkpoint was saved to
  `models/train/MiniMax-H3-30s-smoke/epoch-0.safetensors` (`66.2 GB`).
- The training entrypoint was updated so the CP path explicitly sets
  `train_micro_batch_size_per_gpu=1` before `accelerate.prepare`, because
  Accelerate 1.7 does not forward that field from the YAML-only DeepSpeed
  config when no dataloader is prepared.
- ZeRO-3 runs should keep `--initialize_model_on_cpu`; this lets DeepSpeed
  partition the model before it is moved to GPU and avoids a temporary full
  62GB-per-rank resident state.

Remaining:

- 6.4 memory/communication/throughput measurement at CP=1/2/4/8 is not being
  run. Per-rank sequence activation increases as CP decreases, and the user
  decided not to spend GPU time on the CP scaling comparison after the CP=8
  smoke test passed.
- 6.5 training-curve comparison against the non-CP baseline is also not being
  run in this pass. The current smoke run is a single-sample, single-step
  launch validation, not a multi-step training curve.
