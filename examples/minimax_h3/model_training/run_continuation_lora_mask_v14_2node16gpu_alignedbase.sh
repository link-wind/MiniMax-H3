#!/usr/bin/env bash
set -euo pipefail

# MiniMax-H3 mask-v14 continuation LoRA training on 2 nodes x 8 GPUs (16 GPUs).
#
# Launch this script once on EACH node of the 2-node job (Gemini taskrole1).
# Every node runs its own 8-rank torchrun; rendezvous uses the platform master
# address/port/rank variables below.
#
# Memory strategy:
#   - Measured on 16 ranks (2 nodes): a 345-frame continuation step uses
#     ~23GiB forward but ~71.5GiB peak in backward. CP=2 (DP=8) is only 1-2GiB
#     short of the 80GiB ceiling and OOMs; CP=4 (DP=4, quarter-window) roughly
#     halves that again and is the safe default. DP groups run different samples.
#   - 16 ranks also halve per-rank ZeRO-3 parameter/gradient storage vs 8 ranks.
#   - If a CP=4 run reports plenty of headroom, CP_WORLD_SIZE=2 may be retried.
#
# DeepSpeed config:
#   - Default is the "no cpu checkpoint" config (param offload none, plain
#     activation checkpointing) that carried the successful 30s 719-frame runs.
#   - DS_CPU_OFFLOAD=1 switches to cpu-param-offload + partition/cpu
#     activation-checkpointing. WARNING: that config was measured to explode in
#     backward (~23GiB forward -> ~71GiB backward peak) regardless of CP size
#     and OOMs even on 16 GPUs; keep it off.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"

export PATH="${H3_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
# Prefer the current allocator variable and avoid the deprecated alias leaking
# in from a parent shell.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# --- multi-node rendezvous (Gemini platform) -------------------------------
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PLATFORM_NODES="${GEMINI_TASK_ROLE_TASK_COUNT_taskrole1:-}"
export MASTER_ADDR="${MASTER_ADDR:-${GEMINI_IP_taskrole1_0:?GEMINI_IP_taskrole1_0 must be set on multi-node jobs}}"
export MASTER_PORT="${MASTER_PORT:-${GEMINI_taskrole1_0_http_PORT:-29500}}"
export NODE_RANK="${NODE_RANK:-${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:?GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX must be set on multi-node jobs}}"
if [[ -n "${PLATFORM_NODES}" && "${PLATFORM_NODES}" != "${NNODES}" ]]; then
  echo "[error] this job was allocated ${PLATFORM_NODES} nodes but NNODES=${NNODES}" >&2
  exit 1
fi
TOTAL_PROCESSES=$((NNODES * NPROC_PER_NODE))

# --- accelerate / DeepSpeed -------------------------------------------------
# Accelerate reads these settings when train.py creates its Accelerator.
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_MIXED_PRECISION=bf16
# Defaulted here (not only in the run-parameters block below) because this export comes
# first and the script runs under ``set -u``: an unset GRAD_ACCUM would abort the launch.
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM:-8}"
export ACCELERATE_DEEPSPEED_ZERO3_INIT=true
export ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true
export ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE=none
if [[ "${DS_CPU_OFFLOAD:-0}" == "1" ]]; then
  # Configuration used by the recent 8-GPU continuation attempts: params on
  # CPU, activations partitioned and checkpointed to CPU. Max GPU headroom but
  # the most CPU traffic; requires the ZeRO-3 stub-device fix in runner.py.
  DS_CONFIG="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${REPO_ROOT}/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8.json}"
  export ACCELERATE_DEEPSPEED_CONFIG_FILE="${DS_CONFIG}"
  export ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=cpu
else
  # Config that carried the successful 719-frame 30s runs (24 GPUs / CP=2):
  # ZeRO-3 without param offload and without DeepSpeed cpu checkpointing.
  DS_CONFIG="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${REPO_ROOT}/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8_no_cpu_checkpoint.json}"
  export ACCELERATE_DEEPSPEED_CONFIG_FILE="${DS_CONFIG}"
  export ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=none
fi

# --- run parameters ---------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/caches/h3_masked_av_v14_mix100k_16gpu}"
# ContinuationLatentDataset appends <split>/manifest.jsonl when this is a
# directory, so pass the cache root (not the train subdirectory).
MANIFEST="${MANIFEST:-${DATA_ROOT}}"
CP_WORLD_SIZE="${CP_WORLD_SIZE:-4}"
if [[ "${CP_WORLD_SIZE}" -lt 1 || $((TOTAL_PROCESSES % CP_WORLD_SIZE)) -ne 0 ]]; then
  echo "[error] CP_WORLD_SIZE=${CP_WORLD_SIZE} must divide total processes ${TOTAL_PROCESSES}" >&2
  exit 1
fi
DP_WORLD_SIZE=$((TOTAL_PROCESSES / CP_WORLD_SIZE))
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n${NNODES}g_cp${CP_WORLD_SIZE}_dp${DP_WORLD_SIZE}}"
# PRESET_LORA_PATH removed: acceleration fused into base transformer_lora600_v4
PRESET_LORA_PATH="${PRESET_LORA_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${DIFFSYNTH_MODEL_BASE_PATH}/MiniMaxH3/FL2VA/processor}"
MAX_ITEMS="${MAX_ITEMS:-99936}"
# Effective batch per optimizer step = DP_WORLD_SIZE (4) x GRAD_ACCUM.
# v3 ran GRAD_ACCUM=1 (4 samples/step, loss noisy); 8 is the new default.
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LORA_RANK="${LORA_RANK:-32}"
SEED="${SEED:-42}"
LAMBDA_AUDIO="${LAMBDA_AUDIO:-1.0}"
# Prefix input form for masked-av-v14 training: noised-at-t (v3/v5,
# default), clean (inference-form clean prefix + timestep=1), or mixed
# (50/50 per micro-batch).  mixed is the clean-prefix experiment.
PREFIX_MODE="${PREFIX_MODE:-noised}"
# Long-horizon memory (M1-fast, 2026-09-20).  "on" feeds the clean latent block
# that the cache carries under ``memory_latents`` (built with
# ``build_continuation_cache.py --memory-frames 39``); "off" reproduces the
# previous packed layout bit for bit and is the A/B control.
CONTINUATION_MEMORY_MODE="${CONTINUATION_MEMORY_MODE:-off}"
# Slot names the cache must provide.  Set to "stm,ltm" for the memory-only
# dual-slot objective; empty disables the check (the legacy single-slot path).
CONTINUATION_MEMORY_EXPECT_SLOTS="${CONTINUATION_MEMORY_EXPECT_SLOTS:-}"

mkdir -p "${OUTPUT_PATH}"
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] Python environment not found: ${PYTHON}" >&2
  exit 1
fi
if [[ -d "${MANIFEST}" ]]; then
  MANIFEST_CHECK="${MANIFEST}/train/manifest.jsonl"
else
  MANIFEST_CHECK="${MANIFEST}"
fi
if [[ ! -f "${MANIFEST_CHECK}" ]]; then
  echo "[error] continuation manifest not found: ${MANIFEST_CHECK}" >&2
  exit 1
fi
if [[ ! -f "${PROCESSOR_PATH}/preprocessor_config.json" ]]; then
  echo "[error] H3 processor not found: ${PROCESSOR_PATH}" >&2
  echo "        Set PROCESSOR_PATH to the directory containing preprocessor_config.json." >&2
  exit 1
fi

TRAIN_SCRIPT="${REPO_ROOT}/examples/minimax_h3/model_training/train.py"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_PATH}/continuation.pt}"
if [[ "${NODE_RANK}" == "0" ]]; then
  export LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/train.log}"
else
  LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/train.node${NODE_RANK}.log}"
fi

ARGS=(
  "${TRAIN_SCRIPT}"
  --dataset_base_path "${REPO_ROOT}"
  --dataset_metadata_path /unused
  --num_frames 345
  --task continuation_sft
  --continuation_manifest "${MANIFEST}"
  --continuation_split train
  --continuation_max_items "${MAX_ITEMS}"
  --num_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --save_steps "${SAVE_STEPS}"
  --seed "${SEED}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --bf16
  --initialize_model_on_cpu
  --use_gradient_checkpointing
  --training_cfg_scale 1.0
  --continuation_conditioning masked-av-v14
  --continuation_overlap_steps 12
  --continuation_hard_core_steps 12
  --continuation_transition_steps 0
  --continuation_first_suffix_steps 5
  --continuation_transition_weight 0.5
  --continuation_first_suffix_weight 3.0
  --continuation_suffix_weight 1.0
  --continuation_lambda_audio "${LAMBDA_AUDIO}"
  --continuation_prefix_present_mode "${PREFIX_MODE}"
  --continuation_memory_mode "${CONTINUATION_MEMORY_MODE}"
  --lora_base_model dit
  --trainable_models dit
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2
  --lora_rank "${LORA_RANK}"
  --processor_path "${PROCESSOR_PATH}"
  --model_id_with_origin_paths "MiniMaxH3:FL2VA/text_encoder/model*.safetensors,MiniMaxH3:FL2VA/video_vae/source/model.safetensors,MiniMaxH3:FL2VA/audio_vae/model.safetensors,MiniMaxH3:FL2VA/transformer_lora600_v4/model*.safetensors"
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_PATH}"
  --enable_csv_log
  --continuation_checkpoint_save_path "${CHECKPOINT_PATH}"
  --continuation_checkpoint_interval "${SAVE_STEPS}"
  --cp_world_size "${CP_WORLD_SIZE}"
)

if [[ -n "${CONTINUATION_MEMORY_EXPECT_SLOTS}" ]]; then
  ARGS+=(--continuation_memory_expect_slots "${CONTINUATION_MEMORY_EXPECT_SLOTS}")
fi

# TensorBoard is optional: CSV logging and loss.png do not depend on it.
ENABLE_TENSORBOARD_LOG="${ENABLE_TENSORBOARD_LOG:-auto}"
if [[ "${ENABLE_TENSORBOARD_LOG}" == "1" || "${ENABLE_TENSORBOARD_LOG,,}" == "true" ]]; then
  ARGS+=(--enable_tensorboard_log)
elif [[ "${ENABLE_TENSORBOARD_LOG,,}" == "auto" ]]; then
  if "${PYTHON}" -c 'import tensorboard' >/dev/null 2>&1; then
    ARGS+=(--enable_tensorboard_log)
  else
    echo "[warn] tensorboard is not installed; continue with CSV/loss.png logging"
  fi
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME_PATH:-}" ]]; then
  ARGS+=(--continuation_resume_path "${RESUME_PATH}")
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  printf "%s\n" \
    "[train] MiniMax-H3 mask-v14 continuation LoRA (2-node / 16-GPU)" \
    "[train] started $(date -u +%FT%TZ)" >> "${LOG_PATH}"
  echo "[train] nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} total=${TOTAL_PROCESSES}"
  echo "[train] CP=${CP_WORLD_SIZE} DP=${DP_WORLD_SIZE} GAS=${GRAD_ACCUM} (effective batch per optimizer step = $((DP_WORLD_SIZE * GRAD_ACCUM)) samples)"
  echo "[train] master=${MASTER_ADDR}:${MASTER_PORT} node_rank=${NODE_RANK}"
  echo "[train] manifest=${MANIFEST} max_items=${MAX_ITEMS} frames=345 overlap=39 (12 latent tokens)"
  echo "[train] deepspeed config=${DS_CONFIG}"
  printf "[train] deepspeed config=%s\n" "${DS_CONFIG}" >> "${LOG_PATH}"
  echo "[train] offload_param=${ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE}"
  echo "[train] processor=${PROCESSOR_PATH}"
  echo "[train] output=${OUTPUT_PATH}"
  echo "[train] log=${LOG_PATH}"
  echo "[hint] smoke: OUTPUT_PATH=... MAX_STEPS=2 MAX_ITEMS=8 GRAD_ACCUM=1 DIFFSYNTH_MEMORY_LOG=1 bash $0"
  echo "[hint] Memory measured: CP=2 peaks ~71.5GiB/80GiB (OOM by ~2GiB); default CP=4 is safe. DS_CPU_OFFLOAD makes it worse; keep it unset."
fi

# Each node runs an identical 8-rank torchrun; the platform runs this script on
# every node of taskrole1. Clean any stale .deepspeed_env before launching.
rm -f "${REPO_ROOT}/.deepspeed_env"

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

# Post-processing is done once from node 0 only.
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "[train] finished; loss CSV: ${OUTPUT_PATH}/loss.csv"
  if [[ -f "${OUTPUT_PATH}/loss.csv" ]]; then
    "${PYTHON}" examples/minimax_h3/model_training/plot_training_loss.py \
      --log-dir "${OUTPUT_PATH}" \
      --output "${OUTPUT_PATH}/loss.png" 2>&1 | tee -a "${LOG_PATH}"
  else
    echo "[warn] loss.csv was not produced; skip loss plot" | tee -a "${LOG_PATH}"
  fi
fi
