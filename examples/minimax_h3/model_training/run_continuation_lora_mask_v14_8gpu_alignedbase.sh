#!/usr/bin/env bash
set -euo pipefail

# MiniMax-H3 mask-v14 continuation LoRA training.
# This runner uses 8-GPU data parallelism and deliberately keeps CP disabled:
# each rank processes a complete 345-frame sample.

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

# Accelerate reads these settings when train.py creates its Accelerator.
export ACCELERATE_USE_DEEPSPEED=true
# This existing config only enables ZeRO-3; the filename is historical and does
# not turn on Context Parallel. CP is explicitly disabled below with cp=1.
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${REPO_ROOT}/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8.json}"
export ACCELERATE_MIXED_PRECISION=bf16
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export ACCELERATE_DEEPSPEED_ZERO3_INIT=true
export ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true
export ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE=none
export ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=cpu

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/caches/h3_masked_av_v14_cache_8gpu}"
# ContinuationLatentDataset appends <split>/manifest.jsonl when this is a
# directory, so pass the cache root (not the train subdirectory).
MANIFEST="${MANIFEST:-${DATA_ROOT}}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/continuation_lora/h3_continuation_lora_mask_v14_dp8_40691}"
# PRESET_LORA_PATH removed: acceleration fused into base transformer_lora600_v4
PRESET_LORA_PATH="${PRESET_LORA_PATH:-}"
# With local-model mode enabled, pass the processor explicitly. Otherwise the
# default origin pattern can resolve to an empty glob ([]) and Transformers
# interprets it as an invalid Hugging Face repo id.
PROCESSOR_PATH="${PROCESSOR_PATH:-${DIFFSYNTH_MODEL_BASE_PATH}/MiniMaxH3/FL2VA/processor}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
# Context parallel size. 1 = pure data parallelism (each rank runs the full
# 345-frame window). Set CP_WORLD_SIZE=2 to split each 345-frame window across
# a pair of GPUs (author's formal config), which roughly halves per-rank
# activation memory if the pure-DP run still OOMs on backward.
CP_WORLD_SIZE="${CP_WORLD_SIZE:-1}"
MAX_ITEMS="${MAX_ITEMS:-40691}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LORA_RANK="${LORA_RANK:-32}"
SEED="${SEED:-42}"

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
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "[error] NUM_PROCESSES must be positive, got ${NUM_PROCESSES}" >&2
  exit 1
fi

TRAIN_SCRIPT="${REPO_ROOT}/examples/minimax_h3/model_training/train.py"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_PATH}/continuation.pt}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/train.log}"

# Long-horizon memory (memory-only objective, see 长时记忆模块训练方案.md §17).
# "memory-only" feeds the cached slots and forbids any context rows, so the
# whole window is generated from memory + text; "off" reproduces the previous
# packed layout bit for bit.
CONTINUATION_MEMORY_MODE="${CONTINUATION_MEMORY_MODE:-off}"
# Slot names the cache must provide.  Doubles as the ablation switch: naming a
# subset drops the other slots, so one cache serves the stm+ltm / stm / ltm arms
# without re-encoding (stm,ltm for the full arm).
CONTINUATION_MEMORY_EXPECT_SLOTS="${CONTINUATION_MEMORY_EXPECT_SLOTS:-}"

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
  --continuation_lambda_audio 0.5
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
  --continuation_memory_mode "${CONTINUATION_MEMORY_MODE}"
)

if [[ -n "${CONTINUATION_MEMORY_EXPECT_SLOTS}" ]]; then
  ARGS+=(--continuation_memory_expect_slots "${CONTINUATION_MEMORY_EXPECT_SLOTS}")
fi

# TensorBoard is optional: CSV logging and loss.png do not depend on it.
# Avoid aborting a long run when the training environment lacks the package.
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

echo "[train] MiniMax-H3 mask-v14 continuation LoRA"
echo "[train] GPUs=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES} CP=${CP_WORLD_SIZE} DP=$((NUM_PROCESSES / CP_WORLD_SIZE))"
echo "[train] manifest=${MANIFEST} max_items=${MAX_ITEMS} frames=345 overlap=39 (12 latent tokens)"
echo "[train] processor=${PROCESSOR_PATH}"
echo "[train] output=${OUTPUT_PATH}"
echo "[train] log=${LOG_PATH}"
if [[ "${CP_WORLD_SIZE}" -gt 1 ]]; then
  if [[ $((NUM_PROCESSES % CP_WORLD_SIZE)) -ne 0 ]]; then
    echo "[error] NUM_PROCESSES must be divisible by CP_WORLD_SIZE (${NUM_PROCESSES} % ${CP_WORLD_SIZE})" >&2
    exit 1
  fi
  echo "[train] CP=${CP_WORLD_SIZE} enabled: each 345-frame window is split across ${CP_WORLD_SIZE} GPUs"
fi
echo "[hint] If backward OOMs again, retry with CP_WORLD_SIZE=2; enable DIFFSYNTH_MEMORY_LOG=1 for per-step GPU memory milestones."

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_PROCESSES}" \
  "${ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "[train] finished; loss CSV: ${OUTPUT_PATH}/loss.csv"
if [[ -f "${OUTPUT_PATH}/loss.csv" ]]; then
  "${PYTHON}" examples/minimax_h3/model_training/plot_training_loss.py \
    --log-dir "${OUTPUT_PATH}" \
    --output "${OUTPUT_PATH}/loss.png" 2>&1 | tee -a "${LOG_PATH}"
else
  echo "[warn] loss.csv was not produced; skip loss plot" | tee -a "${LOG_PATH}"
fi
