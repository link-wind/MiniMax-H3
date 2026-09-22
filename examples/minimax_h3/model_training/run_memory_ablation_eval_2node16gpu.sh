#!/usr/bin/env bash
set -euo pipefail

# Teacher-forced memory-slot ablation on 2 nodes x 8 GPUs (16 GPUs).
#
# Launch this script once on EACH node of the 2-node job (Gemini taskrole1):
# it runs its own 8-rank torchrun and rendezvous via the platform variables.
#
# This does NOT train.  It scores five conditioning arms on the validation
# cache with the trained LoRA loaded, which is the cheapest way to answer
# "does the model actually read the memory slots?" before building any
# autoregressive rollout.
#
#   EVAL_ARMS=both,stm,ltm,none,swap
#   EVAL_REPEAT=3          fixed-seed repeats per sample (t/noise variance)
#   EVAL_MAX_SAMPLES=200   quick screen; leave unset for the whole split
#   EVAL_SKIP_UNSLOTTED=0  set to 1 to score only slotted samples
#
# Run shape mirrors training: CP=4 (DP=4).  CP must stay 4 so the sequence
# split matches what the LoRA was trained on; CP=2 would also change the
# per-rank shape of every collective in the run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"

export PATH="${H3_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
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
# ZeRO-3 keeps the frozen base sharded; without it the 57GB transformer plus
# activations does not fit.  The eval never calls backward, so no optimizer
# state is allocated.
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_MIXED_PRECISION=bf16
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export ACCELERATE_DEEPSPEED_ZERO3_INIT=true
export ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE=none
export ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=none
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${REPO_ROOT}/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8_no_cpu_checkpoint.json}"

# --- run parameters ---------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/memory_only_stage2/cache_dual_slot}"
MANIFEST="${MANIFEST:-${DATA_ROOT}}"
SPLIT="${SPLIT:-validation}"
CP_WORLD_SIZE="${CP_WORLD_SIZE:-4}"
if [[ "${CP_WORLD_SIZE}" -lt 1 || $((TOTAL_PROCESSES % CP_WORLD_SIZE)) -ne 0 ]]; then
  echo "[error] CP_WORLD_SIZE=${CP_WORLD_SIZE} must divide total processes ${TOTAL_PROCESSES}" >&2
  exit 1
fi
DP_WORLD_SIZE=$((TOTAL_PROCESSES / CP_WORLD_SIZE))

OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/continuation_lora/memory_ablation_eval}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${REPO_ROOT}/outputs/continuation_lora/h3_memory_only_stage2_2n16g_v2/step-375.safetensors}"
LORA_RANK="${LORA_RANK:-32}"
PROCESSOR_PATH="${PROCESSOR_PATH:-/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/processor}"
EVAL_ARMS="${EVAL_ARMS:-both,stm,ltm,none,swap}"
EVAL_REPEAT="${EVAL_REPEAT:-3}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
EVAL_SKIP_UNSLOTTED="${EVAL_SKIP_UNSLOTTED:-0}"

if [[ ! -f "${LORA_CHECKPOINT}" ]]; then
  echo "[error] LoRA checkpoint not found: ${LORA_CHECKPOINT}" >&2
  echo "        Set LORA_CHECKPOINT to step-250.safetensors or step-375.safetensors." >&2
  exit 1
fi
if [[ ! -d "${PROCESSOR_PATH}" ]]; then
  echo "[error] H3 processor not found: ${PROCESSOR_PATH}" >&2
  exit 1
fi

EVAL_SCRIPT="${REPO_ROOT}/examples/minimax_h3/model_training/eval_memory_ablation.py"
if [[ "${NODE_RANK}" == "0" ]]; then
  export LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/eval.log}"
else
  LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/eval.node${NODE_RANK}.log}"
fi
mkdir -p "${OUTPUT_PATH}"

ARGS=(
  "${EVAL_SCRIPT}"
  --dataset_base_path "${REPO_ROOT}"
  --dataset_metadata_path /unused
  --num_frames 345
  --task continuation_sft
  --continuation_manifest "${MANIFEST}"
  --continuation_split "${SPLIT}"
  --continuation_max_items 99936
  --bf16
  --initialize_model_on_cpu
  --training_cfg_scale 1.0
  --continuation_conditioning masked-av-v14
  --continuation_overlap_steps 12
  --continuation_hard_core_steps 12
  --continuation_transition_steps 0
  --continuation_first_suffix_steps 5
  --continuation_transition_weight 0.5
  --continuation_first_suffix_weight 3.0
  --continuation_suffix_weight 1.0
  --continuation_lambda_audio 1.0
  --continuation_prefix_present_mode noised
  --continuation_memory_mode memory-only
  --continuation_memory_expect_slots stm,ltm
  --lora_base_model dit
  --trainable_models dit
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2
  --lora_rank "${LORA_RANK}"
  --lora_checkpoint "${LORA_CHECKPOINT}"
  --processor_path "${PROCESSOR_PATH}"
  --model_id_with_origin_paths "MiniMaxH3:FL2VA/text_encoder/model*.safetensors,MiniMaxH3:FL2VA/video_vae/source/model.safetensors,MiniMaxH3:FL2VA/audio_vae/model.safetensors,MiniMaxH3:FL2VA/transformer_lora600_v4/model*.safetensors"
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_PATH}"
  --cp_world_size "${CP_WORLD_SIZE}"
  --eval-arms "${EVAL_ARMS}"
  --eval-repeat "${EVAL_REPEAT}"
)
if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
  ARGS+=(--eval-max-samples "${EVAL_MAX_SAMPLES}")
fi
if [[ "${EVAL_SKIP_UNSLOTTED}" == "1" ]]; then
  ARGS+=(--eval-skip-unslotted)
fi

if [[ "${NODE_RANK}" == "0" ]]; then
  {
    echo "[eval] H3 continuation memory-slot ablation (2-node / 16-GPU)"
    echo "[eval] started $(date -u +%FT%TZ)"
    echo "[eval] cp=${CP_WORLD_SIZE} dp=${DP_WORLD_SIZE} manifest=${MANIFEST} split=${SPLIT}"
    echo "[eval] lora=${LORA_CHECKPOINT} rank=${LORA_RANK}"
    echo "[eval] arms=${EVAL_ARMS} repeats=${EVAL_REPEAT} max_samples=${EVAL_MAX_SAMPLES:-all}"
    echo "[eval] output=${OUTPUT_PATH}"
  } | tee -a "${LOG_PATH}"
fi

rm -f "${REPO_ROOT}/.deepspeed_env"

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"
