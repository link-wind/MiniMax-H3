#!/usr/bin/env bash
set -euo pipefail

# Generative-rollout memory-slot ablation, one arm per GPU.
#
# Run this once on each node with a disjoint ROLLOUT_ARMS list, e.g.
#
#   node 0:  ROLLOUT_ARMS=both,stm,ltm        bash run_memory_rollout_eval_16gpu.sh
#   node 1:  ROLLOUT_ARMS=none,swap,base      bash run_memory_rollout_eval_16gpu.sh
#
# Arms are independent single-process pipelines, so there is no rendezvous and no
# collective anywhere in this job: each arm owns one GPU end to end.  Splitting
# the arm list across nodes is therefore free, and a node runs its arms
# concurrently in the background.
#
# This is the generative counterpart of run_memory_ablation_eval_2node16gpu.sh.
# That one is teacher-forced and shares one DeepSpeed job across 16 ranks; this
# one rolls out and must be per-arm, because the swap arm has to serve donor
# content from a different sequence.
#
#   ROLLOUT_ARMS=both,stm,ltm,none,swap,base
#   PROMPT_MODE=no-subject     strip <SUBJECT> so only the LTM slot carries identity
#   DONOR_CACHE=...            validation cache supplying swap content (needed for swap)
#   SEGMENT_PLAN=...           default is the 5-window / ~65 s trainstyle shot plan
#   NUM_INFERENCE_STEPS=8      matches the distilled base the LoRA was trained on
#
# Run the smoke plan first.  It is two 141-frame windows at the same resolution,
# so it exercises the whole path (slots, donor shapes, decode) in minutes and
# writes peak memory per window to progress_<arm>.jsonl.  Only scale up to the
# 345-frame plan once that peak says the full window fits.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"

export PATH="${H3_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

H3_MODEL_ROOT="${H3_MODEL_ROOT:-/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI}"
H3_BASE="${H3_BASE:-${H3_MODEL_ROOT}/MiniMaxH3/FL2VA}"
TRANSFORMER_DIR="${TRANSFORMER_DIR:-${H3_BASE}/transformer_lora600_v4}"
# A sharded transformer is one model.  The pattern is deliberately left
# unexpanded here and expanded where it is passed on the command line, because
# the Python loader has to receive the *files* -- passing the literal
# ``model*.safetensors`` makes it try to open that string as a path.
CHECKPOINT="${CHECKPOINT:-${TRANSFORMER_DIR}/model*.safetensors}"
if ! compgen -G "${CHECKPOINT}" > /dev/null; then
  echo "[error] no H3 transformer checkpoint matched: ${CHECKPOINT}" >&2
  exit 1
fi
CHECKPOINT_COUNT=$(compgen -G "${CHECKPOINT}" | wc -l)

LORA_CHECKPOINT="${LORA_CHECKPOINT:-${REPO_ROOT}/outputs/continuation_lora/h3_memory_only_stage2_2n16g_v2/step-375.safetensors}"
if [[ ! -f "${LORA_CHECKPOINT}" ]]; then
  echo "[error] LoRA checkpoint not found: ${LORA_CHECKPOINT}" >&2
  echo "        Set LORA_CHECKPOINT to step-250.safetensors or step-375.safetensors." >&2
  exit 1
fi

SEGMENT_PLAN="${SEGMENT_PLAN:-${REPO_ROOT}/examples/minimax_h3/model_inference/h3_continuation_plan_60s_trainstyle_shots.json}"
DONOR_CACHE="${DONOR_CACHE:-${REPO_ROOT}/outputs/memory_only_stage2/cache_dual_slot}"
DONOR_SPLIT="${DONOR_SPLIT:-validation}"
DEFAULT_OUTPUT="${REPO_ROOT}/outputs/continuation_lora/memory_rollout_ablation"
if [[ "${PROMPT_MODE:-full}" == "no-subject" ]]; then
  DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_nosubject"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT}}"

ROLLOUT_ARMS="${ROLLOUT_ARMS:-all}"
if [[ "${ROLLOUT_ARMS}" == "all" ]]; then
  ROLLOUT_ARMS="both,stm,ltm,none,swap,base"
fi
PROMPT_MODE="${PROMPT_MODE:-full}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
# 8, not 50: the training base is ``transformer_lora600_v4``, the distilled
# acceleration DiT whose reviewed references are generated at 8 steps (see
# model_inference/run_single_shot_batch.sh).  The continuation LoRA was trained
# on top of that base, so 50 steps would sample it off its training distribution
# and cost six times the wall clock.
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
OVERLAP_FRAMES="${OVERLAP_FRAMES:-39}"
STM_FRAMES="${STM_FRAMES:-39}"
LTM_FRAMES="${LTM_FRAMES:-39}"
SEED="${SEED:-42}"
# Off by default: the metrics decide the ablation, and a rollout is a few hundred
# frames per arm.  Turn it on for the run whose output someone will actually watch.
SAVE_VIDEO="${SAVE_VIDEO:-0}"
NODE_TAG="${NODE_TAG:-${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-0}}"

SCRIPT="${REPO_ROOT}/examples/minimax_h3/model_training/eval_memory_rollout.py"
mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/rollout.node${NODE_TAG}.log}"

{
  echo "[rollout] generative memory-slot ablation"
  echo "[rollout] started $(date -u +%FT%TZ)"
  echo "[rollout] arms=${ROLLOUT_ARMS} prompt_mode=${PROMPT_MODE} overlap=${OVERLAP_FRAMES}"
  echo "[rollout] stm=${STM_FRAMES} ltm=${LTM_FRAMES} resolution=${HEIGHT}x${WIDTH} steps=${NUM_INFERENCE_STEPS} save_video=${SAVE_VIDEO}"
  echo "[rollout] plan=${SEGMENT_PLAN}"
  echo "[rollout] checkpoint=${CHECKPOINT} (${CHECKPOINT_COUNT} shard(s))"
  echo "[rollout] lora=${LORA_CHECKPOINT}"
  echo "[rollout] donor_cache=${DONOR_CACHE}"
  echo "[rollout] output=${OUTPUT_DIR}"
} | tee -a "${LOG_PATH}"

pids=()
gpu_index=0
IFS=',' read -r -a arms <<< "${ROLLOUT_ARMS}"
for arm in "${arms[@]}"; do
  arm="${arm// /}"
  [[ -z "${arm}" ]] && continue
  if [[ "${arm}" == "swap" && ! -d "${DONOR_CACHE}" ]]; then
    echo "[error] donor cache not found for the swap arm: ${DONOR_CACHE}" >&2
    exit 1
  fi
  arm_output="${OUTPUT_DIR}/${arm}"
  mkdir -p "${arm_output}"
  echo "[rollout] gpu ${gpu_index} -> arm ${arm}" | tee -a "${LOG_PATH}"
  extra_args=()
  if [[ "${SAVE_VIDEO}" == "1" ]]; then
    extra_args+=(--save-video)
  fi
  CUDA_VISIBLE_DEVICES="${gpu_index}" "${PYTHON}" "${SCRIPT}" \
    --segment-plan "${SEGMENT_PLAN}" \
    --checkpoint ${CHECKPOINT} \
    --h3-base "${H3_BASE}" \
    --lora "${LORA_CHECKPOINT}" \
    --donor-cache "${DONOR_CACHE}" \
    --donor-split "${DONOR_SPLIT}" \
    --arms "${arm}" \
    --prompt-mode "${PROMPT_MODE}" \
    --stm-frames "${STM_FRAMES}" \
    --ltm-frames "${LTM_FRAMES}" \
    --overlap-frames "${OVERLAP_FRAMES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --seed "${SEED}" \
    --output-dir "${arm_output}" \
    ${extra_args[@]+"${extra_args[@]}"} \
    >> "${arm_output}/run.log" 2>&1 &
  pids+=($!)
  gpu_index=$((gpu_index + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
    echo "[rollout] a worker failed (pid ${pid}); see ${OUTPUT_DIR}/*/run.log" | tee -a "${LOG_PATH}"
  fi
done

echo "[rollout] finished $(date -u +%FT%TZ) status=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
