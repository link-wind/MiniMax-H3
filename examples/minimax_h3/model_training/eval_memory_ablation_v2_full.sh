#!/usr/bin/env bash
set -euo pipefail

# One-line entry point for the v2 (step-375) memory-slot ablation, full split.
#
#   bash examples/minimax_h3/model_training/eval_memory_ablation_v2_full.sh
#
# Quick smoke first (20 samples, 2 repeats):
#   SMOKE=1 bash examples/minimax_h3/model_training/eval_memory_ablation_v2_full.sh
#
# Every setting below can still be overridden by exporting it beforehand.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export LORA_CHECKPOINT="${LORA_CHECKPOINT:-${REPO_ROOT}/outputs/continuation_lora/h3_memory_only_stage2_2n16g_v2/step-375.safetensors}"
export LORA_RANK="${LORA_RANK:-32}"
export EVAL_ARMS="${EVAL_ARMS:-both,stm,ltm,none,swap}"
export EVAL_REPEAT="${EVAL_REPEAT:-3}"
export OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/continuation_lora/memory_ablation_v2_full}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-20}"
  export EVAL_REPEAT="${SMOKE_REPEAT:-2}"
  export OUTPUT_PATH="${SMOKE_OUTPUT_PATH:-${REPO_ROOT}/outputs/continuation_lora/memory_ablation_v2_smoke}"
fi

if [[ ! -f "${LORA_CHECKPOINT}" ]]; then
  echo "[error] LoRA checkpoint not found: ${LORA_CHECKPOINT}" >&2
  exit 1
fi

echo "[wrapper] lora=${LORA_CHECKPOINT}"
echo "[wrapper] rank=${LORA_RANK} arms=${EVAL_ARMS} repeats=${EVAL_REPEAT} max_samples=${EVAL_MAX_SAMPLES:-all}"
echo "[wrapper] output=${OUTPUT_PATH}"

exec bash "${REPO_ROOT}/examples/minimax_h3/model_training/run_memory_ablation_eval_2node16gpu.sh"
