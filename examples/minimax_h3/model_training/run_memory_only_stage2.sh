#!/usr/bin/env bash
set -euo pipefail

# Stage-2 "memory-only" continuation training (no overlap, no context rows).
#
# Objective:  p(target shot | memory slots, text)
#
# Every sample is exactly one shot.  The whole window is the prediction target,
# and the only cross-shot conditioning is a pair of clean memory slots:
#
#   stm  39 frames immediately before the window   -> contiguous, lead = None
#   ltm  39 opening frames of the source clip      -> constant lead = 36 steps
#
# Why no overlap: with a latent prefix in the sequence, appearance continuity is
# already carried by the context rows, so any gain from the memory block is not
# attributable to it.  Removing the prefix makes the slots the only appearance
# carrier, which is the setting the memory module is actually being tested in.
#
# The LTM slot is anchored at a *constant* distance instead of its true age: its
# content is arbitrarily old, so a real distance would walk it out of the trained
# position range as the rollout grows.  How stale it is is carried by the recency
# embedding instead.
#
# This script only prepares data and then delegates to the 2-node launcher, so
# the memory schedule and the non-memory schedule cannot drift apart.
#
# Usage (once per node):
#   bash examples/minimax_h3/model_training/run_memory_only_stage2.sh
#   INDEX_ONLY=1 bash .../run_memory_only_stage2.sh    # data prep only

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SOURCE_JSONL="${SOURCE_JSONL:-/gemini/platform/public/aigc/human_guozz2/data/LongVideoGen/20260729/data_with_face_and_speech_and_caption.jsonl}"
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/outputs/memory_only_stage2}"
INDEX_JSONL="${INDEX_JSONL:-${WORK_ROOT}/index.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}/caches_dual_slot}"
MAX_INDEX_SAMPLES="${MAX_INDEX_SAMPLES:-}"
# Memory slot geometry.  Both slots use the M1-fast 39-frame unit (12 latent
# steps); the LTM anchor is 3x the STM span so the two slots stay positionally
# distinguishable instead of reading as one contiguous block.
MEMORY_FRAMES="${MEMORY_FRAMES:-39}"
LTM_FRAMES="${LTM_FRAMES:-39}"
LTM_LEAD_STEPS="${LTM_LEAD_STEPS:-36}"

mkdir -p "${WORK_ROOT}"

if [[ ! -f "${INDEX_JSONL}" ]]; then
  echo "[data] building memory-only index -> ${INDEX_JSONL}"
  INDEX_ARGS=(
    "${REPO_ROOT}/examples/minimax_h3/model_training/build_continuation_dataset.py"
    --source-jsonl "${SOURCE_JSONL}"
    --output-jsonl "${INDEX_JSONL}"
    --mode masked-av-v14
    --memory-only
    --progress
  )
  if [[ -n "${MAX_INDEX_SAMPLES}" ]]; then
    # All three caps must be set: the indexer only stops scanning once every
    # split has hit its limit, so leaving validation/test open scans the whole
    # 9.2GB source to write the same 12k rows.
    SIDE_SAMPLES=$(( MAX_INDEX_SAMPLES / 20 > 200 ? MAX_INDEX_SAMPLES / 20 : 200 ))
    INDEX_ARGS+=(
      --max-train "${MAX_INDEX_SAMPLES}"
      --max-validation "${SIDE_SAMPLES}"
      --max-test "${SIDE_SAMPLES}"
    )
  fi
  "${PYTHON}" "${INDEX_ARGS[@]}"
else
  echo "[data] reusing index ${INDEX_JSONL}"
fi

if [[ "${INDEX_ONLY:-0}" == "1" ]]; then
  echo "[data] INDEX_ONLY=1: stopping before the cache build"
  exit 0
fi

if [[ ! -f "${CACHE_ROOT}/train/manifest.jsonl" ]]; then
  echo "[data] building dual-slot latent cache -> ${CACHE_ROOT}"
  CACHE_ARGS=(
    "${REPO_ROOT}/examples/minimax_h3/model_training/build_continuation_cache.py"
    --index-jsonl "${INDEX_JSONL}"
    --output-dir "${CACHE_ROOT}"
    --expanded-index
    --h3-base "${H3_BASE:-/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA}"
    --memory-frames "${MEMORY_FRAMES}"
    --ltm-frames "${LTM_FRAMES}"
    --ltm-lead-steps "${LTM_LEAD_STEPS}"
    --height "${HEIGHT:-480}"
    --width "${WIDTH:-832}"
    --progress
  )
  if [[ -n "${NUM_SHARDS:-}" ]]; then
    CACHE_ARGS+=(--num-shards "${NUM_SHARDS}" --shard-index "${SHARD_INDEX:-0}")
  fi
  "${PYTHON}" "${CACHE_ARGS[@]}"
else
  echo "[data] reusing cache ${CACHE_ROOT}"
fi

echo "[train] delegating to the 2-node launcher with the memory-only contract"
CONTINUATION_MEMORY_MODE=memory-only \
CONTINUATION_MEMORY_EXPECT_SLOTS=stm,ltm \
DATA_ROOT="${CACHE_ROOT}" \
MANIFEST="${CACHE_ROOT}" \
exec bash "${REPO_ROOT}/examples/minimax_h3/model_training/run_continuation_lora_mask_v14_2node16gpu_alignedbase.sh"
