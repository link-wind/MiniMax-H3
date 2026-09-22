#!/usr/bin/env bash
set -euo pipefail

# Dual-slot memory cache builder: 8 local GPUs, an explicit shard range, and no
# node / rendezvous / cross-machine logic at all.
#
# The index is split by ``hash(sample_id) % NUM_SHARDS``, so a shard range is a
# complete, self-contained unit of work.  Two independent machines just take two
# disjoint ranges (0-7 and 8-15) and write into the same OUTPUT_ROOT; because
# every shard owns its own ``shard_<i>/`` directory and its own samples, nothing
# is shared and nothing needs to be coordinated.
#
# Each sample encodes three VAE clips: the window plus two memory slots --
# ``stm`` (the 39 frames before the window, contiguous) and ``ltm`` (the opening
# 39 frames of the clip, anchored at a constant distance).  See
# 长时记忆模块训练方案.md §17.
#
# Env knobs:
#   SHARD_START     first shard this machine owns      (default: 0)
#   NUM_SHARDS      total shards across all machines   (default: 16)
#   GPUS            GPUs on this machine               (default: 8)
#   INDEX_JSONL     expanded memory-only index         (default: outputs/memory_only_stage2/index.jsonl)
#   OUTPUT_ROOT     cache output root (shared across machines)
#   H3_BASE         VAE root (vae/ + audio_vae/, or FL2VA video_vae/ + audio_vae/)
#   MEMORY_FRAMES   STM slot frames, 17n+5             (default: 39; 0 disables)
#   LTM_FRAMES      LTM slot frames, 17n+5             (default: 39; 0 disables)
#   LTM_LEAD_STEPS  LTM anchor in latent steps         (default: 36)
#   CPU_PREFETCH / CPU_PREFETCH_WORKERS                (default: 4 / 4 per GPU)
#   FAKE            1 = CPU fake encoders, no weights  (default: 0)
#
# Example (machine A, then machine B):
#   SHARD_START=0 bash .../run_memory_only_cache_8gpu.sh
#   SHARD_START=8 bash .../run_memory_only_cache_8gpu.sh
# Then merge once, from either machine:
#   PYTHONPATH="$PWD" python .../merge_continuation_manifests.py --output-root <OUTPUT_ROOT>

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python}"

SHARD_START="${SHARD_START:-0}"
NUM_SHARDS="${NUM_SHARDS:-16}"
GPUS="${GPUS:-8}"

INDEX_JSONL="${INDEX_JSONL:-${REPO_ROOT}/outputs/memory_only_stage2/index.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/memory_only_stage2/cache_dual_slot}"
H3_BASE="${H3_BASE:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/MiniMax-H3-diffusers}"

MEMORY_FRAMES="${MEMORY_FRAMES:-39}"
LTM_FRAMES="${LTM_FRAMES:-39}"
LTM_LEAD_STEPS="${LTM_LEAD_STEPS:-36}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
VIDEO_TILING="${VIDEO_TILING:-1}"
VIDEO_TILE_HEIGHT="${VIDEO_TILE_HEIGHT:-480}"
VIDEO_TILE_WIDTH="${VIDEO_TILE_WIDTH:-832}"
CPU_PREFETCH="${CPU_PREFETCH:-4}"
CPU_PREFETCH_WORKERS="${CPU_PREFETCH_WORKERS:-$CPU_PREFETCH}"
OVERWRITE="${OVERWRITE:-0}"
FAKE="${FAKE:-0}"

SHARD_END=$((SHARD_START + GPUS))
if (( SHARD_END > NUM_SHARDS )); then
  echo "[error] shards ${SHARD_START}..$((SHARD_END-1)) exceed NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${INDEX_JSONL}" ]]; then
  echo "[error] index not found: ${INDEX_JSONL}" >&2
  exit 1
fi
if [[ "${FAKE}" != "1" && ! -d "${H3_BASE}/vae" && ! -d "${H3_BASE}/video_vae" ]]; then
  echo "[error] no VAE under ${H3_BASE} (need vae/+audio_vae/ or video_vae/+audio_vae/)" >&2
  exit 1
fi
if [[ "${MEMORY_FRAMES}" -le 0 && "${LTM_FRAMES}" -le 0 ]]; then
  echo "[error] MEMORY_FRAMES and LTM_FRAMES are both 0: this builds the non-memory cache" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"
echo "[cache] local machine owns shards ${SHARD_START}..$((SHARD_END-1)) of ${NUM_SHARDS} (${GPUS} GPUs)"
echo "[cache] memory     : stm=${MEMORY_FRAMES} ltm=${LTM_FRAMES} lead=${LTM_LEAD_STEPS}"
echo "[cache] index      : ${INDEX_JSONL}"
echo "[cache] output     : ${OUTPUT_ROOT}"
echo "[cache] geometry   : ${HEIGHT}x${WIDTH}   fake: ${FAKE}"

video_tiling_args=()
if [[ "${VIDEO_TILING}" == "1" ]]; then
  video_tiling_args+=(--video-tile-height "${VIDEO_TILE_HEIGHT}" --video-tile-width "${VIDEO_TILE_WIDTH}")
else
  video_tiling_args+=(--disable-video-tiling)
fi
ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

pids=()
for shard in $(seq "$SHARD_START" "$((SHARD_END - 1))"); do
  local_gpu=$((shard - SHARD_START))
  shard_root="${OUTPUT_ROOT}/shard_${shard}"
  mkdir -p "${shard_root}"
  echo "[cache] shard ${shard}/${NUM_SHARDS} on local gpu ${local_gpu}"
  args=(
    "${REPO_ROOT}/examples/minimax_h3/model_training/build_continuation_cache.py"
    --index-jsonl "${INDEX_JSONL}" --expanded-index --output-dir "${shard_root}"
    --device cuda --height "${HEIGHT}" --width "${WIDTH}"
    --num-shards "${NUM_SHARDS}" --shard-index "${shard}"
    --memory-frames "${MEMORY_FRAMES}" --ltm-frames "${LTM_FRAMES}" --ltm-lead-steps "${LTM_LEAD_STEPS}"
    --cpu-audio-vae --cpu-prefetch "${CPU_PREFETCH}" --cpu-prefetch-workers "${CPU_PREFETCH_WORKERS}"
    --progress
  )
  if [[ "${FAKE}" == "1" ]]; then
    args+=(--fake)
  else
    args+=(--h3-base "${H3_BASE}")
  fi
  args+=("${video_tiling_args[@]}")
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${local_gpu}" \
  PYTORCH_ALLOC_CONF="${ALLOC_CONF}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${args[@]}" \
    >"${OUTPUT_ROOT}/logs/shards_${SHARD_START}_${shard}.log" 2>&1 &
  pids+=("$!")
done

# Progress over this machine's own shards.  The shard total is only known once
# encoding finishes, so 1/NUM_SHARDS of the index is used as the target.
total_samples="$(awk 'NF {n++} END {print n+0}' "$INDEX_JSONL")"
local_total=$(( (total_samples + NUM_SHARDS - 1) / NUM_SHARDS * GPUS ))
started_at="$(date +%s)"
while :; do
  done_samples=0
  alive=0
  for shard in $(seq "$SHARD_START" "$((SHARD_END - 1))"); do
    log_file="${OUTPUT_ROOT}/logs/shards_${SHARD_START}_${shard}.log"
    if [[ -s "$log_file" ]]; then
      shard_done="$(tr '\r' '\n' < "$log_file" | sed -nE 's/.*shard [0-9]+\/[0-9]+: ([0-9]+)sample.*/\1/p' | tail -1)"
      done_samples=$((done_samples + ${shard_done:-0}))
    fi
  done
  for pid in "${pids[@]}"; do
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ -n "$state" && "$state" != Z* ]] && alive=1
  done
  elapsed=$(( $(date +%s) - started_at ))
  if (( done_samples > 0 && elapsed > 0 )); then
    (( done_samples > local_total )) && done_samples="$local_total"
    eta=$(( (local_total - done_samples) * elapsed / done_samples ))
    printf '\r[shards %s-%s] %d/%d samples (%d%%) | %.2f sample/s | ETA %02d:%02d:%02d' \
      "$SHARD_START" "$((SHARD_END-1))" "$done_samples" "$local_total" \
      "$((done_samples * 100 / local_total))" \
      "$(awk -v n="$done_samples" -v e="$elapsed" 'BEGIN{printf "%.2f", n/e}')" \
      $((eta/3600)) $(((eta%3600)/60)) $((eta%60))
  else
    printf '\r[shards %s-%s] running ...' "$SHARD_START" "$((SHARD_END-1))"
  fi
  (( alive == 0 )) && break
  sleep 10
done
printf '\n'

failed=0
for shard in $(seq "$SHARD_START" "$((SHARD_END - 1))"); do
  if ! wait "${pids[$((shard - SHARD_START))]}"; then
    failed=1
    log_file="${OUTPUT_ROOT}/logs/shards_${SHARD_START}_${shard}.log"
    echo "[cache] shard ${shard} failed; last 40 log lines:" >&2
    [[ -s "$log_file" ]] && tail -n 40 "$log_file" >&2 || echo "  (no log written)"
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[cache] rerun this same command to resume; finished shards are skipped" >&2
  exit 1
fi

echo "[cache] shards ${SHARD_START}..$((SHARD_END-1)) done"
echo "[cache] once every machine has finished, merge from either one:"
echo "  PYTHONPATH=\"${REPO_ROOT}\" \"${PYTHON_BIN}\" \\"
echo "    ${REPO_ROOT}/examples/minimax_h3/model_training/merge_continuation_manifests.py \\"
echo "    --output-root \"${OUTPUT_ROOT}\""
