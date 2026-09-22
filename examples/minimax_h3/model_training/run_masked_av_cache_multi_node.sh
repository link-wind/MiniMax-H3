#!/usr/bin/env bash
set -euo pipefail

# Multi-node H3 masked-av-v14 cache builder (e.g. 2 nodes x 8 GPUs = 16 shards).
#
# The index is split by ``hash(sample_hash) % NUM_SHARDS`` inside
# build_continuation_cache.py, so every shard is independent and different nodes
# may encode disjoint shards in parallel against a shared filesystem.  Run this
# same script on each node, passing a distinct NODE_INDEX.
#
# All of INDEX_JSONL, OUTPUT_ROOT and H3_BASE must be shared across nodes
# (network filesystem).  Only NODE_INDEX=0 merges the per-shard manifests once
# every shard's split manifest is present.
#
# Env knobs:
#   NODES           total nodes               (default: 2)
#   NODE_INDEX      this node, 0-based        (default: 0)
#   GPUS_PER_NODE   GPUs on this node         (default: 8)
#   INDEX_JSONL     expanded index to encode  (default: outputs/continuation_mix_index/train.jsonl)
#   OUTPUT_ROOT     cache output root         (default: outputs/caches/h3_masked_av_v14_mix100k_16gpu)
#   MAX_SAMPLES     record cap; "" = no cap   (default: empty)
#   GPU_START       optional local GPU offset (default: 0)
#   HEIGHT/WIDTH    latent geometry           (default: 480x832)
#   VIDEO_TILING    1/0                       (default: 1)
#   VIDEO_TILE_HEIGHT / VIDEO_TILE_WIDTH      (default: 480x832)
#   CPU_PREFETCH / CPU_PREFETCH_WORKERS       (default: 4 / 4 per GPU)
#   OVERWRITE       1 re-encode existing .pt  (default: 0)
#   NO_REUSE        1 ignore reused pointers  (default: 0)
#   CPU_AUDIO_VAE   1/0 keep audio VAE on CPU (default: 1)
#   FAKE            1 CPU-only fake encoders  (default: 0)
#
# Memory slots (memory-only objective, see 长时记忆模块训练方案.md §17):
#   MEMORY_FRAMES   STM slot: the N frames before each window (default: 0 = off)
#   LTM_FRAMES      LTM slot: the opening N frames of the clip (default: 0 = off)
#   LTM_LEAD_STEPS  constant anchor of the LTM slot, in latent steps (default: 36)
#
#   Setting MEMORY_FRAMES and LTM_FRAMES together produces the dual-slot layout
#   that ``train.py --continuation_memory_mode memory-only`` reads.  Both slots
#   must satisfy 17n+5; 39 frames = 12 latent steps is the M1-fast unit.

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python}"
H3_BASE="${H3_BASE:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/MiniMax-H3-diffusers}"

NODES="${NODES:-2}"
NODE_INDEX="${NODE_INDEX:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GPU_START="${GPU_START:-0}"
NUM_SHARDS=$((NODES * GPUS_PER_NODE))
SHARD_START=$((NODE_INDEX * GPUS_PER_NODE))
SHARD_END=$((SHARD_START + GPUS_PER_NODE))

INDEX_JSONL="${INDEX_JSONL:-${REPO_ROOT}/outputs/continuation_mix_index/train.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/caches/h3_masked_av_v14_mix100k_16gpu}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
VIDEO_TILING="${VIDEO_TILING:-1}"
VIDEO_TILE_HEIGHT="${VIDEO_TILE_HEIGHT:-480}"
VIDEO_TILE_WIDTH="${VIDEO_TILE_WIDTH:-832}"
CPU_PREFETCH="${CPU_PREFETCH:-4}"
CPU_PREFETCH_WORKERS="${CPU_PREFETCH_WORKERS:-$CPU_PREFETCH}"
MAX_SAMPLES="${MAX_SAMPLES-}"
OVERWRITE="${OVERWRITE:-0}"
NO_REUSE="${NO_REUSE:-0}"
CPU_AUDIO_VAE="${CPU_AUDIO_VAE:-1}"
FAKE="${FAKE:-0}"
MEMORY_FRAMES="${MEMORY_FRAMES:-0}"
LTM_FRAMES="${LTM_FRAMES:-0}"
LTM_LEAD_STEPS="${LTM_LEAD_STEPS:-36}"

if [[ "$NODE_INDEX" -lt 0 || "$NODE_INDEX" -ge "$NODES" ]]; then
  echo "[error] NODE_INDEX=$NODE_INDEX out of range [0,$NODES)" >&2
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
if [[ "${FAKE}" != "1" && ! -d "${H3_BASE}/vae" ]]; then
  echo "[error] H3 VAE not found under ${H3_BASE}/vae" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

echo "[cache] nodes=${NODES} node=${NODE_INDEX} gpus/node=${GPUS_PER_NODE} shards=${NUM_SHARDS} -> node shards ${SHARD_START}..$((SHARD_END-1))"
echo "[cache] memory     : stm=${MEMORY_FRAMES} ltm=${LTM_FRAMES} lead=${LTM_LEAD_STEPS} (0 = no memory slot)"
echo "[cache] index      : ${INDEX_JSONL}"
echo "[cache] output     : ${OUTPUT_ROOT}"
echo "[cache] geometry   : ${HEIGHT}x${WIDTH}   fake: ${FAKE}   max_samples: ${MAX_SAMPLES:-<all>}"
echo "[cache] tiling     : ${VIDEO_TILING} (${VIDEO_TILE_HEIGHT}x${VIDEO_TILE_WIDTH})"
echo "[cache] prefetch   : depth=${CPU_PREFETCH} workers=${CPU_PREFETCH_WORKERS} per GPU"

video_tiling_args=()
if [[ "${VIDEO_TILING}" == "1" ]]; then
  video_tiling_args+=(--video-tile-height "${VIDEO_TILE_HEIGHT}" --video-tile-width "${VIDEO_TILE_WIDTH}")
else
  video_tiling_args+=(--disable-video-tiling)
fi

ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

pids=()
for shard in $(seq "$SHARD_START" "$((SHARD_END - 1))"); do
  local_gpu=$((shard - SHARD_START + GPU_START))
  shard_root="${OUTPUT_ROOT}/shard_${shard}"
  mkdir -p "${shard_root}"
  echo "[cache] shard ${shard}/${NUM_SHARDS} on local gpu ${local_gpu}"
  args=(
    "${REPO_ROOT}/examples/minimax_h3/model_training/build_continuation_cache.py"
    --index-jsonl "${INDEX_JSONL}" --expanded-index --output-dir "${shard_root}"
    --device cuda --height "${HEIGHT}" --width "${WIDTH}"
    --num-shards "${NUM_SHARDS}" --shard-index "${shard}"
    --cpu-audio-vae --cpu-prefetch "${CPU_PREFETCH}" --cpu-prefetch-workers "${CPU_PREFETCH_WORKERS}"
    --progress
  )
  if [[ "${FAKE}" == "1" ]]; then
    args+=(--fake)
  else
    args+=(--h3-base "${H3_BASE}")
  fi
  args+=("${video_tiling_args[@]}")
  if [[ -n "${MAX_SAMPLES}" ]]; then
    args+=(--max-samples "${MAX_SAMPLES}")
  fi
  # Memory slots are encoded inside the same sample loop, so a cache entry is
  # always self-contained and the shard layout is unchanged.
  if [[ "${MEMORY_FRAMES}" -gt 0 || "${LTM_FRAMES}" -gt 0 ]]; then
    args+=(--memory-frames "${MEMORY_FRAMES}" --ltm-frames "${LTM_FRAMES}" --ltm-lead-steps "${LTM_LEAD_STEPS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${NO_REUSE}" == "1" ]]; then
    args+=(--no-reuse)
  fi
  CUDA_VISIBLE_DEVICES="${local_gpu}" \
  PYTORCH_ALLOC_CONF="${ALLOC_CONF}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
  PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${args[@]}" \
    >"${OUTPUT_ROOT}/logs/node${NODE_INDEX}_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

# Aggregate progress across this node's own shards only.
total_samples="$(awk 'NF {n++} END {print n+0}' "$INDEX_JSONL")"
# Approximate this node's share of the index.  The true per-shard total is
# only known once encoding finishes, so 1/NODES is a good steady-state estimate.
node_total=$(( total_samples / NODES ))
progress_start_time="$(date +%s)"
while :; do
  run_completed=0
  alive=0
  for shard in $(seq "$SHARD_START" "$((SHARD_END - 1))"); do
    log_file="${OUTPUT_ROOT}/logs/node${NODE_INDEX}_shard_${shard}.log"
    if [[ -s "$log_file" ]]; then
      shard_completed="$(tr '\r' '\n' < "$log_file" | sed -nE 's/.*shard [0-9]+\/[0-9]+: ([0-9]+)sample.*/\1/p' | tail -1)"
      run_completed=$((run_completed + ${shard_completed:-0}))
    fi
  done
  for pid in "${pids[@]}"; do
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ -n "$state" && "$state" != Z* ]] && alive=1
  done
  now="$(date +%s)"
  elapsed=$((now - progress_start_time))
  if (( run_completed > 0 && elapsed > 0 )); then
    completed_samples=$run_completed
    (( completed_samples > node_total )) && completed_samples="$node_total"
    percent=$((completed_samples * 100 / node_total))
    eta_seconds=$(( (node_total - completed_samples) * elapsed / run_completed ))
    printf '\r[node %s] %d/%d samples (%d%%) | %s sample/s | ETA %02d:%02d:%02d' \
      "$NODE_INDEX" "$completed_samples" "$node_total" "$percent" \
      "$(awk -v n="$run_completed" -v e="$elapsed" 'BEGIN{printf "%.2f", n/e}')" \
      $((eta_seconds/3600)) $(((eta_seconds%3600)/60)) $((eta_seconds%60))
  else
    printf '\r[node %s] running ...' "$NODE_INDEX"
  fi
  (( alive == 0 )) && break
  sleep 10
done
printf '\n'

failed=0
for shard in $(seq "$SHARD_START" "$((SHARD_END - 1))"); do
  idx=$((shard - SHARD_START))
  if ! wait "${pids[$idx]}"; then
    failed=1
    log_file="${OUTPUT_ROOT}/logs/node${NODE_INDEX}_shard_${shard}.log"
    echo "[cache] node ${NODE_INDEX} shard ${shard} failed; last 50 log lines:" >&2
    if [[ -s "$log_file" ]]; then
      tail -n 50 "$log_file" >&2
    else
      echo "[cache] no log written: ${log_file}" >&2
    fi
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[cache] a shard failed on node ${NODE_INDEX}; rerun this script to resume" >&2
  exit 1
fi

# Only node 0 merges, and only once every shard's split manifest exists.
if [[ "$NODE_INDEX" == "0" ]]; then
  echo "[cache] node 0: waiting for all ${NUM_SHARDS} shards before merging"
  all_ready=1
  missing=""
  for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
    if ! compgen -G "${OUTPUT_ROOT}/shard_${shard}/*/manifest.jsonl" >/dev/null; then
      all_ready=0
      missing="${missing}${missing:+ }${shard}"
    fi
  done
  if [[ "$all_ready" == "1" ]]; then
    echo "[merge] merging manifests"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      "$REPO_ROOT/examples/minimax_h3/model_training/merge_continuation_manifests.py" \
      --output-root "$OUTPUT_ROOT"
    echo "[done] $OUTPUT_ROOT"
  else
    echo "[cache] node 0 finished its shards but shards without a manifest: ${missing:-none};"
    echo "[cache] run merge_continuation_manifests.py manually once all nodes complete:"
    echo "  PYTHONPATH=\"$REPO_ROOT\" \"$PYTHON_BIN\" \\"
    echo "    examples/minimax_h3/model_training/merge_continuation_manifests.py --output-root \"$OUTPUT_ROOT\""
  fi
else
  shard_end_p1=$((SHARD_END - 1))
  echo "[cache] node ${NODE_INDEX} done (${SHARD_START}..${shard_end_p1}); wait for node 0 to merge"
fi
