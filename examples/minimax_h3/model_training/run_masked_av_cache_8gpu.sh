#!/usr/bin/env bash
set -euo pipefail

# Resumable eight-GPU H3 masked-av-v14 cache builder.
SOURCE_JSONL="${SOURCE_JSONL:-/gemini/platform/public/aigc/human_guozz2/data/LongVideoGen/20260729/data_with_face_and_speech_and_caption.jsonl}"
H3_BASE="${H3_BASE:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/MiniMax-H3-diffusers}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/caches/h3_masked_av_v14_cache_8gpu}"
INDEX_JSONL="${INDEX_JSONL:-${OUTPUT_ROOT}/index.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-8}"
WINDOW_FRAMES="${WINDOW_FRAMES:-345}"
OVERLAP_FRAMES="${OVERLAP_FRAMES:-39}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
VIDEO_TILING="${VIDEO_TILING:-1}"
# Default 480x832 tiles (whole frame) for H100 80GB: ~2.6x faster video-VAE
# encode vs 256x256, ~74 GB peak; use 480x480 (~48 GB) if less VRAM headroom.
VIDEO_TILE_HEIGHT="${VIDEO_TILE_HEIGHT:-480}"
VIDEO_TILE_WIDTH="${VIDEO_TILE_WIDTH:-832}"
# Retain a few decoded 345-frame windows per GPU process in CPU RAM.  Decoding
# 1080p windows is the slow half of this job, so a single worker per shard was
# the bottleneck; each in-flight window is ~0.5 GB, and NUM_SHARDS * workers
# decoders should stay below the host's physical core count.
CPU_PREFETCH="${CPU_PREFETCH:-4}"
CPU_PREFETCH_WORKERS="${CPU_PREFETCH_WORKERS:-$CPU_PREFETCH}"
# Build a bounded manifest by default.  The source JSONL is very large, so a
# 30k-record run uses a 27k/1.5k/1.5k train/validation/test split.  Set any
# value to an empty string to remove that split's limit.
MAX_TRAIN="${MAX_TRAIN:-27000}"
MAX_VALIDATION="${MAX_VALIDATION:-1500}"
MAX_TEST="${MAX_TEST:-1500}"
# Path validation performs two network filesystem stat calls per source row.
# The corpus paths are already validated in the cache stage, so keep this off
# by default; set REQUIRE_PATHS=1 for strict manifest construction.
REQUIRE_PATHS="${REQUIRE_PATHS:-0}"
PYTHON_BIN="${PYTHON_BIN:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

mkdir -p "$OUTPUT_ROOT/logs"
if [[ ! -s "$INDEX_JSONL" ]]; then
  echo "[index] building $INDEX_JSONL"
  index_tmp="${INDEX_JSONL}.tmp.$$"
  index_limits=()
  index_path_check=()
  [[ -n "$MAX_TRAIN" ]] && index_limits+=(--max-train "$MAX_TRAIN")
  [[ -n "$MAX_VALIDATION" ]] && index_limits+=(--max-validation "$MAX_VALIDATION")
  [[ -n "$MAX_TEST" ]] && index_limits+=(--max-test "$MAX_TEST")
  [[ "$REQUIRE_PATHS" == "1" ]] && index_path_check+=(--require-paths)
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    "$REPO_ROOT/examples/minimax_h3/model_training/build_continuation_dataset.py" \
    --source-jsonl "$SOURCE_JSONL" --output-jsonl "$index_tmp" \
    --mode masked-av-v14 --window-frames "$WINDOW_FRAMES" \
    --overlap-frames "$OVERLAP_FRAMES" --progress \
    "${index_path_check[@]}" "${index_limits[@]}"
  mv -f "$index_tmp" "$INDEX_JSONL"
else
  echo "[index] reusing $INDEX_JSONL"
fi

echo "[cache] launching ${NUM_SHARDS} shards"
echo "[cache] CPU prefetch: depth=${CPU_PREFETCH}, workers=${CPU_PREFETCH_WORKERS} per GPU"
video_tiling_args=()
if [[ "$VIDEO_TILING" == "1" ]]; then
  video_tiling_args+=(--video-tile-height "$VIDEO_TILE_HEIGHT" --video-tile-width "$VIDEO_TILE_WIDTH")
else
  video_tiling_args+=(--disable-video-tiling)
fi
pids=()
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  shard_root="${OUTPUT_ROOT}/shard_${shard}"
  mkdir -p "$shard_root"
  echo "[cache] shard ${shard}/${NUM_SHARDS}"
  CUDA_VISIBLE_DEVICES="$shard" \
  PYTORCH_ALLOC_CONF="$ALLOC_CONF" PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    "$REPO_ROOT/examples/minimax_h3/model_training/build_continuation_cache.py" \
    --index-jsonl "$INDEX_JSONL" --expanded-index --output-dir "$shard_root" \
    --h3-base "$H3_BASE" --device cuda --height "$HEIGHT" --width "$WIDTH" \
    "${video_tiling_args[@]}" --num-shards "$NUM_SHARDS" --shard-index "$shard" \
    --cpu-audio-vae --cpu-prefetch "$CPU_PREFETCH" --cpu-prefetch-workers "$CPU_PREFETCH_WORKERS" --progress >"${OUTPUT_ROOT}/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

# Keep the per-shard tqdm logs for debugging, while also showing one compact
# aggregate progress line in the launching terminal.
total_samples="$(awk 'NF {n++} END {print n+0}' "$INDEX_JSONL")"
cached_before="$(find "$OUTPUT_ROOT" -path '*/shard_*/*' -type f -name '*.pt' | wc -l)"
progress_start_time="$(date +%s)"
while :; do
  run_completed=0
  alive=0
  for (( shard=0; shard<NUM_SHARDS; shard++ )); do
    log_file="${OUTPUT_ROOT}/logs/shard_${shard}.log"
    if [[ -s "$log_file" ]]; then
      shard_completed="$(tr '\r' '\n' < "$log_file" | sed -nE 's/.*shard [0-9]+\/[0-9]+: ([0-9]+)sample.*/\1/p' | tail -1)"
      run_completed=$((run_completed + ${shard_completed:-0}))
    fi
  done
  for pid in "${pids[@]}"; do
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ -n "$state" && "$state" != Z* ]] && alive=1
  done
  completed_samples=$((run_completed + cached_before))
  (( completed_samples > total_samples )) && completed_samples="$total_samples"
  now="$(date +%s)"
  elapsed=$((now - progress_start_time))
  percent=0
  if (( total_samples > 0 )); then
    percent=$((completed_samples * 100 / total_samples))
  fi
  if (( run_completed > 0 && elapsed > 0 )); then
    eta_seconds=$(( (total_samples - completed_samples) * elapsed / run_completed ))
    rate_milli=$(( run_completed * 1000 / elapsed ))
    eta_hours=$((eta_seconds / 3600))
    eta_minutes=$(((eta_seconds % 3600) / 60))
    eta_seconds_remainder=$((eta_seconds % 60))
    elapsed_hours=$((elapsed / 3600))
    elapsed_minutes=$(((elapsed % 3600) / 60))
    elapsed_seconds=$((elapsed % 60))
    printf -v eta '%02d:%02d:%02d' "$eta_hours" "$eta_minutes" "$eta_seconds_remainder"
    printf -v elapsed_text '%02d:%02d:%02d' "$elapsed_hours" "$elapsed_minutes" "$elapsed_seconds"
    printf -v rate_text '%d.%03d' $((rate_milli / 1000)) $((rate_milli % 1000))
    printf '\r[cache] progress %d/%d samples (%d%%) | elapsed %s | %s sample/s | ETA %s' \
      "$completed_samples" "$total_samples" "$percent" \
      "$elapsed_text" "$rate_text" "$eta"
  else
    printf '\r[cache] progress %d/%d samples (%d%%) | elapsed 00:00:00 | ETA calculating...' \
      "$completed_samples" "$total_samples" "$percent"
  fi
  (( alive == 0 )) && break
  sleep 10
done
printf '\n'

failed=0
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  if ! wait "${pids[$shard]}"; then
    failed=1
    log_file="${OUTPUT_ROOT}/logs/shard_${shard}.log"
    echo "[cache] shard ${shard} failed; last 80 log lines follow:" >&2
    if [[ -s "$log_file" ]]; then
      tail -n 80 "$log_file" >&2
    else
      echo "[cache] shard ${shard} did not write a log: ${log_file}" >&2
    fi
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[cache] a shard failed; rerun this script to resume" >&2
  exit 1
fi

echo "[merge] merging manifests"
PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
  "$REPO_ROOT/examples/minimax_h3/model_training/merge_continuation_manifests.py" \
  --output-root "$OUTPUT_ROOT"
echo "[done] $OUTPUT_ROOT"
