#!/usr/bin/env bash
set -euo pipefail

# Single-GPU H3 masked-av-v14 cache builder.
#
# Defaults to the 24-record smoke index so you can check the real VAE encode
# path (tiling / audio-VAE placement / per-sample speed) before committing to
# the 8-GPU run.  Point INDEX_JSONL at the full mix index and clear MAX_SAMPLES
# to encode everything on one card instead.
#
# Env knobs (all optional):
#   INDEX_JSONL   index to encode            (default: smoke index)
#   OUTPUT_ROOT   cache output root          (default: outputs/caches/h3_mix_smoke)
#   MAX_SAMPLES   record cap; "" = no cap    (default: 24)
#   GPU           CUDA_VISIBLE_DEVICES       (default: 0)
#   H3_BASE       FL2VA diffusers root       (default below)
#   HEIGHT/WIDTH  latent geometry            (default: 480x832)
#   VIDEO_TILING  1/0                        (default: 1)
#   VIDEO_TILE_HEIGHT / VIDEO_TILE_WIDTH     (default: 480x832)
#   CPU_PREFETCH / CPU_PREFETCH_WORKERS      (default: 8 / 8, see note below)
#   CPU_AUDIO_VAE 1/0 keep audio VAE on CPU  (default: 1)
#   AUDIO_DIR     per-fragment wav dir; "" auto-resolves <clip_dir>/../audio/raw
#   OVERWRITE     1 re-encode existing .pt   (default: 0, resume-friendly)
#   NO_REUSE      1 ignore reused-record pointers (default: 0)
#   FAKE          1 CPU-only fake encoders, no H3 weights (plumbing check)

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python}"
H3_BASE="${H3_BASE:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/MiniMax-H3-diffusers}"
GPU="${GPU:-0}"

INDEX_JSONL="${INDEX_JSONL:-${REPO_ROOT}/outputs/continuation_mix_index/smoke_index.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/caches/h3_mix_smoke}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
VIDEO_TILING="${VIDEO_TILING:-1}"
# Default 480x832 tiles = the whole 480x832 frame as one tile (no spatial
# splitting).  On an H100 80GB this cuts video-VAE encode time ~2.6x (38s ->
# 14s for 345 frames) versus 256x256 while peaking ~74 GB, so it needs
# PYTORCH_*_ALLOC_CONF=expandable_segments:True (set below).  Drop to 480x480
# (~48 GB peak) if you hit OOM on a card with less headroom.
VIDEO_TILE_HEIGHT="${VIDEO_TILE_HEIGHT:-480}"
VIDEO_TILE_WIDTH="${VIDEO_TILE_WIDTH:-832}"
# Decoding 345 frames of 1080p takes ~10 s per window on one thread, which made
# the single prefetch worker the bottleneck.  Each in-flight window holds about
# 0.5 GB of resized frames, so a depth/worker count of 8 costs ~5 GB of host RAM.
CPU_PREFETCH="${CPU_PREFETCH:-8}"
CPU_PREFETCH_WORKERS="${CPU_PREFETCH_WORKERS:-${CPU_PREFETCH}}"
MAX_SAMPLES="${MAX_SAMPLES-24}"
AUDIO_DIR="${AUDIO_DIR-}"
OVERWRITE="${OVERWRITE:-0}"
NO_REUSE="${NO_REUSE:-0}"
CPU_AUDIO_VAE="${CPU_AUDIO_VAE:-1}"
FAKE="${FAKE:-0}"

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
LOG_PATH="${OUTPUT_ROOT}/logs/single_gpu.log"

ARGS=(
  "${REPO_ROOT}/examples/minimax_h3/model_training/build_continuation_cache.py"
  --index-jsonl "${INDEX_JSONL}" --expanded-index --output-dir "${OUTPUT_ROOT}"
  --device cuda --height "${HEIGHT}" --width "${WIDTH}"
  --cpu-prefetch "${CPU_PREFETCH}" --cpu-prefetch-workers "${CPU_PREFETCH_WORKERS}"
  --progress
)
if [[ "${FAKE}" == "1" ]]; then
  ARGS+=(--fake)
else
  ARGS+=(--h3-base "${H3_BASE}")
fi
if [[ "${VIDEO_TILING}" == "1" ]]; then
  ARGS+=(--video-tile-height "${VIDEO_TILE_HEIGHT}" --video-tile-width "${VIDEO_TILE_WIDTH}")
else
  ARGS+=(--disable-video-tiling)
fi
if [[ "${CPU_AUDIO_VAE}" == "1" && "${FAKE}" != "1" ]]; then
  ARGS+=(--cpu-audio-vae)
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${AUDIO_DIR}" ]]; then
  ARGS+=(--audio-dir "${AUDIO_DIR}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${NO_REUSE}" == "1" ]]; then
  ARGS+=(--no-reuse)
fi

ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

echo "[cache] index      : ${INDEX_JSONL}"
echo "[cache] output     : ${OUTPUT_ROOT}"
echo "[cache] gpu        : ${GPU}   tiling: ${VIDEO_TILING} (${VIDEO_TILE_HEIGHT}x${VIDEO_TILE_WIDTH})"
echo "[cache] geometry   : ${HEIGHT}x${WIDTH}   fake: ${FAKE}   max_samples: ${MAX_SAMPLES:-<all>}"
echo "[cache] prefetch   : depth=${CPU_PREFETCH} workers=${CPU_PREFETCH_WORKERS}   log: ${LOG_PATH}"

cached_before="$(find "${OUTPUT_ROOT}" -path '*/train/*.pt' -type f 2>/dev/null | wc -l)"
start_time="$(date +%s)"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_ALLOC_CONF="${ALLOC_CONF}" PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}" \
PYTHONPATH="${REPO_ROOT}" "${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
status="${PIPESTATUS[0]}"
set -e

elapsed=$(( $(date +%s) - start_time ))
cached_after="$(find "${OUTPUT_ROOT}" -path '*/train/*.pt' -type f 2>/dev/null | wc -l)"
written=$(( cached_after - cached_before ))
rate="n/a"
if (( written > 0 && elapsed > 0 )); then
  rate="$(awk -v e="${elapsed}" -v w="${written}" 'BEGIN { printf "%.2f", e / w }')"
fi

echo ""
echo "[cache] exit=${status} elapsed=${elapsed}s new_pt=${written} sec/sample=${rate}"
if [[ "${status}" -ne 0 ]]; then
  echo "[cache] last 40 log lines:" >&2
  tail -n 40 "${LOG_PATH}" >&2
  exit "${status}"
fi
if [[ -n "${MAX_SAMPLES}" && "${written}" -eq 0 ]]; then
  echo "[cache] nothing new written (existing .pt were skipped); set OVERWRITE=1 to re-encode" >&2
fi

if [[ "${rate}" != "n/a" ]]; then
  eta8="$(awk -v r="${rate}" 'BEGIN { printf "%.1f", 50534 * r / 8 / 3600 }')"
  echo "[cache] extrapolation: 50,534 new windows at ${rate}s/sample on 8 GPUs ~= ${eta8} h"
fi
