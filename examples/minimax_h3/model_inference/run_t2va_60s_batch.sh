#!/usr/bin/env bash
set -euo pipefail

# Batch-generate the 5 T2VA (text->video+audio) 60s prompts in
# h3_t2va_60s_prompts.json. Each entry is a 5-window continuation plan
# (345 frames/window, 39-frame Masked-AV overlap) assembled and trimmed to
# 1440 frames (~60s at 24 fps).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${H3_RUNNER:-$REPO_ROOT/examples/minimax_h3/model_inference/run_30s_local_cp_infer.sh}"
MANIFEST="${H3_PROMPT_MANIFEST:-$REPO_ROOT/examples/minimax_h3/model_inference/prompts/h3_t2va_60s_prompts.json}"
OUTPUT_DIR="${H3_BATCH_OUTPUT_DIR:-$REPO_ROOT/outputs/t2va_60s/h3_t2va_60s_batch}"
PLAN_DIR="${H3_BATCH_PLAN_DIR:-$REPO_ROOT/outputs/t2va_60s/h3_t2va_60s_plans}"
NUM_STEPS="${H3_NUM_INFERENCE_STEPS:-8}"
SEED_BASE="${H3_SEED_BASE:-2000}"
CHECKPOINT="${H3_MERGED_CHECKPOINT:-${H3_CHECKPOINT:-/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA/transformer_lora600_v4}}"
SKIP_EXISTING="${H3_SKIP_EXISTING:-1}"
DRY_RUN=0
T2VA_LORA="${H3_T2VA_LORA:-}"
T2VA_LORA_SCALE="${H3_T2VA_LORA_SCALE:-0.7}"
HEIGHT="${H3_HEIGHT:-768}"
WIDTH="${H3_WIDTH:-1344}"

# T2VA is intentionally safe on a single GPU. Override for multi-GPU CP.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export H3_NPROC_PER_NODE="${H3_NPROC_PER_NODE:-1}"
export H3_CP_WORLD_SIZE="${H3_CP_WORLD_SIZE:-1}"

usage() {
  cat <<'EOF'
Usage: run_t2va_60s_batch.sh [--dry-run] [--force]

Environment overrides:
  H3_PROMPT_MANIFEST, H3_BATCH_OUTPUT_DIR, H3_BATCH_PLAN_DIR
  H3_MERGED_CHECKPOINT (or H3_CHECKPOINT), H3_NUM_INFERENCE_STEPS (8)
  H3_T2VA_LORA (default: none), H3_T2VA_LORA_SCALE (default: 0.7)
  H3_SEED_BASE (2000), H3_SKIP_EXISTING (1)
  H3_HEIGHT (768), H3_WIDTH (1344), H3_RUNNER
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) SKIP_EXISTING=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
[[ -x "$RUNNER" ]] || { echo "runner is not executable: $RUNNER" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$PLAN_DIR"

# Materialize text prompts and continuation plans from the reviewed manifest.
python - "$MANIFEST" "$PLAN_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, staging = map(Path, sys.argv[1:])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
items = payload.get("single_shot_60s")
if not isinstance(items, list) or len(items) != 5:
    raise SystemExit("manifest must contain exactly five single_shot_60s entries")

for old_file in staging.glob("*.json"):
    old_file.unlink()

for item in items:
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id or Path(item_id).name != item_id:
        raise SystemExit(f"invalid prompt id: {item_id!r}")
    global_prompt = item.get("global_prompt")
    segments = item.get("segments")
    if not isinstance(global_prompt, str) or not global_prompt.strip() or not isinstance(segments, list) or len(segments) != 5:
        raise SystemExit(f"{item_id}: invalid global_prompt or needs exactly five segments")
    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not isinstance(segment.get("prompt"), str) or not segment["prompt"].strip():
            raise SystemExit(f"{item_id}: every segment needs a non-empty prompt")
        normalized.append({
            "segment_id": str(segment.get("segment_id") or f"window_{index + 1:02d}"),
            "requested_frames": 345,
            "prompt": segment["prompt"].strip(),
        })
    plan = {"plan_id": item_id, "global_prompt": global_prompt.strip(), "segments": normalized}
    destination = staging / f"{item_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
PY

run_one() {
  local label="$1" output="$2" seed="$3"
  shift 3
  if [[ "$SKIP_EXISTING" == "1" && -s "$output" ]]; then
    echo "[skip] $label -> $output"
    return 0
  fi
  local -a command=("$RUNNER" "$@" --num_inference_steps "$NUM_STEPS" --seed "$seed")
  [[ -n "$CHECKPOINT" ]] && command+=(--checkpoint "$CHECKPOINT")
  if [[ -n "$T2VA_LORA" ]]; then
    command+=(--lora "$T2VA_LORA" --lora-scale "$T2VA_LORA_SCALE")
  fi
  echo "[run]  $label (seed=$seed)"
  printf '       %q ' "${command[@]}"
  printf '\n'
  ((DRY_RUN)) && return 0
  "${command[@]}"
}

seed=$SEED_BASE
shopt -s nullglob
for plan_file in "$PLAN_DIR"/*.json; do
  [[ -s "$plan_file" ]] || { echo "continuation plan missing or empty: $plan_file" >&2; exit 1; }
  id="$(basename "$plan_file" .json)"
  run_one "$id" "$OUTPUT_DIR/$id.mp4" "$seed" \
    --continuation-plan "$plan_file" \
    --continuation-window-frames 345 --continuation-overlap-frames 39 \
    --continuation-mode masked-av-v14 \
    --height "$HEIGHT" --width "$WIDTH" \
    --output-video-frames 1440 \
    --output_path "$OUTPUT_DIR/$id.mp4"
  seed=$((seed + 1))
done

echo "batch complete: outputs under $OUTPUT_DIR"
