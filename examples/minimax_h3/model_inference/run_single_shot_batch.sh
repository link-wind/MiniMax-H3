#!/usr/bin/env bash
set -euo pipefail

# Batch-generate the 10 prompts in h3_single_shot_prompts.json.
# 30s: two 362-frame windows with native 39-frame Masked-AV overlap. 60s:
# five 345-frame windows, dynamic appearance memory, and a final 60s trim.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${H3_RUNNER:-$REPO_ROOT/examples/minimax_h3/model_inference/run_30s_local_cp_infer.sh}"
MANIFEST="${H3_PROMPT_MANIFEST:-$REPO_ROOT/examples/minimax_h3/model_inference/prompts/h3_single_shot_prompts.json}"
OUTPUT_DIR="${H3_BATCH_OUTPUT_DIR:-$REPO_ROOT/outputs/single_shot/h3_single_shot_batch}"
# Keep plans outside the generated-video tree.  A cleanup/retry of OUTPUT_DIR
# must never remove the JSON that an already-started torchrun process needs.
PLAN_DIR="${H3_BATCH_PLAN_DIR:-$REPO_ROOT/outputs/single_shot/h3_single_shot_plans}"
NUM_STEPS="${H3_NUM_INFERENCE_STEPS:-8}"
SEED_BASE="${H3_SEED_BASE:-1000}"
CHECKPOINT="${H3_MERGED_CHECKPOINT:-${H3_CHECKPOINT:-/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA/transformer_lora600_v4}}"
SKIP_EXISTING="${H3_SKIP_EXISTING:-1}"
DRY_RUN=0

# This batch is intentionally safe to run on one GPU. Override these exports
# when using a multi-GPU CP launch.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export H3_NPROC_PER_NODE="${H3_NPROC_PER_NODE:-1}"
export H3_CP_WORLD_SIZE="${H3_CP_WORLD_SIZE:-1}"

usage() {
  cat <<'EOF'
Usage: run_single_shot_batch.sh [--dry-run] [--force]

Environment overrides:
  H3_PROMPT_MANIFEST, H3_BATCH_OUTPUT_DIR, H3_BATCH_PLAN_DIR
  H3_MERGED_CHECKPOINT (or H3_CHECKPOINT): merged acceleration DiT weight;
    default: transformer_lora600_v4 used by the reviewed reference videos
  H3_NUM_INFERENCE_STEPS (default: 8), H3_SEED_BASE (default: 1000)
  H3_SKIP_EXISTING (default: 1)
  H3_RUNNER (default: run_30s_local_cp_infer.sh)
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
mkdir -p "$OUTPUT_DIR/30s" "$OUTPUT_DIR/60s" "$PLAN_DIR/30s" "$PLAN_DIR/60s"

# Materialize text prompts and continuation plans from one reviewed manifest.
python - "$MANIFEST" "$PLAN_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, staging = map(Path, sys.argv[1:])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(payload.get("single_shot_30s"), list) or not isinstance(payload.get("single_shot_60s"), list):
    raise SystemExit("manifest must contain single_shot_30s and single_shot_60s lists")
if len(payload["single_shot_30s"]) != 5 or len(payload["single_shot_60s"]) != 5:
    raise SystemExit("manifest must contain exactly five 30s prompts and five 60s plans")

for directory, suffix in ((staging / "30s", ".txt"), (staging / "30s", ".json"), (staging / "60s", ".json")):
    for old_file in directory.glob(f"*{suffix}"):
        old_file.unlink()

def validate_id(item):
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id or Path(item_id).name != item_id:
        raise SystemExit(f"invalid prompt id: {item_id!r}")
    return item_id

for item in payload["single_shot_30s"]:
    item_id = validate_id(item)
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit(f"{item_id}: prompt must be a non-empty string")
    plan = {
        "plan_id": item_id,
        "global_prompt": prompt.strip(),
        "segments": [
            {"segment_id": "window_01", "requested_frames": 362, "prompt": prompt.strip()},
            {
                "segment_id": "window_02",
                "requested_frames": 362,
                "prompt": "[Same shot, direct continuation] Continue from the exact final frame. "
                + prompt.strip(),
            },
        ],
    }
    destination = staging / "30s" / f"{item_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)

for item in payload["single_shot_60s"]:
    item_id = validate_id(item)
    global_prompt = item.get("global_prompt")
    segments = item.get("segments")
    if not isinstance(global_prompt, str) or not global_prompt.strip() or not isinstance(segments, list) or not segments:
        raise SystemExit(f"{item_id}: invalid global_prompt or segments")
    if len(segments) != 5:
        raise SystemExit(f"{item_id}: the reviewed 60s profile requires exactly five windows")
    normalized_segments = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not isinstance(segment.get("prompt"), str) or not segment["prompt"].strip():
            raise SystemExit(f"{item_id}: every segment needs a non-empty prompt")
        normalized_segments.append({
            "segment_id": str(segment.get("segment_id") or f"window_{index + 1:02d}"),
            "requested_frames": 345,
            "prompt": segment["prompt"].strip(),
        })
    plan = {"plan_id": item_id, "global_prompt": global_prompt.strip(), "segments": normalized_segments}
    destination = staging / "60s" / f"{item_id}.json"
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
  echo "[run]  $label (seed=$seed)"
  printf '       %q ' "${command[@]}"
  printf '\n'
  ((DRY_RUN)) && return 0
  "${command[@]}"
}

seed=$SEED_BASE
shopt -s nullglob
for plan_file in "$PLAN_DIR/30s"/*.json; do
  [[ -s "$plan_file" ]] || { echo "continuation plan missing or empty: $plan_file" >&2; exit 1; }
  id="$(basename "$plan_file" .json)"
  run_one "$id" "$OUTPUT_DIR/30s/$id.mp4" "$seed" \
    --continuation-plan "$plan_file" --continuation-window-frames 362 \
    --continuation-overlap-frames 39 --continuation-mode masked-av-v14 \
    --output_path "$OUTPUT_DIR/30s/$id.mp4"
  seed=$((seed + 1))
done

for plan_file in "$PLAN_DIR/60s"/*.json; do
  [[ -s "$plan_file" ]] || { echo "continuation plan missing or empty: $plan_file" >&2; exit 1; }
  id="$(basename "$plan_file" .json)"
  run_one "$id" "$OUTPUT_DIR/60s/$id.mp4" "$seed" \
    --continuation-plan "$plan_file" --continuation-window-frames 345 \
    --continuation-overlap-frames 39 --continuation-mode masked-av-v14 \
    --appearance-memory-mode dynamic --appearance-trusted-anchor-frames 2 \
    --appearance-memory-frames 0 --appearance-boundary-reference 1 \
    --appearance-max-visual-references 4 --height 832 --width 480 \
    --output-video-frames 1440 \
    --output_path "$OUTPUT_DIR/60s/$id.mp4"
  seed=$((seed + 1))
done

echo "batch complete: outputs under $OUTPUT_DIR"
