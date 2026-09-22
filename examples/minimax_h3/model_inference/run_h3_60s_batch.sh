#!/usr/bin/env bash
# Unified 60s H3 batch: runs the 5 T2VA prompts (single-shot) AND the AI2V
# multi-shot song task (化风行万里 100s), all at 768x1344 landscape.
#
#   bash examples/minimax_h3/model_inference/run_h3_60s_batch.sh
#   RUN_T2VA=0 bash ...run_h3_60s_batch.sh        # AI2V only
#   RUN_AI2V=0 bash ...run_h3_60s_batch.sh        # T2VA only
#   H3_T2VA_MANIFEST=...multishot.json bash ...   # use a multi-shot T2VA manifest
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3ROOT=/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA
VENV=/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python

# T2VA section resolution (landscape 1344x768, divisible by 32).
HEIGHT="${H3_HEIGHT:-768}"
WIDTH="${H3_WIDTH:-1344}"

# AI2V section uses the input image's native portrait resolution
# (549x976 -> aligned to 544x960, divisible by 32).
AI2V_HEIGHT="${H3_AI2V_HEIGHT:-960}"
AI2V_WIDTH="${H3_AI2V_WIDTH:-544}"

# Which sections to run (default both).
RUN_T2VA="${RUN_T2VA:-1}"
RUN_AI2V="${RUN_AI2V:-1}"

# T2VA section tuning.
H3_T2VA_MANIFEST="${H3_T2VA_MANIFEST:-$REPO_ROOT/examples/minimax_h3/model_inference/prompts/h3_t2va_60s_prompts.json}"
T2VA_STEPS="${H3_NUM_INFERENCE_STEPS:-8}"
T2VA_SEED="${H3_SEED_BASE:-2000}"
T2VA_CHECKPOINT="${H3_MERGED_CHECKPOINT:-${H3_CHECKPOINT:-$H3ROOT/transformer_lora600_v4}}"
T2VA_LORA="${H3_T2VA_LORA:-}"
T2VA_LORA_SCALE="${H3_T2VA_LORA_SCALE:-0.7}"

# AI2V multi-shot section tuning.
AI2V_IMG="${H3_AI2V_IMG:-/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/data/diffsynth_example_dataset/化风行万里/4.jpg}"
AI2V_WAV="${H3_AI2V_WAV:-/tmp/h3_song_100s.wav}"
AI2V_LORA="${H3_AI2V_LORA:-$REPO_ROOT/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp4_v5_gas8/step-500.safetensors}"
AI2V_LORA_SCALE="${H3_AI2V_LORA_SCALE:-0.7}"
AI2V_STEPS="${H3_AI2V_STEPS:-8}"
AI2V_WINDOWS="${H3_AI2V_WINDOWS:-8}"
AI2V_OUT="${H3_AI2V_OUT:-$REPO_ROOT/outputs/h3_ai2v_song100s_multishot_960x544.mp4}"
AI2V_GPU="${H3_AI2V_GPU:-0}"
AI2V_INJECT_IMAGE="${H3_AI2V_INJECT_IMAGE:-0}"
AI2V_SHOT_WINDOWS="${H3_AI2V_SHOT_WINDOWS:-}"

usage() {
  cat <<'EOF'
Unified 60s H3 batch (T2VA 5 prompts + AI2V multi-shot) at 768x1344.

Environment overrides:
  H3_HEIGHT (768), H3_WIDTH (1344)  # T2VA resolution
  H3_AI2V_HEIGHT (960), H3_AI2V_WIDTH (544)  # AI2V image-native portrait
  RUN_T2VA (1), RUN_AI2V (1)
  H3_T2VA_MANIFEST, H3_NUM_INFERENCE_STEPS (8), H3_SEED_BASE (2000)
  H3_MERGED_CHECKPOINT / H3_CHECKPOINT
  H3_T2VA_LORA (default: none), H3_T2VA_LORA_SCALE (default: 0.7)
  H3_AI2V_IMG, H3_AI2V_WAV, H3_AI2V_LORA, H3_AI2V_LORA_SCALE (0.7)
  H3_AI2V_STEPS (8), H3_AI2V_WINDOWS (8), H3_AI2V_OUT, H3_AI2V_GPU (0)
  H3_AI2V_INJECT_IMAGE (0; 1 = inject first-frame image every window)
  H3_AI2V_SHOT_WINDOWS (empty; e.g. "0,3,6" = window-boundary hard cuts with intra-shot continuation)
EOF
}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo "==== H3 60s unified batch @ ${WIDTH}x${HEIGHT} ===="

# ---------- Section 1: T2VA 5 prompts ----------
if [[ "$RUN_T2VA" == "1" ]]; then
  echo ""
  echo "==== [1/2] T2VA batch (5 prompts) ===="
  env H3_HEIGHT="$HEIGHT" H3_WIDTH="$WIDTH" \
    H3_PROMPT_MANIFEST="$H3_T2VA_MANIFEST" \
    H3_MERGED_CHECKPOINT="$T2VA_CHECKPOINT" \
    H3_NUM_INFERENCE_STEPS="$T2VA_STEPS" \
    H3_SEED_BASE="$T2VA_SEED" \
    H3_T2VA_LORA="$T2VA_LORA" H3_T2VA_LORA_SCALE="$T2VA_LORA_SCALE" \
    CUDA_VISIBLE_DEVICES=0 \
    bash "$REPO_ROOT/examples/minimax_h3/model_inference/run_t2va_60s_batch.sh"
fi

# ---------- Section 2: AI2V multi-shot 100s ----------
if [[ "$RUN_AI2V" == "1" ]]; then
  echo ""
  echo "==== [2/2] AI2V multi-shot (化风行万里 100s) ===="
  for p in "$H3ROOT" "$H3ROOT/transformer_lora600_v4" "$AI2V_IMG" "$AI2V_WAV" "$AI2V_LORA"; do
    if [[ ! -e "$p" ]]; then
      echo "[error] missing: $p" >&2
      exit 1
    fi
  done
  inject_args=()
  if [[ "$AI2V_INJECT_IMAGE" == "1" ]]; then
    inject_args+=(--inject-image-every-window)
  fi
  shot_args=()
  if [[ -n "$AI2V_SHOT_WINDOWS" ]]; then
    shot_args+=(--shot-window-starts "$AI2V_SHOT_WINDOWS")
  fi
  env PYTHONPATH="$REPO_ROOT" CUDA_VISIBLE_DEVICES="$AI2V_GPU" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$VENV" -u "$REPO_ROOT/examples/minimax_h3/model_inference/MiniMax-H3-AI2V-Masked-1min.py" \
      --h3-root "$H3ROOT" \
      --checkpoint "$H3ROOT/transformer_lora600_v4" \
      --image "$AI2V_IMG" \
      --audio "$AI2V_WAV" \
      --lora "$AI2V_LORA" \
      --lora-scale "$AI2V_LORA_SCALE" \
      --height "$AI2V_HEIGHT" --width "$AI2V_WIDTH" \
      --window-frames 345 --overlap-frames 39 \
      --windows "$AI2V_WINDOWS" --steps "$AI2V_STEPS" --seed 0 \
      --multi-shot \
      "${inject_args[@]}" \
      "${shot_args[@]}" \
      --match-audio-duration \
      --output "$AI2V_OUT"
fi

echo ""
echo "==== H3 60s unified batch complete ===="
