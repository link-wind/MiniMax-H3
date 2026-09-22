#!/usr/bin/env bash
# AI2V capability test for the half-trained v9 mixedprefix checkpoint
# (step-1000) at LoRA scale 1.0. Multi-shot, per-window image injection.
set -euo pipefail

cd /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio

H3ROOT=/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA
LORA="${H3_AI2V_LORA:-$PWD/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp2_v9_mixedprefix_100k_lambda0p5/step-1000.safetensors}"
LORA_SCALE="${H3_AI2V_LORA_SCALE:-1.0}"
STEPS="${H3_AI2V_STEPS:-8}"
GPU="${H3_AI2V_GPU:-0}"
HEIGHT="${H3_AI2V_HEIGHT:-960}"
WIDTH="${H3_AI2V_WIDTH:-544}"
REFERENCE_SHORT_EDGE="${H3_AI2V_REFERENCE_SHORT_EDGE:-256}"
ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
IMG='/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/data/diffsynth_example_dataset/化风行万里/4.jpg'
WAV=/tmp/h3_song_100s.wav
VENV=/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python

for p in "$H3ROOT/transformer_lora600_v4" "$LORA" "$IMG" "$WAV"; do
  [[ -e "$p" ]] || { echo "[error] missing: $p" >&2; exit 1; }
done

OUT="${H3_TEST_OUT:-$PWD/outputs/h3_ai2v_v9s1000_ls1p0_multishot_inject.mp4}"
echo "[run] LoRA=$LORA"
echo "[run] scale=$LORA_SCALE steps=$STEPS gpu=$GPU size=${WIDTH}x${HEIGHT}"
echo "[run] reference-short-edge=$REFERENCE_SHORT_EDGE out=$OUT"

env PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_ALLOC_CONF="$ALLOC_CONF" PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  "$VENV" -u examples/minimax_h3/model_inference/MiniMax-H3-AI2V-Masked-1min.py \
    --h3-root "$H3ROOT" \
    --checkpoint "$H3ROOT/transformer_lora600_v4" \
    --image "$IMG" \
    --audio "$WAV" \
    --lora "$LORA" \
    --lora-scale "$LORA_SCALE" \
    --height "$HEIGHT" --width "$WIDTH" \
    --reference-short-edge "$REFERENCE_SHORT_EDGE" \
    --window-frames 345 --overlap-frames 39 \
    --windows 8 --steps "$STEPS" --seed 0 \
    --multi-shot \
    --inject-image-every-window \
    --match-audio-duration \
    --output "$OUT"
