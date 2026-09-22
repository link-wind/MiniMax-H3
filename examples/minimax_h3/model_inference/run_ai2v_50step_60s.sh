#!/usr/bin/env bash
# AI2V 60s @ 50 diffusion steps using the step-1000 continuation LoRA (scale 1.0).
# Multi-shot + per-window image injection. 5 windows (~63s, cropped to the 60s audio).
set -euo pipefail

cd /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio

H3ROOT=/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA
LORA="${H3_AI2V_LORA:-$PWD/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp2_v9_mixedprefix_100k_lambda0p5/step-1000.safetensors}"
LORA_SCALE="${H3_AI2V_LORA_SCALE:-1.0}"
IMG='/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/data/diffsynth_example_dataset/化风行万里/4.jpg'
SRC_WAV=/tmp/h3_song_100s.wav
WAV="${H3_AI2V_60S_WAV:-/tmp/h3_song_60s.wav}"   # 60s audio
VENV=/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python
OUT="${H3_AI2V_OUT:-$PWD/outputs/h3_ai2v_v9s1000_ls1p0_60s_50step.mp4}"

# 1) Crop source audio to 60s (idempotent).
[[ -f "$WAV" ]] || ffmpeg -y -loglevel error -i "$SRC_WAV" -t 60 -ar 32000 -ac 2 "$WAV"

for p in "$H3ROOT/transformer_lora600_v4" "$LORA" "$IMG" "$WAV"; do
  [[ -e "$p" ]] || { echo "[error] missing: $p" >&2; exit 1; }
done

echo "[run] LoRA=$LORA scale=$LORA_SCALE steps=50 windows=5"
echo "[run] audio=$WAV out=$OUT"

env PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="${H3_AI2V_GPU:-0}" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$VENV" -u examples/minimax_h3/model_inference/MiniMax-H3-AI2V-Masked-1min.py \
    --h3-root "$H3ROOT" \
    --checkpoint "$H3ROOT/transformer_lora600_v4" \
    --image "$IMG" \
    --audio "$WAV" \
    --lora "$LORA" \
    --lora-scale "$LORA_SCALE" \
    --height 960 --width 544 \
    --window-frames 345 --overlap-frames 39 \
    --windows 5 --steps 50 --seed 0 \
    --multi-shot \
    --inject-image-every-window \
    --match-audio-duration \
    --output "$OUT"
