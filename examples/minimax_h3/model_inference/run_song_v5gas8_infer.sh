#!/usr/bin/env bash
# MiniMax-H3 AI2V 100s 推理：v5_gas8 LoRA (step-500) + lora-scale + camera-mode 可调。
# 用法:
#   bash examples/minimax_h3/model_inference/run_song_v5gas8_infer.sh
#                                      # scale 0.7, camera fixed (locked-off)
#   LORA_SCALE=0.8 bash ...run_song_v5gas8_infer.sh
#   CAMERA_MODE=normal bash ...        # normal music-video camera motion
#   CAMERA_MODE=multi  bash ...        # multi-shot (Shot 1..N hard cuts)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3ROOT=/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA
IMG='/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/data/diffsynth_example_dataset/化风行万里/4.jpg'
WAV=/tmp/h3_song_100s.wav
LORA="$REPO_ROOT/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp4_v5_gas8/step-500.safetensors"
VENV=/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python

LORA_SCALE="${LORA_SCALE:-0.7}"
# Camera mode: fixed (default) | normal (--normal-camera-motion) | multi (--multi-shot)
CAMERA_MODE="${CAMERA_MODE:-fixed}"
CAMERA_ARG=()
case "$CAMERA_MODE" in
  fixed)  ;;
  normal) CAMERA_ARG=(--normal-camera-motion) ;;
  multi)  CAMERA_ARG=(--multi-shot) ;;
  *) echo "[error] unknown CAMERA_MODE=$CAMERA_MODE (use fixed|normal|multi)" >&2; exit 1 ;;
esac
OUTPUT="${OUTPUT:-$REPO_ROOT/outputs/h3_ai2v_song100s_v5gas8_cam${CAMERA_MODE}_s500_ls${LORA_SCALE}.mp4}"

# ---- 输入检查 ----
for p in "$H3ROOT" "$H3ROOT/transformer_lora600_v4" "$IMG" "$WAV" "$LORA"; do
  if [[ ! -e "$p" ]]; then
    echo "[error] missing: $p" >&2
    exit 1
  fi
done

echo "[run] LORA=$LORA"
echo "[run] LORA_SCALE=$LORA_SCALE"
echo "[run] CAMERA_MODE=$CAMERA_MODE"
echo "[run] OUTPUT=$OUTPUT"

env PYTHONPATH="$REPO_ROOT" CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"$VENV" -u "$REPO_ROOT/examples/minimax_h3/model_inference/MiniMax-H3-AI2V-Masked-1min.py" \
  --h3-root "$H3ROOT" \
  --checkpoint "$H3ROOT/transformer_lora600_v4" \
  --image "$IMG" \
  --audio "$WAV" \
  --lora "$LORA" \
  --lora-scale "$LORA_SCALE" \
  --height 960 --width 544 \
  --window-frames 345 --overlap-frames 39 \
  --windows 8 --steps 8 --seed 0 \
  --match-audio-duration \
  "${CAMERA_ARG[@]}" \
  --output "$OUTPUT"
