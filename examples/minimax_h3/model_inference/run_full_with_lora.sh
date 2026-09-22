#!/usr/bin/env bash
# Run the FULL H3 batch (T2VA 5 prompts with continuation LoRA + AI2V
# multi-shot with per-window image injection) at the verified config.
# Paths are hard-coded so interactive line-wrapping cannot corrupt them.
set -euo pipefail

cd /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio

LORA="$PWD/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp4_v5_gas8/step-500.safetensors"
if [[ ! -f "$LORA" ]]; then
  echo "[error] LoRA not found: $LORA" >&2
  exit 1
fi

export H3_T2VA_LORA="$LORA"
export H3_T2VA_LORA_SCALE=0.7
export H3_AI2V_INJECT_IMAGE=1
export H3_SKIP_EXISTING=0
export H3_AI2V_OUT="$PWD/outputs/h3_ai2v_song100s_multishot_injectimg.mp4"

exec bash examples/minimax_h3/model_inference/run_h3_60s_batch.sh
