#!/usr/bin/env bash
# Run ONLY the AI2V multi-shot task with per-window image injection.
# Paths are hard-coded so interactive line-wrapping cannot corrupt them.
set -euo pipefail

cd /gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio

LORA="$PWD/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp4_v5_gas8/step-500.safetensors"
if [[ ! -f "$LORA" ]]; then
  echo "[error] LoRA not found: $LORA" >&2
  exit 1
fi

export RUN_T2VA=0
export H3_AI2V_INJECT_IMAGE=1
# Window-boundary shot plan: 0,3,6 -> three shots over windows 0-2 / 3-5 / 6-7.
# Inside a shot windows keep the masked 39-frame continuation; at each listed
# window the shot hard-cuts (no latent tail) while the first-frame image keeps
# the person/scene consistent. Override with H3_AI2V_SHOT_WINDOWS, e.g. "0,4"
# for two ~50s shots or "0,2,4,6" for four ~25s shots.
# NOTE: use ${VAR-default} (not :-) so an explicitly empty H3_AI2V_SHOT_WINDOWS=""
# disables window-boundary cuts and keeps the legacy behaviour.
export H3_AI2V_SHOT_WINDOWS="${H3_AI2V_SHOT_WINDOWS-0,3,6}"
export H3_AI2V_OUT="$PWD/outputs/h3_ai2v_song100s_multishot_injectimg.mp4"
exec bash examples/minimax_h3/model_inference/run_h3_60s_batch.sh
