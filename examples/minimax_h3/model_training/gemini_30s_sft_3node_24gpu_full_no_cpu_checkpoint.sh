#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT=/gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio
export H3_DS_CONFIG="$REPO_ROOT/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8_no_cpu_checkpoint.json"
exec bash "$REPO_ROOT/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_full.sh"
