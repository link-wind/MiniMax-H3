#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT=/gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio
export H3_CP_SIZE=2
exec bash "$REPO_ROOT/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_smoke.sh"
