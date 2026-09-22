#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT=/gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio
export H3_CP_SIZE=2
export H3_DATASET_REPEAT=20
export H3_SAVE_STEPS=10
export H3_OUTPUT_PATH="$REPO_ROOT/models/train/MiniMax-H3-30s-multi-3node-cp2-full-repeat20-no-cpu-checkpoint"
export H3_DS_CONFIG="$REPO_ROOT/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8_no_cpu_checkpoint.json"
exec bash "$REPO_ROOT/examples/minimax_h3/model_training/gemini_30s_sft_3node_24gpu_full.sh"
