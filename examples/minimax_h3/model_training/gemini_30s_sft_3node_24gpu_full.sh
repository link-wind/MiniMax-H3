#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT=/gemini/platform/public/aigc/lss/Interns/lsj/DiffSynth-Studio
cd "$REPO_ROOT"

export H3_VENV=/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv
export DIFFSYNTH_MODEL_BASE_PATH=/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI
export DIFFSYNTH_SKIP_DOWNLOAD=True
export PATH="$H3_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export H3_CP_TRACE=1

export MASTER_ADDR="${GEMINI_IP_taskrole1_0:?GEMINI_IP_taskrole1_0 is required}"
export MASTER_PORT="${GEMINI_taskrole1_0_http_PORT:-29500}"
export NODE_RANK="${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:?GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX is required}"
export NNODES=3
export H3_CP_SIZE="${H3_CP_SIZE:-8}"
export H3_DATASET_REPEAT="${H3_DATASET_REPEAT:-3}"
export H3_SAVE_STEPS="${H3_SAVE_STEPS:-}"
export H3_OUTPUT_PATH="${H3_OUTPUT_PATH:-${GEMINI_DATA_OUT:-$REPO_ROOT/models/train/MiniMax-H3-30s-multi-3node-full-repeat${H3_DATASET_REPEAT:-3}}}"

rm -f "$REPO_ROOT/.deepspeed_env"

export ACCELERATE_USE_DEEPSPEED=true
export H3_DS_CONFIG="${H3_DS_CONFIG:-$REPO_ROOT/examples/minimax_h3/model_training/full/deepspeed_zero3_cp8.json}"
export ACCELERATE_DEEPSPEED_CONFIG_FILE="$H3_DS_CONFIG"
export ACCELERATE_MIXED_PRECISION=bf16
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export ACCELERATE_DEEPSPEED_ZERO3_INIT=true
export ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=true
export ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE=none
export ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE=none

save_steps_args=()
if [ -n "$H3_SAVE_STEPS" ]; then
  save_steps_args=(--save_steps "$H3_SAVE_STEPS")
fi

torchrun \
  --nnodes "$NNODES" \
  --nproc_per_node=8 \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$REPO_ROOT/examples/minimax_h3/model_training/train.py" \
  --dataset_base_path "$REPO_ROOT/models/train/MiniMax-H3-30s-multi-cache" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 719 \
  --dataset_repeat "$H3_DATASET_REPEAT" \
  --dataset_num_workers 8 \
  --model_id_with_origin_paths "MiniMaxH3:FL2VA/transformer/model*.safetensors" \
  --processor_path "/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI/MiniMaxH3/FL2VA/processor" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$H3_OUTPUT_PATH" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --initialize_model_on_cpu \
  --cp_world_size "$H3_CP_SIZE" \
  "${save_steps_args[@]}" \
  --task "sft:train"
