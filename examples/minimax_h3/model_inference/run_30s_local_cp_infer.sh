#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv/bin/python"
NPROC_PER_NODE="${H3_NPROC_PER_NODE:-8}"
CP_WORLD_SIZE="${H3_CP_WORLD_SIZE:-8}"

cd "$REPO_ROOT"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PYTHONPATH="$REPO_ROOT" \
  "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port=29537 \
  examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-30s-local-cp.py \
  --cp_world_size "$CP_WORLD_SIZE" \
  "$@"
