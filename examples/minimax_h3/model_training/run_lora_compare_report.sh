#!/usr/bin/env bash
set -euo pipefail

# Summarize a run_lora_compare_eval_16gpu.sh output tree into one table.
# Run it on either machine -- it only reads the output tree, no GPU needed.
#
#   OUTPUT_ROOT=outputs/continuation_lora/lora_compare bash run_lora_compare_report.sh
#   BASE=base bash run_lora_compare_report.sh        # which arm the Δ columns use

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/continuation_lora/lora_compare}"
BASE="${BASE:-base}"

"${PYTHON}" "${REPO_ROOT}/examples/minimax_h3/model_training/report_lora_compare.py" \
  --root "${OUTPUT_ROOT}" --base "${BASE}" "$@"
