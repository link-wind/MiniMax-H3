#!/usr/bin/env bash
set -euo pipefail

# Summarize a run_vbench_long_16gpu.sh tree into one table.
# Run it on either machine -- it only reads the output tree, no GPU needed.
#
#   OUTPUT_ROOT=outputs/continuation_lora/lora_compare bash run_vbench_long_report.sh
#   BASE=base bash run_vbench_long_report.sh        # which arm the Δ column uses

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# vbench 装在系统 python 上（不是 H3 那个 venv），汇总脚本也一样用它。
VBENCH_PYTHON="${VBENCH_PYTHON:-/usr/bin/python3}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/continuation_lora/lora_compare}"
VBENCH_OUTPUT="${VBENCH_OUTPUT:-${OUTPUT_ROOT}/vbench_long}"
BASE="${BASE:-base}"

"${VBENCH_PYTHON}" "${REPO_ROOT}/examples/minimax_h3/model_training/report_vbench_long.py" \
  --root "${VBENCH_OUTPUT}" --base "${BASE}" "$@"
