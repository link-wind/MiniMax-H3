#!/usr/bin/env bash
set -euo pipefail

# Long-video continuation comparison of several continuation LoRAs
# (typically base / v5 / v9) over a suite of segment plans.
#
# One job is one (arm, plan) pair and owns one GPU end to end: a single-process
# pipeline, no rendezvous, no collective anywhere.  Nothing is shared between
# jobs, so there is nothing to coordinate -- each machine is simply told which
# arms and which plans it is responsible for and runs them on its own 8 GPUs.
# No node index, no rank, no cross-machine anything.
#
# One command per machine, no line continuations, nothing to type but the split:
#
#   SPLIT=a bash run_lora_compare_eval_16gpu.sh     # on machine A
#   SPLIT=b bash run_lora_compare_eval_16gpu.sh     # on machine B
#
# SPLIT is which slice of the plan grid this machine owns: 'a' and 'b' are the
# two halves, 'all' (the default) is the whole grid on one machine, and a
# comma-separated list of catalogue tags works too.  The arms are baked in
# (base / v5 / v9).  If the job grid is larger than the GPU count the script
# queues it: GPUS_PER_NODE jobs run at a time and the next one starts as soon as
# a GPU frees up.
#
# Optional:
#   SPLIT       a | b | all | tag,tag,...  default all (see the catalogue below)
#   LORA_ARMS   NAME=PATH[@SCALE],...      default base=,v5=<...>,v9=<...>
#               An empty PATH is the un-adapted base model; SCALE defaults to 1.0.
#   PLANS       TAG=PATH,...               overrides SPLIT for a one-off plan
#   GPUS_PER_NODE                          default 8 (also the concurrency limit)
#   PROMPT_MODE full|no-subject            default full
#   HEIGHT, WIDTH, NUM_INFERENCE_STEPS, SEED, OVERLAP_FRAMES, SAVE_VIDEO, FORCE
#   RUN_TAG     log file suffix           default the hostname, so two machines
#                                        writing the same OUTPUT_ROOT do not fight
#                                        over one log
#
# Every arm runs with memory *off*.  v5/v9 are plain continuation LoRAs trained
# without memory slots, so turning the injector on would condition them on
# something they never saw and the difference would stop being about training.
#
# What the v5 / v9 pair does and does not control (read off the two
# training_args.json files, not assumed):
#
#   property            v5 (40 691 samples)   v9 (99 936 samples)
#   epochs              1                     1
#   lora rank           32                    32
#   learning rate       1e-5                  1e-5
#   effective batch     cp4 x gas8  = 32      cp2 x gas4  = 32
#
# So epochs, effective batch, rank and LR all match and both runs are exactly one
# pass over their data: v9 simply saw 2.46x more distinct samples.  The one hard
# confound is the prefix presentation.  v5's training_args.json has no
# ``continuation_prefix_present_mode`` key, so it ran the old default ``noised``,
# while the inference path (masked-av-v14) feeds the overlap prefix as a *clean*
# latent -- a train/inference gap -- and v9 was trained with ``mixed``, which
# exists precisely to close that gap.  A v9 win is therefore consistent with
# either more data or a better-matched objective.  Separating them needs a
# retrain at a fixed prefix mode; report this as "the v9 recipe vs the v5
# recipe", never as "100k beats 40k".

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"

export PATH="${H3_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

H3_MODEL_ROOT="${H3_MODEL_ROOT:-/gemini/platform/public/aigc/human_guozz2/model/MiniMaxAI}"
H3_BASE="${H3_BASE:-${H3_MODEL_ROOT}/MiniMaxH3/FL2VA}"
TRANSFORMER_DIR="${TRANSFORMER_DIR:-${H3_BASE}/transformer_lora600_v4}"
# A sharded transformer is one model.  The glob stays unexpanded here and is
# expanded where it reaches the command line, because the loader has to receive
# the *files* -- the literal 'model*.safetensors' is opened as a path and fails.
CHECKPOINT="${CHECKPOINT:-${TRANSFORMER_DIR}/model*.safetensors}"
if ! compgen -G "${CHECKPOINT}" > /dev/null; then
  echo "[error] no H3 transformer checkpoint matched: ${CHECKPOINT}" >&2
  exit 1
fi

# ---- 预设：不传环境变量也能一行命令跑起来 ------------------------------- #
# The arms are fixed for this comparison, so they live here rather than in the
# command line.  An empty path is the un-adapted base model; v5/v9 are the two
# continuation LoRAs under comparison.  Set LORA_ARMS to override.
LORA_ARMS="${LORA_ARMS:-base=,v5=${REPO_ROOT}/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp4_v5_gas8/step-1271.safetensors,v9=${REPO_ROOT}/outputs/continuation_lora/h3_continuation_lora_mask_v14_2n16g_cp2_v9_mixedprefix_100k_lambda0p5/step-3123.safetensors}"

# Plan catalogue.  Tags are what shows up in the report, so they are stable.
PLAN_DIR="${PLAN_DIR:-${REPO_ROOT}/examples/minimax_h3/model_inference}"
declare -A plan_path_of=(
  [trainstyle65s]="${PLAN_DIR}/h3_continuation_plan_60s_trainstyle_shots.json"
  [single345]="${PLAN_DIR}/h3_continuation_plan_345_single_shot.json"
  [trainstyle345]="${PLAN_DIR}/h3_continuation_plan_345_trainstyle_shots.json"
  [singing60s]="${PLAN_DIR}/h3_continuation_plan_60s_singing_shots.json"
  [singing345]="${PLAN_DIR}/h3_continuation_plan_345_singing_shots.json"
)

# SPLIT picks which plans this machine owns, so the two machines can be told
# apart with one short word instead of a repeated wall of paths:
#   SPLIT=a     2 plans, 6 jobs    <- one machine
#   SPLIT=b     2 plans, 6 jobs    <- the other machine
#   SPLIT=all   the full grid      <- default, one machine
#   SPLIT=trainstyle65s,single345  any comma-separated list of catalogue tags
# PLANS still overrides everything, for a one-off plan outside the catalogue.
if [[ -z "${PLANS:-}" ]]; then
  case "${SPLIT:-all}" in
    all) plan_tags="trainstyle65s single345 trainstyle345 singing60s singing345" ;;
    a)   plan_tags="trainstyle65s single345" ;;
    b)   plan_tags="trainstyle345 singing60s" ;;
    *)   plan_tags="${SPLIT//,/ }" ;;
  esac
  PLANS=""
  for tag in ${plan_tags}; do
    if [[ -z "${plan_path_of[${tag}]:-}" ]]; then
      echo "[error] unknown SPLIT plan tag '${tag}'; known: ${!plan_path_of[*]}" >&2
      exit 1
    fi
    PLANS="${PLANS:+${PLANS},}${tag}=${plan_path_of[${tag}]}"
  done
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PROMPT_MODE="${PROMPT_MODE:-full}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
# 8, not the pipeline default of 50: the continuation LoRAs were trained on top of
# transformer_lora600_v4, the distilled base whose reviewed references are
# generated at 8 steps.  Sampling at 50 would push every arm off its training
# distribution and cost six times the wall clock.
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
# 39 frames = 12 latent steps = the exact joint H3 head masked-av-v14 needs.
OVERLAP_FRAMES="${OVERLAP_FRAMES:-39}"
SEED="${SEED:-42}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/continuation_lora/lora_compare}"
RUN_TAG="${RUN_TAG:-$(hostname)}"
# Overridable so the queueing itself can be exercised with a stub in place of the
# 14B pipeline, which needs a GPU and half an hour to say anything.
SCRIPT="${SCRIPT:-${REPO_ROOT}/examples/minimax_h3/model_training/eval_memory_rollout.py}"
mkdir -p "${OUTPUT_ROOT}"
LOG_PATH="${OUTPUT_ROOT}/lora_compare.${RUN_TAG}.log"

# ---- 展开 job 网格：臂外层、plan 内层 ---------------------------------- #
declare -A arm_spec_of=()   # 臂名 -> 单臂 NAME=PATH[@SCALE] 规格
jobs=()                     # 每项 "arm|plan_tag|plan_path"
IFS=',' read -r -a arm_list <<< "${LORA_ARMS}"
IFS=',' read -r -a plan_list <<< "${PLANS}"
for arm in "${arm_list[@]}"; do
  arm="$(echo "${arm}" | sed 's/^ *//; s/ *$//')"
  [[ -z "${arm}" ]] && continue
  if [[ "${arm}" != *=* ]]; then
    echo "[error] LORA_ARMS entries need NAME=PATH[@SCALE], got '${arm}'" >&2
    exit 1
  fi
  arm_name="${arm%%=*}"
  if [[ -n "${arm_spec_of[${arm_name}]:-}" ]]; then
    echo "[error] LORA_ARMS repeats arm '${arm_name}'" >&2
    exit 1
  fi
  arm_spec_of["${arm_name}"]="${arm}"
  for plan in "${plan_list[@]}"; do
    plan="$(echo "${plan}" | sed 's/^ *//; s/ *$//')"
    [[ -z "${plan}" ]] && continue
    tag="${plan%%=*}"; path="${plan#*=}"
    if [[ "${tag}" == "${plan}" ]]; then
      echo "[error] PLANS entries need TAG=PATH, got '${plan}'" >&2
      exit 1
    fi
    if [[ ! -f "${path}" ]]; then
      echo "[error] plan not found: ${path}" >&2
      exit 1
    fi
    jobs+=("${arm_name}|${tag}|${path}")
  done
done
if [[ ${#jobs[@]} -eq 0 ]]; then
  echo "[error] empty job grid" >&2
  exit 1
fi

{
  echo "[lora-compare] started $(date -u +%FT%TZ) host=${RUN_TAG}"
  echo "[lora-compare] gpus=${GPUS_PER_NODE} jobs=${#jobs[@]} prompt_mode=${PROMPT_MODE}"
  echo "[lora-compare] resolution=${HEIGHT}x${WIDTH} steps=${NUM_INFERENCE_STEPS} overlap=${OVERLAP_FRAMES} seed=${SEED} save_video=${SAVE_VIDEO}"
  echo "[lora-compare] checkpoint=$(compgen -G "${CHECKPOINT}" | wc -l) shard(s)"
  echo "[lora-compare] lora_arms=${LORA_ARMS}"
  echo "[lora-compare] plans=${PLANS}"
  echo "[lora-compare] output=${OUTPUT_ROOT}"
} | tee -a "${LOG_PATH}"

# ---- 待跑清单：跳过已完成的 -------------------------------------------- #
pending=()
skipped=0
for job in "${jobs[@]}"; do
  IFS='|' read -r arm tag plan_path <<< "${job}"
  job_out="${OUTPUT_ROOT}/${arm}/${tag}"
  mkdir -p "${job_out}"
  if [[ "${FORCE}" != "1" && -f "${job_out}/rollout_ablation_${arm}.json" ]]; then
    # SAVE_VIDEO=1 时"已完成"还得包含视频本身。早期的对比跑是 SAVE_VIDEO=0，只留了 json
    # 没留 mp4；只看 json 的话这些 job 会被全部跳过，VBench 阶段就永远等不到输入。
    if [[ "${SAVE_VIDEO}" != "1" ]] || compgen -G "${job_out}/videos/*.mp4" > /dev/null; then
      echo "[lora-compare] skip  ${arm}/${tag}（已有结果，FORCE=1 可重跑）" | tee -a "${LOG_PATH}"
      skipped=$((skipped + 1))
      continue
    fi
    echo "[lora-compare] redo  ${arm}/${tag}（有 json 但没有 videos/*.mp4，SAVE_VIDEO=1 需要视频）" | tee -a "${LOG_PATH}"
  fi
  pending+=("${job}")
done

if [[ "${DRY_RUN}" == "1" ]]; then
  index=0
  for job in "${pending[@]+"${pending[@]}"}"; do
    IFS='|' read -r arm tag plan_path <<< "${job}"
    gpu=$(( index % GPUS_PER_NODE ))
    echo "[lora-compare] gpu ${gpu} -> ${arm} / ${tag}" | tee -a "${LOG_PATH}"
    echo "[lora-compare]   CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${SCRIPT} \\\\" | tee -a "${LOG_PATH}"
    echo "[lora-compare]     --segment-plan ${plan_path} --checkpoint <${CHECKPOINT##*/} x $(compgen -G "${CHECKPOINT}" | wc -l)> --h3-base ${H3_BASE} \\\\" | tee -a "${LOG_PATH}"
    echo "[lora-compare]     --lora-arms '${arm_spec_of[${arm}]}' --prompt-mode ${PROMPT_MODE} --overlap-frames ${OVERLAP_FRAMES} \\\\" | tee -a "${LOG_PATH}"
    echo "[lora-compare]     --height ${HEIGHT} --width ${WIDTH} --num-inference-steps ${NUM_INFERENCE_STEPS} --seed ${SEED} \\\\" | tee -a "${LOG_PATH}"
    echo "[lora-compare]     --output-dir ${OUTPUT_ROOT}/${arm}/${tag}" | tee -a "${LOG_PATH}"
    index=$((index + 1))
  done
  echo "[lora-compare] dry-run: would run ${index} job(s), ${skipped} already done" | tee -a "${LOG_PATH}"
  exit 0
fi

# ---- 本地 GPU 排队执行 ------------------------------------------------- #
declare -A gpu_of_pid=() label_of_pid=()
free_gpus=()
for ((gpu = 0; gpu < GPUS_PER_NODE; gpu++)); do free_gpus+=("${gpu}"); done
failed=()
completed=0

reap_one() {
  # 收割一个已完成子进程，把它占的卡放回空闲池
  local finished="" gpu=""
  if wait -n -p finished; then :; else
    failed+=("${label_of_pid[${finished}]:-unknown}")
  fi
  gpu="${gpu_of_pid[${finished}]:-}"
  unset "gpu_of_pid[${finished}]" "label_of_pid[${finished}]"
  [[ -n "${gpu}" ]] && free_gpus+=("${gpu}")
  completed=$((completed + 1))
}

for job in "${pending[@]+"${pending[@]}"}"; do
  IFS='|' read -r arm tag plan_path <<< "${job}"
  # 等一张空闲卡；队列满时阻塞在 wait
  while [[ ${#free_gpus[@]} -eq 0 ]]; do
    reap_one
  done
  gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")

  job_out="${OUTPUT_ROOT}/${arm}/${tag}"
  arm_spec="${arm_spec_of[${arm}]:-}"
  if [[ -z "${arm_spec}" ]]; then
    echo "[error] arm '${arm}' is not in LORA_ARMS" >&2
    exit 1
  fi
  extra_args=()
  [[ "${SAVE_VIDEO}" == "1" ]] && extra_args+=(--save-video)

  echo "[lora-compare] gpu ${gpu} -> ${arm} / ${tag}" | tee -a "${LOG_PATH}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT}" \
    --segment-plan "${plan_path}" \
    --checkpoint ${CHECKPOINT} \
    --h3-base "${H3_BASE}" \
    --lora-arms "${arm_spec}" \
    --prompt-mode "${PROMPT_MODE}" \
    --overlap-frames "${OVERLAP_FRAMES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --seed "${SEED}" \
    --output-dir "${job_out}" \
    ${extra_args[@]+"${extra_args[@]}"} \
    >> "${job_out}/run.log" 2>&1 &
  pid=$!
  gpu_of_pid["${pid}"]="${gpu}"
  label_of_pid["${pid}"]="${arm}/${tag}@gpu${gpu}"
done

while [[ ${#gpu_of_pid[@]} -gt 0 ]]; do
  reap_one
done

status=0
if [[ ${#failed[@]} -gt 0 ]]; then
  status=1
  echo "[lora-compare] failed: ${failed[*]}" | tee -a "${LOG_PATH}"
  echo "[lora-compare] 看看 ${OUTPUT_ROOT}/<arm>/<plan>/run.log" | tee -a "${LOG_PATH}"
fi
echo "[lora-compare] finished $(date -u +%FT%TZ) status=${status} completed=${completed} skipped=${skipped}" | tee -a "${LOG_PATH}"
echo "[lora-compare] 汇总：OUTPUT_ROOT=${OUTPUT_ROOT} bash ${REPO_ROOT}/examples/minimax_h3/model_training/run_lora_compare_report.sh" | tee -a "${LOG_PATH}"
exit "${status}"
