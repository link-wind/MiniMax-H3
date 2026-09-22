#!/usr/bin/env bash
set -euo pipefail

# Full VBench-Long run over the base / v5 / v9 continuation LoRAs.
#
# Two stages, both embarrassingly parallel and both free of cross-machine
# coordination -- a job owns one GPU end to end, there is no rendezvous and no
# collective anywhere, so two machines are used by telling each one which slice
# of the job grid it owns:
#
#   stage rollout : generate the videos with SAVE_VIDEO=1 (reuses
#                   run_lora_compare_eval_16gpu.sh; sharded by that script's SPLIT)
#   stage vbench  : run the *official* VBench-Long CLI, one (arm, plan, dimension)
#                   per job, on the mp4s the first stage produced
#
# The second stage needs every mp4, so the two machines must both finish stage
# rollout before either starts stage vbench.
#
# One command per machine, no line continuations:
#
#   SHARD_INDEX=0 SHARD_COUNT=2 SPLIT=a bash run_vbench_long_16gpu.sh   # machine A
#   SHARD_INDEX=1 SHARD_COUNT=2 SPLIT=b bash run_vbench_long_16gpu.sh   # machine B
#
# `SPLIT` only steers stage rollout (which plans this machine generates);
# `SHARD_INDEX/SHARD_COUNT` only steer stage vbench (which of the
# arm x plan x dimension jobs this machine scores).  A single machine can do
# everything with `SHARD_INDEX=0 SHARD_COUNT=1 SPLIT=all`.
#
# Why custom input and not the standard suite: we are not generating from
# VBench's 946 prompts, we are scoring our own rollouts, which is exactly
# `--mode long_custom_input`.  That mode supports 10 of the 16 dimensions;
# `VBenchLong.check_dimension_requires_extra_info` rejects the other six
# (object_class, multiple_objects, color, spatial_relationship, scene,
# appearance_style), so the very real GRiT/mmcv/detectron2 problem does not
# apply here.  `human_action` is excluded by default for a different reason:
# it reads the action label out of the *filename*, and our files are named after
# the arm, so it would report a constant 0.
#
# Optional:
#   STAGE             all | rollout | vbench      default all
#   DIMS              space list                 default the 9 dims below
#   SHARD_INDEX       int                        default 0
#   SHARD_COUNT       int                        default 1
#   GPUS_PER_NODE     int                        default 8 (also concurrency limit)
#   OUTPUT_ROOT       rollout tree               default <repo>/outputs/continuation_lora/lora_compare
#   VBENCH_OUTPUT     results tree               default $OUTPUT_ROOT/vbench_long
#   VBENCH_WORK       scratch tree               default $OUTPUT_ROOT/vbench_work
#   ARMS / PLANS      restrict stage vbench to a subset (default: all found)
#   EXPECTED_MP4S     how many videos stage vbench should see (default: no check).
#                     Set it when the two machines each generate half the videos:
#                     without it, a machine that starts scoring before its peer
#                     finished stage rollout silently drops the missing cells.
#   WAIT_MP4_SECONDS  how long to poll for EXPECTED_MP4S before giving up (default 0)
#   FLICKER_STATIC_FILTER 1|0                    default 1 (matches evaluate_long.sh)
#   LORA_ARMS / SPLIT / PLANS                    passed through to stage rollout
#   FORCE             1 to redo finished jobs    default 0
#   DRY_RUN           1 to print the plan only   default 0
#   VBENCH_RUNNER     path to a stub that replaces the VBench CLI (for tests)
#   VBENCH_PYTHON     python for the VBench stage      default /usr/bin/python3
#   VBENCH_ENV_ROOT   portable VBench tree            default <sj>/vbench-portable
#
# Note the two interpreters.  Stage rollout needs the MiniMax-H3 venv (torch +
# diffsynth); stage vbench needs the *system* python, because vbench 0.1.5 is
# deliberately not in that venv -- the image's torch .so live on the system
# LD_LIBRARY_PATH and a venv-local torch dies on _dlpack_exchange_api.
#
# It also has to find vbench at all.  /usr/local/lib/python3.10/dist-packages and
# ~/.cache live on the container's overlay, i.e. **node-local**: a machine that
# did not build this environment has neither the packages nor the weights, and
# "just pip install it once" does not fix that, because every job may start in a
# fresh container.  So the whole thing lives on the shared filesystem instead:
#
#   $VBENCH_ENV_ROOT/site-packages    vbench + deps, installed with --target
#   $VBENCH_ENV_ROOT/compat           sitecustomize: pkg_resources.packaging,
#                                     moviepy.editor
#   $VBENCH_ENV_ROOT/home/.cache      weights (~11 GB), reached by setting HOME
#
# PYTHONPATH is ordered so our copy wins over anything in the image, and HOME is
# overridden *for the VBench subprocess only* -- vbench resolves CACHE_DIR,
# dreamsim and torch.hub all out of ~/.cache.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
H3_VENV="${H3_VENV:-/gemini/platform/public/aigc/lss/Interns/lsj/MiniMax-H3/.venv-uv}"
PYTHON="${H3_VENV}/bin/python"
VBENCH_PYTHON="${VBENCH_PYTHON:-/usr/bin/python3}"
VBENCH_ENV_ROOT="${VBENCH_ENV_ROOT:-$(dirname "${REPO_ROOT}")/vbench-portable}"
VBENCH_PYTHONPATH="${VBENCH_ENV_ROOT}/site-packages:${VBENCH_ENV_ROOT}/compat"
VBENCH_HOME="${VBENCH_ENV_ROOT}/home"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${VBENCH_HOME}/.cache/vbench}"

export PATH="${H3_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

STAGE="${STAGE:-all}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
FLICKER_STATIC_FILTER="${FLICKER_STATIC_FILTER:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/continuation_lora/lora_compare}"
WAIT_MP4_SECONDS="${WAIT_MP4_SECONDS:-0}"
VBENCH_OUTPUT="${VBENCH_OUTPUT:-${OUTPUT_ROOT}/vbench_long}"
VBENCH_WORK="${VBENCH_WORK:-${OUTPUT_ROOT}/vbench_work}"
RUN_TAG="${RUN_TAG:-$(hostname)}"

# custom input 下允许的 10 维里去掉 human_action（标签取自文件名），剩下这 9 个。
DIMS="${DIMS:-subject_consistency background_consistency aesthetic_quality imaging_quality temporal_style overall_consistency temporal_flickering motion_smoothness dynamic_degree}"

ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-${REPO_ROOT}/examples/minimax_h3/model_training/run_lora_compare_eval_16gpu.sh}"
VCBENCH_ENTRY="vbench2_beta_long.eval_long"

if [[ "${SHARD_COUNT}" -lt 1 ]] || [[ "${SHARD_INDEX}" -lt 0 ]] || [[ "${SHARD_INDEX}" -ge "${SHARD_COUNT}" ]]; then
  echo "[error] need 0 <= SHARD_INDEX < SHARD_COUNT, got ${SHARD_INDEX}/${SHARD_COUNT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${VBENCH_OUTPUT}" "${VBENCH_WORK}"
LOG_PATH="${OUTPUT_ROOT}/vbench_long.${RUN_TAG}.log"

# 先确认这套环境真的能用。失败时最有用的信息是"哪个包从哪儿来"，
# 而不是五分钟后队列里的一堆 ImportError。
VBENCH_ENV_DUMP="$(PYTHONPATH="${VBENCH_PYTHONPATH}" HOME="${VBENCH_HOME}" \
  "${VBENCH_PYTHON}" -c 'import os, vbench, torch; print(vbench.__file__); print(torch.__version__)' 2>&1)" || {
  echo "[error] ${VBENCH_PYTHON} 里 import vbench / torch 失败：" >&2
  echo "${VBENCH_ENV_DUMP}" >&2
  cat >&2 <<HINT
[hint] 这一阶段要的是共享盘上的可移植环境，不是镜像里的 site-packages：
         VBENCH_PYTHON  = ${VBENCH_PYTHON}
         PYTHONPATH     = ${VBENCH_PYTHONPATH}
         HOME           = ${VBENCH_HOME}
       vbench 应该从 ${VBENCH_ENV_ROOT}/site-packages/vbench 被导入。
       目录不存在的话，先按 docs/VBench评测环境与权重说明.md 建一次。
HINT
  exit 1
}
echo "[vbench-long] python=${VBENCH_PYTHON} vbench=$(printf '%s' "${VBENCH_ENV_DUMP}" | head -1) torch=$(printf '%s' "${VBENCH_ENV_DUMP}" | tail -1)" | tee -a "${LOG_PATH}"

# 再确认"写 mp4"这条路径落在共享盘实现上，而不是镜像里的 torchvision。
# 镜像的 torchvision.io.write_video 在本机是靠一层 overlay 里的本地补丁才不炸的
# （PyAV 14 要求 frame.pict_type 是 PictureType 枚举，torchvision 0.20 传的是字符串
# "NONE"），换一台没打过这个补丁的机器，rollout 全跑完才在切 clip 时集体失败。
# 这里探一次：既断言用的是我们的 PyAV 实现，也断言本节点真能写出可解码的 mp4。
WRITER_DUMP="$(PYTHONPATH="${VBENCH_PYTHONPATH}" HOME="${VBENCH_HOME}" \
  "${VBENCH_PYTHON}" -c '
import os, tempfile
import numpy as np
import vbench2_beta_long.utils as vu
mod = vu.write_video.__module__
print("writer_module=" + mod)
assert mod == "vbench2_beta_long._vbench_portable_video", "unexpected writer: " + mod
frames = np.zeros((4, 64, 96, 3), dtype=np.uint8)
frames[..., 0] = 200
path = os.path.join(tempfile.mkdtemp(), "probe.mp4")
vu.write_video(path, frames, fps=8)
import decord
n = len(decord.VideoReader(path, num_threads=1))
print("writer_probe_frames=" + str(n))
assert n == 4, "wrote 4 frames, read back " + str(n)
' 2>&1)" || {
  echo "[error] 本节点写 mp4 的自检没过：" >&2
  echo "${WRITER_DUMP}" >&2
  cat >&2 <<HINT
[hint] vbench2_beta_long 应该用 ${VBENCH_ENV_ROOT}/site-packages 下的 PyAV 版 write_video。
       报 TypeError: an integer is required 说明还在走镜像的 torchvision，
       也就是 PYTHONPATH 没生效（检查 VBENCH_ENV_ROOT / VBENCH_PYTHONPATH）。
HINT
  exit 1
}
echo "[vbench-long] $(printf '%s' "${WRITER_DUMP}" | tr '\n' ' ')" | tee -a "${LOG_PATH}"

# VBench 的 long 侧不随包分发 VBench_full_info.json，preprocess() 又一定要读它，
# 所以从真正被导入的那个 vbench 包里解析路径，而不是写死 site-packages。
FULL_INFO="${FULL_INFO:-$(PYTHONPATH="${VBENCH_PYTHONPATH}" HOME="${VBENCH_HOME}" "${VBENCH_PYTHON}" -c 'import os, vbench; print(os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json"))')}"
if [[ ! -f "${FULL_INFO}" ]]; then
  echo "[error] VBench_full_info.json not found at ${FULL_INFO}" >&2
  exit 1
fi

# ---------------------------------------------------------------- stage rollout
if [[ "${STAGE}" == "all" || "${STAGE}" == "rollout" ]]; then
  echo "[vbench-long] === stage rollout: 先生成 mp4（SAVE_VIDEO=1） ===" | tee -a "${LOG_PATH}"
  # 早点把"跑错机器了"这件事说清楚。不检查的话，登录节点上会先在 rollout 里跑到
  # 模型加载才抛一个 CUDA 错误，看懂它比看懂这一行难得多。
  # 注意 nvidia-smi -L 在没有卡时也返回 0，只是输出为空，所以要看输出而不是退出码。
  if [[ -z "$(nvidia-smi -L 2>/dev/null)" ]]; then
    echo "[error] 这台机器上没有可见的 GPU，stage rollout 跑不了。" >&2
    cat >&2 <<HINT
[hint] stage rollout 要推理，必须在 8 卡的机器上跑。
       如果视频已经在别处生成好了，这台只做打分：STAGE=vbench ...
       只想在没卡的机器上验证脚本流程：STAGE=vbench VBENCH_FORCE_CPU=1 ...
HINT
    exit 1
  fi
  if [[ ! -x "${ROLLOUT_SCRIPT}" && ! -f "${ROLLOUT_SCRIPT}" ]]; then
    echo "[error] rollout script not found: ${ROLLOUT_SCRIPT}" >&2
    exit 1
  fi
  SAVE_VIDEO=1 GPUS_PER_NODE="${GPUS_PER_NODE}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash "${ROLLOUT_SCRIPT}"
fi

if [[ "${STAGE}" == "rollout" ]]; then
  echo "[vbench-long] stage rollout 结束（STAGE=rollout，不跑 VBench）" | tee -a "${LOG_PATH}"
  exit 0
fi

# ----------------------------------------------------------------- stage vbench
# stage vbench 要看到**所有**视频：两台机器各自只生成一半（SPLIT），所以默认会等到
# EXPECTED_MP4S 齐了再开始，否则先跑完的那台会把对方那半当成"不存在"，静默漏掉一些格子。
if [[ -n "${EXPECTED_MP4S:-}" ]]; then
  deadline=$(( $(date +%s) + WAIT_MP4_SECONDS ))
  while :; do
    have=$(find "${OUTPUT_ROOT}" -mindepth 4 -maxdepth 4 -path '*/videos/*.mp4' | wc -l)
    if [[ "${have}" -ge "${EXPECTED_MP4S}" ]]; then
      echo "[vbench-long] 找到 ${have}/${EXPECTED_MP4S} 个 mp4，开始评测" | tee -a "${LOG_PATH}"
      break
    fi
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "[error] 只找到 ${have}/${EXPECTED_MP4S} 个 mp4，等不到另一半（WAIT_MP4_SECONDS=${WAIT_MP4_SECONDS}）" >&2
      find "${OUTPUT_ROOT}" -mindepth 4 -maxdepth 4 -path '*/videos/*.mp4' | sort >&2
      echo "[hint] 先在两台机器上都跑 STAGE=rollout，或者在另一台上把 stage rollout 跑完再回来" >&2
      exit 1
    fi
    echo "[vbench-long] 等另一半视频：${have}/${EXPECTED_MP4S} ..." | tee -a "${LOG_PATH}"
    sleep 30
  done
fi

mapfile -t MP4S < <(find "${OUTPUT_ROOT}" -mindepth 4 -maxdepth 4 -path '*/videos/*.mp4' | sort)
if [[ ${#MP4S[@]} -eq 0 ]]; then
  echo "[error] 在 ${OUTPUT_ROOT} 下没找到 */<plan>/videos/*.mp4 —— 先跑 STAGE=rollout 生成视频" >&2
  exit 1
fi

# 收集 job 网格：(视频路径, 臂, plan, 维度)。臂/plan 直接从 rollout 的目录结构读出来，
# 不再重复一遍 plan 目录表。
jobs=()   # "mp4|arm|plan|dim"
set_names=()
declare -A seen_set=()
for mp4 in "${MP4S[@]}"; do
  plan_dir="$(dirname "${mp4}")"                 # <root>/<arm>/<plan>/videos
  plan="$(basename "$(dirname "${plan_dir}")")"
  arm="$(basename "$(dirname "$(dirname "${plan_dir}")")")"
  if [[ -n "${ARMS:-}" && ",${ARMS}," != *",${arm},"* ]]; then continue; fi
  if [[ -n "${PLANS:-}" && ",${PLANS}," != *",${plan},"* ]]; then continue; fi
  key="${arm}-${plan}"
  if [[ -z "${seen_set[${key}]:-}" ]]; then seen_set["${key}"]=1; set_names+=("${key}"); fi
  for dim in ${DIMS}; do
    jobs+=("${mp4}|${arm}|${plan}|${dim}")
  done
done

if [[ ${#jobs[@]} -eq 0 ]]; then
  echo "[error] ARMS=${ARMS:-<all>} / PLANS=${PLANS:-<all>} 过滤之后没有 job 了" >&2
  exit 1
fi

# 分片：job i 归 shard (i % SHARD_COUNT)。用取模而不是把 plan 切成两半，是因为这一阶段
# 的 job 已经是 (臂, plan, 维度) 三元组，取模能自动均衡，且两台机器不需要商量。
shard_expected=""
for ((k = 0; k < SHARD_COUNT; k++)); do
  [[ "${k}" == "${SHARD_INDEX}" ]] && continue
  shard_expected="${shard_expected}${shard_expected:+, }${k}"
done

shard_jobs=()
for i in "${!jobs[@]}"; do
  if (( i % SHARD_COUNT == SHARD_INDEX )); then
    shard_jobs+=("${jobs[$i]}")
  fi
done

{
  echo "[vbench-long] started $(date -u +%FT%TZ) host=${RUN_TAG}"
  echo "[vbench-long] gpus=${GPUS_PER_NODE} jobs=${#jobs[@]} shard=${SHARD_INDEX}/${SHARD_COUNT} -> ${#shard_jobs[@]} job(s)"
  echo "[vbench-long] dims=${DIMS}"
  echo "[vbench-long] output=${VBENCH_OUTPUT}"
  echo "[vbench-long] work=${VBENCH_WORK}"
  echo "[vbench-long] full_info=${FULL_INFO}"
  echo "[vbench-long] flicker_static_filter=${FLICKER_STATIC_FILTER} force=${FORCE} dry_run=${DRY_RUN}"
  printf '[vbench-long] sets=%s\n' "${set_names[*]}" 
  # SHARD_INDEX 默认 0，如果另一台也漏了它，两台就会做同一片、互相覆盖，而 1/2 那半
  # 永远不会被调度（表现为同一个 vbench_long/<维度>/<臂-plan>/ 下出现两个 results_* 文件）。
  if [[ "${SHARD_COUNT}" -gt 1 ]]; then
    printf '[vbench-long] 提醒：本机是分片 %s/%s，另一台的 SHARD_INDEX 应该传 %s。两台都漏传时默认都是 0，会互相覆盖、剩下的分片永远不会被调度。\n' \
      "${SHARD_INDEX}" "${SHARD_COUNT}" "${shard_expected}"
  fi
} | tee -a "${LOG_PATH}"

if [[ ${#shard_jobs[@]} -eq 0 ]]; then
  echo "[vbench-long] 这个分片没有 job（SHARD_COUNT 太大？）" | tee -a "${LOG_PATH}"
  exit 0
fi

job_is_done() {  # $1=dim $2=set_name
  compgen -G "${VBENCH_OUTPUT}/$1/$2/*_eval_results.json" > /dev/null
}

# 先挑出没做完的 job。已完成的默认跳过（FORCE=1 重跑），因为一个 (维度, 臂, plan)
# 往往要跑几分钟，中断后重开不该从头再来。
pending=()
skipped=0
for spec in "${shard_jobs[@]}"; do
  IFS='|' read -r _mp4 arm plan dim <<< "${spec}"
  if [[ "${FORCE}" != "1" ]] && job_is_done "${dim}" "${arm}-${plan}"; then
    echo "[vbench-long] skip (done) ${dim} ${arm}-${plan}" | tee -a "${LOG_PATH}"
    skipped=$((skipped + 1))
    continue
  fi
  pending+=("${spec}")
done

if [[ "${DRY_RUN}" == "1" ]]; then
  index=0
  for spec in "${pending[@]+"${pending[@]}"}"; do
    IFS='|' read -r mp4 arm plan dim <<< "${spec}"
    set_name="${arm}-${plan}"
    gpu=$(( index % GPUS_PER_NODE ))
    extra_desc=""
    [[ "${dim}" == "temporal_flickering" && "${FLICKER_STATIC_FILTER}" == "1" ]] && extra_desc=" --static_filter_flag"
    echo "[vbench-long] gpu ${gpu} -> ${dim} / ${set_name}" | tee -a "${LOG_PATH}"
    echo "[vbench-long]   CUDA_VISIBLE_DEVICES=${gpu} ${VBENCH_PYTHON} -m ${VCBENCH_ENTRY} \\\\" | tee -a "${LOG_PATH}"
    echo "[vbench-long]     --videos_path ${VBENCH_WORK}/${dim}/${set_name}/videos --dimension ${dim} --mode long_custom_input \\\\" | tee -a "${LOG_PATH}"
    echo "[vbench-long]     --dev_flag --use_semantic_splitting --load_ckpt_from_local True \\\\" | tee -a "${LOG_PATH}"
    echo "[vbench-long]     --full_json_dir ${FULL_INFO} --num_of_samples_per_prompt 1 \\\\" | tee -a "${LOG_PATH}"
    echo "[vbench-long]     --output_path ${VBENCH_OUTPUT}/${dim}/${set_name}${extra_desc}" | tee -a "${LOG_PATH}"
    echo "[vbench-long]   (soft link from ${mp4})" | tee -a "${LOG_PATH}"
    index=$((index + 1))
  done
  echo "[vbench-long] dry-run: would run ${index} job(s), ${skipped} already done" | tee -a "${LOG_PATH}"
  exit 0
fi

# ---- 本地 GPU 排队执行 ------------------------------------------------- #
declare -A gpu_of_pid=() label_of_pid=()
free_gpus=()
for ((gpu = 0; gpu < GPUS_PER_NODE; gpu++)); do free_gpus+=("${gpu}"); done
failed=()
completed=0

reap_one() {
  local finished="" gpu=""
  if wait -n -p finished; then :; else
    failed+=("${label_of_pid[${finished}]:-unknown}")
  fi
  gpu="${gpu_of_pid[${finished}]:-}"
  unset "gpu_of_pid[${finished}]" "label_of_pid[${finished}]"
  [[ -n "${gpu}" ]] && free_gpus+=("${gpu}")
  completed=$((completed + 1))
}

for spec in "${pending[@]+"${pending[@]}"}"; do
  IFS='|' read -r mp4 arm plan dim <<< "${spec}"
  set_name="${arm}-${plan}"
  while [[ ${#free_gpus[@]} -eq 0 ]]; do reap_one; done
  gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")

  job_work="${VBENCH_WORK}/${dim}/${set_name}"
  job_videos="${job_work}/videos"
  out_dir="${VBENCH_OUTPUT}/${dim}/${set_name}"

  if [[ "${FORCE}" == "1" ]]; then rm -rf "${job_work}"; fi
  mkdir -p "${job_videos}" "${out_dir}"
  # 每个 job 一条私有软链：VBench 的 preprocess / 各维度实现会往 videos_path 里写
  # split_clip、split_scene、*_cat_firstframes_videos，共享同一份会互相踩。
  ln -sf "$(readlink -f "${mp4}")" "${job_videos}/${set_name}.mp4"
  printf '{"arm": "%s", "plan": "%s", "dimension": "%s", "source_mp4": "%s"}\n' \
    "${arm}" "${plan}" "${dim}" "${mp4}" > "${out_dir}/job.json"

  extra_args=()
  [[ "${dim}" == "temporal_flickering" && "${FLICKER_STATIC_FILTER}" == "1" ]] && extra_args+=(--static_filter_flag)

  echo "[vbench-long] gpu ${gpu} -> ${dim} / ${set_name}" | tee -a "${LOG_PATH}"
  if [[ -n "${VBENCH_RUNNER:-}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" HOME="${VBENCH_HOME}" PYTHONPATH="${VBENCH_PYTHONPATH}" \
      bash "${VBENCH_RUNNER}" \
      --videos_path "${job_videos}" --dimension "${dim}" --output_path "${out_dir}" \
      --full_json_dir "${FULL_INFO}" \
      ${extra_args[@]+"${extra_args[@]}"} \
      >> "${job_work}/run.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" HOME="${VBENCH_HOME}" \
      PYTHONPATH="${VBENCH_PYTHONPATH}" VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR}" \
      "${VBENCH_PYTHON}" -m "${VCBENCH_ENTRY}" \
      --videos_path "${job_videos}" \
      --dimension "${dim}" \
      --mode long_custom_input \
      --dev_flag \
      --use_semantic_splitting \
      --load_ckpt_from_local True \
      --full_json_dir "${FULL_INFO}" \
      --num_of_samples_per_prompt 1 \
      --output_path "${out_dir}" \
      ${extra_args[@]+"${extra_args[@]}"} \
      >> "${job_work}/run.log" 2>&1 &
  fi
  pid=$!
  gpu_of_pid["${pid}"]="${gpu}"
  label_of_pid["${pid}"]="${dim}/${set_name}@gpu${gpu}"
done

while [[ ${#gpu_of_pid[@]} -gt 0 ]]; do reap_one; done

status=0
if [[ ${#failed[@]} -gt 0 ]]; then
  status=1
  echo "[vbench-long] failed: ${failed[*]}" | tee -a "${LOG_PATH}"
  echo "[vbench-long] 看看 ${VBENCH_WORK}/<dim>/<arm>-<plan>/run.log" | tee -a "${LOG_PATH}"
fi
echo "[vbench-long] finished $(date -u +%FT%TZ) status=${status} completed=${completed} skipped=${skipped}" | tee -a "${LOG_PATH}"

if [[ "${SHARD_COUNT}" -eq 1 ]]; then
  "${VBENCH_PYTHON}" "${REPO_ROOT}/examples/minimax_h3/model_training/report_vbench_long.py" \
    --root "${VBENCH_OUTPUT}" | tee -a "${LOG_PATH}"
else
  echo "[vbench-long] 汇总（等所有分片都结束，在任意一台机器上）：OUTPUT_ROOT=${OUTPUT_ROOT} VBENCH_OUTPUT=${VBENCH_OUTPUT} bash ${REPO_ROOT}/examples/minimax_h3/model_training/run_vbench_long_report.sh" | tee -a "${LOG_PATH}"
fi

exit "${status}"
