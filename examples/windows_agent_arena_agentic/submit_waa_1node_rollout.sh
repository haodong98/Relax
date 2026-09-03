#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 1 node x 4 GPUs x 2 hours = 8 GPU-hours, WAA rollout-only smoke.

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --job-name=relax-waa-4b-rollout
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=460000
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 1 ]; then
    echo "ERROR: rollout-only smoke requires exactly one node" >&2
    exit 1
fi

RELAX_REPO_DIR="${RELAX_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${RELAX_REPO_DIR:?RELAX_REPO_DIR or SLURM_SUBMIT_DIR is required}"
script_dir="${RELAX_REPO_DIR}/examples/windows_agent_arena_agentic"
: "${WAA_REPO_DIR:?WAA_REPO_DIR is required}"
: "${WAA_GOLDEN_STORAGE:?WAA_GOLDEN_STORAGE is required}"
WAA_PYTHON="${WAA_PYTHON:-/usr/bin/python3.11}"
: "${EDF_TOML:?EDF_TOML is required}"
: "${RELAX_ENV_ROOT:?RELAX_ENV_ROOT is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"
EXP_DIR="${EXP_DIR%/}/job-${SLURM_JOB_ID}"
export EXP_DIR
export MEGATRON_DIR="${RELAX_ENV_ROOT}/Megatron-LM"
export TRANSFER_QUEUE_DIR="${RELAX_ENV_ROOT}/TransferQueue"
export VENV_BIN="${RELAX_ENV_ROOT}/relax_venv/bin"
export CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-${RELAX_ENV_ROOT}/relax_venv/lib/python3.12/site-packages/nvidia/cudnn/lib}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/relax-flashinfer/${SLURM_JOB_ID}}"
export SGLANG_NUMA_LIBRARY="${SGLANG_NUMA_LIBRARY:-/host_usr_lib64/libnuma.so.1.0.0}"
container_mounts="/usr/lib64:/host_usr_lib64:ro"

for required_path in \
    "${RELAX_REPO_DIR}" \
    "${script_dir}/scripts/build_dataset.py" \
    "${WAA_REPO_DIR}" \
    "${WAA_GOLDEN_STORAGE}/data.img" \
    "${WAA_PYTHON}" \
    "${EDF_TOML}" \
    "${MEGATRON_DIR}" \
    "${TRANSFER_QUEUE_DIR}/transfer_queue" \
    "${VENV_BIN}" \
    "${CUDNN_LIB_DIR}" \
    "${MODEL_DIR}/Qwen3-VL-4B-Instruct"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: required path not found: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${DATA_DIR}/waa"
if [ -e "${EXP_DIR}" ]; then
    echo "ERROR: EXP_DIR must be a fresh path for this job: ${EXP_DIR}" >&2
    exit 1
fi
mkdir -p "${EXP_DIR}"
export WAA_BROKER_MANIFEST_DIR="${EXP_DIR}/brokers"
export WAA_BROKER_EVENT_DIR="${EXP_DIR}/broker_events"
export WAA_BROKER_TOKEN_FILE="${EXP_DIR}/broker.token"
export WAA_TRUSTED_REGISTRY="${DATA_DIR}/waa/trusted_registry.json"
export WAA_ASSET_CACHE="${DATA_DIR}/waa/assets"
mkdir -p "${WAA_BROKER_MANIFEST_DIR}" "${WAA_BROKER_EVENT_DIR}"
chmod 700 "${WAA_BROKER_MANIFEST_DIR}"
umask 077
od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"${WAA_BROKER_TOKEN_FILE}"
chmod 600 "${WAA_BROKER_TOKEN_FILE}"

"${WAA_PYTHON}" -c 'import requests'
/usr/bin/python3.11 "${script_dir}/scripts/build_dataset.py" \
    --waa-repo "${WAA_REPO_DIR}" \
    --output-dir "${DATA_DIR}/waa"
/usr/bin/python3.11 "${script_dir}/scripts/cache_assets.py" \
    --registry "${WAA_TRUSTED_REGISTRY}" \
    --output-dir "${WAA_ASSET_CACHE}" \
    --task-id "366de66e-cbae-4d72-b042-26390db2b145-WOS"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=4 --overlap --exact \
    --environment="${EDF_TOML}" --container-mounts="${container_mounts}" --export=ALL \
    bash "${script_dir}/scripts/enter_training_env.sh" \
    python3 "${script_dir}/scripts/preflight_training_env.py" \
    --model "${MODEL_DIR}/Qwen3-VL-4B-Instruct" --output-dir "${EXP_DIR}/training_env_preflight"

export RELAX_REPO_DIR WAA_REPO_DIR WAA_GOLDEN_STORAGE WAA_PYTHON WAA_TRUSTED_REGISTRY WAA_ASSET_CACHE
broker_step_pid=""
train_step_pid=""
cleanup() {
    exit_code=$?
    cleanup_failed=0
    trap - EXIT INT TERM
    if [ -n "${train_step_pid}" ]; then
        kill -TERM "${train_step_pid}" 2>/dev/null || true
        wait "${train_step_pid}" 2>/dev/null || true
    fi
    if [ -n "${broker_step_pid}" ]; then
        kill -TERM "${broker_step_pid}" 2>/dev/null || true
        wait "${broker_step_pid}" 2>/dev/null || true
        if ! srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --overlap --exact \
            bash "${script_dir}/scripts/audit_node_cleanup.sh"; then
            cleanup_failed=1
        fi
    fi
    if [ "${exit_code}" -eq 0 ] && [ "${cleanup_failed}" -ne 0 ]; then
        exit_code=1
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --cpus-per-task=16 --overlap --exact --kill-on-bad-exit=1 \
    --export=ALL,CUDA_VISIBLE_DEVICES= \
    bash "${script_dir}/scripts/start_node_broker.sh" &
broker_step_pid=$!

/usr/bin/python3.11 "${script_dir}/scripts/wait_brokers.py" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" --expected 1 --timeout 300

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --overlap --exact \
    --environment="${EDF_TOML}" \
    python3 "${script_dir}/scripts/wait_brokers.py" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" --expected 1 --timeout 30

hosts_file="${EXP_DIR}/hosts"
scontrol show hostnames "${SLURM_JOB_NODELIST}" >"${hosts_file}"
head_host=$(head -n 1 "${hosts_file}")
master_addr=$(srun --nodes=1 --ntasks=1 --overlap --exact -w "${head_host}" hostname -I | awk '{print $1}')
export MASTER_ADDR="${master_addr}"
export WORLD_SIZE=1
export NUM_GPUS=4
export NUM_GPUS_TOTAL=4
export RELAX_SPMD_COMPLETION_FILE="${EXP_DIR}/spmd_completion"
export RAY_CLUSTER_READY_TIMEOUT_S=300
export RAY_GCS_WAIT_ATTEMPTS=60
export RAY_JOIN_ATTEMPTS=20
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export WAA_MAX_STEPS="${WAA_MAX_STEPS:-8}"
export WAA_PROMPT_DATA="${DATA_DIR}/waa/smoke_train.jsonl"
export RUN_SCRIPT="${RELAX_REPO_DIR}/scripts/training/multimodal/run-qwen3-vl-4b-waa-1node-rollout.sh"
unset RAY_NO_WAIT

train_cpus_per_task="${TRAIN_CPUS_PER_TASK:-256}"
if [ "${train_cpus_per_task}" -lt 64 ]; then
    echo "ERROR: TRAIN_CPUS_PER_TASK must be at least 64 for Ray and managed sessions" >&2
    exit 1
fi

if [ -z "${SLURM_JOB_END_TIME:-}" ]; then
    echo "ERROR: SLURM_JOB_END_TIME is required for cleanup budgeting" >&2
    exit 1
fi
training_budget_s=$((SLURM_JOB_END_TIME - $(date +%s) - 900))
if [ "${training_budget_s}" -lt 600 ]; then
    echo "ERROR: less than ten minutes remain before the cleanup reserve" >&2
    exit 1
fi

timeout --signal=TERM --kill-after=120s "${training_budget_s}s" \
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gpus-per-task=4 --cpus-per-task="${train_cpus_per_task}" \
    --overlap --exact --kill-on-bad-exit=1 \
    --environment="${EDF_TOML}" --container-mounts="${container_mounts}" --export=ALL \
    bash -lc '
        HOST_IP=$(hostname -I | awk "{print \$1}")
        export HOST_IP POD_NAME="${HOST_IP}"
        exec bash "${RELAX_REPO_DIR}/examples/windows_agent_arena_agentic/scripts/enter_training_env.sh" \
            bash "${RELAX_REPO_DIR}/scripts/entrypoint/spmd-multinode.sh" "${RUN_SCRIPT}"
    ' &
train_step_pid=$!

sleep 15
/usr/bin/python3.11 "${script_dir}/scripts/wait_brokers.py" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" --expected 1 --timeout 30

wait "${train_step_pid}"
train_step_pid=""
