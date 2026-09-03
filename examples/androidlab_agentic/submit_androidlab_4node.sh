#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# AndroidLab operation smoke: 4 nodes x 4 GPUs/node x 2 hours = 32 GPU-hours.

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --job-name=relax-androidlab-8b
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=460000
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

if [ "${SLURM_JOB_NUM_NODES:-0}" -ne 4 ]; then
    echo "ERROR: AndroidLab functional smoke requires exactly four nodes" >&2
    exit 1
fi

RELAX_REPO_DIR="${RELAX_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${RELAX_REPO_DIR:?RELAX_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${ANDROIDLAB_REPO_DIR:?ANDROIDLAB_REPO_DIR is required}"
: "${ANDROIDLAB_BROKER_START_COMMAND_JSON:?ANDROIDLAB_BROKER_START_COMMAND_JSON is required}"
: "${EDF_TOML:?EDF_TOML is required}"
: "${RELAX_ENV_ROOT:?RELAX_ENV_ROOT is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${SAVE_DIR:?SAVE_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"

EXP_DIR="${EXP_DIR%/}/job-${SLURM_JOB_ID}"
if [ -e "${EXP_DIR}" ]; then
    echo "ERROR: EXP_DIR must be fresh for this job: ${EXP_DIR}" >&2
    exit 1
fi
mkdir -p "${EXP_DIR}" "${DATA_DIR}/androidlab" "${SAVE_DIR}"
export EXP_DIR
export MEGATRON_DIR="${RELAX_ENV_ROOT}/Megatron-LM"
export TRANSFER_QUEUE_DIR="${RELAX_ENV_ROOT}/TransferQueue"
export VENV_BIN="${RELAX_ENV_ROOT}/relax_venv/bin"
export CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-${RELAX_ENV_ROOT}/relax_venv/lib/python3.12/site-packages/nvidia/cudnn/lib}"
export SGLANG_NUMA_LIBRARY="${SGLANG_NUMA_LIBRARY:-/host_usr_lib64/libnuma.so.1.0.0}"
container_mounts="/usr/lib64:/host_usr_lib64:ro"
export ANDROIDLAB_TRUSTED_REGISTRY="${DATA_DIR}/androidlab/trusted_registry.json"
export ANDROIDLAB_BROKER_MANIFEST_DIR="${EXP_DIR}/brokers"
export ANDROIDLAB_BROKER_EVENT_DIR="${EXP_DIR}/broker_events"
export ANDROIDLAB_BROKER_TOKEN_FILE="${EXP_DIR}/broker.token"
export ANDROIDLAB_BROKER_LEASE_ROOT="/tmp/relax-androidlab-${SLURM_JOB_ID}"
mkdir -p "${ANDROIDLAB_BROKER_MANIFEST_DIR}" "${ANDROIDLAB_BROKER_EVENT_DIR}"
chmod 700 "${ANDROIDLAB_BROKER_MANIFEST_DIR}"
umask 077
od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"${ANDROIDLAB_BROKER_TOKEN_FILE}"
chmod 600 "${ANDROIDLAB_BROKER_TOKEN_FILE}"

/usr/bin/python3.11 "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/build_dataset.py" \
    --androidlab-repo "${ANDROIDLAB_REPO_DIR}" --output-dir "${DATA_DIR}/androidlab"

for required_path in \
    "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/start_node_broker.sh" \
    "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/enter_training_env.sh" \
    "${ANDROIDLAB_REPO_DIR}/evaluation/config/setting.yaml" \
    "${MEGATRON_DIR}" "${TRANSFER_QUEUE_DIR}/transfer_queue" "${VENV_BIN}" "${CUDNN_LIB_DIR}" \
    "${MODEL_DIR}/Qwen3-VL-8B-Instruct" \
    "${EDF_TOML}"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: required path not found: ${required_path}" >&2
        exit 1
    fi
done

broker_step_pid=""
train_step_pid=""
cleanup() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ -n "${train_step_pid}" ]; then
        kill -TERM "${train_step_pid}" 2>/dev/null || true
        wait "${train_step_pid}" 2>/dev/null || true
    fi
    if [ -n "${broker_step_pid}" ]; then
        kill -TERM "${broker_step_pid}" 2>/dev/null || true
        wait "${broker_step_pid}" 2>/dev/null || true
    fi
    if find "${ANDROIDLAB_BROKER_MANIFEST_DIR}" -name 'broker-*.json' -print -quit | grep -q .; then
        echo "ERROR: AndroidLab broker manifests remained after cleanup" >&2
        exit_code=1
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

export RELAX_REPO_DIR ANDROIDLAB_REPO_DIR ANDROIDLAB_BROKER_START_COMMAND_JSON ANDROIDLAB_TRUSTED_REGISTRY
export ANDROIDLAB_BROKER_MANIFEST_DIR ANDROIDLAB_BROKER_EVENT_DIR ANDROIDLAB_BROKER_TOKEN_FILE
export ANDROIDLAB_BROKER_LEASE_ROOT ANDROIDLAB_BROKER_STOP_COMMAND_JSON ANDROIDLAB_QUERY_JUDGE_URL
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=16 --overlap --exact --kill-on-bad-exit=1 \
    --export=ALL,CUDA_VISIBLE_DEVICES= \
    bash "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/start_node_broker.sh" &
broker_step_pid=$!

/usr/bin/python3.11 "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/wait_brokers.py" \
    --manifest-dir "${ANDROIDLAB_BROKER_MANIFEST_DIR}" --expected 4 --timeout 600

hosts_file="${EXP_DIR}/hosts"
scontrol show hostnames "${SLURM_JOB_NODELIST}" >"${hosts_file}"
head_host=$(head -n 1 "${hosts_file}")
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --overlap --exact -w "${head_host}" hostname -I | awk '{print $1}')
export WORLD_SIZE=4 NUM_GPUS=4 NUM_GPUS_TOTAL=4 NUM_ROLLOUT=3
export ANDROIDLAB_PROMPT_DATA="${DATA_DIR}/androidlab/operation.jsonl"
export RUN_SCRIPT="${RELAX_REPO_DIR}/scripts/training/multimodal/run-qwen3-vl-8b-androidlab-4node-async.sh"
export RELAX_SPMD_COMPLETION_FILE="${EXP_DIR}/spmd_completion"
unset RAY_NO_WAIT

if [ -z "${SLURM_JOB_END_TIME:-}" ]; then
    echo "ERROR: SLURM_JOB_END_TIME is required for cleanup budgeting" >&2
    exit 1
fi
training_budget_s=$((SLURM_JOB_END_TIME - $(date +%s) - 900))
if [ "${training_budget_s}" -lt 600 ]; then
    echo "ERROR: less than ten minutes remain before cleanup reserve" >&2
    exit 1
fi
timeout --signal=TERM --kill-after=120s "${training_budget_s}s" \
    srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task="${TRAIN_CPUS_PER_TASK:-256}" \
        --overlap --exact --kill-on-bad-exit=1 --environment="${EDF_TOML}" --container-mounts="${container_mounts}" --export=ALL \
        bash -lc '
            HOST_IP=$(hostname -I | awk "{print \$1}")
            export HOST_IP POD_NAME="${HOST_IP}"
            exec bash "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/enter_training_env.sh" \
                bash "${RELAX_REPO_DIR}/scripts/entrypoint/spmd-multinode.sh" "${RUN_SCRIPT}"
        ' &
train_step_pid=$!

sleep 15
/usr/bin/python3.11 "${RELAX_REPO_DIR}/examples/androidlab_agentic/scripts/wait_brokers.py" \
    --manifest-dir "${ANDROIDLAB_BROKER_MANIFEST_DIR}" --expected 4 --timeout 30
wait "${train_step_pid}"
train_step_pid=""
