#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Zero-GPU live WAA environment gate; this does not consume the GPU-hour budget.

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --job-name=relax-waa-env
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120

set -euo pipefail

RELAX_REPO_DIR="${RELAX_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${RELAX_REPO_DIR:?RELAX_REPO_DIR or SLURM_SUBMIT_DIR is required}"
script_dir="${RELAX_REPO_DIR}/examples/windows_agent_arena_agentic"
: "${WAA_REPO_DIR:?WAA_REPO_DIR is required}"
: "${WAA_GOLDEN_STORAGE:?WAA_GOLDEN_STORAGE is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"
WAA_PYTHON="${WAA_PYTHON:-/usr/bin/python3.11}"

for required_path in \
    "${RELAX_REPO_DIR}" \
    "${script_dir}/scripts/build_dataset.py" \
    "${WAA_REPO_DIR}" \
    "${WAA_GOLDEN_STORAGE}/data.img" \
    "${WAA_PYTHON}"; do
    if [ ! -e "${required_path}" ]; then
        echo "ERROR: required path not found: ${required_path}" >&2
        exit 1
    fi
done
"${WAA_PYTHON}" -c 'import requests'

mkdir -p "${DATA_DIR}/waa" "${EXP_DIR}/brokers"
chmod 700 "${EXP_DIR}/brokers"
export WAA_BROKER_MANIFEST_DIR="${EXP_DIR}/brokers"
export WAA_BROKER_TOKEN_FILE="${EXP_DIR}/broker.token"
export WAA_TRUSTED_REGISTRY="${DATA_DIR}/waa/trusted_registry.json"
export WAA_ASSET_CACHE="${DATA_DIR}/waa/assets"
umask 077
od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"${WAA_BROKER_TOKEN_FILE}"
chmod 600 "${WAA_BROKER_TOKEN_FILE}"

/usr/bin/python3.11 "${script_dir}/scripts/build_dataset.py" \
    --waa-repo "${WAA_REPO_DIR}" --output-dir "${DATA_DIR}/waa"
/usr/bin/python3.11 "${script_dir}/scripts/cache_assets.py" \
    --registry "${WAA_TRUSTED_REGISTRY}" --output-dir "${WAA_ASSET_CACHE}" \
    --task-id "366de66e-cbae-4d72-b042-26390db2b145-WOS"

export RELAX_REPO_DIR WAA_REPO_DIR WAA_GOLDEN_STORAGE WAA_PYTHON WAA_TRUSTED_REGISTRY WAA_ASSET_CACHE
broker_pid=""
cleanup() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ -n "${broker_pid}" ]; then
        kill -TERM "${broker_pid}" 2>/dev/null || true
        wait "${broker_pid}" 2>/dev/null || true
    fi
    if ! bash "${script_dir}/scripts/audit_node_cleanup.sh"; then
        exit_code=1
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

bash "${script_dir}/scripts/start_node_broker.sh" &
broker_pid=$!
/usr/bin/python3.11 "${script_dir}/scripts/wait_brokers.py" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" --expected 1 --timeout 420
PYTHONPATH="${script_dir}:${PYTHONPATH:-}" /usr/bin/python3.11 "${script_dir}/scripts/smoke_broker_episode.py" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" \
    --token-file "${WAA_BROKER_TOKEN_FILE}" \
    --registry "${WAA_TRUSTED_REGISTRY}" \
    --output "${EXP_DIR}/live-broker-smoke.json"
