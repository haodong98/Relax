#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${RELAX_REPO_DIR:?RELAX_REPO_DIR is required}"
: "${WAA_REPO_DIR:?WAA_REPO_DIR is required}"
: "${WAA_GOLDEN_STORAGE:?WAA_GOLDEN_STORAGE is required}"
: "${WAA_TRUSTED_REGISTRY:?WAA_TRUSTED_REGISTRY is required}"
: "${WAA_BROKER_MANIFEST_DIR:?WAA_BROKER_MANIFEST_DIR is required}"
: "${WAA_BROKER_TOKEN_FILE:?WAA_BROKER_TOKEN_FILE is required}"
: "${WAA_ASSET_CACHE:?WAA_ASSET_CACHE is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

node_name=$(hostname -s)
node_ip=$(hostname -I | awk '{print $1}')
node_root="${WAA_NODE_ROOT_BASE:-/tmp/relax-waa}/${SLURM_JOB_ID}-${node_name}"

cd "${RELAX_REPO_DIR}"
export PYTHONPATH="${RELAX_REPO_DIR}/examples/windows_agent_arena_agentic:${WAA_REPO_DIR}/src/win-arena-container/client:${PYTHONPATH:-}"
exec "${WAA_PYTHON:-/usr/bin/python3.11}" -m app.node_broker \
    --waa-repo "${WAA_REPO_DIR}" \
    --golden-storage "${WAA_GOLDEN_STORAGE}" \
    --node-root "${node_root}" \
    --registry "${WAA_TRUSTED_REGISTRY}" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" \
    --token-file "${WAA_BROKER_TOKEN_FILE}" \
    --advertise-host "${node_ip}" \
    --ready-timeout "${WAA_READY_TIMEOUT_S:-300}" \
    --lease-ttl "${WAA_LEASE_TTL_S:-2700}"
