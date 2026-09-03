#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
node_name=$(hostname -s)
node_root="${WAA_NODE_ROOT_BASE:-/tmp/relax-waa}/${SLURM_JOB_ID}-${node_name}"
container_prefix="relax-waa-${SLURM_JOB_ID}-${node_name}-"

if [ -e "${node_root}" ]; then
    echo "ERROR: WAA node root remains after broker shutdown: ${node_root}" >&2
    exit 1
fi
if podman ps -a --format '{{.Names}}' | awk -v prefix="${container_prefix}" 'index($0, prefix) == 1 {found=1} END {exit found ? 0 : 1}'; then
    echo "ERROR: WAA container remains after broker shutdown: ${container_prefix}*" >&2
    exit 1
fi

: "${WAA_BROKER_MANIFEST_DIR:?WAA_BROKER_MANIFEST_DIR is required}"
: "${WAA_BROKER_TOKEN_FILE:?WAA_BROKER_TOKEN_FILE is required}"
: "${EXP_DIR:?EXP_DIR is required}"
if find "${WAA_BROKER_MANIFEST_DIR}" -maxdepth 1 -type f -name 'broker-*.json' -print -quit | grep -q .; then
    echo "ERROR: broker manifest remains after broker shutdown" >&2
    exit 1
fi
rm -f -- "${WAA_BROKER_TOKEN_FILE}"
/usr/bin/python3.11 "${RELAX_REPO_DIR}/examples/windows_agent_arena_agentic/scripts/write_cleanup_audit.py" \
    --output-dir "${EXP_DIR}/cleanup_audit" \
    --node-root "${node_root}" \
    --container-prefix "${container_prefix}" \
    --manifest-dir "${WAA_BROKER_MANIFEST_DIR}" \
    --token-file "${WAA_BROKER_TOKEN_FILE}"
