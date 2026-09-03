#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${RELAX_REPO_DIR:?RELAX_REPO_DIR is required}"
: "${ANDROIDLAB_REPO_DIR:?ANDROIDLAB_REPO_DIR is required}"
: "${ANDROIDLAB_TRUSTED_REGISTRY:?ANDROIDLAB_TRUSTED_REGISTRY is required}"
: "${ANDROIDLAB_BROKER_TOKEN_FILE:?ANDROIDLAB_BROKER_TOKEN_FILE is required}"
: "${ANDROIDLAB_BROKER_MANIFEST_DIR:?ANDROIDLAB_BROKER_MANIFEST_DIR is required}"
: "${ANDROIDLAB_BROKER_EVENT_DIR:?ANDROIDLAB_BROKER_EVENT_DIR is required}"
: "${ANDROIDLAB_BROKER_LEASE_ROOT:?ANDROIDLAB_BROKER_LEASE_ROOT is required}"
: "${ANDROIDLAB_BROKER_START_COMMAND_JSON:?ANDROIDLAB_BROKER_START_COMMAND_JSON is required}"

host_name=$(hostname)
work_root="${ANDROIDLAB_BROKER_WORK_ROOT:-${ANDROIDLAB_BROKER_LEASE_ROOT}/work}"
mkdir -p "${work_root}" "${ANDROIDLAB_BROKER_LEASE_ROOT}" "${ANDROIDLAB_BROKER_EVENT_DIR}"

cd "${RELAX_REPO_DIR}/examples/androidlab_agentic"
broker_args=(
    --registry "${ANDROIDLAB_TRUSTED_REGISTRY}" \
    --token-file "${ANDROIDLAB_BROKER_TOKEN_FILE}" \
    --manifest-dir "${ANDROIDLAB_BROKER_MANIFEST_DIR}" \
    --work-root "${work_root}" \
    --lease-root "${ANDROIDLAB_BROKER_LEASE_ROOT}" \
    --event-path "${ANDROIDLAB_BROKER_EVENT_DIR}/broker-${host_name}.jsonl" \
    --adb "${ANDROIDLAB_ADB:-adb}" \
    --androidlab-repo "${ANDROIDLAB_REPO_DIR}" \
    --lease-ttl "${ANDROIDLAB_LEASE_TTL_S:-1800}" \
    --start-command-json "${ANDROIDLAB_BROKER_START_COMMAND_JSON}"
)
if [ -n "${ANDROIDLAB_BROKER_STOP_COMMAND_JSON:-}" ]; then
    broker_args+=(--stop-command-json "${ANDROIDLAB_BROKER_STOP_COMMAND_JSON}")
fi
if [ -n "${ANDROIDLAB_QUERY_JUDGE_URL:-}" ]; then
    broker_args+=(--query-judge-url "${ANDROIDLAB_QUERY_JUDGE_URL}")
fi
exec "${ANDROIDLAB_BROKER_PYTHON:-/usr/bin/python3.11}" -m app.node_broker "${broker_args[@]}"
