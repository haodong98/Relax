#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
: "${RELAX_REPO_DIR:?RELAX_REPO_DIR is required}"
: "${OSWORLD_REPO_DIR:?OSWORLD_REPO_DIR is required}"
: "${OSWORLD_TRUSTED_REGISTRY:?OSWORLD_TRUSTED_REGISTRY is required}"
: "${OSWORLD_BROKER_TOKEN_FILE:?OSWORLD_BROKER_TOKEN_FILE is required}"
: "${OSWORLD_BROKER_MANIFEST_DIR:?OSWORLD_BROKER_MANIFEST_DIR is required}"
: "${OSWORLD_BROKER_START_COMMAND_JSON:?OSWORLD_BROKER_START_COMMAND_JSON is required}"
: "${OSWORLD_EVALUATE_COMMAND_JSON:?OSWORLD_EVALUATE_COMMAND_JSON is required}"

cd "${RELAX_REPO_DIR}/examples/osworld_agentic"
export PYTHONPATH="${RELAX_REPO_DIR}/examples/osworld_agentic:${PYTHONPATH:-}"

broker_args=(
    --registry "${OSWORLD_TRUSTED_REGISTRY}"
    --token-file "${OSWORLD_BROKER_TOKEN_FILE}"
    --manifest-dir "${OSWORLD_BROKER_MANIFEST_DIR}"
    --lease-root "${OSWORLD_BROKER_LEASE_ROOT:-/tmp/relax-osworld-leases}"
    --start-command-json "${OSWORLD_BROKER_START_COMMAND_JSON}"
    --evaluate-command-json "${OSWORLD_EVALUATE_COMMAND_JSON}"
)
if [ -n "${OSWORLD_BROKER_ADVERTISE_HOST:-}" ]; then
    broker_args+=(--advertise-host "${OSWORLD_BROKER_ADVERTISE_HOST}")
fi
if [ -n "${OSWORLD_BROKER_STOP_COMMAND_JSON:-}" ]; then
    broker_args+=(--stop-command-json "${OSWORLD_BROKER_STOP_COMMAND_JSON}")
fi
exec "${OSWORLD_BROKER_PYTHON:-python3.11}" -m app.node_broker "${broker_args[@]}"
