#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${RELAX_BASE_URL:?RELAX_BASE_URL is required}"
: "${RELAX_SESSION_ID:?RELAX_SESSION_ID is required}"
: "${RELAX_INPUT_JSON:?RELAX_INPUT_JSON is required}"
: "${RELAX_OUTPUT_JSON:?RELAX_OUTPUT_JSON is required}"
: "${ANDROIDLAB_BROKER_MANIFEST_DIR:?ANDROIDLAB_BROKER_MANIFEST_DIR is required}"
: "${ANDROIDLAB_BROKER_TOKEN_FILE:?ANDROIDLAB_BROKER_TOKEN_FILE is required}"

exec "${RELAX_AGENT_PYTHON:-python}" -m app.client \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
