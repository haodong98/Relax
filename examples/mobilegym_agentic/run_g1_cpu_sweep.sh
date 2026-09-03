#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Run this from an allocated CPU node (for example, an Slurm debug allocation).
# It intentionally makes no GPU or Ray request: G1 validates MobileGym browser
# concurrency before any Relax training topology is admitted.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MOBILEGYM_REPO_DIR="${MOBILEGYM_REPO_DIR:?set MOBILEGYM_REPO_DIR}"
MOBILEGYM_PYTHON="${MOBILEGYM_PYTHON:?set MOBILEGYM_PYTHON}"
MOBILEGYM_ENV_URL="${MOBILEGYM_ENV_URL:?set MOBILEGYM_ENV_URL}"
G1_OUTPUT_ROOT="${G1_OUTPUT_ROOT:?set G1_OUTPUT_ROOT to a fresh output directory}"

exec "${MOBILEGYM_PYTHON}" "${SCRIPT_DIR}/g1_cpu_sweep.py" \
    --mobilegym-repo "${MOBILEGYM_REPO_DIR}" \
    --mobilegym-python "${MOBILEGYM_PYTHON}" \
    --env-url "${MOBILEGYM_ENV_URL}" \
    --output-root "${G1_OUTPUT_ROOT}" \
    --concurrency 1 \
    --concurrency 1 \
    --concurrency 8 \
    --concurrency 32 \
    --concurrency 64
