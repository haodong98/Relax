#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${RELAX_REPO_DIR:?RELAX_REPO_DIR is required}"
: "${MEGATRON_DIR:?MEGATRON_DIR is required}"
: "${TRANSFER_QUEUE_DIR:?TRANSFER_QUEUE_DIR is required}"
: "${VENV_BIN:?VENV_BIN is required}"
: "${CUDNN_LIB_DIR:?CUDNN_LIB_DIR is required}"
: "${SGLANG_NUMA_LIBRARY:?SGLANG_NUMA_LIBRARY is required}"

if [ ! -r "${SGLANG_NUMA_LIBRARY}" ]; then
    echo "ERROR: SGLang NUMA library is missing in the EDF container: ${SGLANG_NUMA_LIBRARY}" >&2
    exit 1
fi
export PATH="${VENV_BIN}:${PATH}"
export LD_PRELOAD="${SGLANG_NUMA_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
venv_site="$("${VENV_BIN}/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="${RELAX_REPO_DIR}:${MEGATRON_DIR}:${venv_site}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -d "${TRANSFER_QUEUE_DIR}/transfer_queue" ]; then
    export PYTHONPATH="${TRANSFER_QUEUE_DIR}:${PYTHONPATH}"
fi
export MEGATRON="${MEGATRON_DIR}"
export RELAX="${RELAX_REPO_DIR}"
export RELAX_REQUIRE_WEIGHT_PUBLICATION="${RELAX_REQUIRE_WEIGHT_PUBLICATION:-1}"

exec "$@"
