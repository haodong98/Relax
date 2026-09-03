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
# Preload the exact cuDNN component libraries so the correct version wins symbol
# resolution before any wheel-bundled cuDNN is mapped (LD_LIBRARY_PATH alone is
# insufficient when a foreign libcudnn loads first). This mirrors the MobileGym
# runtime gate; missing components are a hard error.
cudnn_libraries=(
    libcudnn.so.9
    libcudnn_graph.so.9
    libcudnn_engines_runtime_compiled.so.9
    libcudnn_ops.so.9
    libcudnn_cnn.so.9
    libcudnn_adv.so.9
    libcudnn_engines_precompiled.so.9
    libcudnn_heuristic.so.9
)
cudnn_preload_paths=()
for cudnn_library in "${cudnn_libraries[@]}"; do
    if [ ! -r "${CUDNN_LIB_DIR}/${cudnn_library}" ]; then
        echo "ERROR: missing cuDNN preload library: ${CUDNN_LIB_DIR}/${cudnn_library}" >&2
        exit 1
    fi
    cudnn_preload_paths+=("${CUDNN_LIB_DIR}/${cudnn_library}")
done
cudnn_preload="$(IFS=:; echo "${cudnn_preload_paths[*]}")"
export PATH="${VENV_BIN}:${PATH}"
export LD_PRELOAD="${SGLANG_NUMA_LIBRARY}:${cudnn_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
venv_site="$("${VENV_BIN}/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="${RELAX_REPO_DIR}:${MEGATRON_DIR}:${venv_site}${PYTHONPATH:+:${PYTHONPATH}}"
if [ -d "${TRANSFER_QUEUE_DIR}/transfer_queue" ]; then
    export PYTHONPATH="${TRANSFER_QUEUE_DIR}:${PYTHONPATH}"
fi
export MEGATRON="${MEGATRON_DIR}"
export RELAX="${RELAX_REPO_DIR}"
# A functional fully-async gate must fail if the rollout service cannot accept
# a trained weight version; partial actor-only progress is not success.
export RELAX_REQUIRE_WEIGHT_PUBLICATION="${RELAX_REQUIRE_WEIGHT_PUBLICATION:-1}"

exec "$@"
