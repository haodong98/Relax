#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

REASONING_TRIGGER="${REASONING_TRIGGER:-terminal_once}"
EXP_DIR="${EXP_DIR:?EXP_DIR must be set by submit_mobilegym_e2e.sh}"
export G2_JUDGE_SMOKE_DIR="${EXP_DIR}/g2_judge_tp2_smoke"
export RELAX_JUDGE_GPU_SAMPLE_DIR="${EXP_DIR}/gpu_samples"
export RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S="${RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S:-0.2}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDNN_LIB_DIR,LD_LIBRARY_PATH,RELAX_JUDGE_GPU_SAMPLE_DIR"

JUDGE_SERVICES_CONFIG="${SCRIPT_DIR}/judge_services_e2e_${REASONING_TRIGGER}.json"
if [ ! -f "${JUDGE_SERVICES_CONFIG}" ]; then
    echo "ERROR: no judge config at ${JUDGE_SERVICES_CONFIG}" >&2
    exit 1
fi

exec python3 "${SCRIPT_DIR}/g2_judge_tp2_smoke.py" \
    --skip-hf-validate \
    --num-gpus-per-node 4 \
    --actor-num-gpus-per-node 1 \
    --rollout-num-gpus-per-engine 1 \
    --rollout-batch-size 1 \
    --global-batch-size 1 \
    --n-samples-per-prompt 1 \
    --num-rollout 1 \
    --resource '{"judge_accuracy":[1,2],"judge_multiturn_vlm":[1,2]}' \
    --rm-type dual-agentic-judge \
    --judge-services-config "$(tr -d '\n' < "${JUDGE_SERVICES_CONFIG}")" \
    --use-agentic-rollout \
    --agent-command /bin/true \
    --agent-cwd "${SCRIPT_DIR}" \
    --reward-key score \
    "${MODEL_ARGS[@]}"
