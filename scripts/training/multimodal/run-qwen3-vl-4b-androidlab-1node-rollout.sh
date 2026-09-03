#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-VL-4B AndroidLab operation rollout-only smoke on one node.

set -euo pipefail
set -x

timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${repo_dir}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"
: "${ANDROIDLAB_BROKER_MANIFEST_DIR:?ANDROIDLAB_BROKER_MANIFEST_DIR is required}"
: "${ANDROIDLAB_BROKER_TOKEN_FILE:?ANDROIDLAB_BROKER_TOKEN_FILE is required}"
: "${CUDNN_LIB_DIR:?CUDNN_LIB_DIR is required}"

agent_dir="${repo_dir}/examples/androidlab_agentic"
prompt_set="${ANDROIDLAB_PROMPT_DATA:-${DATA_DIR}/androidlab/operation.jsonl}"
mkdir -p "${EXP_DIR}/timeline" "${EXP_DIR}/rollout_result" "${EXP_DIR}/debug_rollout"

: "${RUNTIME_ENV_JSON:?RUNTIME_ENV_JSON must be set by the Relax cluster entrypoint}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDNN_LIB_DIR,LD_LIBRARY_PATH,FLASHINFER_WORKSPACE_BASE,RELAX_PROPAGATE_ENV_VARS"
RUNTIME_ENV_JSON="$(
    python3 -c '
import json
import os
import sys

runtime_env = json.load(sys.stdin)
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "CUDNN_LIB_DIR",
    "LD_LIBRARY_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    "RELAX_PROPAGATE_ENV_VARS",
):
    env_vars[name] = os.environ[name]
json.dump(runtime_env, sys.stdout, separators=(",", ":"))
' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"rollout":[1,4]}' --num-gpus-per-node 4 \
    --rollout-num-gpus-per-engine 1 --rollout-engine-init-timeout 300 \
    --sglang-router-policy round_robin --sglang-mem-fraction-static 0.6 \
    --hf-checkpoint "${MODEL_DIR}/Qwen3-VL-4B-Instruct" \
    --prompt-data "${prompt_set}" --input-key input --metadata-key metadata \
    --multimodal-keys '{"image":"images"}' --use-agentic-rollout \
    --agent-command ". ${agent_dir}/run_agent_app.sh" --agent-cwd "${agent_dir}" \
    --agent-env "ANDROIDLAB_BROKER_MANIFEST_DIR=${ANDROIDLAB_BROKER_MANIFEST_DIR}" \
        "ANDROIDLAB_BROKER_TOKEN_FILE=${ANDROIDLAB_BROKER_TOKEN_FILE}" \
        "ANDROIDLAB_MAX_STEPS=${ANDROIDLAB_MAX_STEPS:-8}" \
        "ANDROIDLAB_LEASE_WAIT_S=${ANDROIDLAB_LEASE_WAIT_S:-600}" \
        "ANDROIDLAB_CHAT_TIMEOUT_S=${ANDROIDLAB_CHAT_TIMEOUT_S:-900}" \
    --agent-timeout 1200 --num-rollout "${NUM_ROLLOUT:-1}" \
    --rollout-batch-size 1 --n-samples-per-prompt 1 --global-batch-size 1 \
    --rollout-max-prompt-len 16384 --rollout-max-response-len 512 \
    --rollout-max-context-len 32768 --rollout-temperature 1 \
    --rollout-seed "${SEED:-42}" --seed "${SEED:-42}" --debug-rollout-only \
    --save-debug-rollout-data "${EXP_DIR}/debug_rollout/{rollout_id}.pt" \
    --num-data-storage-units 1 --fully-async --no-rope-fusion --transformer-impl local \
    --use-metrics-service --timeline-dump-dir "${EXP_DIR}/timeline" \
    --rollout-result-dir "${EXP_DIR}/rollout_result" \
    --tb-project-name "Relax/dev/androidlab-rollout" \
    --tb-experiment-name "qwen3-vl-4b-androidlab-rollout-${timestamp}"
