#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# G3-A/G4-A: production TP1 rollout engines across four-GPU nodes.
# This is deliberately plain inference, with no MobileGym browser, Judge, or
# trainer dependency, so failures isolate Ray placement/router/rendezvous.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

for required_name in MODEL_DIR EXP_DIR RUNTIME_ENV_JSON CUDNN_LIB_DIR FLASHINFER_WORKSPACE_BASE; do
    if [ -z "${!required_name:-}" ]; then
        echo "ERROR: ${required_name} must be set." >&2
        exit 1
    fi
done

mkdir -p "${EXP_DIR}/timeline" "${EXP_DIR}/rollout_result" "${EXP_DIR}/debug_rollout" "${EXP_DIR}/gpu_samples"
ROLLOUT_ENGINE_COUNT="${ROLLOUT_ENGINE_COUNT:-8}"
ROLLOUT_SAMPLE_COUNT="${ROLLOUT_SAMPLE_COUNT:-8}"
if ! [[ "${ROLLOUT_ENGINE_COUNT}" =~ ^[1-9][0-9]*$ && "${ROLLOUT_SAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ROLLOUT_ENGINE_COUNT and ROLLOUT_SAMPLE_COUNT must be positive integers." >&2
    exit 1
fi
PROMPT_FILE="${EXP_DIR}/rollout_topology_prompts.jsonl"
python3 -c '
import json
import sys

path = sys.argv[1]
count = int(sys.argv[2])
with open(path, "w", encoding="utf-8") as output:
    for index in range(count):
        prompt = (
            f"Request {index}: emit a numbered technical checklist with exactly 256 short items. "
            "Do not stop early and do not use tools."
        )
        output.write(json.dumps({"prompt": prompt}) + "\n")
' "${PROMPT_FILE}" "${ROLLOUT_SAMPLE_COUNT}"

export RELAX_JUDGE_GPU_SAMPLE_DIR="${EXP_DIR}/gpu_samples"
export RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S="${RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S:-0.1}"
export RELAX_EXTRA_ENV_ALLOWLIST="${RELAX_EXTRA_ENV_ALLOWLIST:+${RELAX_EXTRA_ENV_ALLOWLIST},}RELAX_ENV_ROOT,RELAX_SPMD_COMPLETION_FILE"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDNN_LIB_DIR,LD_LIBRARY_PATH,FLASHINFER_WORKSPACE_BASE,RELAX_EXTRA_ENV_ALLOWLIST,RELAX_JUDGE_GPU_SAMPLE_DIR,RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S,RELAX_PROPAGATE_ENV_VARS"
RUNTIME_ENV_JSON="$(
    python3 -c '
import json
import os
import sys

runtime_env = json.load(sys.stdin)
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "CUDNN_LIB_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "RELAX_EXTRA_ENV_ALLOWLIST",
    "RELAX_JUDGE_GPU_SAMPLE_DIR",
    "RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S",
    "RELAX_PROPAGATE_ENV_VARS",
):
    env_vars[name] = os.environ[name]
json.dump(runtime_env, sys.stdout, separators=(",", ":"))
' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

DRIVER_LOG="${EXP_DIR}/rollout_topology_driver.log"
# This marker belongs to the outer Slurm SPMD process. The nested Ray Job has
# its own lifecycle and must not warn about or propagate this launcher-only key.
unset RELAX_SPMD_COMPLETION_FILE
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "{\"rollout\":[1,${ROLLOUT_ENGINE_COUNT}]}" \
    --num-gpus-per-node 4 \
    --rollout-num-gpus-per-engine 1 \
    --rollout-engine-init-timeout 300 \
    --sglang-router-policy round_robin \
    --sglang-mem-fraction-static 0.6 \
    --hf-checkpoint "${MODEL_DIR}/Qwen3-VL-4B-Instruct" \
    --prompt-data "${PROMPT_FILE}" \
    --input-key prompt \
    --apply-chat-template \
    --rm-type dummy \
    --reward-key score \
    --num-rollout 1 \
    --rollout-batch-size "${ROLLOUT_SAMPLE_COUNT}" \
    --n-samples-per-prompt 1 \
    --global-batch-size "${ROLLOUT_SAMPLE_COUNT}" \
    --rollout-max-prompt-len 512 \
    --rollout-max-response-len 1024 \
    --rollout-max-context-len 1536 \
    --rollout-temperature 1 \
    --debug-rollout-only \
    --save-debug-rollout-data "${EXP_DIR}/debug_rollout/{rollout_id}.pt" \
    --num-data-storage-units 1 \
    --fully-async \
    --use-metrics-service \
    --timeline-dump-dir "${EXP_DIR}/timeline" \
    --rollout-result-dir "${EXP_DIR}/rollout_result" \
    "${MODEL_ARGS[@]}" \
    2>&1 | tee "${DRIVER_LOG}"

python3 "${SCRIPT_DIR}/scripts/check_g3_rollout8.py" \
    --exp-dir "${EXP_DIR}" \
    --driver-log "${DRIVER_LOG}" \
    --expected-samples "${ROLLOUT_SAMPLE_COUNT}" \
    --expected-engines "${ROLLOUT_ENGINE_COUNT}"
