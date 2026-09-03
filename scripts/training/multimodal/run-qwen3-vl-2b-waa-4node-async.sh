#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-VL-2B + WindowsAgentArena, fully asynchronous, 4-node/16-GPU smoke.

set -euo pipefail
set -x

timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${repo_dir}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-2B-instruct.sh"

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${SAVE_DIR:?SAVE_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"
: "${WAA_BROKER_MANIFEST_DIR:?WAA_BROKER_MANIFEST_DIR is required}"
: "${WAA_BROKER_TOKEN_FILE:?WAA_BROKER_TOKEN_FILE is required}"

agent_dir="${repo_dir}/examples/windows_agent_arena_agentic"
prompt_set="${WAA_PROMPT_DATA:-${DATA_DIR}/waa/smoke_train.jsonl}"
model_path="${MODEL_DIR}/Qwen3-VL-2B-Instruct"
num_rollout="${NUM_ROLLOUT:-3}"
if [ ! -f "${prompt_set}" ]; then
    echo "ERROR: prompt data not found: ${prompt_set}" >&2
    exit 1
fi
if [ "${num_rollout}" -ne 3 ]; then
    echo "ERROR: functional smoke requires exactly three rollout rounds" >&2
    exit 1
fi
mkdir -p "${SAVE_DIR}" "${EXP_DIR}/timeline" "${EXP_DIR}/rollout_result"

ckpt_args=(
    --hf-checkpoint "${model_path}"
    --megatron-to-hf-mode bridge
    --save "${SAVE_DIR}/Qwen3-VL-2B-WAA-Checkpoint"
    --save-interval 100
    --max-actor-ckpt-to-keep 1
    --warm-hf-checkpoint-page-cache
)

rollout_args=(
    --prompt-data "${prompt_set}"
    --input-key input
    --metadata-key metadata
    --multimodal-keys '{"image":"images"}'
    --use-agentic-rollout
    --agent-command ". ${agent_dir}/run_agent_app.sh"
    --agent-cwd "${agent_dir}"
    --agent-env
        "WAA_BROKER_MANIFEST_DIR=${WAA_BROKER_MANIFEST_DIR}"
        "WAA_BROKER_TOKEN_FILE=${WAA_BROKER_TOKEN_FILE}"
        "WAA_MAX_STEPS=${WAA_MAX_STEPS:-8}"
        "WAA_LEASE_WAIT_S=${WAA_LEASE_WAIT_S:-1200}"
        "WAA_CHAT_TIMEOUT_S=${WAA_CHAT_TIMEOUT_S:-1800}"
    --agent-timeout 2400
    --num-rollout "${num_rollout}"
    --rollout-batch-size 1
    --n-samples-per-prompt 4
    --over-sampling-batch-size 1
    --global-batch-size 4
    --rollout-max-prompt-len 16384
    --rollout-max-response-len 512
    --rollout-max-context-len 32768
    --rollout-temperature 1
    --rollout-shuffle
    --rollout-seed "${SEED:-42}"
    --seed "${SEED:-42}"
)

grpo_args=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

optimizer_args=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

sglang_args=(
    --rollout-num-gpus-per-engine 1
    --rollout-engine-init-timeout 300
    --sglang-mem-fraction-static 0.6
    --sglang-router-policy round_robin
)

megatron_args=(
    --transformer-impl local
    --no-rope-fusion
    --no-gradient-accumulation-fusion
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --distributed-timeout-minutes 5
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)

resource_args=(
    # Resource tuple is [num_serves,total_gpus]; Controller currently requires
    # num_serves=1. Twelve TP=1 rollout engines can span the remaining 3 nodes.
    --resource '{"actor":[1,4],"rollout":[1,12],"advantages":[1,0]}'
    --num-gpus-per-node 4
    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --max-staleness 0
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --fully-async
    --use-health-check
)

log_args=(
    --use-metrics-service
    --timeline-dump-dir "${EXP_DIR}/timeline"
    --rollout-result-dir "${EXP_DIR}/rollout_result"
    --tb-project-name "${PROJECT_NAME:-Relax/dev/waa}"
    --tb-experiment-name "${EXP_NAME:-qwen3-vl-2b-waa-async-${timestamp}}"
)

mkdir -p "${EXP_DIR}/logs"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${resource_args[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ckpt_args[@]}" \
    "${rollout_args[@]}" \
    "${grpo_args[@]}" \
    "${optimizer_args[@]}" \
    "${sglang_args[@]}" \
    "${megatron_args[@]}" \
    "${log_args[@]}" \
    2>&1 | tee "${EXP_DIR}/logs/driver-${timestamp}.log"
