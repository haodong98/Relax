#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-VL-8B + AndroidLab operation tasks, fully asynchronous, 4 nodes / 16 GPUs.

set -euo pipefail
set -x

timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${repo_dir}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-8B.sh"

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${SAVE_DIR:?SAVE_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"
: "${ANDROIDLAB_BROKER_MANIFEST_DIR:?ANDROIDLAB_BROKER_MANIFEST_DIR is required}"
: "${ANDROIDLAB_BROKER_TOKEN_FILE:?ANDROIDLAB_BROKER_TOKEN_FILE is required}"

agent_dir="${repo_dir}/examples/androidlab_agentic"
prompt_set="${ANDROIDLAB_PROMPT_DATA:-${DATA_DIR}/androidlab/operation.jsonl}"
model_path="${MODEL_DIR}/Qwen3-VL-8B-Instruct"
num_rollout="${NUM_ROLLOUT:-3}"
if [ ! -f "${prompt_set}" ]; then
    echo "ERROR: prompt data not found: ${prompt_set}" >&2
    exit 1
fi
if [ "${num_rollout}" -ne 3 ]; then
    echo "ERROR: functional smoke requires exactly three rollout rounds" >&2
    exit 1
fi
mkdir -p "${SAVE_DIR}" "${EXP_DIR}/timeline" "${EXP_DIR}/rollout_result" "${EXP_DIR}/logs"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor":[1,4],"rollout":[1,12],"advantages":[1,0]}' \
    --num-gpus-per-node 4 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --fully-async --max-staleness 0 --num-iters-per-train-update 1 --use-health-check \
    --hf-checkpoint "${model_path}" --megatron-to-hf-mode bridge \
    --save "${SAVE_DIR}/Qwen3-VL-8B-AndroidLab-Checkpoint" --save-interval 100 --max-actor-ckpt-to-keep 1 \
    --prompt-data "${prompt_set}" --input-key input --metadata-key metadata \
    --multimodal-keys '{"image":"images"}' \
    --use-agentic-rollout --agent-command ". ${agent_dir}/run_agent_app.sh" --agent-cwd "${agent_dir}" \
    --agent-env "ANDROIDLAB_BROKER_MANIFEST_DIR=${ANDROIDLAB_BROKER_MANIFEST_DIR}" \
    "ANDROIDLAB_BROKER_TOKEN_FILE=${ANDROIDLAB_BROKER_TOKEN_FILE}" \
    "ANDROIDLAB_MAX_STEPS=${ANDROIDLAB_MAX_STEPS:-12}" \
    "ANDROIDLAB_LEASE_WAIT_S=${ANDROIDLAB_LEASE_WAIT_S:-1200}" \
    "ANDROIDLAB_CHAT_TIMEOUT_S=${ANDROIDLAB_CHAT_TIMEOUT_S:-1800}" \
    --agent-timeout 2400 --num-rollout "${num_rollout}" --rollout-batch-size 1 \
    --n-samples-per-prompt 4 --over-sampling-batch-size 1 --global-batch-size 4 \
    --rollout-max-prompt-len 16384 --rollout-max-response-len 512 --rollout-max-context-len 32768 \
    --rollout-temperature 1 --rollout-shuffle --seed "${SEED:-42}" --rollout-seed "${SEED:-42}" \
    --advantage-estimator grpo --kl-loss-coef 0 --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --rollout-num-gpus-per-engine 1 --rollout-engine-init-timeout 300 --sglang-mem-fraction-static 0.6 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 32768 --recompute-granularity full --recompute-method uniform \
    --recompute-num-layers 1 --no-rope-fusion --no-gradient-accumulation-fusion \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --use-metrics-service --timeline-dump-dir "${EXP_DIR}/timeline" --rollout-result-dir "${EXP_DIR}/rollout_result" \
    --tb-project-name "${PROJECT_NAME:-Relax/dev/androidlab}" \
    --tb-experiment-name "${EXP_NAME:-qwen3-vl-8b-androidlab-async-${timestamp}}" \
    "${MODEL_ARGS[@]}" 2>&1 | tee "${EXP_DIR}/logs/driver-${timestamp}.log"
