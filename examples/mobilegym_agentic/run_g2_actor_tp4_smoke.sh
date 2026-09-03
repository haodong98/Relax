#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# G2-C actor-only Qwen3-VL TP4 smoke. Replays one completed MobileGym debug
# rollout without starting rollout or Judge services. Run once with
# G2C_MODE=train, then in a fresh allocation with G2C_MODE=reload and the same
# G2C_ROOT to verify model/optimizer/RNG checkpoint restoration.

set -euo pipefail
set -o xtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

for required_name in MODEL_DIR G2C_ROOT CAPTURE_PT RUNTIME_ENV_JSON CUDNN_LIB_DIR; do
    if [ -z "${!required_name:-}" ]; then
        echo "ERROR: ${required_name} must be set." >&2
        exit 1
    fi
done
if [ ! -s "${CAPTURE_PT}" ]; then
    echo "ERROR: completed debug rollout is missing or empty: ${CAPTURE_PT}" >&2
    exit 1
fi

G2C_MODE="${G2C_MODE:-train}"
if [ "${G2C_MODE}" != "train" ] && [ "${G2C_MODE}" != "reload" ]; then
    echo "ERROR: G2C_MODE must be train or reload, got ${G2C_MODE}." >&2
    exit 1
fi

CHECKPOINT_DIR="${G2C_ROOT}/checkpoint"
TIMELINE_DIR="${G2C_ROOT}/${G2C_MODE}_timeline"
mkdir -p "${G2C_ROOT}" "${TIMELINE_DIR}"
if [ "${G2C_MODE}" = "train" ] && [ -e "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: G2-C train requires a fresh checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [ "${G2C_MODE}" = "reload" ] && [ ! -s "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: G2-C reload requires a completed checkpoint: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDNN_LIB_DIR,LD_LIBRARY_PATH,RELAX_PROPAGATE_ENV_VARS"
G2C_LD_LIBRARY_PATH="${CUDNN_LIB_DIR}"
while IFS= read -r library_dir; do
    if [ -n "${library_dir}" ] && [ "${library_dir}" != "${CUDNN_LIB_DIR}" ]; then
        G2C_LD_LIBRARY_PATH="${G2C_LD_LIBRARY_PATH}:${library_dir}"
    fi
done < <(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}")
export G2C_LD_LIBRARY_PATH
RUNTIME_ENV_JSON="$(
    python3 -c '
import json
import os
import sys

runtime_env = json.load(sys.stdin)
env_vars = runtime_env.setdefault("env_vars", {})
for name in ("CUDNN_LIB_DIR", "RELAX_PROPAGATE_ENV_VARS"):
    env_vars[name] = os.environ[name]
json.dump(runtime_env, sys.stdout, separators=(",", ":"))
' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-VL-4B-Instruct"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --save "${CHECKPOINT_DIR}"
    --save-interval 1
    --max-actor-ckpt-to-keep 2
)
if [ "${G2C_MODE}" = "reload" ]; then
    CKPT_ARGS+=(--load "${CHECKPOINT_DIR}")
fi

DATA_ARGS=(
    --load-debug-rollout-data "${CAPTURE_PT}"
    --load-debug-rollout-data-subsample 0.5
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --num-rollout 2
    --rollout-batch-size 1
    --n-samples-per-prompt 2
    --global-batch-size 2
    --rollout-max-context-len 32768
    --disable-rewards-normalization
)

ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

MEGATRON_ARGS=(
    --transformer-impl local
    --no-rope-fusion
    --no-gradient-accumulation-fusion
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- env LD_LIBRARY_PATH="${G2C_LD_LIBRARY_PATH}" python3 -m relax.entrypoints.train \
    --resource '{"actor":[1,4]}' \
    --num-gpus-per-node 4 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    --use-metrics-service \
    --timeline-dump-dir "${TIMELINE_DIR}" \
    "${MEGATRON_ARGS[@]}"

if [ ! -s "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: G2-C completed without a checkpoint tracker: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [ "$(tr -d '[:space:]' <"${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")" != "1" ]; then
    echo "ERROR: G2-C expected checkpoint iteration 1." >&2
    exit 1
fi
if [ ! -d "${CHECKPOINT_DIR}/iter_0000001" ] || [ -z "$(find "${CHECKPOINT_DIR}/iter_0000001" -type f -print -quit)" ]; then
    echo "ERROR: G2-C checkpoint iteration 1 is missing or empty." >&2
    exit 1
fi
