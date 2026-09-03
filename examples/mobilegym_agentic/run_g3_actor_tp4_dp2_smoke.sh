#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# G3-B: actor-only Qwen3-VL TP4 x DP2 across two four-GPU nodes. Replays
# completed MobileGym rollout tensors without starting rollout/Judge services.
# Run once with G3C_MODE=train, then from a fresh allocation with
# G3C_MODE=reload and the same G3C_ROOT.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

for required_name in MODEL_DIR G3C_ROOT CAPTURE_PT RUNTIME_ENV_JSON CUDNN_LIB_DIR; do
    if [ -z "${!required_name:-}" ]; then
        echo "ERROR: ${required_name} must be set." >&2
        exit 1
    fi
done
if [ ! -s "${CAPTURE_PT}" ]; then
    echo "ERROR: completed debug rollout is missing or empty: ${CAPTURE_PT}" >&2
    exit 1
fi
python3 -c '
import sys

import torch

capture = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
samples = capture.get("samples")
if not isinstance(samples, list) or len(samples) != 4:
    raise RuntimeError(f"expected four replay samples, got {type(samples).__name__}:{len(samples or [])}")
for index, sample in enumerate(samples):
    if sample.get("status") != "completed" or not sample.get("tokens"):
        raise RuntimeError(f"replay sample {index} is not completed or has no tokens")
    multimodal = sample.get("multimodal_train_inputs") or {}
    for key in ("pixel_values", "image_grid_thw"):
        value = multimodal.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            raise RuntimeError(f"replay sample {index} lacks tensor {key}")
' "${CAPTURE_PT}"

G3C_MODE="${G3C_MODE:-train}"
if [ "${G3C_MODE}" != "train" ] && [ "${G3C_MODE}" != "reload" ]; then
    echo "ERROR: G3C_MODE must be train or reload, got ${G3C_MODE}." >&2
    exit 1
fi

CHECKPOINT_DIR="${G3C_ROOT}/checkpoint"
TIMELINE_DIR="${G3C_ROOT}/${G3C_MODE}_timeline"
DRIVER_LOG="${G3C_ROOT}/${G3C_MODE}_driver.log"
mkdir -p "${G3C_ROOT}" "${TIMELINE_DIR}"
if [ "${G3C_MODE}" = "train" ] && [ -e "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: G3 actor train requires a fresh checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [ "${G3C_MODE}" = "reload" ] && [ ! -s "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: G3 actor reload requires a completed checkpoint: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

export RELAX_EXTRA_ENV_ALLOWLIST="${RELAX_EXTRA_ENV_ALLOWLIST:+${RELAX_EXTRA_ENV_ALLOWLIST},}RELAX_ENV_ROOT,RELAX_SPMD_COMPLETION_FILE"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDNN_LIB_DIR,LD_LIBRARY_PATH,RELAX_EXTRA_ENV_ALLOWLIST,RELAX_PROPAGATE_ENV_VARS"
G3C_LD_LIBRARY_PATH="${CUDNN_LIB_DIR}"
while IFS= read -r library_dir; do
    if [ -n "${library_dir}" ] && [ "${library_dir}" != "${CUDNN_LIB_DIR}" ]; then
        G3C_LD_LIBRARY_PATH="${G3C_LD_LIBRARY_PATH}:${library_dir}"
    fi
done < <(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}")
export G3C_LD_LIBRARY_PATH
RUNTIME_ENV_JSON="$(
    python3 -c '
import json
import os
import sys

runtime_env = json.load(sys.stdin)
env_vars = runtime_env.setdefault("env_vars", {})
for name in ("CUDNN_LIB_DIR", "RELAX_EXTRA_ENV_ALLOWLIST", "RELAX_PROPAGATE_ENV_VARS"):
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
if [ "${G3C_MODE}" = "reload" ]; then
    CKPT_ARGS+=(--load "${CHECKPOINT_DIR}")
fi

DATA_ARGS=(
    --load-debug-rollout-data "${CAPTURE_PT}"
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --num-rollout 2
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --global-batch-size 4
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

# The completion marker belongs to the outer Slurm SPMD process. Do not expose
# it to the nested Ray Job; the runtime allow-list is retained for Ray actors.
unset RELAX_SPMD_COMPLETION_FILE
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- env LD_LIBRARY_PATH="${G3C_LD_LIBRARY_PATH}" python3 -m relax.entrypoints.train \
    --resource '{"actor":[1,8]}' \
    --num-gpus-per-node 4 \
    --actor-num-nodes 2 \
    --actor-num-gpus-per-node 4 \
    --distributed-timeout-minutes 5 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    --use-metrics-service \
    --timeline-dump-dir "${TIMELINE_DIR}" \
    "${MEGATRON_ARGS[@]}" \
    2>&1 | tee "${DRIVER_LOG}"

python3 "${SCRIPT_DIR}/scripts/check_g3_actor_tp4_dp2.py" \
    --root "${G3C_ROOT}" \
    --mode "${G3C_MODE}" \
    --driver-log "${DRIVER_LOG}"
