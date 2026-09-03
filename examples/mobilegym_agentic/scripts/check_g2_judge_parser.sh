#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Validate the dedicated G2 Judge driver's normal Relax argument path without
# starting Ray or allocating Judge placement groups. Run inside the EDF image.

set -euo pipefail

RELAX_REPO_DIR="${RELAX_REPO_DIR:-/users/${USER}/haodong/framework/Relax}"
RELAX_ENV_ROOT="${RELAX_ENV_ROOT:-/iopsstor/scratch/cscs/${USER}/mobilegym_e2e/g2_relax_env_te214_sm90_cuda13_v2}"
VENV_BIN="${RELAX_ENV_ROOT}/relax_venv/bin"
VENV_SITE="$("${VENV_BIN}/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDNN_LIB_DIR="${RELAX_ENV_ROOT}/relax_venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
export CUDNN_LIB_DIR

export PATH="${VENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${RELAX_REPO_DIR}:${RELAX_ENV_ROOT}/Megatron-LM:${RELAX_ENV_ROOT}/TransferQueue:${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
export RELAX_JUDGE_GPU_SAMPLE_DIR="${RELAX_ENV_ROOT}/g2_parser_gpu_samples"
export RELAX_PROPAGATE_ENV_VARS="LD_LIBRARY_PATH,RELAX_JUDGE_GPU_SAMPLE_DIR"

cd "${RELAX_REPO_DIR}"
python3.12 -m py_compile examples/mobilegym_agentic/g2_judge_tp2_smoke.py
python3.12 - <<'PY'
import importlib.util
import json
import os
import sys

import yaml

from relax.utils.arguments import parse_args
from relax.utils.judge_config import parse_judge_services_config
from relax.utils.utils import post_process_env

driver_spec = importlib.util.spec_from_file_location(
    "g2_judge_tp2_smoke", "examples/mobilegym_agentic/g2_judge_tp2_smoke.py"
)
assert driver_spec is not None and driver_spec.loader is not None
driver_module = importlib.util.module_from_spec(driver_spec)
driver_spec.loader.exec_module(driver_module)

with open("examples/mobilegym_agentic/judge_services_e2e_terminal_once.json", encoding="utf-8") as file:
    judge_config = file.read().replace("\n", "")

sys.argv = [
    "g2-judge-parser",
    "--skip-hf-validate",
    "--num-gpus-per-node",
    "4",
    "--actor-num-gpus-per-node",
    "1",
    "--rollout-num-gpus-per-engine",
    "1",
    "--rollout-batch-size",
    "1",
    "--global-batch-size",
    "1",
    "--n-samples-per-prompt",
    "1",
    "--num-rollout",
    "1",
    "--resource",
    '{"judge_accuracy":[1,2],"judge_multiturn_vlm":[1,2]}',
    "--rm-type",
    "dual-agentic-judge",
    "--judge-services-config",
    judge_config,
    "--use-agentic-rollout",
    "--agent-command",
    "/bin/true",
    "--agent-cwd",
    "examples/mobilegym_agentic",
    "--reward-key",
    "score",
    "--swiglu",
    "--num-layers",
    "36",
    "--hidden-size",
    "2560",
    "--ffn-hidden-size",
    "9728",
    "--num-attention-heads",
    "32",
    "--group-query-attention",
    "--num-query-groups",
    "8",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--rotary-base",
    "5000000",
    "--vocab-size",
    "151936",
    "--kv-channels",
    "128",
    "--qk-layernorm",
]
args = parse_args()
driver_module._prepend_unique_library_path(os.environ["CUDNN_LIB_DIR"])
assert set(args.resource) == {"judge_accuracy", "judge_multiturn_vlm"}
assert args.judge_services.by_role("judge_accuracy").num_gpus_per_engine == 2
assert args.judge_services.by_role("judge_multiturn_vlm").num_gpus_per_engine == 2
for trigger in ("terminal_once", "per_turn"):
    with open(f"examples/mobilegym_agentic/judge_services_e2e_{trigger}.json", encoding="utf-8") as file:
        config = parse_judge_services_config(json.load(file))
    assert config.reasoning_trigger == trigger
    assert all(config.by_role(role).num_gpus_per_engine == 2 for role in ("judge_accuracy", "judge_multiturn_vlm"))
with open("configs/env.yaml", encoding="utf-8") as file:
    runtime_env = post_process_env(args, yaml.safe_load(file))
assert runtime_env["env_vars"]["LD_LIBRARY_PATH"].split(":")[0].endswith("nvidia/cudnn/lib")
assert runtime_env["env_vars"]["RELAX_JUDGE_GPU_SAMPLE_DIR"].endswith("g2_parser_gpu_samples")
print(f"G2 parser passed: {args.resource}")
PY
