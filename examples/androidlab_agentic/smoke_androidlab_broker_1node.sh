#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Zero-GPU AndroidLab environment smoke; query tasks are intentionally excluded.

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --job-name=relax-androidlab-env
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

repo_dir="${RELAX_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${repo_dir:?RELAX_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${ANDROIDLAB_REPO_DIR:?ANDROIDLAB_REPO_DIR is required}"
: "${ANDROIDLAB_BROKER_START_COMMAND_JSON:?ANDROIDLAB_BROKER_START_COMMAND_JSON is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"

adapter_dir="${repo_dir}/examples/androidlab_agentic"
export RELAX_REPO_DIR="${repo_dir}"
export ANDROIDLAB_TRUSTED_REGISTRY="${DATA_DIR}/androidlab/trusted_registry.json"
export ANDROIDLAB_BROKER_MANIFEST_DIR="${EXP_DIR}/brokers"
export ANDROIDLAB_BROKER_EVENT_DIR="${EXP_DIR}/broker_events"
export ANDROIDLAB_BROKER_TOKEN_FILE="${EXP_DIR}/broker.token"
export ANDROIDLAB_BROKER_LEASE_ROOT="${EXP_DIR}/leases"
mkdir -p "${ANDROIDLAB_BROKER_MANIFEST_DIR}" "${ANDROIDLAB_BROKER_EVENT_DIR}"
umask 077
od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"${ANDROIDLAB_BROKER_TOKEN_FILE}"
chmod 600 "${ANDROIDLAB_BROKER_TOKEN_FILE}"

/usr/bin/python3.11 "${adapter_dir}/scripts/build_dataset.py" \
    --androidlab-repo "${ANDROIDLAB_REPO_DIR}" --output-dir "${DATA_DIR}/androidlab"

broker_pid=""
cleanup() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ -n "${broker_pid}" ]; then
        kill -TERM "${broker_pid}" 2>/dev/null || true
        wait "${broker_pid}" 2>/dev/null || true
    fi
    if find "${ANDROIDLAB_BROKER_MANIFEST_DIR}" -name 'broker-*.json' -print -quit | grep -q .; then
        echo "ERROR: AndroidLab broker manifests remained after cleanup" >&2
        exit_code=1
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

bash "${adapter_dir}/scripts/start_node_broker.sh" &
broker_pid=$!
/usr/bin/python3.11 "${adapter_dir}/scripts/wait_brokers.py" \
    --manifest-dir "${ANDROIDLAB_BROKER_MANIFEST_DIR}" --expected 1 --timeout 600

# The operation smoke is deliberately action-only and does not require a query judge.
PYTHONPATH="${adapter_dir}:${PYTHONPATH:-}" /usr/bin/python3.11 - <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

manifest_dir = Path(os.environ["ANDROIDLAB_BROKER_MANIFEST_DIR"])
manifest = json.loads(next(manifest_dir.glob("broker-*.json")).read_text())
token = Path(os.environ["ANDROIDLAB_BROKER_TOKEN_FILE"]).read_text().strip()
registry = json.loads(Path(os.environ["ANDROIDLAB_TRUSTED_REGISTRY"]).read_text())
task_id = next(task_id for task_id, task in registry["tasks"].items() if task.get("metric_type") != "query_detect")
task = registry["tasks"][task_id]
base = manifest["broker_url"].rstrip("/")

def post(path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())

lease = post("/v1/lease", {"request_id": "env-smoke", "task_id": task_id,
                           "task_manifest_digest": task["task_manifest_digest"]})
fields = {"generation": lease["generation"], "lease_id": lease["lease_id"]}
try:
    assert lease["width"] > 0 and lease["height"] > 0
    waited = post("/v1/action", {**fields, "action": {"type": "wait"}})
    assert waited["terminal"] is False
    done = post("/v1/action", {**fields, "action": {"type": "done"}})
    assert done["terminal"] is True
    outcome = post("/v1/evaluate", fields)
    assert isinstance(outcome.get("score"), (int, float))
finally:
    post("/v1/release", fields)
print(json.dumps({"task_id": task_id, "outcome": outcome, "timestamp": time.time()}))
PY
