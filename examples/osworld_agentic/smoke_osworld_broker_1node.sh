#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Zero-GPU OSWorld environment smoke; native start/evaluate commands are trusted.

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

repo_dir="${RELAX_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${repo_dir:?RELAX_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${OSWORLD_REPO_DIR:?OSWORLD_REPO_DIR is required}"
: "${OSWORLD_BROKER_START_COMMAND_JSON:?OSWORLD_BROKER_START_COMMAND_JSON is required}"
: "${OSWORLD_EVALUATE_COMMAND_JSON:?OSWORLD_EVALUATE_COMMAND_JSON is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${EXP_DIR:?EXP_DIR is required}"

adapter_dir="${repo_dir}/examples/osworld_agentic"
export RELAX_REPO_DIR="${repo_dir}" OSWORLD_TRUSTED_REGISTRY="${DATA_DIR}/osworld/trusted_registry.json"
export OSWORLD_BROKER_MANIFEST_DIR="${EXP_DIR}/brokers" OSWORLD_BROKER_TOKEN_FILE="${EXP_DIR}/broker.token"
export OSWORLD_BROKER_LEASE_ROOT="${EXP_DIR}/leases"
mkdir -p "${OSWORLD_BROKER_MANIFEST_DIR}" "${OSWORLD_BROKER_LEASE_ROOT}"
umask 077
od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"${OSWORLD_BROKER_TOKEN_FILE}"
chmod 600 "${OSWORLD_BROKER_TOKEN_FILE}"
/usr/bin/python3.11 "${adapter_dir}/scripts/build_dataset.py" --osworld-repo "${OSWORLD_REPO_DIR}" --output-dir "${DATA_DIR}/osworld"

broker_pid=""
cleanup() {
    exit_code=$?
    trap - EXIT INT TERM
    if [ -n "${broker_pid}" ]; then kill -TERM "${broker_pid}" 2>/dev/null || true; wait "${broker_pid}" 2>/dev/null || true; fi
    if compgen -G "${OSWORLD_BROKER_MANIFEST_DIR}/broker-*.json" >/dev/null; then exit_code=1; fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM
bash "${adapter_dir}/scripts/start_node_broker.sh" &
broker_pid=$!
for _ in $(seq 1 120); do
    compgen -G "${OSWORLD_BROKER_MANIFEST_DIR}/broker-*.json" >/dev/null && break
    sleep 1
done
manifest=$(printf '%s\n' "${OSWORLD_BROKER_MANIFEST_DIR}"/broker-*.json)
[ -f "${manifest}" ] || { echo "ERROR: OSWorld broker did not become ready" >&2; exit 1; }

/usr/bin/python3.11 - <<'PY'
import json
import os
import urllib.request
import uuid
from pathlib import Path

manifest = json.loads(next(Path(os.environ["OSWORLD_BROKER_MANIFEST_DIR"]).glob("broker-*.json")).read_text())
token = Path(os.environ["OSWORLD_BROKER_TOKEN_FILE"]).read_text().strip()
registry = json.loads(Path(os.environ["OSWORLD_TRUSTED_REGISTRY"]).read_text())
task_id, task = next(iter(registry["tasks"].items()))
base = manifest["broker_url"].rstrip("/")

def post(path, value):
    request = urllib.request.Request(base + path, data=json.dumps(value).encode(), method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())

lease = post("/v1/lease", {"request_id": uuid.uuid4().hex, "task_id": task_id,
                           "task_manifest_digest": task["task_manifest_digest"]})
fields = {"lease_id": lease["lease_id"], "generation": lease["generation"]}
try:
    assert lease["width"] == 1920 and lease["height"] == 1080
    result = post("/v1/action", {**fields, "action": {"type": "wait"}})
    assert result["terminal"] is False
    print(json.dumps({"task_id": task_id, "status": "environment_ready"}))
finally:
    post("/v1/release", fields)
PY
