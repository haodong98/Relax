# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the pinned structured-action episode against one live WAA broker."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path

from app.client import _png_dimensions, _post_json
from app.notepad_smoke_env import PINNED_NOTEPAD_TASK_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = sorted(args.manifest_dir.glob("broker-*.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one broker manifest, found {len(manifests)}")
    broker_url = json.loads(manifests[0].read_text(encoding="utf-8"))["broker_url"].rstrip("/")
    token = args.token_file.read_text(encoding="utf-8").strip()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    task = registry["tasks"][PINNED_NOTEPAD_TASK_ID]
    acquire_started = time.monotonic()
    request_id = uuid.uuid4().hex
    lease = _post_json(
        f"{broker_url}/v1/lease",
        {
            "request_id": request_id,
            "task_id": PINNED_NOTEPAD_TASK_ID,
            "task_manifest_digest": task["task_manifest_digest"],
        },
        token=token,
        timeout=360,
    )
    acquire_seconds = time.monotonic() - acquire_started
    lease_fields = {"generation": lease["generation"], "lease_id": lease["lease_id"]}
    evaluated: dict | None = None
    try:
        dimensions = _png_dimensions(lease["screenshot"])
        if dimensions != (int(lease["width"]), int(lease["height"])):
            raise RuntimeError("live WAA lease returned unexpected screenshot dimensions")
        waited = _post_json(
            f"{broker_url}/v1/action",
            {**lease_fields, "action": {"type": "wait"}},
            token=token,
            timeout=180,
        )
        if waited.get("terminal") is not False or _png_dimensions(waited["screenshot"]) != dimensions:
            raise RuntimeError("live WAA wait action failed its screenshot contract")
        done = _post_json(
            f"{broker_url}/v1/action",
            {**lease_fields, "action": {"type": "done"}},
            token=token,
            timeout=180,
        )
        if done.get("terminal") is not True:
            raise RuntimeError("live WAA done action was not terminal")
        evaluated = _post_json(f"{broker_url}/v1/evaluate", lease_fields, token=token, timeout=300)
        score = evaluated.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RuntimeError("live WAA evaluator returned an invalid score")
        if float(score) != 0.0 or evaluated.get("reason") != "and_short_circuit":
            raise RuntimeError(f"clean golden image unexpectedly scored the no-op episode: {evaluated}")
    finally:
        _post_json(f"{broker_url}/v1/release", lease_fields, token=token, timeout=240)

    result = {
        "actions": ["wait", "done"],
        "acquire_seconds": acquire_seconds,
        "evaluator": evaluated,
        "schema_version": "waa.live_broker_smoke.v1",
        "screenshot_dimensions": list(dimensions),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task_id": PINNED_NOTEPAD_TASK_ID,
        "timestamp_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
