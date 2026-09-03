# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify the exact four-node WAA fully-async functional smoke from
artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_STEPS = 3
EXPECTED_SAMPLES_PER_STEP = 4
EXPECTED_LEASES = EXPECTED_STEPS * EXPECTED_SAMPLES_PER_STEP
EXPECTED_PROVIDER_CONTRACT = {
    "mrope_section": [24, 20, 20],
    "position_embedding_type": "mrope",
    "rotary_base": 5_000_000,
    "share_embeddings_and_output_weights": True,
}
FATAL_MARKERS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "DistNetworkError",
    "NCCL error",
    "strict weight-publication health gate failed",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"non-object JSONL row in {path}")
    return rows


def _validate_preflight(exp_dir: Path, hosts: set[str]) -> dict[str, Any]:
    paths = sorted((exp_dir / "training_env_preflight").glob("*.json"))
    rows = [_read_json(path) for path in paths]
    observed_hosts = {str(row.get("hostname")) for row in rows}
    if len(rows) != 4 or observed_hosts != hosts:
        raise RuntimeError(f"training preflight does not cover the four allocated hosts: {observed_hosts} != {hosts}")
    for row in rows:
        if row.get("schema_version") != "waa.training_env_preflight.v1":
            raise RuntimeError(f"invalid training preflight schema: {row}")
        if row.get("contract") != EXPECTED_PROVIDER_CONTRACT:
            raise RuntimeError(f"Qwen3-VL provider mismatch on {row.get('hostname')}: {row.get('contract')}")
        if not all(row.get(key) not in (None, "", "unknown") for key in ("ray_version", "sglang_version")):
            raise RuntimeError(f"missing runtime version in training preflight: {row}")
    return {"preflight_hosts": sorted(observed_hosts), "provider_contract": EXPECTED_PROVIDER_CONTRACT}


def _validate_brokers(exp_dir: Path, hosts: set[str]) -> dict[str, Any]:
    paths = sorted((exp_dir / "broker_events").glob("broker-*.jsonl"))
    rows_by_host = {path.stem.removeprefix("broker-"): _read_jsonl(path) for path in paths}
    if set(rows_by_host) != hosts:
        raise RuntimeError(f"broker event hosts differ from allocation: {set(rows_by_host)} != {hosts}")
    rows = [row for host_rows in rows_by_host.values() for row in host_rows]
    invalid_schema = [row for row in rows if row.get("schema_version") != "waa.broker_event.v1"]
    if invalid_schema:
        raise RuntimeError(f"invalid broker event schema: {invalid_schema[:1]}")
    if any(row.get("event") == "lease_failed" for row in rows):
        raise RuntimeError("at least one WAA lease failed during the formal smoke")

    by_event: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[str(row.get("event"))].append(row)
    ready = by_event["lease_ready"]
    evaluated = by_event["lease_evaluated"]
    released = by_event["lease_released"]
    if (len(ready), len(evaluated), len(released)) != (EXPECTED_LEASES, EXPECTED_LEASES, EXPECTED_LEASES):
        raise RuntimeError(
            "formal smoke requires exactly 12 ready/evaluated/released leases: "
            f"ready={len(ready)}, evaluated={len(evaluated)}, released={len(released)}"
        )
    ready_ids = {str(row.get("lease_id")) for row in ready}
    evaluated_ids = {str(row.get("lease_id")) for row in evaluated}
    released_ids = {str(row.get("lease_id")) for row in released}
    if len(ready_ids) != EXPECTED_LEASES or evaluated_ids != ready_ids or released_ids != ready_ids:
        raise RuntimeError("ready/evaluated/released lease identities do not match exactly")
    if any(not 0 < float(row.get("acquire_seconds", math.inf)) < 360 for row in ready):
        raise RuntimeError("at least one WAA fresh lease missed the 360-second cold-start gate")
    if any(int(row.get("width", 0)) <= 0 or int(row.get("height", 0)) <= 0 for row in ready):
        raise RuntimeError("at least one WAA lease lacks valid screenshot dimensions")
    scores = [row.get("score") for row in evaluated]
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 1
        for score in scores
    ):
        raise RuntimeError(f"invalid native WAA scores: {scores}")
    ready_per_host = Counter(str(row.get("hostname")) for row in ready)
    if set(ready_per_host) != hosts or any(count <= 0 for count in ready_per_host.values()):
        raise RuntimeError(f"not every broker served a fresh lease: {ready_per_host}")
    return {
        "broker_ready_per_host": dict(sorted(ready_per_host.items())),
        "lease_count": len(ready),
        "native_rewards": [float(score) for score in scores],
    }


def _validate_rollouts(exp_dir: Path) -> dict[str, Any]:
    paths = sorted((exp_dir / "rollout_result" / "train").glob("*.jsonl"))
    rows_by_step = {int(path.stem): _read_jsonl(path) for path in paths if path.stem.isdigit()}
    if set(rows_by_step) != set(range(EXPECTED_STEPS)):
        raise RuntimeError(f"rollout result steps are not exactly 0..2: {set(rows_by_step)}")
    rewards: list[float] = []
    weight_versions: dict[int, list[str]] = {}
    for step, rows in rows_by_step.items():
        if len(rows) != EXPECTED_SAMPLES_PER_STEP:
            raise RuntimeError(f"rollout {step} has {len(rows)} samples instead of four")
        groups = Counter(row.get("group_index") for row in rows)
        if sorted(groups.values()) != [EXPECTED_SAMPLES_PER_STEP]:
            raise RuntimeError(f"rollout {step} is not one GRPO group of four: {groups}")
        for row in rows:
            if row.get("status") != "completed":
                raise RuntimeError(f"non-completed rollout row: {row.get('status')}")
            reward = row.get("reward")
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(float(reward))
                or not 0 <= float(reward) <= 1
            ):
                raise RuntimeError(f"invalid committed reward: {reward}")
            rewards.append(float(reward))
            if int(row.get("image_count", 0)) <= 0:
                raise RuntimeError("committed agentic rollout lacks screenshot input")
            mm = row.get("multimodal_train_inputs")
            if not isinstance(mm, dict) or not {"pixel_values", "image_grid_thw"}.issubset(mm):
                raise RuntimeError(f"committed rollout lacks processed Qwen3-VL tensors: {mm}")
            versions = row.get("weight_versions")
            if not isinstance(versions, list) or not versions:
                raise RuntimeError("committed rollout lacks policy weight-version provenance")
        weight_versions[step] = sorted({str(version) for row in rows for version in row["weight_versions"]})
    return {"committed_rewards": rewards, "rollout_samples": len(rewards), "weight_versions": weight_versions}


def _timeline_steps(exp_dir: Path, name: str) -> set[int]:
    steps: set[int] = set()
    for path in (exp_dir / "timeline").glob("*.json"):
        value = _read_json(path)
        if not isinstance(value, list):
            continue
        for event in value:
            if not isinstance(event, dict) or event.get("name") != name:
                continue
            step = (event.get("args") or {}).get("step")
            if isinstance(step, int) and not isinstance(step, bool):
                steps.add(step)
    return steps


def _validate_training(exp_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    log_paths = sorted((exp_dir / "logs").glob("driver-*.log"))
    if len(log_paths) != 1:
        raise RuntimeError(f"expected exactly one driver log, found {len(log_paths)}")
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
    fatal = [marker for marker in FATAL_MARKERS if marker in log_text]
    if fatal:
        raise RuntimeError(f"driver log contains fatal markers: {fatal}")
    required_flags = ("--fully-async", "--max-staleness 0", "--num-rollout 3")
    missing_flags = [flag for flag in required_flags if flag not in log_text]
    if missing_flags:
        raise RuntimeError(f"driver log lacks frozen fully-async flags: {missing_flags}")

    expected = set(range(EXPECTED_STEPS))
    optimizer_steps = _timeline_steps(exp_dir, "critical_path.optimizer_step")
    weight_steps = _timeline_steps(exp_dir, "critical_path.weight_update")
    ready_steps = _timeline_steps(exp_dir, "critical_path.weight_serving_ready")
    if optimizer_steps != expected or weight_steps != expected or ready_steps != expected:
        raise RuntimeError(
            "training timeline is incomplete: "
            f"optimizer={optimizer_steps}, weight_update={weight_steps}, serving_ready={ready_steps}"
        )
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text(encoding="utf-8").strip()) != EXPECTED_STEPS - 1:
        raise RuntimeError(f"final checkpoint tracker is missing or wrong: {tracker}")
    return {
        "checkpoint_iteration": EXPECTED_STEPS - 1,
        "optimizer_steps": sorted(optimizer_steps),
        "weight_serving_ready_steps": sorted(ready_steps),
        "weight_update_steps": sorted(weight_steps),
    }


def _validate_cleanup(exp_dir: Path, hosts: set[str]) -> dict[str, Any]:
    paths = sorted((exp_dir / "cleanup_audit").glob("*.json"))
    rows = [_read_json(path) for path in paths]
    observed_hosts = {str(row.get("hostname")) for row in rows}
    if len(rows) != 4 or observed_hosts != hosts:
        raise RuntimeError(f"cleanup audit does not cover four hosts: {observed_hosts} != {hosts}")
    for row in rows:
        if row.get("schema_version") != "waa.cleanup_audit.v1":
            raise RuntimeError(f"invalid cleanup audit schema: {row}")
        if row.get("containers") or row.get("broker_manifests"):
            raise RuntimeError(f"WAA cleanup left containers/manifests: {row}")
        if row.get("node_root_absent") is not True or row.get("token_absent") is not True:
            raise RuntimeError(f"WAA cleanup left overlay/token state: {row}")
    return {"cleanup_hosts": sorted(observed_hosts), "cleanup_status": "zero_residuals"}


def _slurm_accounting(job_id: str) -> dict[str, Any]:
    output = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ElapsedRaw,AllocTRES",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    rows = [line.split("|") for line in output.splitlines() if line.strip()]
    matches = [row for row in rows if row[0] == job_id]
    if len(matches) != 1 or len(matches[0]) < 4:
        raise RuntimeError(f"cannot identify top-level Slurm accounting row for {job_id}: {rows}")
    _, state, elapsed_raw, alloc_tres = matches[0][:4]
    gpu_match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", alloc_tres)
    if state != "COMPLETED" or gpu_match is None:
        raise RuntimeError(f"formal Slurm job did not complete with GPU accounting: state={state}, TRES={alloc_tres}")
    elapsed_seconds = int(elapsed_raw)
    gpu_count = int(gpu_match.group(1))
    if gpu_count != 16:
        raise RuntimeError(f"formal smoke allocated {gpu_count} GPUs instead of 16")
    return {
        "alloc_tres": alloc_tres,
        "elapsed_seconds": elapsed_seconds,
        "formal_gpu_hours": elapsed_seconds * gpu_count / 3600,
        "gpu_count": gpu_count,
        "slurm_state": state,
    }


def validate(
    exp_dir: Path,
    checkpoint_dir: Path,
    job_id: str,
    *,
    preliminary_gpu_hours: float,
    gpu_hour_budget: float,
) -> dict[str, Any]:
    hosts_path = exp_dir / "hosts"
    hosts = {line.strip() for line in hosts_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(hosts) != 4:
        raise RuntimeError(f"formal smoke did not allocate four distinct hosts: {hosts}")
    accounting = _slurm_accounting(job_id)
    total_gpu_hours = preliminary_gpu_hours + accounting["formal_gpu_hours"]
    if total_gpu_hours > gpu_hour_budget:
        raise RuntimeError(f"GPU-hour budget exceeded: {total_gpu_hours:.6f} > {gpu_hour_budget:.6f}")

    report: dict[str, Any] = {
        "claim_scope": "pinned_notepad_fully_async_functional_smoke_not_learning_or_full_corpus",
        "gpu_hour_budget": gpu_hour_budget,
        "job_id": job_id,
        "preliminary_gpu_hours": preliminary_gpu_hours,
        "schema_version": "waa.formal_verification.v1",
        "status": "passed",
        "total_gpu_hours": total_gpu_hours,
    }
    report.update(accounting)
    report.update(_validate_preflight(exp_dir, hosts))
    report.update(_validate_brokers(exp_dir, hosts))
    report.update(_validate_rollouts(exp_dir))
    report.update(_validate_training(exp_dir, checkpoint_dir))
    report.update(_validate_cleanup(exp_dir, hosts))
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--preliminary-gpu-hours", type=float, default=76 * 4 / 3600)
    parser.add_argument("--gpu-hour-budget", type=float, default=80.0)
    args = parser.parse_args()
    output = args.exp_dir / "formal_verification.json"
    try:
        report = validate(
            args.exp_dir,
            args.checkpoint_dir,
            args.job_id,
            preliminary_gpu_hours=args.preliminary_gpu_hours,
            gpu_hour_budget=args.gpu_hour_budget,
        )
    except Exception as exc:
        _write_report(
            output,
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "job_id": args.job_id,
                "schema_version": "waa.formal_verification.v1",
                "status": "failed",
            },
        )
        raise
    _write_report(output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
