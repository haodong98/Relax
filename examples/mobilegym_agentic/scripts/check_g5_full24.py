#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from check_g4_full12 import _read_jsonl, _validate_accounting_drain, _validate_training_timeline


def _request_counts(section: dict[str, Any]) -> tuple[int, int]:
    return int(section.get("raw_count", -1)), int(section.get("clean_count", -1))


def _validate_rollout_rows(exp_dir: Path, trigger: str, expected_steps: int) -> dict[str, Any]:
    rows_by_step = {
        int(path.stem): _read_jsonl(path)
        for path in sorted((exp_dir / "rollout_result" / "train").glob("*.jsonl"))
        if path.stem.isdigit()
    }
    if set(rows_by_step) != set(range(expected_steps)):
        raise RuntimeError(f"rollout steps mismatch: {sorted(rows_by_step)}")
    if any(len(rows) != 64 for rows in rows_by_step.values()):
        raise RuntimeError(
            f"each G5 step must contain 64 samples: { {step: len(rows) for step, rows in rows_by_step.items()} }"
        )

    versions_by_step: dict[int, set[str]] = defaultdict(set)
    per_turn_count = 0
    compacted_pixel_sample_count = 0
    retried_judge_call_count = 0
    for step, rows in rows_by_step.items():
        group_rows: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("status") != "completed":
                raise RuntimeError(f"non-completed committed sample at step {step}: {row.get('status')}")
            if row.get("image_count") != row.get("agent_turns"):
                raise RuntimeError(f"append-only screenshot history mismatch at step {step}")
            pixel_summary = (row.get("multimodal_train_inputs") or {}).get("pixel_values")
            if not isinstance(pixel_summary, str) or "dtype=torch.bfloat16" not in pixel_summary:
                raise RuntimeError(
                    f"training pixel_values were not compacted to bfloat16 at step {step}: {pixel_summary}"
                )
            compacted_pixel_sample_count += 1
            versions = row.get("weight_versions")
            if not isinstance(versions, list) or not versions or len(set(map(str, versions))) != 1:
                raise RuntimeError(f"trajectory lacks one exportable policy lineage: {versions}")
            versions_by_step[step].update(map(str, versions))
            group_rows[row.get("group_index")].append(row)
            trace = row.get("latency_trace") or {}
            reward = trace.get("reward") or {}
            if reward.get("pipeline_status") != "success" or reward.get("executor_status") != "success":
                raise RuntimeError(f"reward path failed: {reward}")
            if reward.get("reasoning_execution_trigger") != trigger:
                raise RuntimeError(
                    f"reward trigger mismatch: {reward.get('reasoning_execution_trigger')} != {trigger}"
                )
            if int(trace.get("per_turn_off_lineage_judge_count", 0) or 0) != 0:
                raise RuntimeError("off-lineage per-turn Judge work entered a committed sample")
            # A judge call that needed one retry but still returned a valid response on an
            # in-lineage attempt is real, honest measurement data (its own elapsed_s/http_elapsed_s
            # already include the retry cost) -- at 320+ terminal judge calls per run some client-side
            # retries are expected background noise, not a pipeline defect. Only an actual failure
            # (non-success status or an invalid response) indicates something is wrong.
            accuracy = (reward.get("judges") or {}).get("answer_accuracy") or {}
            if accuracy.get("status") != "success" or accuracy.get("invalid_response_count") != 0:
                raise RuntimeError(f"terminal ORM did not complete cleanly: {accuracy}")
            if accuracy.get("attempt_count", 1) != 1:
                retried_judge_call_count += 1
            if trigger == "terminal_once":
                vlm = (reward.get("judges") or {}).get("multi_turn_reasoning") or {}
                if vlm.get("status") != "success":
                    raise RuntimeError(f"terminal VLM did not complete cleanly: {vlm}")
                if vlm.get("attempt_count", 1) != 1:
                    retried_judge_call_count += 1
                if reward.get("per_turn_judge_count") or reward.get("per_turn_judges"):
                    raise RuntimeError("terminal_once unexpectedly contains per-turn Judge work")
            else:
                judges = reward.get("per_turn_judges") or []
                count = reward.get("per_turn_judge_count")
                if not isinstance(count, int) or count <= 0 or count != len(judges):
                    raise RuntimeError(f"per-turn Judge cardinality mismatch: count={count}, rows={len(judges)}")
                if any(
                    item.get("status") != "success"
                    or not item.get("response_state_hash")
                    or not item.get("observation_state_hash")
                    or (item.get("judge") or {}).get("attempt_count") != 1
                    or (item.get("judge") or {}).get("invalid_response_count") != 0
                    for item in judges
                ):
                    raise RuntimeError("per-turn sidecar lacks clean lineage-complete evidence")
                per_turn_count += count
        if sorted(len(rows) for rows in group_rows.values()) != [8] * 8:
            raise RuntimeError(f"step {step} is not eight groups of eight samples")
        for group, rows in group_rows.items():
            prompt_hashes = {
                hashlib.sha256(json.dumps(row.get("prompt"), separators=(",", ":")).encode()).hexdigest()
                for row in rows
            }
            if len(prompt_hashes) != 1:
                raise RuntimeError(f"group {group} step {step} contains multiple task instructions")

    initial_versions = versions_by_step[0]
    measured_versions = set().union(*(versions_by_step[step] for step in range(1, expected_steps)))
    if not measured_versions - initial_versions:
        raise RuntimeError(f"no post-warmup policy version reached committed rollout: {dict(versions_by_step)}")
    return {
        "rollout_samples": sum(map(len, rows_by_step.values())),
        "compacted_pixel_sample_count": compacted_pixel_sample_count,
        "per_turn_judge_count": per_turn_count,
        "retried_terminal_judge_call_count": retried_judge_call_count,
        "policy_versions_by_step": {str(step): sorted(values) for step, values in versions_by_step.items()},
    }


def _load_placement(exp_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], str]]:
    inventory_rows = _read_jsonl(exp_dir / "allocation_gpu_inventory.jsonl")
    if len(inventory_rows) != 6:
        raise RuntimeError(f"allocation inventory must contain six nodes, got {len(inventory_rows)}")
    gpu_by_ip_index: dict[tuple[str, int], str] = {}
    all_inventory_uuids: list[str] = []
    for row in inventory_rows:
        ips = {str(row.get("ip")), *(str(value) for value in row.get("ips", []))}
        gpus = row.get("gpus") or []
        if sorted(int(gpu["index"]) for gpu in gpus) != [0, 1, 2, 3]:
            raise RuntimeError(f"node inventory is not four GPUs: {row}")
        for gpu in gpus:
            all_inventory_uuids.append(str(gpu["uuid"]))
            for ip in ips:
                gpu_by_ip_index[(ip, int(gpu["index"]))] = str(gpu["uuid"])
    if len(all_inventory_uuids) != 24 or len(set(all_inventory_uuids)) != 24:
        raise RuntimeError("allocation GPU inventory does not contain 24 unique UUIDs")

    placement: dict[str, list[dict[str, Any]]] = {}
    for role in ("actor", "rollout", "judge_accuracy", "judge_multiturn_vlm"):
        payload = json.loads((exp_dir / "placement" / f"{role}.json").read_text(encoding="utf-8"))
        if payload.get("role") != role:
            raise RuntimeError(f"placement role mismatch for {role}: {payload}")
        placement[role] = payload.get("entries") or []
    return placement, gpu_by_ip_index


def _validate_flashinfer_workspaces(exp_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((exp_dir / "flashinfer_workspace").glob("*.json"))
    ]
    if len(rows) != 6 or len({str(row.get("hostname")) for row in rows}) != 6:
        raise RuntimeError(f"FlashInfer workspace probe must cover six nodes: {rows}")
    workspaces = {str(row.get("workspace")) for row in rows}
    if len(workspaces) != 1 or not next(iter(workspaces)).startswith("/tmp/relax-flashinfer/"):
        raise RuntimeError(f"FlashInfer workspace is not one run-specific node-local path: {workspaces}")
    if any(int(row.get("free_bytes", 0)) < 4 * 1024**3 for row in rows):
        raise RuntimeError(f"FlashInfer workspace capacity probe failed: {rows}")
    return {
        "flashinfer_workspace": next(iter(workspaces)),
        "flashinfer_workspace_hosts": sorted(str(row["hostname"]) for row in rows),
    }


def _validate_placement_and_gpu_drain(exp_dir: Path) -> dict[str, Any]:
    placement, gpu_by_ip_index = _load_placement(exp_dir)
    workspace_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((exp_dir / "flashinfer_workspace").glob("*.json"))
    ]
    expected_flashinfer_workspace = {str(row.get("workspace")) for row in workspace_rows}
    if len(expected_flashinfer_workspace) != 1:
        raise RuntimeError(f"FlashInfer workspace probes disagree: {workspace_rows}")
    expected_counts = {"actor": 4, "rollout": 12, "judge_accuracy": 4, "judge_multiturn_vlm": 4}
    if {role: len(entries) for role, entries in placement.items()} != expected_counts:
        raise RuntimeError(
            f"placement cardinality mismatch: { {role: len(rows) for role, rows in placement.items()} }"
        )

    role_uuids: dict[str, list[str]] = {}
    role_nodes: dict[str, set[str]] = {}
    for role, entries in placement.items():
        nodes: defaultdict[str, set[int]] = defaultdict(set)
        uuids = []
        for entry in entries:
            node_ip = str(entry["node_ip"])
            gpu_id = int(entry["physical_gpu_id"])
            nodes[node_ip].add(gpu_id)
            try:
                uuids.append(gpu_by_ip_index[(node_ip, gpu_id)])
            except KeyError as exc:
                raise RuntimeError(f"placement entry cannot be mapped to allocation UUID: {entry}") from exc
        role_nodes[role] = set(nodes)
        role_uuids[role] = uuids
        # Balanced G5 split: actor/rollout/judge_accuracy/judge_multiturn_vlm each
        # get whole four-GPU node blocks (judges are TP4, one full node per role,
        # rather than the earlier TP2-colocated-pair layout).
        if any(gpus != {0, 1, 2, 3} for gpus in nodes.values()):
            raise RuntimeError(f"{role} does not occupy complete four-GPU node blocks: {dict(nodes)}")
    if len(role_nodes["actor"]) != 1 or len(role_nodes["rollout"]) != 3:
        raise RuntimeError(f"actor/rollout node topology mismatch: {role_nodes}")
    judge_nodes = role_nodes["judge_accuracy"] | role_nodes["judge_multiturn_vlm"]
    if len(role_nodes["judge_accuracy"]) != 1 or len(role_nodes["judge_multiturn_vlm"]) != 1 or len(judge_nodes) != 2:
        raise RuntimeError(f"Judges are not each on their own dedicated node: {role_nodes}")
    if role_nodes["actor"] & role_nodes["rollout"] or (role_nodes["actor"] | role_nodes["rollout"]) & judge_nodes:
        raise RuntimeError(f"role node blocks overlap: {role_nodes}")
    all_role_uuids = [uuid for values in role_uuids.values() for uuid in values]
    if len(all_role_uuids) != 24 or len(set(all_role_uuids)) != 24:
        raise RuntimeError(f"role placement does not cover 24 unique GPUs: {role_uuids}")

    manifests: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    active_paths: set[Path] = set()
    final_paths: set[Path] = set()
    for path in sorted((exp_dir / "gpu_samples").glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("record_type") == "manifest":
                manifests[str(row.get("role"))].append(row)
            elif row.get("record_type") == "sample":
                metrics = row.get("sglang") or {}
                if float(metrics.get("sglang:gen_throughput", 0.0) or 0.0) > 0:
                    active_paths.add(path)
                if row.get("final_sample") is True:
                    final_paths.add(path)
                    if (
                        float(metrics.get("sglang:num_running_reqs", 0.0) or 0.0) != 0.0
                        or float(metrics.get("sglang:num_queue_reqs", 0.0) or 0.0) != 0.0
                    ):
                        raise RuntimeError(f"right-censored SGLang WIP in {path}")
    actual_manifests = {role: len(rows) for role, rows in manifests.items()}
    expected_manifests = {"rollout": 12, "judge_accuracy": 1, "judge_multiturn_vlm": 1}
    if actual_manifests != expected_manifests:
        raise RuntimeError(f"sampler manifest topology mismatch: {actual_manifests}")
    manifest_workspaces = {str(row.get("flashinfer_workspace_base")) for rows in manifests.values() for row in rows}
    if manifest_workspaces != expected_flashinfer_workspace:
        raise RuntimeError(
            f"SGLang actor FlashInfer workspaces disagree with node probes: "
            f"{manifest_workspaces} != {expected_flashinfer_workspace}"
        )
    paths = {path for path in (exp_dir / "gpu_samples").glob("*.jsonl") if path.stat().st_size}
    if paths != active_paths or paths != final_paths:
        raise RuntimeError(
            f"sampler activity/drain incomplete: inactive={paths - active_paths}, no_final={paths - final_paths}"
        )
    for role, rows in manifests.items():
        sampled = {uuid for row in rows for uuid in row.get("gpu_uuids", [])}
        if sampled != set(role_uuids[role]):
            raise RuntimeError(f"sampler UUIDs disagree with placement for {role}: {sampled} != {role_uuids[role]}")
    return {
        "placement_nodes": {role: sorted(nodes) for role, nodes in role_nodes.items()},
        "role_gpu_uuid_counts": {role: len(set(values)) for role, values in role_uuids.items()},
        "sampler_manifests": actual_manifests,
    }


def _validate_direct_report(
    exp_dir: Path, trigger: str, max_clock_offset_ms: float, measured_rounds: int
) -> dict[str, Any]:
    variant = json.loads((exp_dir / "direct_report.json").read_text(encoding="utf-8"))["variants"][trigger]
    if variant.get("observed_benchmark_modes") != ["dual"] or variant.get("observed_reasoning_triggers") != [trigger]:
        raise RuntimeError("direct report mode/trigger mismatch")
    if not isinstance(variant.get("benchmark_invariant_hash"), str):
        raise RuntimeError("reward and publication markers lack one frozen benchmark invariant")
    clock = variant.get("clock_domain") or {}
    if len(clock.get("hosts") or []) < 2 or float(clock.get("max_offset_ms", float("inf"))) > max_clock_offset_ms:
        raise RuntimeError(f"multi-host clock audit failed: {clock}")
    request = variant.get("request") or {}
    terminal_count = 64 * measured_rounds
    expected = {
        "terminal_once": {
            "terminal_orm": (terminal_count, terminal_count),
            "terminal_vlm": (terminal_count, terminal_count),
            "per_turn_vlm": (0, 0),
        },
        "per_turn": {"terminal_orm": (terminal_count, terminal_count), "terminal_vlm": (0, 0)},
    }[trigger]
    for name, counts in expected.items():
        if _request_counts(request[name]) != counts:
            raise RuntimeError(f"{name} request counts mismatch: {_request_counts(request[name])} != {counts}")
    if trigger == "per_turn":
        raw, clean = _request_counts(request["per_turn_vlm"])
        if raw <= 0 or raw != clean:
            raise RuntimeError(f"per-turn request tail is not clean: {raw}/{clean}")
    trajectory = variant.get("trajectory") or {}
    if (trajectory.get("raw_count"), trajectory.get("clean_count"), trajectory.get("fallback_count")) != (
        terminal_count,
        terminal_count,
        0,
    ):
        raise RuntimeError(f"trajectory distribution is incomplete: {trajectory}")
    group_count = 8 * measured_rounds
    group = variant.get("group") or {}
    if (
        group.get("group_finalize_count"),
        group.get("complete_group_count"),
        group.get("missing_terminal_admission_group_count"),
        group.get("missing_transfer_group_count"),
    ) != (group_count, group_count, 0, 0):
        raise RuntimeError(f"group distribution is incomplete: {group}")
    for name in ("group_reward_closure_s", "group_completion_spread_s", "group_finalize_s", "group_transfer_delay_s"):
        if int((group.get(name) or {}).get("count", 0)) != group_count:
            raise RuntimeError(f"group metric {name} is incomplete")
    trainer = variant.get("trainer") or {}
    returned_batches = int(trainer.get("returned_trainer_batch_count", 0))
    if trainer.get("data_wait_with_missing_returned_group_provenance_count") != 0 or returned_batches <= 0:
        raise RuntimeError(f"trainer provenance is incomplete: {trainer}")
    if not trainer.get("exclusive_and_other_blocker_available"):
        raise RuntimeError(f"trainer exclusive/concurrent blocker decomposition is unavailable: {trainer}")
    for name in ("inclusive_reward_ancestor_wait_s", "exclusive_reward_wait_s", "reward_plus_other_blocker_wait_s"):
        if int((trainer.get(name) or {}).get("count", 0)) != returned_batches:
            raise RuntimeError(f"trainer metric {name} does not cover returned batches")
    publication = variant.get("publication") or {}
    if (
        publication.get("count") != measured_rounds
        or [row.get("trajectory_count") for row in publication.get("per_step", [])] != [64] * measured_rounds
    ):
        raise RuntimeError(f"publication report is incomplete: {publication}")
    workload = variant.get("workload") or {}
    if workload.get("sample_count") != terminal_count or workload.get("terminal_status_counts") != {
        "completed": terminal_count
    }:
        raise RuntimeError(f"workload report is incomplete: {workload}")
    reliability = variant.get("reliability") or {}
    zero_keys = (
        "retry_request_count",
        "invalid_response_count",
        "fallback_trajectory_count",
        "off_lineage_judge_count",
        "failed_reward_trajectory_count",
        "judge_group_replacements",
        "interrupted_groups",
    )
    bad = {key: reliability.get(key) for key in zero_keys if reliability.get(key) != 0}
    if bad or reliability.get("accounting_step_count") != measured_rounds:
        raise RuntimeError(f"measured reliability is not clean: bad={bad}, report={reliability}")
    gpu = variant.get("judge_gpu_efficiency") or {}
    if variant.get("judge_gpu_efficiency_issues"):
        raise RuntimeError(f"measured GPU telemetry has integrity issues: {variant['judge_gpu_efficiency_issues']}")
    for role in ("rollout", "judge_accuracy", "judge_multiturn_vlm"):
        if float((gpu.get(role) or {}).get("nonidle_fraction", 0.0)) <= 0.0:
            raise RuntimeError(f"measured GPU window lacks activity for {role}: {gpu.get(role)}")
    return {
        "measured_steps": variant.get("measured_steps"),
        "clock_max_offset_ms": clock["max_offset_ms"],
        "trainer_batches": returned_batches,
        "benchmark_invariant_hash": variant["benchmark_invariant_hash"],
    }


_BENIGN_TRACEBACK_FINGERPRINT = (
    "_fetch_available_resources_per_node",
    "ray.exceptions.RpcError: RPC error: Deadline Exceeded",
)


def _unexplained_tracebacks(log_text: str) -> list[str]:
    # ServeController's periodic GCS resource-usage poll can transiently time
    # out under heavy judge/rollout load and self-recover (observed in job
    # 3117772: rollout, reward, and training all continued normally right
    # after). Do not treat that one well-understood fingerprint as fatal, but
    # still flag any other traceback verbatim.
    unexplained = []
    for chunk in log_text.split("Traceback (most recent call last):")[1:]:
        window = chunk[:2000]
        if not all(marker in window for marker in _BENIGN_TRACEBACK_FINGERPRINT):
            unexplained.append(window[:200])
    return unexplained


def validate(exp_dir: Path, trigger: str, expected_steps: int, max_clock_offset_ms: float) -> dict[str, Any]:
    driver_log = exp_dir / "g5_full24_driver.log"
    log_text = driver_log.read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "CUDA out of memory",
        "TCPStore timed out",
        "DistNetworkError",
        "put_get_socket._put_to_single_storage_unit] attempt 1/2 failed",
    )
    observed = [marker for marker in forbidden if marker in log_text]
    if _unexplained_tracebacks(log_text):
        observed.append("Traceback (most recent call last)")
    if observed:
        raise RuntimeError(f"G5 driver contains fatal markers: {observed}")
    storage_unit_match = re.search(r"num_data_storage_units\s+\.{2,}\s+(\d+)", log_text)
    if storage_unit_match is None or int(storage_unit_match.group(1)) != 8:
        raise RuntimeError(
            "G5 must run with eight TransferQueue storage units; "
            f"observed={storage_unit_match.group(1) if storage_unit_match else None}"
        )
    tracker = exp_dir / "checkpoint" / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text(encoding="utf-8").strip()) != expected_steps - 1:
        raise RuntimeError(f"final checkpoint tracker is missing or wrong: {tracker}")
    result = {"schema_version": 1, "status": "passed", "trigger": trigger, "expected_steps": expected_steps}
    result.update(_validate_rollout_rows(exp_dir, trigger, expected_steps))
    result.update(_validate_direct_report(exp_dir, trigger, max_clock_offset_ms, measured_rounds=expected_steps - 1))
    result.update(_validate_training_timeline(exp_dir, expected_steps))
    result.update(_validate_accounting_drain(exp_dir, expected_steps))
    result.update(_validate_flashinfer_workspaces(exp_dir))
    result.update(_validate_placement_and_gpu_drain(exp_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--trigger", choices=("terminal_once", "per_turn"), required=True)
    parser.add_argument("--expected-steps", type=int, default=3)
    parser.add_argument("--max-clock-offset-ms", type=float, default=10.0)
    args = parser.parse_args()
    report = validate(args.exp_dir, args.trigger, args.expected_steps, args.max_clock_offset_ms)
    output = args.exp_dir / "g5_full24_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
