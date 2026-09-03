#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _request_counts(section: dict[str, Any]) -> tuple[int, int]:
    return int(section.get("raw_count", -1)), int(section.get("clean_count", -1))


def _validate_rollout_rows(exp_dir: Path, trigger: str, expected_steps: int) -> dict[str, Any]:
    rows_by_step = {
        int(path.stem): _read_jsonl(path)
        for path in sorted((exp_dir / "rollout_result" / "train").glob("*.jsonl"))
        if path.stem.isdigit()
    }
    missing = sorted(set(range(expected_steps)) - set(rows_by_step))
    if missing:
        raise RuntimeError(f"rollout_result lacks trained steps: {missing}")
    rows = [row for step in range(expected_steps) for row in rows_by_step[step]]
    if any(len(rows_by_step[step]) != 4 for step in range(expected_steps)):
        raise RuntimeError(
            f"each trained step must contain four samples: { {k: len(v) for k, v in rows_by_step.items()} }"
        )
    if any(row.get("status") != "completed" for row in rows):
        raise RuntimeError(f"rollout rows are not cleanly completed: {Counter(row.get('status') for row in rows)}")
    if any(not isinstance(row.get("weight_versions"), list) or not row["weight_versions"] for row in rows):
        raise RuntimeError("at least one rollout row lacks policy weight_versions provenance")

    per_turn_count = 0
    for step, step_rows in rows_by_step.items():
        if step >= expected_steps:
            continue
        groups: defaultdict[Any, int] = defaultdict(int)
        for row in step_rows:
            groups[row.get("group_index")] += 1
            trace = row.get("latency_trace") or {}
            reward = trace.get("reward") or {}
            if reward.get("pipeline_status") != "success" or reward.get("executor_status") != "success":
                raise RuntimeError(f"reward path failed: {reward}")
            if reward.get("reasoning_execution_trigger") != trigger:
                raise RuntimeError(
                    f"reward trigger mismatch: {reward.get('reasoning_execution_trigger')} != {trigger}"
                )
            if reward.get("terminal_outcome") in {"resident_discarded", "cancelled"}:
                raise RuntimeError(f"right-censored reward outcome entered committed rows: {reward}")
            accuracy = (reward.get("judges") or {}).get("answer_accuracy") or {}
            if accuracy.get("status") != "success" or accuracy.get("attempt_count") != 1:
                raise RuntimeError(f"terminal ORM is not a clean first attempt: {accuracy}")
            if trigger == "terminal_once":
                vlm = (reward.get("judges") or {}).get("multi_turn_reasoning") or {}
                if vlm.get("status") != "success" or vlm.get("attempt_count") != 1:
                    raise RuntimeError(f"terminal VLM is not a clean first attempt: {vlm}")
                if reward.get("per_turn_judge_count") or reward.get("per_turn_judges"):
                    raise RuntimeError("terminal_once unexpectedly contains per-turn Judge work")
            else:
                judges = reward.get("per_turn_judges") or []
                count = reward.get("per_turn_judge_count")
                if not isinstance(count, int) or count <= 0 or count != len(judges):
                    raise RuntimeError(f"per-turn Judge cardinality mismatch: count={count}, rows={len(judges)}")
                if any(
                    item.get("status") != "success"
                    or (item.get("judge") or {}).get("attempt_count") != 1
                    or (item.get("judge") or {}).get("invalid_response_count") != 0
                    for item in judges
                ):
                    raise RuntimeError(f"per-turn sidecar is not clean: {judges}")
                per_turn_count += count
        if sorted(groups.values()) != [2, 2]:
            raise RuntimeError(f"step {step} is not two groups of two samples: {dict(groups)}")
    return {"rollout_samples": len(rows), "per_turn_judge_count": per_turn_count}


def _validate_gpu_drain(exp_dir: Path) -> dict[str, Any]:
    manifests: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    active_paths: set[Path] = set()
    final_paths: set[Path] = set()
    for path in sorted((exp_dir / "gpu_samples").glob("*.jsonl")):
        rows = _read_jsonl(path)
        for row in rows:
            if row.get("record_type") == "manifest":
                manifests[str(row.get("role"))].append(row)
            elif row.get("record_type") == "sample":
                metrics = row.get("sglang") or {}
                if float(metrics.get("sglang:gen_throughput", 0.0) or 0.0) > 0:
                    active_paths.add(path)
                if row.get("final_sample") is True:
                    final_paths.add(path)
                    running = float(metrics.get("sglang:num_running_reqs", -1) or 0.0)
                    queued = float(metrics.get("sglang:num_queue_reqs", -1) or 0.0)
                    if running != 0.0 or queued != 0.0:
                        raise RuntimeError(f"right-censored SGLang WIP in final sample {path}: {metrics}")
    expected = {"rollout": 4, "judge_accuracy": 1, "judge_multiturn_vlm": 1}
    actual = {role: len(rows) for role, rows in manifests.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected rollout/Judge sampler topology: {actual}")
    paths = {path for path in (exp_dir / "gpu_samples").glob("*.jsonl") if path.stat().st_size}
    if active_paths != paths or final_paths != paths:
        raise RuntimeError(
            f"every engine needs active and final drain samples: inactive={paths - active_paths}, no_final={paths - final_paths}"
        )
    manifest_rows = [row for rows in manifests.values() for row in rows]
    uuids = [uuid for row in manifest_rows for uuid in row.get("gpu_uuids", [])]
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise RuntimeError(f"rollout/Judge GPU UUIDs overlap: {uuids}")
    return {"sampler_manifests": actual, "rollout_judge_gpu_uuids": len(set(uuids))}


def _validate_accounting_drain(exp_dir: Path, expected_steps: int) -> dict[str, Any]:
    path = exp_dir / "latency_markers" / "agentic_accounting_end.jsonl"
    rows = _read_jsonl(path)
    if len(rows) != expected_steps:
        raise RuntimeError(f"expected {expected_steps} accounting snapshots, got {len(rows)}")
    final = rows[-1].get("snapshot") or {}
    zero_keys = (
        "reward_waiting_groups",
        "reward_waiting_records",
        "reward_ready_groups",
        "reward_completed_groups",
        "reward_inflight_sample_rewards",
        "reward_inflight_group_rewards",
        "transfer_buffer_groups",
        "transfer_tasks",
        "transfer_ready_groups",
        "runtime_groups",
        "runtime_slots",
        "runtime_ready_materialized_batch_groups",
        "runtime_ready_materialized_groups",
    )
    bad = {key: final.get(key) for key in zero_keys if final.get(key) != 0}
    if bad:
        raise RuntimeError(f"final agentic accounting snapshot is not drained: {bad}")
    return {"accounting_steps": len(rows), "final_drain_zero_keys": list(zero_keys)}


def _validate_direct_report(exp_dir: Path, trigger: str, max_clock_offset_ms: float) -> dict[str, Any]:
    report = json.loads((exp_dir / "direct_report.json").read_text(encoding="utf-8"))
    variant = report.get("variants", {}).get(trigger) or {}
    if variant.get("observed_benchmark_modes") != ["dual"]:
        raise RuntimeError(f"unexpected direct benchmark mode: {variant.get('observed_benchmark_modes')}")
    if variant.get("observed_reasoning_triggers") != [trigger]:
        raise RuntimeError(f"unexpected direct trigger: {variant.get('observed_reasoning_triggers')}")
    clock = variant.get("clock_domain") or {}
    if len(clock.get("hosts") or []) < 2 or not isinstance(clock.get("max_offset_ms"), (int, float)):
        raise RuntimeError(f"direct report lacks multi-host clock audit: {clock}")
    if float(clock["max_offset_ms"]) > max_clock_offset_ms:
        raise RuntimeError(f"multi-host clock offset bound exceeds {max_clock_offset_ms:g} ms: {clock}")

    request = variant.get("request") or {}
    expected = {
        "terminal_once": {"terminal_orm": (4, 4), "terminal_vlm": (4, 4), "per_turn_vlm": (0, 0)},
        "per_turn": {"terminal_orm": (4, 4), "terminal_vlm": (0, 0)},
    }[trigger]
    for name, counts in expected.items():
        if _request_counts(request[name]) != counts:
            raise RuntimeError(f"{name} request counts mismatch: {_request_counts(request[name])} != {counts}")
    if trigger == "per_turn":
        raw, clean = _request_counts(request["per_turn_vlm"])
        if raw <= 0 or raw != clean:
            raise RuntimeError(f"per-turn operational tail is not clean: raw={raw}, clean={clean}")
    trajectory = variant.get("trajectory") or {}
    if (trajectory.get("raw_count"), trajectory.get("clean_count"), trajectory.get("fallback_count")) != (4, 4, 0):
        raise RuntimeError(f"measured trajectory distribution is not four clean samples: {trajectory}")
    group = variant.get("group") or {}
    if (
        group.get("group_finalize_count"),
        group.get("complete_group_count"),
        group.get("missing_terminal_admission_group_count"),
    ) != (2, 2, 0):
        raise RuntimeError(f"measured group distribution is incomplete: {group}")
    trainer = variant.get("trainer") or {}
    if trainer.get("data_wait_with_missing_returned_group_provenance_count") != 0:
        raise RuntimeError(f"trainer data-wait provenance is incomplete: {trainer}")
    if int(trainer.get("returned_trainer_batch_count", 0)) <= 0:
        raise RuntimeError(f"no actual returned trainer batch was measured: {trainer}")
    if int((trainer.get("inclusive_reward_ancestor_wait_s") or {}).get("count", 0)) <= 0:
        raise RuntimeError(f"trainer reward-ancestor overlap is unavailable: {trainer}")
    return {
        "measured_steps": variant.get("measured_steps"),
        "clock_max_offset_ms": clock["max_offset_ms"],
        "trainer_batches": trainer["returned_trainer_batch_count"],
    }


def _validate_training_timeline(exp_dir: Path, expected_steps: int) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for path in (exp_dir / "timeline").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            events.extend(row for row in value if isinstance(row, dict))
    optimizer_steps = {
        int((event.get("args") or {}).get("step"))
        for event in events
        if event.get("name") == "critical_path.optimizer_step" and (event.get("args") or {}).get("global_rank") == 0
    }
    if optimizer_steps != set(range(expected_steps)):
        raise RuntimeError(f"optimizer timeline lacks real rank-0 steps: {optimizer_steps}")
    ready_rows = _read_jsonl(exp_dir / "latency_markers" / "weight_serving_ready.jsonl")
    if {int(row["step"]) for row in ready_rows} != set(range(expected_steps)):
        raise RuntimeError(f"weight-serving ready markers mismatch: {ready_rows}")
    if not (exp_dir / "latency_markers" / "reward_workload.jsonl").is_file():
        raise RuntimeError("reward workload marker is missing")
    return {"optimizer_steps": sorted(optimizer_steps), "ready_markers": len(ready_rows)}


def validate(exp_dir: Path, trigger: str, expected_steps: int, max_clock_offset_ms: float) -> dict[str, Any]:
    if trigger not in {"terminal_once", "per_turn"}:
        raise ValueError(f"unsupported trigger: {trigger}")
    driver_log = exp_dir / "g4_full12_driver.log"
    log_text = driver_log.read_text(encoding="utf-8", errors="replace")
    forbidden = ("Traceback (most recent call last)", "CUDA out of memory", "TCPStore timed out", "DistNetworkError")
    observed = [marker for marker in forbidden if marker in log_text]
    if observed:
        raise RuntimeError(f"G4 full driver contains fatal markers: {observed}")
    checkpoint = exp_dir / "checkpoint"
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text(encoding="utf-8").strip()) != expected_steps - 1:
        raise RuntimeError(f"final checkpoint tracker is missing or wrong: {tracker}")

    result = {"schema_version": 1, "status": "passed", "trigger": trigger, "expected_steps": expected_steps}
    result.update(_validate_rollout_rows(exp_dir, trigger, expected_steps))
    result.update(_validate_direct_report(exp_dir, trigger, max_clock_offset_ms))
    result.update(_validate_training_timeline(exp_dir, expected_steps))
    result.update(_validate_accounting_drain(exp_dir, expected_steps))
    result.update(_validate_gpu_drain(exp_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--trigger", choices=("terminal_once", "per_turn"), required=True)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--max-clock-offset-ms", type=float, default=10.0)
    args = parser.parse_args()
    report = validate(args.exp_dir, args.trigger, args.expected_steps, args.max_clock_offset_ms)
    output = args.exp_dir / "g4_full12_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
