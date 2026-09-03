#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed artifact gate for G3 rollout4 + ORM TP2 + VLM TP2."""

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


def _validate_gpu_samples(exp_dir: Path) -> dict[str, Any]:
    manifests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_files: set[Path] = set()
    for path in sorted((exp_dir / "gpu_samples").glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("record_type") == "manifest":
                manifests[str(row.get("role"))].append(row)
            elif row.get("record_type") == "sample":
                gpu_busy = any(float(gpu.get("util_percent", 0.0) or 0.0) > 0.0 for gpu in row.get("gpu", []))
                metrics = row.get("sglang") or {}
                generated = float(metrics.get("sglang:gen_throughput", 0.0) or 0.0) > 0.0
                if gpu_busy and generated:
                    active_files.add(path)

    expected_counts = {"rollout": 4, "judge_accuracy": 1, "judge_multiturn_vlm": 1}
    actual_counts = {role: len(rows) for role, rows in manifests.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected GPU sampler manifest topology: {actual_counts}")
    all_manifests = [row for rows in manifests.values() for row in rows]
    if any(not row.get("nvml_enabled") for row in all_manifests):
        raise RuntimeError("at least one rollout/Judge sampler lacks NVML")
    expected_gpus = {"rollout": 1, "judge_accuracy": 2, "judge_multiturn_vlm": 2}
    for role, rows in manifests.items():
        if any(int(row.get("num_gpus_per_engine", -1)) != expected_gpus[role] for row in rows):
            raise RuntimeError(f"{role} sampler reports the wrong TP allocation: {rows}")

    uuids = [uuid for row in all_manifests for uuid in row.get("gpu_uuids", [])]
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise RuntimeError(f"rollout/Judge GPU UUID allocations overlap or are incomplete: {uuids}")
    manifest_paths = {path for path in (exp_dir / "gpu_samples").glob("*.jsonl") if path.stat().st_size}
    if active_files != manifest_paths:
        inactive = sorted(str(path) for path in manifest_paths - active_files)
        raise RuntimeError(f"not every rollout/Judge engine served measured generation: {inactive}")
    return {"manifest_counts": actual_counts, "allocated_gpu_uuids": len(set(uuids))}


def _validate_reward_rows(exp_dir: Path, trigger: str) -> dict[str, Any]:
    result_paths = sorted((exp_dir / "rollout_result" / "train").glob("*.jsonl"))
    rows = [row for path in result_paths for row in _read_jsonl(path)]
    if len(rows) != 4 or any(row.get("status") != "completed" for row in rows):
        raise RuntimeError(f"expected four completed rollout rows, got {Counter(row.get('status') for row in rows)}")

    per_turn_count = 0
    for row in rows:
        reward = row.get("latency_trace", {}).get("reward", {})
        if reward.get("pipeline_status") != "success" or reward.get("executor_status") != "success":
            raise RuntimeError(f"reward pipeline/executor did not succeed: {reward}")
        if reward.get("reasoning_execution_trigger") != trigger:
            raise RuntimeError(f"reward trigger mismatch: {reward.get('reasoning_execution_trigger')} != {trigger}")
        judges = reward.get("judges") or {}
        accuracy = judges.get("answer_accuracy") or {}
        if accuracy.get("status") != "success" or accuracy.get("attempt_count") != 1:
            raise RuntimeError(f"terminal ORM is not clean first-attempt success: {accuracy}")
        if trigger == "terminal_once":
            vlm = judges.get("multi_turn_reasoning") or {}
            if vlm.get("status") != "success" or vlm.get("attempt_count") != 1:
                raise RuntimeError(f"terminal VLM is not clean first-attempt success: {vlm}")
            if reward.get("per_turn_judges") or reward.get("per_turn_judge_count"):
                raise RuntimeError("terminal_once unexpectedly contains per-turn Judge requests")
        else:
            turn_judges = reward.get("per_turn_judges") or []
            expected_count = int(reward.get("per_turn_judge_count", -1))
            if expected_count <= 0 or expected_count != len(turn_judges):
                raise RuntimeError(
                    f"per-turn Judge cardinality mismatch: expected={expected_count} rows={len(turn_judges)}"
                )
            for turn in turn_judges:
                judge = turn.get("judge") or {}
                if turn.get("status") != "success" or judge.get("attempt_count") != 1:
                    raise RuntimeError(f"per-turn sidecar is not clean first-attempt success: {turn}")
                if judge.get("invalid_response_count") != 0:
                    raise RuntimeError(f"per-turn sidecar returned invalid JSON: {turn}")
                if not turn.get("response_state_hash") or not turn.get("observation_state_hash"):
                    raise RuntimeError(f"per-turn sidecar lacks exportable lineage hashes: {turn}")
            per_turn_count += expected_count
    return {"completed_samples": len(rows), "per_turn_judge_count": per_turn_count}


def validate(exp_dir: Path, trigger: str) -> dict[str, Any]:
    if trigger not in {"terminal_once", "per_turn"}:
        raise ValueError(f"unsupported trigger: {trigger}")
    report = json.loads((exp_dir / "direct_report.json").read_text(encoding="utf-8"))
    if report.get("analysis_mode") != "standalone_direct":
        raise RuntimeError(f"expected standalone direct analysis, got {report.get('analysis_mode')}")
    variant = report.get("variants", {}).get(trigger)
    if not variant:
        raise RuntimeError(f"direct report lacks variant {trigger}")
    if variant.get("observed_benchmark_modes") != ["dual"]:
        raise RuntimeError(f"unexpected benchmark modes: {variant.get('observed_benchmark_modes')}")
    if variant.get("observed_reasoning_triggers") != [trigger]:
        raise RuntimeError(f"unexpected reasoning triggers: {variant.get('observed_reasoning_triggers')}")

    request = variant["request"]
    expected = {
        "terminal_once": {"terminal_orm": (4, 4), "terminal_vlm": (4, 4), "per_turn_vlm": (0, 0)},
        "per_turn": {"terminal_orm": (4, 4), "terminal_vlm": (0, 0)},
    }[trigger]
    for name, counts in expected.items():
        if _request_counts(request[name]) != counts:
            raise RuntimeError(f"{name} request counts mismatch: {_request_counts(request[name])} != {counts}")
    if trigger == "per_turn":
        raw, clean = _request_counts(request["per_turn_vlm"])
        if raw <= 0 or clean != raw:
            raise RuntimeError(f"per-turn request distribution is not clean: raw={raw} clean={clean}")

    trajectory = variant["trajectory"]
    if (trajectory.get("raw_count"), trajectory.get("clean_count"), trajectory.get("fallback_count")) != (4, 4, 0):
        raise RuntimeError(f"trajectory distribution is not four clean samples without fallback: {trajectory}")
    group = variant["group"]
    group_counts = (
        group.get("group_finalize_count"),
        group.get("complete_group_count"),
        group.get("missing_terminal_admission_group_count"),
    )
    if group_counts != (2, 2, 0):
        raise RuntimeError(f"group closure distribution is incomplete: {group_counts}")

    debug_paths = sorted((exp_dir / "debug_rollout").glob("*.pt"))
    if len(debug_paths) != 1 or debug_paths[0].stat().st_size == 0:
        raise RuntimeError(f"expected one non-empty debug capture, got {debug_paths}")
    timeline_paths = [path for path in (exp_dir / "timeline").rglob("*") if path.is_file() and path.stat().st_size]
    if not timeline_paths:
        raise RuntimeError("G3 dual timeline is empty")

    result = {
        "schema_version": 1,
        "status": "passed",
        "trigger": trigger,
        "request_counts": {name: _request_counts(section) for name, section in request.items()},
        "trajectory_count": 4,
        "group_count": 2,
        "timeline_file_count": len(timeline_paths),
    }
    result.update(_validate_reward_rows(exp_dir, trigger))
    result.update(_validate_gpu_samples(exp_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--trigger", choices=("terminal_once", "per_turn"), required=True)
    args = parser.parse_args()

    report = validate(args.exp_dir, args.trigger)
    output = args.exp_dir / "g3_dual8_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
