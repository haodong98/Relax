# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Analyze critical-path timeline traces for reward-model latency experiments.

``--direct`` analyzes every ``--variant`` independently.  It emits raw
request, trajectory, group, and trainer distributions and intentionally does
not select a baseline or derive a paired delta.  Without ``--direct`` the
legacy paired fixed-K analysis remains available for archived experiments.

A variant path may be one
Chrome timeline JSON file, a directory containing ``timeline_step_*.json``, or
a rollout-result directory containing JSONL records with ``latency_trace``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE_ORDER = (
    "request",
    "rollout_orchestration",
    "rollout_admission_wait",
    "rollout_queue",
    "generation",
    "evaluation",
    "turn_judge",
    "reward",
    "transfer",
    "data_wait",
    "training",
    "weight_gate_wait",
    "weight_update",
)

REQUIRED_CRITICAL_EVENT_NAMES = (
    "critical_path.rollout_generation",
    "critical_path.reward",
    "critical_path.transfer",
    "critical_path.training_schedule",
    "critical_path.optimizer_step",
    "critical_path.weight_update",
    "critical_path.weight_serving_ready",
)

BENCHMARK_MODE_STATUS = {
    "recorded": ("bypassed", "bypassed"),
    "accuracy": ("success", "success"),
    "accuracy_shadow": ("success", "success"),
    "dual": ("success", "success"),
    "dual_shadow": ("success", "success"),
}

BENCHMARK_MODE_BRANCHES = {
    "recorded": Counter(),
    "accuracy": Counter({"answer_accuracy": 1}),
    "accuracy_shadow": Counter({"answer_accuracy": 1}),
    "dual": Counter({"answer_accuracy": 1, "multi_turn_reasoning": 1}),
    "dual_shadow": Counter({"answer_accuracy": 1, "multi_turn_reasoning": 1}),
}
REASONING_TRIGGERS = {"terminal_once", "per_turn"}


def _stage_for_event(name: str) -> str | None:
    if name.startswith("critical_path.judge_request"):
        return "request"
    if name == "critical_path.session_terminal_admission":
        return "rollout_orchestration"
    if name in {"critical_path.sample_workload", "critical_path.agentic_accounting_end"}:
        return "rollout_orchestration"
    if name == "critical_path.rollout_evaluation":
        return "evaluation"
    if name in {"critical_path.rollout_queue", "critical_path.rollout_materialize_wait"}:
        return "rollout_queue"
    if name == "critical_path.rollout_generation":
        return "generation"
    if name == "critical_path.rollout_admission_wait":
        return "rollout_admission_wait"
    if name.startswith("critical_path.rollout_"):
        return "rollout_orchestration"
    if name.startswith("critical_path.turn_judge"):
        return "turn_judge"
    if name.startswith("critical_path.reward"):
        return "reward"
    if name.startswith("critical_path.transfer"):
        return "transfer"
    if name.startswith("critical_path.data_wait"):
        return "data_wait"
    if name.startswith("critical_path.training") or name == "critical_path.optimizer_step":
        return "training"
    if name == "critical_path.weight_gate_wait":
        return "weight_gate_wait"
    if name.startswith("critical_path.weight_"):
        return "weight_update"
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _statistics(values: list[float], *, unit_suffix: str = "_s") -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": float(len(values)),
        f"mean{unit_suffix}": sum(values) / len(values),
        f"p50{unit_suffix}": _percentile(values, 50),
        f"p90{unit_suffix}": _percentile(values, 90),
        f"p95{unit_suffix}": _percentile(values, 95),
        f"p99{unit_suffix}": _percentile(values, 99),
        f"max{unit_suffix}": max(values),
    }


def _event_step(event: dict[str, Any]) -> int | None:
    args = event.get("args")
    value = args.get("step") if isinstance(args, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _normalize_timeline_event(event: dict[str, Any]) -> dict[str, Any] | None:
    name = event.get("name")
    stage = _stage_for_event(name) if isinstance(name, str) else None
    timestamp_us = event.get("ts")
    duration_us = event.get("dur")
    if (
        stage is None
        or isinstance(timestamp_us, bool)
        or not isinstance(timestamp_us, (int, float))
        or isinstance(duration_us, bool)
        or not isinstance(duration_us, (int, float))
        or duration_us < 0
    ):
        return None
    start_s = float(timestamp_us) / 1e6
    return {
        "name": name,
        "stage": stage,
        "start_s": start_s,
        "end_s": start_s + float(duration_us) / 1e6,
        "step": _event_step(event),
        "pid": event.get("pid"),
        "tid": event.get("tid"),
        "attributes": dict(event.get("args", {})) if isinstance(event.get("args"), dict) else {},
    }


def _events_from_latency_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("latency_trace")
    if not isinstance(trace, dict):
        return []
    reward_trace = trace.get("reward")
    rollout_id = (
        reward_trace.get("train_partition_step")
        if isinstance(reward_trace, dict) and reward_trace.get("train_partition_step") is not None
        else record.get("rollout_id", record.get("collection_rollout_id"))
    )
    sample_index = (
        record.get("trace_index", 0)
        if record.get("record_type") == "reward_terminal_trace"
        else record.get("sample_index", 0)
    )
    events: list[dict[str, Any]] = []
    reward_attributes = {
        key: reward_trace.get(key)
        for key in (
            "terminal_outcome",
            "pipeline_status",
            "executor_status",
            "executor_error_code",
            "benchmark_mode",
            "group_index",
            "sample_index",
            "context_hash",
            "trajectory_hash",
            "recorded_reward_hash",
            "benchmark_invariant_hash",
            "reasoning_trigger",
            "reasoning_execution_trigger",
            "per_turn_judge_count",
            "per_turn_assistant_turn_count",
            "per_turn_off_lineage_judge_count",
            "sample_key",
            "group_key",
            "expected_trainer_components",
        )
        if isinstance(reward_trace, dict)
    }
    assistant_turn_count = trace.get("per_turn_assistant_turn_count", record.get("agent_turns"))
    if reward_attributes.get("per_turn_assistant_turn_count") is None and isinstance(assistant_turn_count, int):
        reward_attributes["per_turn_assistant_turn_count"] = assistant_turn_count
    off_lineage_count = trace.get("per_turn_off_lineage_judge_count")
    if reward_attributes.get("per_turn_off_lineage_judge_count") is None and isinstance(off_lineage_count, int):
        reward_attributes["per_turn_off_lineage_judge_count"] = off_lineage_count

    def add(
        name: str,
        start_s: Any,
        end_s: Any,
        pid: int,
        clock_host: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)) or end_s < start_s:
            return
        events.append(
            {
                "name": name,
                "ph": "X",
                "ts": int(float(start_s) * 1e6),
                "dur": int((float(end_s) - float(start_s)) * 1e6),
                "pid": pid,
                "tid": sample_index,
                "args": {
                    "step": rollout_id,
                    "clock_host": clock_host,
                    **reward_attributes,
                    **(attributes or {}),
                },
            }
        )

    def span_clock_host(events: dict[str, Any], start_key: str, end_key: str) -> str | None:
        start_host = events.get(f"{start_key}__clock_host")
        end_host = events.get(f"{end_key}__clock_host")
        return start_host if isinstance(start_host, str) and start_host and start_host == end_host else None

    trace_events = trace.get("events")
    if isinstance(trace_events, dict):
        for name, start_key, end_key, pid in (
            ("critical_path.rollout_finalize", "finalize_start_at", "finalize_end_at", 2001),
            (
                "critical_path.rollout_materialize_wait",
                "materialize_ready_at",
                "materialize_harvest_at",
                2001,
            ),
            (
                "critical_path.reward_context_build",
                "reward_context_build_start_at",
                "reward_context_build_end_at",
                2002,
            ),
            ("critical_path.reward", "reward_arrive_at", "reward_end_at", 2002),
            (
                "critical_path.session_terminal_admission",
                "session_terminal_admission_at",
                "session_terminal_admission_at",
                2001,
            ),
            (
                "critical_path.reward_group_finalize",
                "group_finalize_start_at",
                "group_finalize_end_at",
                2002,
            ),
            (
                "critical_path.transfer_buffer_wait",
                "transfer_buffer_enter_at",
                "transfer_release_start_at",
                2003,
            ),
            ("critical_path.transfer", "transfer_release_start_at", "transfer_release_end_at", 2003),
        ):
            add(
                name,
                trace_events.get(start_key),
                trace_events.get(end_key),
                pid,
                span_clock_host(trace_events, start_key, end_key),
            )
        add(
            "critical_path.turn_judge_barrier",
            trace_events.get("turn_judge_barrier_start_at"),
            trace_events.get("turn_judge_barrier_release_at"),
            2001,
            span_clock_host(trace_events, "turn_judge_barrier_start_at", "turn_judge_barrier_release_at"),
        )
        judges = reward_trace.get("judges") if isinstance(reward_trace, dict) else None
        if isinstance(judges, dict):
            for component, judge in judges.items():
                if not isinstance(component, str) or not isinstance(judge, dict):
                    continue
                start_key = f"judge_{component}_queue_enter_at"
                end_key = f"judge_{component}_end_at"
                add(
                    "critical_path.judge_request",
                    trace_events.get(start_key),
                    trace_events.get(end_key),
                    2002,
                    span_clock_host(trace_events, start_key, end_key),
                    {
                        "component": component,
                        "request_kind": "terminal_orm" if component == "answer_accuracy" else "terminal_vlm",
                        "request_status": judge.get("status"),
                        "attempt_count": judge.get("attempt_count"),
                        "invalid_response_count": judge.get("invalid_response_count"),
                        "queue_elapsed_s": judge.get("queue_elapsed_s"),
                        "http_elapsed_s": judge.get("http_elapsed_s"),
                        "payload_prep_elapsed_s": judge.get("payload_prep_elapsed_s"),
                        "parse_elapsed_s": judge.get("parse_elapsed_s"),
                        "backoff_elapsed_s": judge.get("backoff_elapsed_s"),
                        "server": judge.get("server"),
                    },
                )
                # Sub-spans mirroring relax/agentic/rollout.py's
                # critical_path.reward.{component}.queue/.http: the client-side
                # concurrency-limit wait on self._semaphores[spec.role]
                # (relax/engine/rewards/dual_agentic_judge.py) and the last HTTP
                # attempt (http_start_at/http_end_at are overwritten per retry, so
                # this does not capture cumulative http_elapsed_s across retries).
                add(
                    "critical_path.judge_request.queue",
                    trace_events.get(f"judge_{component}_queue_enter_at"),
                    trace_events.get(f"judge_{component}_queue_acquired_at"),
                    2002,
                    span_clock_host(
                        trace_events,
                        f"judge_{component}_queue_enter_at",
                        f"judge_{component}_queue_acquired_at",
                    ),
                    {"component": component, "queue_elapsed_s": judge.get("queue_elapsed_s")},
                )
                add(
                    "critical_path.judge_request.http",
                    trace_events.get(f"judge_{component}_http_start_at"),
                    trace_events.get(f"judge_{component}_http_end_at"),
                    2002,
                    span_clock_host(
                        trace_events,
                        f"judge_{component}_http_start_at",
                        f"judge_{component}_http_end_at",
                    ),
                    {
                        "component": component,
                        "attempt_count": judge.get("attempt_count"),
                        "server": judge.get("server"),
                    },
                )
    turns = trace.get("turns")
    reward_turn_judges = reward_trace.get("per_turn_judges") if isinstance(reward_trace, dict) else None
    turn_judges_by_index = {
        turn_index: judge
        for judge in (reward_turn_judges if isinstance(reward_turn_judges, list) else [])
        if isinstance(judge, dict)
        if isinstance((turn_index := judge.get("turn_index")), int) and not isinstance(turn_index, bool)
    }
    for turn_index, turn in enumerate(turns if isinstance(turns, list) else []):
        turn_judge = turn.get("judge") if isinstance(turn, dict) else None
        if not isinstance(turn_judge, dict):
            turn_judge = turn_judges_by_index.get(turn_index)
        judge_events = turn_judge.get("events") if isinstance(turn_judge, dict) else None
        if isinstance(judge_events, dict):
            add(
                "critical_path.turn_judge",
                judge_events.get("turn_judge_trigger_at"),
                judge_events.get("turn_judge_end_at"),
                2002,
                span_clock_host(judge_events, "turn_judge_trigger_at", "turn_judge_end_at"),
                {
                    "turn_index": turn_index,
                    "judge_role": turn_judge.get("role"),
                    "judge_status": turn_judge.get("status"),
                    "response_state_hash": turn_judge.get("response_state_hash"),
                    "observation_state_hash": turn_judge.get("observation_state_hash"),
                },
            )
            request_end_key = (
                "turn_judge_request_end_at" if "turn_judge_request_end_at" in judge_events else "turn_judge_end_at"
            )
            add(
                "critical_path.judge_request",
                judge_events.get("turn_judge_queue_enter_at", judge_events.get("turn_judge_trigger_at")),
                judge_events.get(request_end_key),
                2002,
                span_clock_host(
                    judge_events,
                    "turn_judge_queue_enter_at"
                    if "turn_judge_queue_enter_at" in judge_events
                    else "turn_judge_trigger_at",
                    request_end_key,
                ),
                {
                    "component": "multi_turn_reasoning",
                    "request_kind": "per_turn_vlm",
                    "turn_index": turn_index,
                    "request_status": turn_judge.get("status"),
                    "attempt_count": (turn_judge.get("judge") or {}).get("attempt_count"),
                    "invalid_response_count": (turn_judge.get("judge") or {}).get("invalid_response_count"),
                    "queue_elapsed_s": (turn_judge.get("judge") or {}).get("queue_elapsed_s"),
                    "http_elapsed_s": (turn_judge.get("judge") or {}).get("http_elapsed_s"),
                    "payload_prep_elapsed_s": (turn_judge.get("judge") or {}).get("payload_prep_elapsed_s"),
                    "parse_elapsed_s": (turn_judge.get("judge") or {}).get("parse_elapsed_s"),
                    "backoff_elapsed_s": (turn_judge.get("judge") or {}).get("backoff_elapsed_s"),
                    "server": (turn_judge.get("judge") or {}).get("server"),
                    "judge_role": turn_judge.get("role"),
                },
            )
        turn_events = turn.get("events") if isinstance(turn, dict) else None
        if isinstance(turn_events, dict):
            # Mirrors the split in relax/agentic/rollout.py
            # (_build_agentic_critical_path_timeline_events): the former
            # single `rollout_pre_generation` span conflated admission setup,
            # the genuine per-session admission-gate wait, and dispatch
            # bookkeeping into one bucket that was invisible to stall
            # attribution (it fell through to the generic `rollout_orchestration`
            # stage, outside REWARD/ROLLOUT_GEN/TRANSFER_STAGES).
            add(
                "critical_path.rollout_admission_setup",
                turn_events.get("chat_request_arrive_at"),
                turn_events.get("ir_created_at"),
                2001,
                span_clock_host(turn_events, "chat_request_arrive_at", "ir_created_at"),
            )
            add(
                "critical_path.rollout_admission_wait",
                turn_events.get("ir_created_at"),
                turn_events.get("ir_activated_at"),
                2001,
                span_clock_host(turn_events, "ir_created_at", "ir_activated_at"),
            )
            add(
                "critical_path.rollout_dispatch",
                turn_events.get("ir_activated_at"),
                turn_events.get("generation_queue_enter_at"),
                2001,
                span_clock_host(turn_events, "ir_activated_at", "generation_queue_enter_at"),
            )
            add(
                "critical_path.rollout_queue",
                turn_events.get("generation_queue_enter_at"),
                turn_events.get("generation_start_at"),
                2001,
                span_clock_host(turn_events, "generation_queue_enter_at", "generation_start_at"),
            )
            add(
                "critical_path.rollout_generation",
                turn_events.get("generation_start_at"),
                turn_events.get("generation_end_at"),
                2001,
                span_clock_host(turn_events, "generation_start_at", "generation_end_at"),
            )
            add(
                "critical_path.rollout_post_generation",
                turn_events.get("generation_end_at"),
                turn_events.get("chat_end_at"),
                2001,
                span_clock_host(turn_events, "generation_end_at", "chat_end_at"),
            )
            add(
                "critical_path.rollout_managed_session",
                turn_events.get("managed_session_runner_start_at"),
                turn_events.get("managed_session_runner_end_at"),
                2001,
                span_clock_host(
                    turn_events,
                    "managed_session_runner_start_at",
                    "managed_session_runner_end_at",
                ),
            )
    if isinstance(trace_events, dict) and record.get("record_type") != "reward_terminal_trace":
        workload_at = trace_events.get("reward_end_at", trace_events.get("finalize_end_at"))
        workload_host = trace_events.get("reward_end_at__clock_host", trace_events.get("finalize_end_at__clock_host"))
        add(
            "critical_path.sample_workload",
            workload_at,
            workload_at,
            2004,
            workload_host if isinstance(workload_host, str) else None,
            {
                "rollout_status": record.get("status"),
                "agent_turns": record.get("agent_turns"),
                "input_tokens": record.get("prompt_token_count", record.get("prompt_length")),
                "output_tokens": record.get("response_token_count", record.get("response_length")),
                "total_tokens": record.get("total_token_count", record.get("total_length")),
                "image_count": record.get("image_count"),
                "image_tokens": record.get("image_token_count", record.get("multimodal_token_count")),
                "weight_versions": record.get("weight_versions"),
            },
        )
    return events


def _discover_variant_files(
    path: Path, *, include_rollout_with_timeline: bool = False
) -> tuple[list[Path], list[Path]]:
    if path.is_file():
        return ([path], []) if path.suffix == ".json" else ([], [path])
    timeline_files = sorted(path.rglob("timeline_step_*.json"))
    rollout_files = sorted(path.rglob("*.jsonl")) if include_rollout_with_timeline or not timeline_files else []
    return timeline_files, rollout_files


def load_variant_events(
    path: Path,
    *,
    include_rollout_with_timeline: bool = False,
    ready_marker_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    timeline_files, rollout_files = _discover_variant_files(
        path,
        include_rollout_with_timeline=include_rollout_with_timeline,
    )
    raw_events: list[dict[str, Any]] = []
    for timeline_file in timeline_files:
        payload = json.loads(timeline_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"timeline file must contain a JSON list: {timeline_file}")
        raw_events.extend(event for event in payload if isinstance(event, dict))
    for rollout_file in rollout_files:
        with rollout_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("event") == "agentic_accounting_end":
                        wall_time_s = record.get("wall_time_s")
                        if isinstance(wall_time_s, (int, float)) and not isinstance(wall_time_s, bool):
                            raw_events.append(
                                {
                                    "name": "critical_path.agentic_accounting_end",
                                    "ph": "X",
                                    "ts": int(float(wall_time_s) * 1e6),
                                    "dur": 0,
                                    "pid": record.get("pid"),
                                    "tid": 0,
                                    "args": {
                                        "step": record.get("step"),
                                        "clock_host": record.get("clock_host"),
                                        **(record.get("snapshot") or {}),
                                    },
                                }
                            )
                    else:
                        raw_events.extend(_events_from_latency_trace(record))
    if ready_marker_path is not None:
        with ready_marker_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                marker = json.loads(line)
                step = marker.get("step")
                wall_time_s = marker.get("wall_time_s")
                clock_host = marker.get("clock_host")
                if (
                    marker.get("event") != "weight_serving_ready"
                    or not isinstance(step, int)
                    or isinstance(step, bool)
                    or not isinstance(wall_time_s, (int, float))
                    or isinstance(wall_time_s, bool)
                    or not isinstance(clock_host, str)
                    or not clock_host
                ):
                    raise ValueError(
                        f"invalid weight-serving-ready marker at {ready_marker_path}:{line_number}: {marker}"
                    )
                raw_events.append(
                    {
                        "name": "critical_path.weight_serving_ready",
                        "ph": "X",
                        "ts": int(float(wall_time_s) * 1e6),
                        "dur": 0,
                        "pid": marker.get("pid"),
                        "tid": 0,
                        "args": dict(marker),
                    }
                )

    # Timeline files are cumulative in the current metrics service; deduplicate
    # complete events repeated in later step dumps.
    unique_events: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in raw_events:
        args = event.get("args", {})
        canonical_args = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = (
            event.get("name"),
            event.get("ts"),
            event.get("dur"),
            event.get("pid"),
            event.get("tid"),
            _event_step(event),
            canonical_args,
        )
        unique_events[key] = event
    normalized = [item for event in unique_events.values() if (item := _normalize_timeline_event(event))]
    source = (
        "timeline+rollout_jsonl"
        if timeline_files and rollout_files
        else "timeline"
        if timeline_files
        else "rollout_jsonl"
    )
    if ready_marker_path is not None:
        source += "+ready_markers"
    return normalized, source


def _direct_identity(value: Any, *, fallback: Any) -> str:
    """Produce a stable key without assuming the runtime's tuple encoding."""
    if value is None:
        value = fallback
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
        if isinstance(value, (dict, list, str, int, float, bool))
        else repr(value)
    )


def _direct_distribution(values: Iterable[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    report: dict[str, Any] = _statistics(finite)
    report["ecdf"] = [
        {"value_s": value, "cumulative_probability": (index + 1) / len(finite)} for index, value in enumerate(finite)
    ]
    return report


def _direct_value_distribution(values: Iterable[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    report: dict[str, Any] = _statistics(finite, unit_suffix="")
    report["ecdf"] = [
        {"value": value, "cumulative_probability": (index + 1) / len(finite)} for index, value in enumerate(finite)
    ]
    return report


def _direct_interval_overlap(
    interval: tuple[float, float],
    intervals: Iterable[tuple[float, float]],
) -> float:
    start, end = interval
    return _union_duration((max(start, other_start), min(end, other_end)) for other_start, other_end in intervals)


def _direct_clean(attributes: dict[str, Any]) -> bool:
    """Clean means one successful attempt with no
    fallback/replacement/error."""
    if attributes.get("request_status") is not None and attributes.get("request_status") != "success":
        return False
    if attributes.get("attempt_count") is not None and attributes.get("attempt_count") != 1:
        return False
    if attributes.get("invalid_response_count", 0) != 0:
        return False
    if attributes.get("reasoning_execution_trigger") == "terminal_once_fallback":
        return False
    if attributes.get("per_turn_off_lineage_judge_count") not in {None, 0}:
        return False
    return attributes.get("terminal_outcome") in {None, "success"} and attributes.get("executor_status") in {
        None,
        "success",
    }


def _direct_window(
    events: list[dict[str, Any]],
    *,
    warmup_steps: int,
    measure_updates: int | None,
    expected_step_stride: int,
) -> tuple[tuple[float, float] | None, list[int]]:
    if measure_updates is None:
        steps = sorted({event["step"] for event in events if isinstance(event.get("step"), int)})
        return None, steps
    if warmup_steps == 0:
        boundaries = _ready_boundaries(events)
        ready_steps = sorted(boundaries)
        if len(ready_steps) < measure_updates + 1:
            raise ValueError("direct ready-to-ready measurement requires one boundary plus measured ready steps")
        selected = ready_steps[: measure_updates + 1]
        if any(next_step - step != expected_step_stride for step, next_step in zip(selected, selected[1:])):
            raise ValueError(f"unexpected serving-ready step stride in direct window: {selected}")
        measured_steps = selected[1:]
        measured_events = [
            event
            for event in events
            if event.get("step") in measured_steps
            and event["name"]
            not in {
                "critical_path.weight_serving_ready",
                "critical_path.sample_workload",
                "critical_path.agentic_accounting_end",
            }
        ]
        start = min((event["start_s"] for event in measured_events), default=boundaries[selected[0]])
        return (start, boundaries[selected[-1]]), measured_steps
    _window, measured_steps, boundary_step = _fixed_k_window(
        events,
        warmup_updates=warmup_steps,
        measure_updates=measure_updates,
        expected_step_stride=expected_step_stride,
    )
    boundaries = _ready_boundaries(events)
    measured_events = [
        event
        for event in events
        if event.get("step") in measured_steps
        and event["name"]
        not in {
            "critical_path.weight_serving_ready",
            "critical_path.sample_workload",
            "critical_path.agentic_accounting_end",
        }
    ]
    start = min((event["start_s"] for event in measured_events), default=boundaries[boundary_step])
    return (start, boundaries[measured_steps[-1]]), measured_steps


def _direct_request_report(events: list[dict[str, Any]], measured_steps: list[int]) -> dict[str, Any]:
    records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["name"] != "critical_path.judge_request" or event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        kind = attributes.get("request_kind")
        if kind not in {"terminal_orm", "terminal_vlm", "per_turn_vlm"}:
            continue
        records[kind].append(
            {
                "client_branch_sojourn_s": event["end_s"] - event["start_s"],
                "queue_s": attributes.get("queue_elapsed_s"),
                "http_s": attributes.get("http_elapsed_s"),
                "payload_prep_s": attributes.get("payload_prep_elapsed_s"),
                "parse_s": attributes.get("parse_elapsed_s"),
                "backoff_s": attributes.get("backoff_elapsed_s"),
                "server": attributes.get("server"),
                "clean": _direct_clean(attributes),
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        def metric(name: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
            return _direct_distribution(item[name] for item in subset)

        result = {
            "raw_count": len(items),
            "clean_count": sum(item["clean"] for item in items),
            "operational": {},
            "clean": {},
        }
        for label, subset in (("operational", items), ("clean", [item for item in items if item["clean"]])):
            for name in ("client_branch_sojourn_s", "queue_s", "http_s", "payload_prep_s", "parse_s", "backoff_s"):
                result[label][name] = metric(name, subset)
            server_keys = sorted(
                {
                    key
                    for item in subset
                    for key, value in (item["server"] or {}).items()
                    if isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            result[label]["server"] = {
                key: _direct_distribution((item["server"] or {}).get(key) for item in subset) for key in server_keys
            }
        return result

    return {kind: summarize(records[kind]) for kind in ("terminal_orm", "terminal_vlm", "per_turn_vlm")}


def _direct_trajectory_report(events: list[dict[str, Any]], measured_steps: list[int]) -> dict[str, Any]:
    by_sample: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        sample_key = _direct_identity(
            attributes.get("sample_key"),
            fallback=[attributes.get("group_index"), attributes.get("sample_index"), event.get("step")],
        )
        by_sample[sample_key].append(event)
    rows: list[dict[str, Any]] = []
    for sample_events in by_sample.values():
        reward_attributes = [
            event.get("attributes", {}) for event in sample_events if event["name"] == "critical_path.reward"
        ]
        if not reward_attributes:
            continue
        attributes: dict[str, Any] = {}
        for item in reward_attributes:
            attributes.update({key: value for key, value in item.items() if value is not None})
        intervals = {
            event["name"]: (event["start_s"], event["end_s"])
            for event in sample_events
            if event["name"]
            in {"critical_path.reward", "critical_path.reward_context_build", "critical_path.turn_judge_barrier"}
        }
        terminal_reward = intervals.get("critical_path.reward")
        if terminal_reward is None:
            continue
        terminal_orm_requests = [
            (event["start_s"], event["end_s"])
            for event in sample_events
            if event["name"] == "critical_path.judge_request"
            and event.get("attributes", {}).get("request_kind") == "terminal_orm"
        ]
        trigger = attributes.get("reasoning_trigger", "terminal_once")
        sidecars = [
            (event["start_s"], event["end_s"])
            for event in sample_events
            if event["name"] == "critical_path.judge_request"
            and event.get("attributes", {}).get("request_kind") == "per_turn_vlm"
        ]
        sidecar_union = _union_duration(sidecars)
        barrier = intervals.get("critical_path.turn_judge_barrier")
        gate_intervals = [terminal_reward]
        context_build = intervals.get("critical_path.reward_context_build")
        if context_build is not None:
            gate_intervals.append(context_build)
        if trigger == "per_turn" and barrier is not None:
            gate_intervals.append(barrier)
        residual = _direct_interval_overlap(barrier, sidecars) if barrier is not None else 0.0
        clean = _direct_clean(attributes) and all(
            _direct_clean(event.get("attributes", {}))
            for event in sample_events
            if event["name"] == "critical_path.judge_request"
        )
        rows.append(
            {
                "reward_gate_s": _union_duration(gate_intervals),
                "terminal_orm_s": _union_duration(terminal_orm_requests),
                "sidecar_interval_union_s": sidecar_union,
                "terminal_barrier_residual_s": barrier[1] - barrier[0] if barrier is not None else 0.0,
                "sidecar_terminal_residual_fraction": residual / sidecar_union if sidecar_union else 0.0,
                "per_turn_judge_count": attributes.get("per_turn_judge_count", 0),
                "assistant_turn_count": attributes.get("per_turn_assistant_turn_count"),
                "off_lineage_judge_count": attributes.get("per_turn_off_lineage_judge_count"),
                "fallback": attributes.get("reasoning_execution_trigger") == "terminal_once_fallback",
                "clean": clean,
            }
        )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: _direct_distribution(row[key] for row in subset)
            for key in (
                "reward_gate_s",
                "terminal_orm_s",
                "sidecar_interval_union_s",
                "terminal_barrier_residual_s",
                "sidecar_terminal_residual_fraction",
                "per_turn_judge_count",
                "assistant_turn_count",
                "off_lineage_judge_count",
            )
        }

    return {
        "raw_count": len(rows),
        "clean_count": sum(row["clean"] for row in rows),
        "fallback_count": sum(row["fallback"] for row in rows),
        "operational": summarize(rows),
        "clean": summarize([row for row in rows if row["clean"]]),
    }


def _direct_group_intervals(
    events: list[dict[str, Any]], measured_steps: list[int]
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    admissions: defaultdict[str, list[float]] = defaultdict(list)
    finalizations: defaultdict[str, set[tuple[float, float]]] = defaultdict(set)
    transfers: defaultdict[str, set[tuple[float, float]]] = defaultdict(set)
    for event in events:
        if event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        group_key = _direct_identity(
            attributes.get("group_key"), fallback=[attributes.get("group_index"), event.get("step")]
        )
        if event["name"] == "critical_path.session_terminal_admission":
            admissions[group_key].append(event["start_s"])
        elif event["name"] == "critical_path.reward_group_finalize":
            finalizations[group_key].add((event["start_s"], event["end_s"]))
        elif event["name"] == "critical_path.transfer":
            transfers[group_key].add((event["start_s"], event["end_s"]))
    inconsistent = {key: values for key, values in finalizations.items() if len(values) > 1}
    if inconsistent:
        raise ValueError(f"group-finalize timestamps differ across duplicated samples: {sorted(inconsistent)}")
    intervals = {
        key: (max(admissions[key]), next(iter(values))[1])
        for key, values in finalizations.items()
        if admissions.get(key) and next(iter(values))[1] >= max(admissions[key])
    }
    complete_keys = sorted(intervals)
    return intervals, {
        "group_finalize_count": len(finalizations),
        "complete_group_count": len(intervals),
        "missing_terminal_admission_group_count": len(set(finalizations) - set(intervals)),
        "missing_transfer_group_count": len(set(intervals) - set(transfers)),
        "group_completion_spread_s": _direct_distribution(
            max(admissions[key]) - min(admissions[key]) for key in complete_keys
        ),
        "group_finalize_s": _direct_distribution(
            next(iter(finalizations[key]))[1] - next(iter(finalizations[key]))[0] for key in complete_keys
        ),
        "group_transfer_delay_s": _direct_distribution(
            max(end for _start, end in transfers[key]) - next(iter(finalizations[key]))[1]
            for key in complete_keys
            if transfers.get(key)
        ),
    }


def _direct_trainer_report(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    group_intervals: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    inclusive: list[float] = []
    exclusive: list[float] = []
    plus_other: list[float] = []
    missing_provenance = 0
    ignored_non_batch_waits = 0
    offline_other_by_group: defaultdict[str, set[tuple[float, float]]] = defaultdict(set)
    for event in events:
        if event.get("step") not in measured_steps:
            continue
        if not (
            event["name"].startswith("critical_path.rollout_") or event["name"].startswith("critical_path.transfer")
        ):
            continue
        group_key = event.get("attributes", {}).get("group_key")
        if group_key is None:
            continue
        offline_other_by_group[_direct_identity(group_key, fallback=group_key)].add((event["start_s"], event["end_s"]))
    batches: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["name"] != "critical_path.data_wait" or event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        if attributes.get("returned_batch") is False:
            ignored_non_batch_waits += 1
            continue
        batch_id = attributes.get("trainer_batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            missing_provenance += 1
            continue
        batches[batch_id].append(event)

    for batch_id, batch_events in batches.items():
        returned_groups: list[Any] = []
        seen_groups: set[str] = set()
        missing_batch_keys = False
        for event in batch_events:
            event_groups = event.get("attributes", {}).get("returned_group_keys")
            if not isinstance(event_groups, list):
                missing_batch_keys = True
                break
            for value in event_groups:
                identity = _direct_identity(value, fallback=value)
                if identity not in seen_groups:
                    seen_groups.add(identity)
                    returned_groups.append(value)
        if missing_batch_keys:
            missing_provenance += 1
            continue
        reward_intervals = [
            group_intervals[key]
            for value in returned_groups
            if (key := _direct_identity(value, fallback=value)) in group_intervals
        ]
        wait = (
            min(event["start_s"] for event in batch_events),
            max(event["end_s"] for event in batch_events),
        )
        inclusive.append(_direct_interval_overlap(wait, reward_intervals))
        raw_other = next(
            (
                event.get("attributes", {}).get("observed_other_blocker_intervals")
                for event in batch_events
                if isinstance(event.get("attributes", {}).get("observed_other_blocker_intervals"), list)
            ),
            None,
        )
        if isinstance(raw_other, list):
            other = [
                (float(item["start_s"]), float(item["end_s"]))
                for item in raw_other
                if isinstance(item, dict)
                and isinstance(item.get("start_s"), (int, float))
                and isinstance(item.get("end_s"), (int, float))
            ]
        else:
            other = [
                interval
                for value in returned_groups
                for interval in offline_other_by_group.get(_direct_identity(value, fallback=value), set())
            ]
            if not other:
                continue
        reward_overlap_segments = [
            (max(wait[0], start), min(wait[1], end))
            for start, end in reward_intervals
            if min(wait[1], end) > max(wait[0], start)
        ]
        plus_other.append(sum(_direct_interval_overlap(segment, other) for segment in reward_overlap_segments))
        exclusive.append(
            sum(_union_duration(_subtract_intervals(segment, other)) for segment in reward_overlap_segments)
        )
    return {
        "data_wait_with_missing_returned_group_provenance_count": missing_provenance,
        "ignored_non_batch_data_wait_count": ignored_non_batch_waits,
        "returned_trainer_batch_count": len(batches),
        "inclusive_reward_ancestor_wait_s": _direct_distribution(inclusive),
        "exclusive_reward_wait_s": _direct_distribution(exclusive),
        "reward_plus_other_blocker_wait_s": _direct_distribution(plus_other),
        "exclusive_and_other_blocker_available": bool(exclusive or plus_other),
        "other_blocker_definition": (
            "group-keyed rollout and transfer ancestor intervals for the groups actually returned by this trainer batch"
        ),
    }


def _direct_publication_report(events: list[dict[str, Any]], measured_steps: list[int]) -> dict[str, Any]:
    boundaries = _ready_boundaries(events)
    ordered_ready_steps = sorted(boundaries)
    rows: list[dict[str, Any]] = []
    for step in measured_steps:
        step_events = [event for event in events if event.get("step") == step]
        step_index = ordered_ready_steps.index(step) if step in ordered_ready_steps else -1
        previous_step = ordered_ready_steps[step_index - 1] if step_index > 0 else None
        request_events = [event for event in step_events if event["name"] == "critical_path.judge_request"]
        sample_keys = {
            _direct_identity(
                event.get("attributes", {}).get("sample_key"),
                fallback=[
                    event.get("attributes", {}).get("group_index"),
                    event.get("attributes", {}).get("sample_index"),
                    step,
                ],
            )
            for event in step_events
            if event["name"] == "critical_path.reward"
        }
        group_keys = {
            _direct_identity(
                event.get("attributes", {}).get("group_key"),
                fallback=[event.get("attributes", {}).get("group_index"), step],
            )
            for event in step_events
            if event["name"] == "critical_path.reward_group_finalize"
        }
        reward_intervals = [
            (event["start_s"], event["end_s"])
            for event in step_events
            if event["name"] in {"critical_path.reward", "critical_path.reward_context_build"}
        ]
        sidecar_intervals = [
            (event["start_s"], event["end_s"])
            for event in request_events
            if event.get("attributes", {}).get("request_kind") == "per_turn_vlm"
        ]
        barrier_intervals = [
            (event["start_s"], event["end_s"])
            for event in step_events
            if event["name"] == "critical_path.turn_judge_barrier"
        ]
        rows.append(
            {
                "step": step,
                "previous_ready_step": previous_step,
                "ready_interval_s": (
                    boundaries[step] - boundaries[previous_step] if previous_step is not None else None
                ),
                "reward_union_s": _union_duration(reward_intervals),
                "sidecar_union_s": _union_duration(sidecar_intervals),
                "barrier_union_s": _union_duration(barrier_intervals),
                "reward_sidecar_barrier_union_s": _union_duration(
                    [*reward_intervals, *sidecar_intervals, *barrier_intervals]
                ),
                "request_count": len(request_events),
                "trajectory_count": len(sample_keys),
                "group_count": len(group_keys),
            }
        )
    return {
        "count": len(rows),
        "per_step": rows,
        "ready_interval_s": _direct_distribution(row["ready_interval_s"] for row in rows),
        "reward_union_s": _direct_distribution(row["reward_union_s"] for row in rows),
        "sidecar_union_s": _direct_distribution(row["sidecar_union_s"] for row in rows),
        "barrier_union_s": _direct_distribution(row["barrier_union_s"] for row in rows),
        "reward_sidecar_barrier_union_s": _direct_distribution(row["reward_sidecar_barrier_union_s"] for row in rows),
        "request_count": _direct_value_distribution(row["request_count"] for row in rows),
        "trajectory_count": _direct_value_distribution(row["trajectory_count"] for row in rows),
        "group_count": _direct_value_distribution(row["group_count"] for row in rows),
    }


def _direct_workload_report(events: list[dict[str, Any]], measured_steps: list[int]) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["name"] != "critical_path.sample_workload" or event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        identity = _direct_identity(
            attributes.get("sample_key"),
            fallback=[attributes.get("group_index"), attributes.get("sample_index"), event.get("step")],
        )
        samples[identity] = attributes
    rows = list(samples.values())
    version_counts: Counter[str] = Counter()
    for row in rows:
        versions = row.get("weight_versions")
        if isinstance(versions, list):
            version_counts.update(str(value) for value in set(versions))
    return {
        "sample_count": len(rows),
        "agent_turns": _direct_value_distribution(row.get("agent_turns") for row in rows),
        "input_tokens": _direct_value_distribution(row.get("input_tokens") for row in rows),
        "output_tokens": _direct_value_distribution(row.get("output_tokens") for row in rows),
        "total_tokens": _direct_value_distribution(row.get("total_tokens") for row in rows),
        "image_count": _direct_value_distribution(row.get("image_count") for row in rows),
        "image_tokens": _direct_value_distribution(row.get("image_tokens") for row in rows),
        "terminal_status_counts": dict(sorted(Counter(str(row.get("rollout_status")) for row in rows).items())),
        "policy_version_sample_counts": dict(sorted(version_counts.items())),
    }


def _direct_reliability_report(events: list[dict[str, Any]], measured_steps: list[int]) -> dict[str, Any]:
    request_events = [
        event
        for event in events
        if event["name"] == "critical_path.judge_request" and event.get("step") in measured_steps
    ]
    rewards: dict[str, dict[str, Any]] = {}
    accounting: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        if event["name"] == "critical_path.reward":
            identity = _direct_identity(
                attributes.get("sample_key"),
                fallback=[attributes.get("group_index"), attributes.get("sample_index"), event.get("step")],
            )
            rewards[identity] = attributes
        elif event["name"] == "critical_path.agentic_accounting_end" and isinstance(event.get("step"), int):
            accounting[event["step"]] = attributes
    status_counts = Counter(str(event.get("attributes", {}).get("request_status")) for event in request_events)
    return {
        "request_count": len(request_events),
        "request_status_counts": dict(sorted(status_counts.items())),
        "retry_request_count": sum(
            int(event.get("attributes", {}).get("attempt_count", 0) or 0) > 1 for event in request_events
        ),
        "invalid_response_count": sum(
            int(event.get("attributes", {}).get("invalid_response_count", 0) or 0) for event in request_events
        ),
        "fallback_trajectory_count": sum(
            row.get("reasoning_execution_trigger") == "terminal_once_fallback" for row in rewards.values()
        ),
        "off_lineage_judge_count": sum(
            int(row.get("per_turn_off_lineage_judge_count", 0) or 0) for row in rewards.values()
        ),
        "failed_reward_trajectory_count": sum(
            row.get("pipeline_status") != "success" or row.get("executor_status") != "success"
            for row in rewards.values()
        ),
        "judge_group_replacements": sum(
            int(row.get("judge_group_replacements", 0) or 0) for row in accounting.values()
        ),
        "interrupted_groups": sum(int(row.get("interrupted_groups", 0) or 0) for row in accounting.values()),
        "accounting_step_count": len(accounting),
    }


def analyze_direct_events(
    events: list[dict[str, Any]],
    *,
    warmup_steps: int = 1,
    measure_updates: int | None = None,
    expected_step_stride: int = 1,
    expected_reasoning_trigger: str | None = None,
    expected_groups_per_round: int | None = None,
    expected_samples_per_group: int | None = None,
    require_clock_host: bool = False,
    allow_synchronized_multi_host_clock: bool = False,
    multi_host_clock_max_offset_ms: float | None = None,
    gpu_sample_manifests: list[dict[str, Any]] | None = None,
    gpu_sample_records: list[dict[str, Any]] | None = None,
    gpu_sample_load_issues: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze one real training run without a baseline/counterfactual."""
    observed_modes = {
        event.get("attributes", {}).get("benchmark_mode")
        for event in events
        if event["name"] == "critical_path.reward" and event.get("attributes", {}).get("benchmark_mode") is not None
    }
    if observed_modes and observed_modes != {"dual"}:
        raise ValueError(
            f"standalone direct analysis accepts only real dual training traces, observed {sorted(observed_modes)}"
        )
    observed_triggers = {
        event.get("attributes", {}).get("reasoning_trigger")
        for event in events
        if event["name"] == "critical_path.reward" and event.get("attributes", {}).get("reasoning_trigger") is not None
    }
    if observed_triggers and not observed_triggers.issubset(REASONING_TRIGGERS):
        raise ValueError(
            f"standalone direct analysis observed unsupported reasoning trigger(s): {sorted(observed_triggers)}"
        )
    if expected_reasoning_trigger is not None and observed_triggers != {expected_reasoning_trigger}:
        raise ValueError(
            "standalone direct reasoning trigger mismatch: "
            f"expected {expected_reasoning_trigger!r}, observed {sorted(observed_triggers)}"
        )
    window, measured_steps = _direct_window(
        events,
        warmup_steps=warmup_steps,
        measure_updates=measure_updates,
        expected_step_stride=expected_step_stride,
    )
    group_intervals, group_coverage = _direct_group_intervals(events, measured_steps)
    if expected_groups_per_round is not None or expected_samples_per_group is not None:
        groups_by_step: defaultdict[int, dict[str, Any]] = defaultdict(dict)
        for event in events:
            if event["name"] != "critical_path.reward_group_finalize" or event.get("step") not in measured_steps:
                continue
            attributes = event.get("attributes", {})
            group_key = attributes.get("group_key")
            identity = _direct_identity(group_key, fallback=[attributes.get("group_index"), event.get("step")])
            groups_by_step[event["step"]][identity] = group_key
        cardinality_issues: list[str] = []
        for step in measured_steps:
            groups = groups_by_step.get(step, {})
            if expected_groups_per_round is not None and len(groups) != expected_groups_per_round:
                cardinality_issues.append(
                    f"step={step} groups expected={expected_groups_per_round} observed={len(groups)}"
                )
            if expected_samples_per_group is not None:
                for identity, group_key in groups.items():
                    observed = len(group_key) if isinstance(group_key, list) else None
                    if observed != expected_samples_per_group:
                        cardinality_issues.append(
                            f"step={step} group={identity} samples "
                            f"expected={expected_samples_per_group} observed={observed}"
                        )
        if cardinality_issues:
            raise ValueError(f"standalone direct fixed-K cardinality mismatch: {cardinality_issues[:8]}")
    clock_domain = _clock_domain_summary(
        events,
        relevant_steps=set(measured_steps),
        require_clock_host=require_clock_host,
        allow_synchronized_multi_host_clock=allow_synchronized_multi_host_clock,
    )
    if len(clock_domain["hosts"]) > 1:
        if multi_host_clock_max_offset_ms is None or multi_host_clock_max_offset_ms < 0:
            raise ValueError(
                "standalone direct multi-host analysis requires a non-negative "
                "multi_host_clock_max_offset_ms audit bound"
            )
        clock_domain["max_offset_ms"] = multi_host_clock_max_offset_ms
    else:
        clock_domain["max_offset_ms"] = 0.0
    reward_invariant_hashes = {
        event.get("attributes", {}).get("benchmark_invariant_hash")
        for event in events
        if event["name"] == "critical_path.reward"
        and event.get("step") in measured_steps
        and event.get("attributes", {}).get("benchmark_invariant_hash") is not None
    }
    ready_invariant_hashes = {
        event.get("attributes", {}).get("benchmark_invariant_hash")
        for event in events
        if event["name"] == "critical_path.weight_serving_ready"
        and event.get("step") in measured_steps
        and event.get("attributes", {}).get("benchmark_invariant_hash") is not None
    }
    benchmark_invariant_hash = None
    if reward_invariant_hashes or ready_invariant_hashes:
        if (
            len(reward_invariant_hashes) != 1
            or len(ready_invariant_hashes) != 1
            or reward_invariant_hashes != ready_invariant_hashes
        ):
            raise ValueError(
                "standalone direct benchmark invariant mismatch between reward and publication markers: "
                f"reward={sorted(reward_invariant_hashes)}, ready={sorted(ready_invariant_hashes)}"
            )
        benchmark_invariant_hash = next(iter(reward_invariant_hashes))
    trajectory_report = _direct_trajectory_report(events, measured_steps)
    measured_events = [event for event in events if event.get("step") in measured_steps]
    gpu_window = window
    if gpu_window is None and measured_events:
        gpu_window = (
            min(event["start_s"] for event in measured_events),
            max(event["end_s"] for event in measured_events),
        )
    gpu_report = None
    gpu_issues = list(gpu_sample_load_issues or [])
    if gpu_window is not None:
        gpu_report, computed_gpu_issues = _judge_gpu_efficiency_report(
            gpu_sample_manifests or [],
            gpu_sample_records or [],
            measurement_window=gpu_window,
            training_sample_count=trajectory_report["raw_count"],
        )
        gpu_issues.extend(computed_gpu_issues)
    return {
        "analysis_mode": "standalone_direct",
        "observed_benchmark_modes": sorted(observed_modes),
        "observed_reasoning_triggers": sorted(observed_triggers),
        "window_s": list(window) if window is not None else None,
        "gpu_measurement_scope": (
            "operational telemetry over the measured-step causal-work envelope through the final serving-ready "
            "marker; it is not request-lineage attribution"
            if window is not None
            else None
        ),
        "measured_steps": measured_steps,
        "clock_domain": clock_domain,
        "benchmark_invariant_hash": benchmark_invariant_hash,
        "request": _direct_request_report(events, measured_steps),
        "trajectory": trajectory_report,
        "group": {
            **group_coverage,
            "group_reward_closure_s": _direct_distribution(end - start for start, end in group_intervals.values()),
        },
        "trainer": _direct_trainer_report(events, measured_steps, group_intervals),
        "publication": _direct_publication_report(events, measured_steps),
        "workload": _direct_workload_report(events, measured_steps),
        "reliability": _direct_reliability_report(events, measured_steps),
        "judge_gpu_efficiency": gpu_report,
        "judge_gpu_efficiency_issues": gpu_issues,
    }


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _active_set_durations(events: list[dict[str, Any]], window: tuple[float, float]) -> dict[str, float]:
    window_start, window_end = window
    boundaries: list[tuple[float, int, str]] = []
    for event in events:
        start = max(window_start, event["start_s"])
        end = min(window_end, event["end_s"])
        if end <= start:
            continue
        boundaries.append((start, 1, event["stage"]))
        boundaries.append((end, -1, event["stage"]))
    boundaries.sort(key=lambda item: item[0])
    active: Counter[str] = Counter()
    durations: defaultdict[str, float] = defaultdict(float)
    previous = window_start
    index = 0
    while index < len(boundaries):
        timestamp = boundaries[index][0]
        if timestamp > previous:
            label = "+".join(stage for stage in STAGE_ORDER if active[stage] > 0) or "no_observed_span"
            durations[label] += timestamp - previous
        while index < len(boundaries) and boundaries[index][0] == timestamp:
            _timestamp, delta, stage = boundaries[index]
            active[stage] += delta
            index += 1
        previous = timestamp
    if window_end > previous:
        label = "+".join(stage for stage in STAGE_ORDER if active[stage] > 0) or "no_observed_span"
        durations[label] += window_end - previous
    return dict(durations)


def _subtract_intervals(
    interval: tuple[float, float],
    cuts: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    remaining = [interval]
    for cut_start, cut_end in sorted(cuts):
        next_remaining = []
        for start, end in remaining:
            if cut_end <= start or cut_start >= end:
                next_remaining.append((start, end))
                continue
            if cut_start > start:
                next_remaining.append((start, min(cut_start, end)))
            if cut_end < end:
                next_remaining.append((max(cut_end, start), end))
        remaining = next_remaining
    return [(start, end) for start, end in remaining if end > start]


def _remove_stream_wait_from_training(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn schedule envelopes into compute/orchestration spans by rank-local
    interval subtraction."""
    waits_by_pid: defaultdict[Any, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event["stage"] == "data_wait":
            waits_by_pid[event["pid"]].append((event["start_s"], event["end_s"]))
    output = []
    for event in events:
        if event["name"] != "critical_path.training_schedule":
            output.append(event)
            continue
        for start, end in _subtract_intervals(
            (event["start_s"], event["end_s"]),
            waits_by_pid[event["pid"]],
        ):
            output.append(
                {
                    **event,
                    "name": "critical_path.training_compute_and_orchestration",
                    "start_s": start,
                    "end_s": end,
                }
            )
    return output


def _ready_boundaries(events: list[dict[str, Any]]) -> dict[int, float]:
    boundaries: dict[int, float] = {}
    for event in events:
        step = event.get("step")
        if event["name"] == "critical_path.weight_serving_ready" and isinstance(step, int):
            boundaries[step] = max(boundaries.get(step, float("-inf")), event["end_s"])
    return boundaries


def _per_update_stage_distributions(
    events: list[dict[str, Any]],
    *,
    boundary_step: int,
    measured_steps: list[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Summarize inclusive stage occupancy for each ready-to-ready interval."""
    boundaries = _ready_boundaries(events)
    interval_durations: list[float] = []
    stage_percentages: dict[str, list[float]] = {stage: [] for stage in STAGE_ORDER}
    previous_step = boundary_step
    for step in measured_steps:
        start = boundaries[previous_step]
        end = boundaries[step]
        duration = end - start
        if duration <= 0:
            raise ValueError(f"non-positive serving-ready interval for steps {previous_step}->{step}: {duration}")
        interval_durations.append(duration)
        for stage in STAGE_ORDER:
            occupied = _union_duration(
                (
                    max(start, event["start_s"]),
                    min(end, event["end_s"]),
                )
                for event in events
                if event["stage"] == stage
            )
            stage_percentages[stage].append(100.0 * occupied / duration)
        previous_step = step
    return (
        _statistics(interval_durations),
        {stage: _statistics(values, unit_suffix="_percent") for stage, values in stage_percentages.items()},
    )


def _load_judge_gpu_samples(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Load one variant's judge/rollout GPU-sampler JSONL sidecar directory
    (see relax.utils.metrics.judge_gpu_sampler and
    GenRMManager.get_gpu_occupancy_snapshot).

    Returns ``(manifest_records, sample_records, issues)`` across every
    ``{role}_{host}_rank{N}.jsonl`` file found directly under ``path``. The
    sidecar remains best-effort, so malformed lines do not abort analysis;
    their count is returned as a report issue rather than being silently
    mistaken for a complete GPU sample stream.
    """
    if not path.is_dir():
        raise ValueError(f"--gpu-samples path must be a directory of sampler JSONL files: {path}")
    manifests: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    malformed_line_count = 0
    for jsonl_path in sorted(path.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_line_count += 1
                    continue
                if not isinstance(record, dict):
                    malformed_line_count += 1
                    continue
                record_type = record.get("record_type")
                if record_type == "sample":
                    samples.append(record)
                elif record_type == "manifest":
                    manifests.append(record)
    issues = [f"malformed_gpu_sample_lines:{malformed_line_count}"] if malformed_line_count else []
    return manifests, samples, issues


def _judge_gpu_efficiency_report(
    manifests: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    measurement_window: tuple[float, float],
    training_sample_count: int | None,
) -> tuple[dict[str, dict[str, Any]] | None, list[str]]:
    """Summarize judge/rollout GPU occupancy for sampler ticks whose ``ts``
    falls inside the measured window, one entry per ``role``.

    This answers "was the judge GPU actually busy", which the timeline trace
    alone cannot: ``no_observed_span`` in ``active_set_percent`` does not
    prove the hardware was idle (see README caveats). It complements, rather
    than replaces, the exposure-ratio/critical-path signal for "did the
    reward call sit on the critical path".

    Returns ``(report_by_role_or_None, issues)``. Never raises: coverage gaps
    are reported as soft ``issues`` strings (mirroring the
    ``validity_issues``/``valid_for_causal_latency_inference`` pattern used
    for paired-comparison concerns) rather than hard-failing the run.
    """
    issues: list[str] = []
    if not samples:
        return None, issues
    start, end = measurement_window
    window_s = end - start
    if window_s <= 0:
        return None, issues

    interval_by_role: dict[str, float] = {}
    manifest_nvml_by_role: defaultdict[str, set[bool]] = defaultdict(set)
    allocated_gpus_by_role_shard: dict[tuple[str, str, int], int] = {}
    for manifest in manifests:
        role = manifest.get("role")
        interval_s = manifest.get("interval_s")
        if (
            isinstance(role, str)
            and role not in interval_by_role
            and isinstance(interval_s, (int, float))
            and not isinstance(interval_s, bool)
            and interval_s > 0
        ):
            interval_by_role[role] = float(interval_s)
        nvml_enabled = manifest.get("nvml_enabled")
        if isinstance(role, str) and isinstance(nvml_enabled, bool):
            manifest_nvml_by_role[role].add(nvml_enabled)
        num_gpus = manifest.get("num_gpus_per_engine")
        engine_rank = manifest.get("engine_rank")
        clock_host = manifest.get("clock_host")
        if (
            isinstance(role, str)
            and isinstance(num_gpus, int)
            and not isinstance(num_gpus, bool)
            and num_gpus > 0
            and isinstance(engine_rank, int)
            and not isinstance(engine_rank, bool)
            and isinstance(clock_host, str)
            and clock_host
        ):
            shard_key = (role, clock_host, engine_rank)
            allocated_gpus_by_role_shard[shard_key] = max(allocated_gpus_by_role_shard.get(shard_key, 0), num_gpus)

    by_role: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in samples:
        ts = record.get("ts")
        role = record.get("role")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not isinstance(role, str):
            continue
        if start <= ts <= end:
            by_role[role].append(record)
    if not by_role:
        issues.append("no_gpu_samples_in_window")
        return None, issues

    report: dict[str, dict[str, Any]] = {}
    for role, records in sorted(by_role.items()):
        util_values: list[float] = []
        nonidle_ticks = 0
        running_reqs_values: list[float] = []
        token_usage_values: list[float] = []
        zero_request_ticks = 0
        sglang_tick_count = 0
        observed_gpu_keys: set[tuple[Any, ...]] = set()
        gpu_sample_tick_count = 0
        for record in records:
            gpus = record.get("gpu") or []
            record_has_gpu_sample = False
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                util = gpu.get("util_percent")
                if isinstance(util, (int, float)) and not isinstance(util, bool):
                    record_has_gpu_sample = True
                    util_values.append(float(util))
                    observed_gpu_keys.add(
                        (
                            record.get("clock_host"),
                            record.get("engine_rank"),
                            gpu.get("uuid") or gpu.get("index"),
                        )
                    )
                    if util > 0:
                        nonidle_ticks += 1
            if record_has_gpu_sample:
                gpu_sample_tick_count += 1
            sglang = record.get("sglang")
            if isinstance(sglang, dict):
                sglang_tick_count += 1
                running = sglang.get("sglang:num_running_reqs")
                if isinstance(running, (int, float)) and not isinstance(running, bool):
                    running_reqs_values.append(float(running))
                    if running == 0:
                        zero_request_ticks += 1
                token_usage = sglang.get("sglang:token_usage")
                if isinstance(token_usage, (int, float)) and not isinstance(token_usage, bool):
                    token_usage_values.append(float(token_usage))

        nvml_values = manifest_nvml_by_role.get(role, set())
        if nvml_values == {False}:
            nvml_available: bool | None = False
            issues.append(f"nvml_unavailable:{role}")
        elif nvml_values:
            nvml_available = True
            if len(nvml_values) > 1:
                issues.append(f"mixed_nvml_availability:{role}")
        else:
            nvml_available = None
        observed_gpu_count = len(observed_gpu_keys)
        allocated_gpu_count = sum(
            gpu_count
            for (manifest_role, _clock_host, _engine_rank), gpu_count in allocated_gpus_by_role_shard.items()
            if manifest_role == role
        )
        if allocated_gpu_count == 0:
            allocated_gpu_count = None
        coverage_fraction = None
        interval_s = interval_by_role.get(role)
        if interval_s and nvml_available is not False:
            expected_ticks = window_s / interval_s
            coverage_fraction = min(1.0, gpu_sample_tick_count / expected_ticks) if expected_ticks > 0 else None
        if coverage_fraction is not None and coverage_fraction < 0.5:
            issues.append(f"low_gpu_sample_coverage:{role}")
        if nvml_available is True and not util_values:
            issues.append(f"no_nvml_gpu_samples:{role}")
        resource_gpu_count = allocated_gpu_count if allocated_gpu_count is not None else observed_gpu_count or None
        report[role] = {
            "sample_tick_count": len(records),
            "gpu_sample_tick_count": gpu_sample_tick_count,
            "gpu_sample_count": len(util_values),
            "nvml_available": nvml_available,
            "gpu_count": observed_gpu_count or None,
            "allocated_gpu_count": allocated_gpu_count,
            "nonidle_fraction": (nonidle_ticks / len(util_values)) if util_values else None,
            "util_percent": _statistics(util_values, unit_suffix="_percent") or None,
            "zero_request_time_fraction": (zero_request_ticks / sglang_tick_count) if sglang_tick_count else None,
            "running_reqs": _statistics(running_reqs_values, unit_suffix="") or None,
            "token_usage": _statistics(token_usage_values, unit_suffix="") or None,
            "gpu_hours_in_window": (
                resource_gpu_count * window_s / 3600.0 if resource_gpu_count is not None else None
            ),
            "allocated_gpu_hours_in_window": (
                resource_gpu_count * window_s / 3600.0 if resource_gpu_count is not None else None
            ),
            "allocated_gpu_seconds_per_training_sample": (
                (resource_gpu_count * window_s) / training_sample_count
                if resource_gpu_count is not None and training_sample_count
                else None
            ),
            "sample_coverage_fraction": coverage_fraction,
        }
    return report, issues


def _reward_identity(event: dict[str, Any]) -> tuple[Any, ...]:
    attributes = event.get("attributes", {})
    trajectory_hash = attributes.get("trajectory_hash")
    workload_hash = (
        trajectory_hash if isinstance(trajectory_hash, str) and trajectory_hash else attributes.get("context_hash")
    )
    return (
        event.get("step"),
        attributes.get("group_index"),
        attributes.get("sample_index"),
        workload_hash,
        attributes.get("recorded_reward_hash"),
    )


def _overlaps(event: dict[str, Any], window: tuple[float, float]) -> bool:
    return event["end_s"] >= window[0] and event["start_s"] <= window[1]


def _observed_reasoning_trigger(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    *,
    measurement_window: tuple[float, float] | None = None,
    expected_reasoning_trigger: str | None = None,
) -> str | None:
    """Validate the trigger carried by every measured reward parent.

    Older traces did not record it; they retain the legacy terminal branch
    semantics when no explicit expectation is requested. New fixed-K A/B runs
    should pass an expectation so a terminal-once/per-turn mix cannot be
    silently analyzed as one variant.
    """
    if expected_reasoning_trigger is not None and expected_reasoning_trigger not in REASONING_TRIGGERS:
        raise ValueError(f"unknown expected reasoning trigger: {expected_reasoning_trigger!r}")
    parents = [
        event
        for event in events
        if event["name"] == "critical_path.reward"
        and (
            event.get("step") in measured_steps
            or (measurement_window is not None and _overlaps(event, measurement_window))
        )
    ]
    observed = {event.get("attributes", {}).get("reasoning_trigger") for event in parents}
    if observed == {None}:
        if expected_reasoning_trigger is not None:
            raise ValueError(
                "reward events are missing reasoning_trigger provenance; cannot validate the requested trigger"
            )
        return None
    if len(observed) != 1:
        raise ValueError(
            "fixed-K measurement requires exactly one observed reasoning trigger, "
            f"got {sorted(str(trigger) for trigger in observed)}"
        )
    trigger = next(iter(observed))
    if trigger not in REASONING_TRIGGERS:
        raise ValueError(f"unknown reasoning trigger in fixed-K measurement window: {trigger!r}")
    if expected_reasoning_trigger is not None and trigger != expected_reasoning_trigger:
        raise ValueError(f"reasoning trigger mismatch: expected {expected_reasoning_trigger!r}, observed {trigger!r}")
    return trigger


def _reasoning_execution_trigger_counts(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    *,
    measurement_window: tuple[float, float] | None = None,
    default_trigger: str | None = None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        if event["name"] != "critical_path.reward" or not (
            event.get("step") in measured_steps
            or (measurement_window is not None and _overlaps(event, measurement_window))
        ):
            continue
        execution_trigger = event.get("attributes", {}).get("reasoning_execution_trigger")
        if not isinstance(execution_trigger, str) or not execution_trigger:
            execution_trigger = default_trigger or "missing"
        counts[execution_trigger] += 1
    return counts


def _validate_reward_outcomes(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    *,
    measurement_window: tuple[float, float] | None = None,
    expected_benchmark_mode: str | None = None,
    observed_reasoning_trigger: str | None = None,
    max_rejected_samples: int = 0,
) -> str:
    parents = [
        event
        for event in events
        if event["name"] == "critical_path.reward"
        and (
            event.get("step") in measured_steps
            or (measurement_window is not None and _overlaps(event, measurement_window))
        )
    ]
    observed_modes = {event.get("attributes", {}).get("benchmark_mode") for event in parents}
    if len(observed_modes) != 1:
        raise ValueError(
            "fixed-K measurement requires exactly one observed reward benchmark mode, "
            f"got {sorted(str(mode) for mode in observed_modes)}"
        )
    observed_mode = next(iter(observed_modes))
    if observed_mode not in BENCHMARK_MODE_STATUS:
        raise ValueError(f"unknown reward benchmark mode in fixed-K measurement window: {observed_mode!r}")
    if expected_benchmark_mode is not None and observed_mode != expected_benchmark_mode:
        raise ValueError(
            f"reward benchmark mode mismatch: expected {expected_benchmark_mode!r}, observed {observed_mode!r}"
        )

    branch_counts: defaultdict[tuple[Any, ...], Counter] = defaultdict(Counter)
    turn_judge_counts: Counter[tuple[Any, ...]] = Counter()
    for event in events:
        prefix = "critical_path.reward."
        if event["name"].startswith(prefix):
            branch_counts[_reward_identity(event)][event["name"][len(prefix) :]] += 1
        if event["name"] == "critical_path.turn_judge":
            turn_judge_counts[_reward_identity(event)] += 1

    expected_pipeline_status, expected_executor_status = BENCHMARK_MODE_STATUS[observed_mode]
    reasoning_trigger = observed_reasoning_trigger or "terminal_once"
    issues = []
    for event in parents:
        attributes = event.get("attributes", {})
        executor_status = attributes.get("executor_status")
        pipeline_status = attributes.get("pipeline_status")
        terminal_outcome = attributes.get("terminal_outcome")
        violations: dict[str, Any] = {}
        execution_trigger = attributes.get("reasoning_execution_trigger")
        if execution_trigger is None:
            # Legacy terminal traces predate this field. New per-turn traces
            # without a fallback marker keep the ordinary per-turn behavior.
            execution_trigger = reasoning_trigger
        valid_execution_triggers = (
            {"terminal_once"} if reasoning_trigger == "terminal_once" else {"per_turn", "terminal_once_fallback"}
        )
        if observed_mode in {"dual", "dual_shadow"} and execution_trigger not in valid_execution_triggers:
            violations["reasoning_execution_trigger"] = {
                "expected": sorted(valid_execution_triggers),
                "observed": execution_trigger,
            }
        expected_branches = BENCHMARK_MODE_BRANCHES[observed_mode].copy()
        if execution_trigger == "per_turn" and observed_mode in {"dual", "dual_shadow"}:
            expected_branches.pop("multi_turn_reasoning", None)
        if (
            executor_status != expected_executor_status
            or pipeline_status != expected_pipeline_status
            or terminal_outcome is not None
            or branch_counts[_reward_identity(event)] != expected_branches
        ):
            violations.update(
                {
                    "executor_status": {"expected": expected_executor_status, "observed": executor_status},
                    "pipeline_status": {"expected": expected_pipeline_status, "observed": pipeline_status},
                    "terminal_outcome": {"expected": None, "observed": terminal_outcome},
                    "judge_branches": {
                        "expected": dict(expected_branches),
                        "observed": dict(branch_counts[_reward_identity(event)]),
                    },
                }
            )
        if execution_trigger == "per_turn" and observed_mode in {"dual", "dual_shadow"}:
            expected_turn_count = attributes.get("per_turn_judge_count")
            observed_turn_count = turn_judge_counts[_reward_identity(event)]
            if (
                isinstance(expected_turn_count, bool)
                or not isinstance(expected_turn_count, int)
                or expected_turn_count <= 0
            ):
                violations["per_turn_judge_count"] = {
                    "expected": "positive integer",
                    "observed": expected_turn_count,
                }
            elif observed_turn_count != expected_turn_count:
                violations["turn_judge_spans"] = {
                    "expected": expected_turn_count,
                    "observed": observed_turn_count,
                }
        elif execution_trigger == "terminal_once_fallback" and observed_mode in {"dual", "dual_shadow"}:
            fallback_turn_count = attributes.get("per_turn_judge_count")
            if (
                isinstance(fallback_turn_count, bool)
                or not isinstance(fallback_turn_count, int)
                or fallback_turn_count != 0
            ):
                violations["per_turn_judge_count"] = {
                    "expected": 0,
                    "observed": fallback_turn_count,
                }
            observed_turn_count = turn_judge_counts[_reward_identity(event)]
            if observed_turn_count != 0:
                violations["turn_judge_spans"] = {
                    "expected": 0,
                    "observed": observed_turn_count,
                }
        if violations:
            issues.append(
                {
                    "step": event.get("step"),
                    "benchmark_mode": observed_mode,
                    "reasoning_trigger": reasoning_trigger,
                    "reasoning_execution_trigger": execution_trigger,
                    "violations": violations,
                }
            )
    if issues:
        # A GRPO group rejected by a required judge (transient exhaustion/backpressure) is
        # sample-local, expected background noise at scale (see agentic_dual_judge/README.md
        # "A required judge failure rejects and replaces the entire GRPO group"), not evidence
        # of a misconfigured measurement. --allow-rejected-samples opts into tolerating a
        # bounded count of exactly this class; anything else (wrong benchmark mode, wrong
        # judge-branch cardinality on an otherwise-successful sample, wrong reasoning trigger)
        # always raises regardless of the tolerance, since those indicate a real config bug.
        def _is_rejection_class(issue: dict[str, Any]) -> bool:
            violations = issue["violations"]
            if set(violations) - {"executor_status", "pipeline_status", "terminal_outcome", "judge_branches"}:
                return False
            return violations.get("terminal_outcome", {}).get("observed") == "group_rejected"

        rejection_issues = [issue for issue in issues if _is_rejection_class(issue)]
        other_issues = [issue for issue in issues if not _is_rejection_class(issue)]
        if other_issues or len(rejection_issues) > max_rejected_samples:
            raise ValueError(
                f"invalid reward outcomes in fixed-K measurement window: {issues[:8]} "
                f"(rejected={len(rejection_issues)}, tolerance={max_rejected_samples}, other={len(other_issues)})"
            )
    return observed_mode


def _benchmark_invariant_hash(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    *,
    measurement_window: tuple[float, float],
    required: bool,
) -> str | None:
    parents = [
        event
        for event in events
        if event["name"] == "critical_path.reward"
        and (event.get("step") in measured_steps or _overlaps(event, measurement_window))
    ]
    values = {event.get("attributes", {}).get("benchmark_invariant_hash") for event in parents}
    if required and (not values or None in values):
        raise ValueError("reward events are missing benchmark invariant hashes")
    nonempty_values = {value for value in values if isinstance(value, str) and value}
    if len(nonempty_values) > 1:
        raise ValueError(f"variant contains mixed benchmark invariant hashes: {sorted(nonempty_values)}")
    return next(iter(nonempty_values)) if nonempty_values else None


def _critical_stage_coverage(events: list[dict[str, Any]], measured_steps: list[int]) -> dict[str, Any]:
    names_by_step: defaultdict[int, set[str]] = defaultdict(set)
    for event in events:
        step = event.get("step")
        if isinstance(step, int) and step in measured_steps:
            names_by_step[step].add(event["name"])
    missing_by_step = {
        step: [name for name in REQUIRED_CRITICAL_EVENT_NAMES if name not in names_by_step[step]]
        for step in measured_steps
    }
    missing_by_step = {step: names for step, names in missing_by_step.items() if names}
    return {
        "complete": not missing_by_step,
        "required_event_names": list(REQUIRED_CRITICAL_EVENT_NAMES),
        "missing_by_step": missing_by_step,
    }


def _sample_identity_coverage(
    events: list[dict[str, Any]], measured_steps: list[int], *, exclude_group_rejected: bool = False
) -> dict[str, Any]:
    """Prove that each measured reward parent has its sample-local prerequisite
    spans."""
    required_exact = (
        "critical_path.reward",
        "critical_path.reward_context_build",
        "critical_path.transfer",
    )
    required_nonempty = ("critical_path.rollout_generation",)
    relevant_names = {*required_exact, *required_nonempty}
    rejected_identities: set[tuple[Any, ...]] = set()
    if exclude_group_rejected:
        # A rejected group is replaced wholesale and never reaches critical_path.transfer
        # (see _workload_signature's matching exclusion above), so none of its sample-local
        # spans should be counted here either -- otherwise this function correctly, but
        # unhelpfully, reports the ordinary "never transferred" consequence of a rejection
        # as a coverage defect.
        rejected_identities = {
            _reward_identity(event)
            for event in events
            if event.get("step") in measured_steps
            and event["name"] == "critical_path.reward"
            and event.get("attributes", {}).get("terminal_outcome") == "group_rejected"
        }
    counts: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    expected_identities: set[tuple[Any, ...]] = set()
    for event in events:
        if event.get("step") not in measured_steps or event["name"] not in relevant_names:
            continue
        identity = _reward_identity(event)
        if identity in rejected_identities:
            continue
        counts[identity][event["name"]] += 1
        if event["name"] == "critical_path.reward":
            expected_identities.add(identity)

    issues = []
    for identity in sorted(expected_identities, key=repr):
        observed = counts[identity]
        violations = {
            name: {"expected": "exactly_one", "observed": observed[name]}
            for name in required_exact
            if observed[name] != 1
        }
        violations.update(
            {
                name: {"expected": "at_least_one", "observed": observed[name]}
                for name in required_nonempty
                if observed[name] < 1
            }
        )
        if violations:
            issues.append({"identity": list(identity), "violations": violations})

    unexpected_identities = [
        {
            "identity": list(identity),
            "event_counts": dict(sorted(event_counts.items())),
        }
        for identity, event_counts in sorted(counts.items(), key=lambda item: repr(item[0]))
        if identity not in expected_identities
    ]
    if unexpected_identities:
        issues.extend(
            {**item, "violations": {"reward_parent": {"expected": "exactly_one", "observed": 0}}}
            for item in unexpected_identities
        )

    identities_per_step = Counter(identity[0] for identity in expected_identities)
    return {
        "complete": bool(expected_identities) and not issues,
        "expected_identity_count": len(expected_identities),
        "expected_identity_count_by_step": {str(step): identities_per_step[step] for step in measured_steps},
        "requirements": {
            **{name: "exactly_one" for name in required_exact},
            **{name: "at_least_one" for name in required_nonempty},
        },
        "issues": issues,
    }


def _clock_domain_summary(
    events: list[dict[str, Any]],
    *,
    relevant_steps: set[int],
    require_clock_host: bool,
    allow_synchronized_multi_host_clock: bool,
) -> dict[str, Any]:
    # Reward work from future partitions may overlap the fixed-K window. Select
    # all such candidates by event kind before consulting wall-clock placement,
    # otherwise an untrusted timestamp could exclude itself from this gate.
    relevant = [
        event
        for event in events
        if event.get("step") in relevant_steps
        or event["name"].startswith("critical_path.reward")
        or event["name"].startswith("critical_path.turn_judge")
    ]
    missing_count = sum(not event.get("attributes", {}).get("clock_host") for event in relevant)
    hosts = sorted(
        {
            str(event.get("attributes", {}).get("clock_host"))
            for event in relevant
            if event.get("attributes", {}).get("clock_host")
        }
    )
    if require_clock_host and missing_count:
        raise ValueError(f"critical-path events are missing clock_host provenance: count={missing_count}")
    if len(hosts) > 1 and not allow_synchronized_multi_host_clock:
        raise ValueError(
            "timeline spans multiple wall-clock hosts; validate NTP/PTP skew and rerun with "
            "--allow-synchronized-multi-host-clock"
        )
    return {
        "hosts": hosts,
        "relevant_ready_steps": sorted(relevant_steps),
        "event_count": len(relevant),
        "reward_wip_candidate_event_count": sum(
            event["name"].startswith("critical_path.reward") or event["name"].startswith("critical_path.turn_judge")
            for event in relevant
        ),
        "event_count_missing_clock_host": missing_count,
        "multi_host_clock_sync_asserted": len(hosts) <= 1 or allow_synchronized_multi_host_clock,
        "selection": "measured_ready_steps_or_any_reward_wip_before_wall_overlap_filter",
    }


def _expected_trainer_components(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    *,
    asserted_components: set[str] | None,
) -> tuple[set[str], dict[str, Any]]:
    parents = [
        event for event in events if event["name"] == "critical_path.reward" and event.get("step") in measured_steps
    ]
    declarations: list[tuple[str, ...]] = []
    missing_count = 0
    for event in parents:
        value = event.get("attributes", {}).get("expected_trainer_components")
        if value is None:
            missing_count += 1
            continue
        if not isinstance(value, (list, tuple, set)) or any(
            not isinstance(component, str) or not component for component in value
        ):
            raise ValueError(f"invalid expected_trainer_components reward attribute: {value!r}")
        declaration = tuple(sorted(set(value)))
        if not declaration:
            raise ValueError("expected_trainer_components reward attribute cannot be empty")
        declarations.append(declaration)
    if declarations and missing_count:
        raise ValueError(
            "expected_trainer_components is missing from some measured reward parents: "
            f"missing={missing_count}, declared={len(declarations)}"
        )
    unique_declarations = set(declarations)
    if len(unique_declarations) > 1:
        raise ValueError(
            f"measured reward parents disagree on expected trainer components: {sorted(unique_declarations)}"
        )
    inferred = set(next(iter(unique_declarations))) if unique_declarations else set()
    asserted = set(asserted_components or ())
    required = {"actor", *inferred, *asserted}
    return required, {
        "inferred_from_reward": sorted(inferred),
        "asserted_by_cli": sorted(asserted),
        "required": sorted(required),
        "reward_parent_count": len(parents),
        "reward_parent_count_missing_declaration": missing_count,
    }


def _workload_signature(
    events: list[dict[str, Any]], measured_steps: list[int], *, exclude_group_rejected: bool = False
) -> dict[int, Counter]:
    signatures: defaultdict[int, Counter] = defaultdict(Counter)
    for event in events:
        step = event.get("step")
        if event["name"] != "critical_path.reward" or step not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        if exclude_group_rejected and attributes.get("terminal_outcome") == "group_rejected":
            # A rejected group's own reward-attempt record can look fully formed (all
            # samples present) or partial, depending on where in the pipeline it was
            # rejected -- either way it is not part of the final committed workload
            # (see _validate_reward_outcomes' rejection-class tolerance above), so it
            # must not count toward this step's group/sample cardinality either.
            continue
        trajectory_hash = attributes.get("trajectory_hash")
        workload_hash = (
            trajectory_hash if isinstance(trajectory_hash, str) and trajectory_hash else attributes.get("context_hash")
        )
        identity = (
            attributes.get("group_index"),
            attributes.get("sample_index"),
            workload_hash,
            attributes.get("recorded_reward_hash"),
        )
        signatures[step][identity] += 1
    return dict(signatures)


def _overlapping_workload_signature(
    events: list[dict[str, Any]],
    measurement_window: tuple[float, float],
) -> Counter:
    return Counter(
        _reward_identity(event)
        for event in events
        if event["name"] == "critical_path.reward" and _overlaps(event, measurement_window)
    )


def _validate_workload_cardinality(
    signature: dict[int, Counter],
    measured_steps: list[int],
    *,
    expected_groups_per_round: int | None,
    expected_samples_per_group: int | None,
    max_rejected_samples: int = 0,
) -> None:
    if (expected_groups_per_round is None) != (expected_samples_per_group is None):
        raise ValueError("expected group count and samples per group must be specified together")
    if expected_groups_per_round is None or expected_samples_per_group is None:
        return
    if expected_groups_per_round < 1 or expected_samples_per_group < 1:
        raise ValueError("expected workload cardinalities must be positive")
    issues = []
    for step in measured_steps:
        group_counts: Counter = Counter()
        for identity, count in signature.get(step, {}).items():
            group_index, sample_index, _context_hash, _recorded_reward_hash = identity
            if group_index is None or sample_index is None:
                issues.append({"step": step, "reason": "missing group/sample index"})
                continue
            group_counts[group_index] += count
        # A GRPO group replacement (see _validate_reward_outcomes above) leaves the
        # rejected attempt's own partial sample records in the signature alongside its
        # full-cardinality replacement, so the step transiently shows one extra,
        # under-populated group. Tolerate a bounded number of those stray records (the
        # same budget as --allow-rejected-samples) rather than requiring the raw event
        # signature to already reflect only the final committed set.
        clean_groups = {index: count for index, count in group_counts.items() if count == expected_samples_per_group}
        stray_sample_count = sum(count for index, count in group_counts.items() if count != expected_samples_per_group)
        if (
            len(clean_groups) == expected_groups_per_round
            and stray_sample_count > 0
            and stray_sample_count <= max_rejected_samples
        ):
            continue
        if len(group_counts) != expected_groups_per_round or any(
            count != expected_samples_per_group for count in group_counts.values()
        ):
            issues.append(
                {
                    "step": step,
                    "observed_group_count": len(group_counts),
                    "observed_samples_per_group": dict(group_counts),
                    "expected_group_count": expected_groups_per_round,
                    "expected_samples_per_group": expected_samples_per_group,
                }
            )
    if issues:
        raise ValueError(f"reward workload cardinality mismatch: {issues[:8]} (tolerance={max_rejected_samples})")


def _optimizer_steps_per_publication_round(
    events: list[dict[str, Any]],
    measured_steps: list[int],
    *,
    require_step_ids: bool,
    expected_components: set[str] | None = None,
) -> dict[str, Any]:
    rank_step_ids: defaultdict[tuple[int, str, Any], set[int]] = defaultdict(set)
    issues = []
    for event in events:
        if event["name"] != "critical_path.optimizer_step" or event.get("step") not in measured_steps:
            continue
        attributes = event.get("attributes", {})
        optimizer_step_id = attributes.get("optimizer_step_id")
        component = attributes.get("component")
        rank = attributes.get("global_rank", event.get("pid"))
        if (
            isinstance(optimizer_step_id, bool)
            or not isinstance(optimizer_step_id, int)
            or not isinstance(component, str)
            or not component
        ):
            issues.append(
                {
                    "step": event.get("step"),
                    "optimizer_step_id": optimizer_step_id,
                    "component": component,
                }
            )
            continue
        rank_step_ids[(event["step"], component, rank)].add(optimizer_step_id)
    if require_step_ids and issues:
        raise ValueError(f"optimizer spans are missing logical step IDs or component provenance: {issues[:8]}")

    observed_training_components = {
        event.get("attributes", {}).get("component")
        for event in events
        if event.get("step") in measured_steps
        and event["name"] in {"critical_path.training_schedule", "critical_path.optimizer_step"}
        and isinstance(event.get("attributes", {}).get("component"), str)
    }
    required_components = set(expected_components or ())
    if require_step_ids:
        required_components.add("actor")
    components = sorted(
        {component for _step, component, _rank in rank_step_ids} | observed_training_components | required_components
    )
    output = {}
    for component in components:
        per_round = []
        expected_ranks = None
        for step in measured_steps:
            rank_to_ids = {
                event_rank: ids
                for (event_step, event_component, event_rank), ids in rank_step_ids.items()
                if event_step == step and event_component == component
            }
            if not rank_to_ids:
                if require_step_ids and component in required_components | observed_training_components:
                    raise ValueError(
                        f"optimizer trace for component {component!r} is missing publication round {step}"
                    )
                continue
            ranks = set(rank_to_ids)
            if expected_ranks is None:
                expected_ranks = ranks
            elif ranks != expected_ranks:
                raise ValueError(
                    f"optimizer rank set changed for {component} at publication round {step}: "
                    f"{sorted(str(rank) for rank in ranks)} != {sorted(str(rank) for rank in expected_ranks)}"
                )
            rank_sets = list(rank_to_ids.values())
            expected_ids = rank_sets[0]
            if any(ids != expected_ids for ids in rank_sets[1:]):
                raise ValueError(
                    f"optimizer step IDs disagree across {component} ranks for publication round {step}: "
                    f"{[sorted(ids) for ids in rank_sets]}"
                )
            per_round.append({"publication_round": step, "optimizer_step_count": len(expected_ids)})
        counts = [float(item["optimizer_step_count"]) for item in per_round]
        output[component] = {
            "per_round": per_round,
            "distribution": _statistics(counts, unit_suffix="_steps"),
        }
    return output


def _fixed_k_window(
    events: list[dict[str, Any]],
    *,
    warmup_updates: int,
    measure_updates: int,
    expected_step_stride: int = 1,
) -> tuple[tuple[float, float], list[int], int]:
    if warmup_updates < 1:
        raise ValueError("ready-to-ready measurement requires at least one warmup update")
    if measure_updates < 1:
        raise ValueError("measure_updates must be positive")
    if expected_step_stride < 1:
        raise ValueError("expected_step_stride must be positive")
    boundaries = _ready_boundaries(events)
    ordered_steps = sorted(boundaries)
    required = warmup_updates + measure_updates
    if len(ordered_steps) < required:
        raise ValueError(f"incomplete timeline: need {required} serving-ready boundaries, found {len(ordered_steps)}")
    selected_boundaries = ordered_steps[:required]
    unexpected_pairs = [
        (previous, current)
        for previous, current in zip(selected_boundaries, selected_boundaries[1:])
        if current - previous != expected_step_stride
    ]
    if unexpected_pairs:
        raise ValueError(
            "baseline serving-ready steps are not contiguous at the expected stride "
            f"{expected_step_stride}: {unexpected_pairs}"
        )
    boundary_step = ordered_steps[warmup_updates - 1]
    measured_steps = ordered_steps[warmup_updates:required]
    return (boundaries[boundary_step], boundaries[measured_steps[-1]]), measured_steps, boundary_step


def _required_ready_window(
    events: list[dict[str, Any]],
    required_steps: list[int],
) -> tuple[float, float]:
    """Resolve one baseline-selected ready-step sequence in another
    timeline."""
    if len(required_steps) < 2:
        raise ValueError("a fixed-K ready window requires one boundary step and at least one measured step")
    boundaries = _ready_boundaries(events)
    missing = [step for step in required_steps if step not in boundaries]
    if missing:
        raise ValueError(f"timeline is missing serving-ready steps: {missing}")
    ordered_steps = sorted(boundaries)
    boundary_index = ordered_steps.index(required_steps[0])
    observed_steps = ordered_steps[boundary_index : boundary_index + len(required_steps)]
    if observed_steps != required_steps:
        raise ValueError(
            f"serving-ready update sequence differs from baseline: expected {required_steps}, got {observed_steps}"
        )
    return boundaries[required_steps[0]], boundaries[required_steps[-1]]


def analyze_events(
    events: list[dict[str, Any]],
    *,
    warmup_steps: int = 0,
    measure_updates: int | None = None,
    source: str = "timeline",
    expected_step_stride: int = 1,
    required_ready_steps: list[int] | None = None,
    require_complete_coverage: bool = False,
    expected_benchmark_mode: str | None = None,
    expected_reasoning_trigger: str | None = None,
    expected_groups_per_round: int | None = None,
    expected_samples_per_group: int | None = None,
    require_clock_host: bool = False,
    allow_synchronized_multi_host_clock: bool = False,
    require_benchmark_invariant_hash: bool = False,
    require_optimizer_step_ids: bool = False,
    expected_optimizer_components: set[str] | None = None,
    gpu_sample_manifests: list[dict[str, Any]] | None = None,
    gpu_sample_records: list[dict[str, Any]] | None = None,
    gpu_sample_load_issues: list[str] | None = None,
    max_rejected_samples: int = 0,
) -> dict[str, Any]:
    original_events = events
    events = _remove_stream_wait_from_training(events)
    steps = sorted({event["step"] for event in events if event["step"] is not None})
    measured_steps = steps[warmup_steps:]
    fixed_window = None
    boundary_step = None
    per_update_ready_interval = None
    per_update_stage_occupancy = None
    critical_stage_coverage = None
    sample_identity_coverage = None
    observed_benchmark_mode = None
    observed_reasoning_trigger = None
    reasoning_execution_trigger_counts: Counter[str] | None = None
    clock_domains = None
    benchmark_invariant_hash = None
    optimizer_steps_per_publication_round = None
    expected_trainer_component_summary = None
    if measure_updates is not None:
        if source != "timeline":
            raise ValueError("fixed-K E2E measurement requires timeline input, not rollout JSONL")
        if required_ready_steps is None:
            fixed_window, measured_steps, boundary_step = _fixed_k_window(
                events,
                warmup_updates=warmup_steps,
                measure_updates=measure_updates,
                expected_step_stride=expected_step_stride,
            )
        else:
            if len(required_ready_steps) != measure_updates + 1:
                raise ValueError(
                    "baseline-selected ready-step sequence must contain "
                    f"measure_updates + 1 entries, got {len(required_ready_steps)}"
                )
            fixed_window = _required_ready_window(events, required_ready_steps)
            boundary_step = required_ready_steps[0]
            measured_steps = required_ready_steps[1:]
        clock_domains = _clock_domain_summary(
            original_events,
            relevant_steps={boundary_step, *measured_steps},
            require_clock_host=require_clock_host,
            allow_synchronized_multi_host_clock=allow_synchronized_multi_host_clock,
        )
        observed_reasoning_trigger = _observed_reasoning_trigger(
            events,
            measured_steps,
            measurement_window=fixed_window,
            expected_reasoning_trigger=expected_reasoning_trigger,
        )
        observed_benchmark_mode = _validate_reward_outcomes(
            events,
            measured_steps,
            measurement_window=fixed_window,
            expected_benchmark_mode=expected_benchmark_mode,
            observed_reasoning_trigger=observed_reasoning_trigger,
            max_rejected_samples=max_rejected_samples,
        )
        reasoning_execution_trigger_counts = _reasoning_execution_trigger_counts(
            events,
            measured_steps,
            measurement_window=fixed_window,
            default_trigger=observed_reasoning_trigger,
        )
        benchmark_invariant_hash = _benchmark_invariant_hash(
            events,
            measured_steps,
            measurement_window=fixed_window,
            required=require_benchmark_invariant_hash,
        )
        _validate_workload_cardinality(
            _workload_signature(events, measured_steps, exclude_group_rejected=max_rejected_samples > 0),
            measured_steps,
            expected_groups_per_round=expected_groups_per_round,
            expected_samples_per_group=expected_samples_per_group,
            max_rejected_samples=max_rejected_samples,
        )
        critical_stage_coverage = _critical_stage_coverage(original_events, measured_steps)
        sample_identity_coverage = _sample_identity_coverage(
            original_events, measured_steps, exclude_group_rejected=max_rejected_samples > 0
        )
        if require_complete_coverage and (
            not critical_stage_coverage["complete"] or not sample_identity_coverage["complete"]
        ):
            raise ValueError(
                "incomplete critical-stage trace coverage or sample-identity coverage in fixed-K window: "
                f"stage={critical_stage_coverage['missing_by_step']}, "
                f"sample={sample_identity_coverage['issues'][:8]}"
            )
        per_update_ready_interval, per_update_stage_occupancy = _per_update_stage_distributions(
            events,
            boundary_step=boundary_step,
            measured_steps=measured_steps,
        )
        required_optimizer_components, expected_trainer_component_summary = _expected_trainer_components(
            original_events,
            measured_steps,
            asserted_components=expected_optimizer_components,
        )
        optimizer_steps_per_publication_round = _optimizer_steps_per_publication_round(
            original_events,
            measured_steps,
            require_step_ids=require_optimizer_step_ids,
            expected_components=required_optimizer_components,
        )
    elif measured_steps:
        events = [event for event in events if event["step"] in measured_steps]
    if not events:
        return {"steps": measured_steps, "event_count": 0, "coverage": source}

    per_step = defaultdict(list)
    per_step_events = (
        [event for event in events if event.get("step") in measured_steps] if fixed_window is not None else events
    )
    for event in per_step_events:
        per_step[event["step"]].append(event)
    step_windows: dict[int | None, tuple[float, float]] = {}
    step_makespans: list[float] = []
    for step, step_events in per_step.items():
        start = min(event["start_s"] for event in step_events)
        serving_ready_ends = [
            event["end_s"] for event in step_events if event["name"] == "critical_path.weight_serving_ready"
        ]
        end = max(serving_ready_ends) if serving_ready_ends else max(event["end_s"] for event in step_events)
        if end >= start:
            step_windows[step] = (start, end)
            step_makespans.append(end - start)

    if fixed_window is None:
        measurement_start = min(event["start_s"] for event in events)
        serving_ready_ends = [
            event["end_s"] for event in events if event["name"] == "critical_path.weight_serving_ready"
        ]
        measurement_end = max(serving_ready_ends) if serving_ready_ends else max(event["end_s"] for event in events)
    else:
        measurement_start, measurement_end = fixed_window
        events = [
            event for event in events if event["end_s"] >= measurement_start and event["start_s"] <= measurement_end
        ]
    measurement_window = (measurement_start, measurement_end)
    total_window_s = measurement_end - measurement_start
    training_sample_count = (
        expected_groups_per_round * expected_samples_per_group * len(measured_steps)
        if expected_groups_per_round is not None and expected_samples_per_group is not None and measured_steps
        else None
    )
    judge_gpu_efficiency, judge_gpu_efficiency_issues = _judge_gpu_efficiency_report(
        gpu_sample_manifests or [],
        gpu_sample_records or [],
        measurement_window=measurement_window,
        training_sample_count=training_sample_count,
    )
    judge_gpu_efficiency_issues = [*(gpu_sample_load_issues or []), *judge_gpu_efficiency_issues]
    if clock_domains is not None and judge_gpu_efficiency is not None:
        known_hosts = set(clock_domains.get("hosts", []))
        sample_hosts = {
            record.get("clock_host")
            for record in (gpu_sample_records or [])
            if isinstance(record.get("ts"), (int, float))
            and not isinstance(record.get("ts"), bool)
            and measurement_start <= record["ts"] <= measurement_end
        }
        if known_hosts and sample_hosts - known_hosts:
            judge_gpu_efficiency_issues.append("gpu_sample_clock_host_mismatch")
    stage_occupancy_s: dict[str, float] = {}
    event_name_durations: defaultdict[str, list[float]] = defaultdict(list)
    event_name_clipped_durations: defaultdict[str, list[float]] = defaultdict(list)
    for stage in STAGE_ORDER:
        stage_occupancy_s[stage] = _union_duration(
            (
                max(measurement_start, event["start_s"]),
                min(measurement_end, event["end_s"]),
            )
            for event in events
            if event["stage"] == stage
        )
    for event in events:
        event_name_durations[event["name"]].append(event["end_s"] - event["start_s"])
        event_name_clipped_durations[event["name"]].append(
            max(0.0, min(measurement_end, event["end_s"]) - max(measurement_start, event["start_s"]))
        )

    active_set_s = defaultdict(float, _active_set_durations(events, measurement_window))

    # Equal-split attribution is explicitly a wall-time accounting view, not a
    # causal critical-path proof. It is useful because it sums to 100% while
    # preserving overlap separately in active_set_percent.
    split_attribution_s: defaultdict[str, float] = defaultdict(float)
    for active_label, duration in active_set_s.items():
        if active_label == "no_observed_span":
            split_attribution_s["no_observed_span"] += duration
            continue
        active_stages = active_label.split("+")
        for stage in active_stages:
            split_attribution_s[stage] += duration / len(active_stages)

    def percentages(values: dict[str, float]) -> dict[str, float]:
        return {
            key: (100.0 * value / total_window_s if total_window_s > 0 else 0.0)
            for key, value in sorted(values.items())
        }

    report = {
        "steps": measured_steps,
        "ready_boundary_step": boundary_step,
        "event_count": len(events),
        "coverage": "full_e2e" if source == "timeline" else "rollout_only",
        "total_window_s": total_window_s,
        "measurement_makespan_s": total_window_s,
        "fixed_k_ready_to_ready_makespan_s": total_window_s if fixed_window is not None else None,
        "per_update_ready_interval": per_update_ready_interval,
        "per_update_inclusive_occupancy_percent": per_update_stage_occupancy,
        "critical_stage_coverage": critical_stage_coverage,
        "sample_identity_coverage": sample_identity_coverage,
        "observed_benchmark_mode": observed_benchmark_mode,
        "observed_reasoning_trigger": observed_reasoning_trigger,
        "reasoning_execution_trigger_counts": (
            dict(reasoning_execution_trigger_counts) if reasoning_execution_trigger_counts is not None else None
        ),
        "per_turn_fallback_sample_count": (
            float(reasoning_execution_trigger_counts.get("terminal_once_fallback", 0))
            if reasoning_execution_trigger_counts is not None
            else None
        ),
        "clock_domains": clock_domains,
        "benchmark_invariant_hash": benchmark_invariant_hash,
        "expected_trainer_components": expected_trainer_component_summary,
        "optimizer_steps_per_publication_round": optimizer_steps_per_publication_round,
        "step_makespan": _statistics(step_makespans),
        "inclusive_occupancy_percent": percentages(stage_occupancy_s),
        "active_set_percent": percentages(dict(active_set_s)),
        "equal_split_observed_wall_percent": percentages(dict(split_attribution_s)),
        "overlap_split_wall_time_percent": percentages(dict(split_attribution_s)),
        "event_latency_by_name": {name: _statistics(values) for name, values in sorted(event_name_durations.items())},
        "event_clipped_duration_by_name": {
            name: _statistics(values) for name, values in sorted(event_name_clipped_durations.items())
        },
        "judge_gpu_efficiency": judge_gpu_efficiency,
        "judge_gpu_efficiency_issues": judge_gpu_efficiency_issues,
        "caveat": (
            "overlap_split_wall_time_percent is additive wall-time accounting, not a causal critical-path share; "
            "paired makespan_delta is the exposed end-to-end effect. judge_gpu_efficiency is a best-effort "
            "secondary signal from an unvalidated sidecar channel, not a hard-gated benchmark input."
        ),
    }
    if source != "timeline":
        return {
            "steps": measured_steps,
            "event_count": len(events),
            "coverage": "rollout_only",
            "observed_rollout_window_s": total_window_s,
            "observed_inclusive_occupancy_percent": percentages(stage_occupancy_s),
            "event_latency_by_name": report["event_latency_by_name"],
            "event_clipped_duration_by_name": report["event_clipped_duration_by_name"],
            "judge_gpu_efficiency": judge_gpu_efficiency,
            "judge_gpu_efficiency_issues": judge_gpu_efficiency_issues,
            "caveat": "rollout JSONL has no trainer/weight spans and cannot support E2E makespan or stage shares.",
        }
    return report


def _stage_occupancy_seconds(variant_report: dict[str, Any], stage: str) -> float | None:
    """Recover one stage's absolute wall-seconds from a variant's
    ``analyze_events`` report (which only stores the percent view), for the
    exposure-ratio calculation in ``main()``.

    Returns None when the report lacks either field, e.g. a rollout-JSONL
    variant whose reduced report has no ``inclusive_occupancy_percent``.
    """
    occupancy_percent = variant_report.get("inclusive_occupancy_percent")
    total_window_s = variant_report.get("total_window_s")
    if not isinstance(occupancy_percent, dict) or not isinstance(total_window_s, (int, float)):
        return None
    value = occupancy_percent.get(stage)
    if not isinstance(value, (int, float)):
        return None
    return value / 100.0 * total_window_s


def _exposure_ratio(baseline_report: dict[str, Any], candidate_report: dict[str, Any]) -> dict[str, Any]:
    """Estimate whether the reward workload actually sat on the critical path,
    by comparing the candidate-vs-baseline delta in ``data_wait`` (trainer
    starved waiting on data) against the total judge-work delta. Terminal
    reward and asynchronous per-turn VLM work remain separate report fields;
    they are combined only for this exposure denominator.

    ``exposure_ratio`` near 0 means reward growth was hidden by async overlap;
    near 1 means it was fully exposed as added trainer stall. This is the
    primary answer to "is reward on the critical path", complementing
    ``paired_makespan_delta_vs_baseline`` (the "how much slower overall").
    """
    base_data_wait = _stage_occupancy_seconds(baseline_report, "data_wait")
    cand_data_wait = _stage_occupancy_seconds(candidate_report, "data_wait")
    base_reward = _stage_occupancy_seconds(baseline_report, "reward")
    cand_reward = _stage_occupancy_seconds(candidate_report, "reward")
    base_turn_judge = _stage_occupancy_seconds(baseline_report, "turn_judge")
    cand_turn_judge = _stage_occupancy_seconds(candidate_report, "turn_judge")
    # Pre-per-turn timeline reports do not carry this new stage. They have no
    # hidden interaction spans, so interpret a missing field as zero when the
    # terminal reward stage itself is present.
    if base_turn_judge is None and base_reward is not None:
        base_turn_judge = 0.0
    if cand_turn_judge is None and cand_reward is not None:
        cand_turn_judge = 0.0
    entry: dict[str, Any] = {
        "delta_data_wait_s": None,
        "delta_reward_occupancy_s": None,
        "delta_turn_judge_occupancy_s": None,
        "delta_total_judge_occupancy_s": None,
        "exposure_ratio": None,
    }
    if base_data_wait is not None and cand_data_wait is not None:
        entry["delta_data_wait_s"] = cand_data_wait - base_data_wait
    if base_reward is not None and cand_reward is not None:
        entry["delta_reward_occupancy_s"] = cand_reward - base_reward
    if base_turn_judge is not None and cand_turn_judge is not None:
        entry["delta_turn_judge_occupancy_s"] = cand_turn_judge - base_turn_judge
    delta_reward = entry["delta_reward_occupancy_s"]
    delta_turn_judge = entry["delta_turn_judge_occupancy_s"]
    if delta_reward is not None and delta_turn_judge is not None:
        entry["delta_total_judge_occupancy_s"] = delta_reward + delta_turn_judge
    delta_data_wait = entry["delta_data_wait_s"]
    delta_judge = entry["delta_total_judge_occupancy_s"]
    if delta_judge is not None and delta_data_wait is not None and delta_judge != 0:
        entry["exposure_ratio"] = delta_data_wait / delta_judge
    return entry


def _paired_makespan_delta(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    warmup_steps: int = 0,
    measure_updates: int | None = None,
    expected_step_stride: int = 1,
    expected_groups_per_round: int | None = None,
    expected_samples_per_group: int | None = None,
    expected_baseline_mode: str | None = None,
    expected_candidate_mode: str | None = None,
    expected_baseline_reasoning_trigger: str | None = None,
    expected_candidate_reasoning_trigger: str | None = None,
    require_benchmark_invariant_hash: bool = False,
    allow_overlapping_workload_mismatch: bool = False,
    allow_reward_changing_pair: bool = False,
) -> dict[str, Any]:
    if measure_updates is not None:
        base_window, paired_steps, boundary_step = _fixed_k_window(
            baseline,
            warmup_updates=warmup_steps,
            measure_updates=measure_updates,
            expected_step_stride=expected_step_stride,
        )
        base_boundaries = _ready_boundaries(baseline)
        candidate_boundaries = _ready_boundaries(candidate)
        required_steps = [boundary_step, *paired_steps]
        candidate_window = _required_ready_window(candidate, required_steps)
        baseline_reasoning_trigger = _observed_reasoning_trigger(
            baseline,
            paired_steps,
            measurement_window=base_window,
            expected_reasoning_trigger=expected_baseline_reasoning_trigger,
        )
        candidate_reasoning_trigger = _observed_reasoning_trigger(
            candidate,
            paired_steps,
            measurement_window=candidate_window,
            expected_reasoning_trigger=expected_candidate_reasoning_trigger,
        )
        baseline_mode = _validate_reward_outcomes(
            baseline,
            paired_steps,
            measurement_window=base_window,
            expected_benchmark_mode=expected_baseline_mode,
            observed_reasoning_trigger=baseline_reasoning_trigger,
        )
        candidate_mode = _validate_reward_outcomes(
            candidate,
            paired_steps,
            measurement_window=candidate_window,
            expected_benchmark_mode=expected_candidate_mode,
            observed_reasoning_trigger=candidate_reasoning_trigger,
        )
        reward_preserving_modes = {"recorded", "accuracy_shadow", "dual_shadow"}
        training_reward_paired = baseline_mode in reward_preserving_modes and candidate_mode in reward_preserving_modes
        if not training_reward_paired and not allow_reward_changing_pair:
            raise ValueError(
                "paired E2E latency requires recorded/shadow modes so policy updates consume the same reward; "
                "use --allow-reward-changing-pair only for an explicitly confounded operational comparison"
            )
        baseline_invariant_hash = _benchmark_invariant_hash(
            baseline,
            paired_steps,
            measurement_window=base_window,
            required=require_benchmark_invariant_hash,
        )
        candidate_invariant_hash = _benchmark_invariant_hash(
            candidate,
            paired_steps,
            measurement_window=candidate_window,
            required=require_benchmark_invariant_hash,
        )
        if baseline_invariant_hash != candidate_invariant_hash:
            raise ValueError(
                "paired variants changed benchmark invariants other than benchmark_mode/reasoning_trigger: "
                f"{baseline_invariant_hash!r} != {candidate_invariant_hash!r}"
            )
        base_signature = _workload_signature(baseline, paired_steps)
        candidate_signature = _workload_signature(candidate, paired_steps)
        missing_workload_steps = {
            "baseline": [step for step in paired_steps if not base_signature.get(step)],
            "candidate": [step for step in paired_steps if not candidate_signature.get(step)],
        }
        if any(missing_workload_steps.values()):
            raise ValueError(f"paired timelines are missing reward workload identities: {missing_workload_steps}")
        missing_context_hashes = [
            (step, identity[:2])
            for signature in (base_signature, candidate_signature)
            for step, identities in signature.items()
            for identity in identities
            if identity[2] is None
        ]
        if missing_context_hashes:
            raise ValueError(
                "paired reward workload identities are missing trajectory_hash "
                f"(or legacy context_hash): {missing_context_hashes[:8]}"
            )
        if baseline_mode in reward_preserving_modes or candidate_mode in reward_preserving_modes:
            missing_recorded_reward_hashes = [
                (step, identity[:2])
                for signature in (base_signature, candidate_signature)
                for step, identities in signature.items()
                for identity in identities
                if identity[3] is None
            ]
            if missing_recorded_reward_hashes:
                raise ValueError(
                    "recorded/shadow pairing requires hashes of the numeric training reward: "
                    f"{missing_recorded_reward_hashes[:8]}"
                )
        duplicate_identities = [
            (step, identity[:2], count)
            for signature in (base_signature, candidate_signature)
            for step, identities in signature.items()
            for identity, count in identities.items()
            if count != 1
        ]
        if duplicate_identities:
            raise ValueError(f"paired reward workload identities are not unique: {duplicate_identities[:8]}")
        _validate_workload_cardinality(
            base_signature,
            paired_steps,
            expected_groups_per_round=expected_groups_per_round,
            expected_samples_per_group=expected_samples_per_group,
        )
        _validate_workload_cardinality(
            candidate_signature,
            paired_steps,
            expected_groups_per_round=expected_groups_per_round,
            expected_samples_per_group=expected_samples_per_group,
        )
        if base_signature and base_signature != candidate_signature:
            raise ValueError("paired reward workload sample counts, identities, or contents differ")
        baseline_overlapping_signature = _overlapping_workload_signature(baseline, base_window)
        candidate_overlapping_signature = _overlapping_workload_signature(candidate, candidate_window)
        overlapping_workload_equal = baseline_overlapping_signature == candidate_overlapping_signature
        if not overlapping_workload_equal and not allow_overlapping_workload_mismatch:
            raise ValueError(
                "reward work overlapping the ready-to-ready window differs; use a fixed admission manifest "
                "or use --allow-overlapping-workload-mismatch only for an exploratory invalid comparison"
            )
        validity_issues = []
        if not training_reward_paired:
            validity_issues.append("training_reward_not_paired")
        if not overlapping_workload_equal:
            validity_issues.append("overlapping_reward_workload_mismatch")
        base_makespan = base_window[1] - base_window[0]
        candidate_makespan = candidate_boundaries[paired_steps[-1]] - candidate_boundaries[boundary_step]
        per_step_deltas = []
        for previous_step, step in zip(required_steps, paired_steps):
            base_interval = base_boundaries[step] - base_boundaries[previous_step]
            candidate_interval = candidate_boundaries[step] - candidate_boundaries[previous_step]
            per_step_deltas.append(candidate_interval - base_interval)
        return {
            "paired_steps": paired_steps,
            "ready_boundary_step": boundary_step,
            "baseline_benchmark_mode": baseline_mode,
            "candidate_benchmark_mode": candidate_mode,
            "baseline_reasoning_trigger": baseline_reasoning_trigger,
            "candidate_reasoning_trigger": candidate_reasoning_trigger,
            "reasoning_trigger_changed": baseline_reasoning_trigger != candidate_reasoning_trigger,
            "benchmark_invariant_hash": baseline_invariant_hash,
            "training_reward_paired": training_reward_paired,
            "valid_for_causal_latency_inference": not validity_issues,
            "validity_issues": validity_issues,
            "overlapping_reward_workload": {
                "equal": overlapping_workload_equal,
                "baseline_parent_count": sum(baseline_overlapping_signature.values()),
                "candidate_parent_count": sum(candidate_overlapping_signature.values()),
                "baseline_only_count": sum(
                    (baseline_overlapping_signature - candidate_overlapping_signature).values()
                ),
                "candidate_only_count": sum(
                    (candidate_overlapping_signature - baseline_overlapping_signature).values()
                ),
            },
            "baseline_makespan_s": base_makespan,
            "candidate_makespan_s": candidate_makespan,
            "global_delta_s": candidate_makespan - base_makespan,
            "global_delta_percent": (
                100.0 * (candidate_makespan - base_makespan) / base_makespan if base_makespan > 0 else 0.0
            ),
            "per_update_interval_delta": _statistics(per_step_deltas),
        }

    def makespans(events: list[dict[str, Any]]) -> dict[int, float]:
        grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            if event["step"] is not None:
                grouped[event["step"]].append(event)
        output = {}
        for step, step_events in grouped.items():
            start = min(event["start_s"] for event in step_events)
            ready = [event["end_s"] for event in step_events if event["name"] == "critical_path.weight_serving_ready"]
            end = max(ready) if ready else max(event["end_s"] for event in step_events)
            output[step] = end - start
        return output

    base = makespans(baseline)
    other = makespans(candidate)
    paired_steps = sorted(set(base) & set(other))[warmup_steps:]
    deltas = [other[step] - base[step] for step in paired_steps]

    def measurement_makespan(events: list[dict[str, Any]]) -> float:
        selected = [event for event in events if event["step"] in paired_steps]
        if not selected:
            return 0.0
        start = min(event["start_s"] for event in selected)
        ready = [event["end_s"] for event in selected if event["name"] == "critical_path.weight_serving_ready"]
        end = max(ready) if ready else max(event["end_s"] for event in selected)
        return end - start

    baseline_makespan = measurement_makespan(baseline)
    candidate_makespan = measurement_makespan(candidate)
    return {
        "paired_steps": paired_steps,
        "baseline_makespan_s": baseline_makespan,
        "candidate_makespan_s": candidate_makespan,
        "global_delta_s": candidate_makespan - baseline_makespan,
        "global_delta_percent": (
            100.0 * (candidate_makespan - baseline_makespan) / baseline_makespan if baseline_makespan > 0 else 0.0
        ),
        "per_step_delta": _statistics(deltas),
    }


def _parse_variant(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("variant must use NAME=PATH")
    path = Path(raw_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"variant path does not exist: {path}")
    return name, path


def _parse_expected_mode(value: str) -> tuple[str, str]:
    name, separator, mode = value.partition("=")
    if not separator or not name or mode not in BENCHMARK_MODE_STATUS:
        raise argparse.ArgumentTypeError(
            f"expected mode must use NAME=MODE where MODE is one of {sorted(BENCHMARK_MODE_STATUS)}"
        )
    return name, mode


def _parse_expected_reasoning_trigger(value: str) -> tuple[str, str]:
    name, separator, trigger = value.partition("=")
    if not separator or not name or trigger not in REASONING_TRIGGERS:
        raise argparse.ArgumentTypeError(
            f"expected reasoning trigger must use NAME=TRIGGER where TRIGGER is one of {sorted(REASONING_TRIGGERS)}"
        )
    return name, trigger


def _reject_duplicate_names(items: list[tuple[str, Any]], *, label: str) -> None:
    counts = Counter(name for name, _value in items)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} names are not allowed: {duplicates}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, type=_parse_variant, help="NAME=PATH")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="analyze each variant independently; never select a baseline or compute paired/exposure deltas",
    )
    parser.add_argument(
        "--gpu-samples",
        action="append",
        default=[],
        type=_parse_variant,
        help="NAME=DIR; optional judge/rollout GPU-sampler JSONL sidecar directory for a --variant NAME",
    )
    parser.add_argument(
        "--ready-markers",
        action="append",
        default=[],
        type=_parse_variant,
        help="NAME=FILE; weight_serving_ready JSONL used to delimit a fixed-K direct window",
    )
    parser.add_argument(
        "--expected-mode",
        action="append",
        default=[],
        type=_parse_expected_mode,
        help="NAME=MODE; inferred when NAME is itself a benchmark mode",
    )
    parser.add_argument(
        "--expected-reasoning-trigger",
        action="append",
        default=[],
        type=_parse_expected_reasoning_trigger,
        help=(
            "NAME=TRIGGER; validate terminal_once or per_turn for a fixed-K variant. "
            "Use this explicitly for the terminal-once vs per-turn A/B."
        ),
    )
    parser.add_argument(
        "--warmup-publication-rounds",
        "--warmup-steps",
        "--warmup-updates",
        dest="warmup_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--measure-publication-rounds",
        "--measure-updates",
        dest="measure_updates",
        type=int,
        help="number of ready-to-ready weight-publication rounds (not optimizer steps)",
    )
    parser.add_argument("--expected-step-stride", type=int, default=1)
    parser.add_argument("--expected-groups-per-round", type=int)
    parser.add_argument("--expected-samples-per-group", type=int)
    parser.add_argument(
        "--expected-optimizer-component",
        action="append",
        choices=("actor", "critic"),
        default=[],
        help="trainer component required in every measured publication round; actor is always required",
    )
    parser.add_argument(
        "--allow-synchronized-multi-host-clock",
        action="store_true",
        help="assert that NTP/PTP skew is acceptable for cross-host wall-clock placement",
    )
    parser.add_argument(
        "--multi-host-clock-max-offset-ms",
        type=float,
        default=None,
        help="Audited upper bound on inter-host wall-clock offset; required for multi-host --direct analysis",
    )
    parser.add_argument(
        "--allow-overlapping-workload-mismatch",
        action="store_true",
        help="emit an exploratory invalid report when future reward work overlapping the fixed-K window differs",
    )
    parser.add_argument(
        "--allow-reward-changing-pair",
        action="store_true",
        help="allow a confounded comparison involving accuracy/dual modes that change the training reward",
    )
    parser.add_argument(
        "--allow-rejected-samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "tolerate up to N terminal_outcome=group_rejected samples in the measured window "
            "(a required judge transiently rejecting one GRPO group) without failing validation; "
            "any other reward-outcome violation still always raises. Default 0 preserves the "
            "original strict behavior"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        _reject_duplicate_names(args.variant, label="variant")
        _reject_duplicate_names(args.expected_mode, label="expected-mode")
        _reject_duplicate_names(args.expected_reasoning_trigger, label="expected-reasoning-trigger")
        _reject_duplicate_names(args.gpu_samples, label="gpu-samples")
        _reject_duplicate_names(args.ready_markers, label="ready-markers")
    except ValueError as exc:
        parser.error(str(exc))
    ready_marker_paths = dict(args.ready_markers)
    loaded: dict[str, tuple[list[dict[str, Any]], str]] = {
        name: load_variant_events(
            path,
            include_rollout_with_timeline=args.direct,
            ready_marker_path=ready_marker_paths.get(name),
        )
        for name, path in args.variant
    }
    unknown_gpu_sample_names = sorted({name for name, _path in args.gpu_samples} - set(loaded))
    if unknown_gpu_sample_names:
        parser.error(f"--gpu-samples names do not match any variant: {unknown_gpu_sample_names}")
    unknown_ready_marker_names = sorted(set(ready_marker_paths) - set(loaded))
    if unknown_ready_marker_names:
        parser.error(f"--ready-markers names do not match any variant: {unknown_ready_marker_names}")
    gpu_samples_loaded: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]] = {
        name: _load_judge_gpu_samples(path) for name, path in args.gpu_samples
    }
    expected_modes = {name: name for name, _path in args.variant if name in BENCHMARK_MODE_STATUS}
    expected_modes.update(dict(args.expected_mode))
    expected_reasoning_triggers = {name: name for name, _path in args.variant if name in REASONING_TRIGGERS}
    expected_reasoning_triggers.update(dict(args.expected_reasoning_trigger))
    unknown_mode_names = sorted(set(expected_modes) - set(loaded))
    if unknown_mode_names:
        parser.error(f"--expected-mode names do not match any variant: {unknown_mode_names}")
    unknown_trigger_names = sorted(set(expected_reasoning_triggers) - set(loaded))
    if unknown_trigger_names:
        parser.error(f"--expected-reasoning-trigger names do not match any variant: {unknown_trigger_names}")
    if args.direct:
        if args.allow_overlapping_workload_mismatch or args.allow_reward_changing_pair:
            parser.error("--direct does not accept paired-comparison override flags")
        report = {
            "analysis_mode": "standalone_direct",
            "variants": {
                name: {
                    "source": source,
                    **analyze_direct_events(
                        events,
                        warmup_steps=args.warmup_steps,
                        measure_updates=args.measure_updates,
                        expected_step_stride=args.expected_step_stride,
                        expected_reasoning_trigger=expected_reasoning_triggers.get(name),
                        expected_groups_per_round=args.expected_groups_per_round,
                        expected_samples_per_group=args.expected_samples_per_group,
                        require_clock_host=True,
                        allow_synchronized_multi_host_clock=args.allow_synchronized_multi_host_clock,
                        multi_host_clock_max_offset_ms=args.multi_host_clock_max_offset_ms,
                        gpu_sample_manifests=gpu_samples_loaded.get(name, ([], [], []))[0],
                        gpu_sample_records=gpu_samples_loaded.get(name, ([], [], []))[1],
                        gpu_sample_load_issues=[
                            *gpu_samples_loaded.get(name, ([], [], []))[2],
                            *(
                                ["no_gpu_sample_records"]
                                if name in dict(args.gpu_samples) and not gpu_samples_loaded.get(name, ([], [], []))[1]
                                else []
                            ),
                        ],
                    ),
                }
                for name, (events, source) in loaded.items()
            },
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output is None:
            sys.stdout.write(rendered + "\n")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return
    if len(loaded) > 1 and args.measure_updates is None:
        parser.error("paired E2E comparison requires --measure-updates")
    if args.measure_updates is not None:
        if args.expected_groups_per_round is None or args.expected_samples_per_group is None:
            parser.error(
                "fixed-K measurement requires --expected-groups-per-round and "
                "--expected-samples-per-group to prove trace completeness"
            )
        non_timeline = [name for name, (_events, source) in loaded.items() if source != "timeline"]
        if non_timeline:
            parser.error(f"fixed-K E2E comparison requires timeline inputs: {non_timeline}")
    if args.measure_updates is not None:
        missing_mode_names = sorted(set(loaded) - set(expected_modes))
        if missing_mode_names:
            parser.error(
                "fixed-K measurement requires an expected benchmark mode for every variant; "
                f"use --expected-mode NAME=MODE for {missing_mode_names}"
            )
        if expected_reasoning_triggers:
            missing_trigger_names = sorted(set(loaded) - set(expected_reasoning_triggers))
            if missing_trigger_names:
                parser.error(
                    "when validating a reasoning trigger pair, provide --expected-reasoning-trigger for every "
                    f"variant; missing {missing_trigger_names}"
                )
    baseline_name = args.variant[0][0]
    baseline_events = loaded[baseline_name][0]
    required_ready_steps = None
    if args.measure_updates is not None:
        _window, measured_steps, boundary_step = _fixed_k_window(
            baseline_events,
            warmup_updates=args.warmup_steps,
            measure_updates=args.measure_updates,
            expected_step_stride=args.expected_step_stride,
        )
        required_ready_steps = [boundary_step, *measured_steps]
    variants_report: dict[str, dict[str, Any]] = {
        name: {
            "source": source,
            **analyze_events(
                events,
                warmup_steps=args.warmup_steps,
                measure_updates=args.measure_updates,
                source=source,
                expected_step_stride=args.expected_step_stride,
                required_ready_steps=required_ready_steps,
                require_complete_coverage=True,
                expected_benchmark_mode=expected_modes.get(name),
                expected_reasoning_trigger=expected_reasoning_triggers.get(name),
                expected_groups_per_round=args.expected_groups_per_round,
                expected_samples_per_group=args.expected_samples_per_group,
                require_clock_host=True,
                allow_synchronized_multi_host_clock=args.allow_synchronized_multi_host_clock,
                require_benchmark_invariant_hash=True,
                require_optimizer_step_ids=True,
                expected_optimizer_components={"actor", *args.expected_optimizer_component},
                gpu_sample_manifests=gpu_samples_loaded.get(name, ([], [], []))[0],
                gpu_sample_records=gpu_samples_loaded.get(name, ([], [], []))[1],
                gpu_sample_load_issues=gpu_samples_loaded.get(name, ([], [], []))[2],
                max_rejected_samples=args.allow_rejected_samples,
            ),
        }
        for name, (events, source) in loaded.items()
    }
    report: dict[str, Any] = {
        "baseline": baseline_name,
        "measurement_unit": "weight_publication_round",
        "variants": variants_report,
        "paired_makespan_delta_vs_baseline": {
            name: _paired_makespan_delta(
                baseline_events,
                events,
                warmup_steps=args.warmup_steps,
                measure_updates=args.measure_updates,
                expected_step_stride=args.expected_step_stride,
                expected_groups_per_round=args.expected_groups_per_round,
                expected_samples_per_group=args.expected_samples_per_group,
                expected_baseline_mode=expected_modes.get(baseline_name),
                expected_candidate_mode=expected_modes.get(name),
                expected_baseline_reasoning_trigger=expected_reasoning_triggers.get(baseline_name),
                expected_candidate_reasoning_trigger=expected_reasoning_triggers.get(name),
                require_benchmark_invariant_hash=True,
                allow_overlapping_workload_mismatch=args.allow_overlapping_workload_mismatch,
                allow_reward_changing_pair=args.allow_reward_changing_pair,
            )
            for name, (events, _source) in loaded.items()
            if name != baseline_name
        },
        "exposure_vs_baseline": {
            name: _exposure_ratio(variants_report[baseline_name], variants_report[name])
            for name in loaded
            if name != baseline_name
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        sys.stdout.write(rendered + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
