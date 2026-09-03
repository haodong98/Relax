# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.agentic.rollout import (
    _aggregate_rollout_timing_from_agentic_trace,
    _build_agentic_critical_path_timeline_events,
)
from relax.utils.types import Sample


def _sample(*, index: int, arrive: float, start: float, end: float) -> Sample:
    sample = Sample(
        session_id=f"session-{index}",
        group_index=4,
        index=index,
        metadata={
            "agentic_trace": {
                "events": {
                    "finalize_start_at": 99.0,
                    "finalize_end_at": 99.5,
                    "reward_context_build_start_at": 99.1,
                    "reward_context_build_end_at": 99.4,
                    "reward_arrive_at": arrive,
                    "reward_start_at": start,
                    "reward_end_at": end,
                    "group_reward_ready_at": 107.0,
                    "group_finalize_start_at": 106.0,
                    "group_finalize_end_at": 107.0,
                    "transfer_buffer_enter_at": 107.0,
                    "transfer_release_start_at": 108.0,
                    "transfer_release_end_at": 109.0,
                },
                "turns": [],
                "reward": {
                    "schema_version": 1,
                    "sample_key": [f"session-{index}", index, None],
                    "group_key": [["session-0", 0, None], ["session-1", 1, None]],
                    "context_hash": f"context-{index}",
                    "benchmark_mode": "dual_shadow",
                    "pipeline_status": "success",
                    "executor_status": "success",
                    "group_index": 4,
                    "sample_index": index,
                    "expected_trainer_components": ["actor", "critic"],
                    "global_queue_elapsed_s": 1.0,
                    "pipeline_elapsed_s": end - start,
                    "executor_elapsed_s": end - start - 0.1,
                    "answer_accuracy_projection_elapsed_s": 0.01,
                    "multi_turn_reasoning_projection_elapsed_s": 0.02,
                    "judges": {
                        "answer_accuracy": {
                            "status": "success",
                            "queue_elapsed_s": 0.2,
                            "payload_prep_elapsed_s": 0.04,
                            "http_elapsed_s": 2.0,
                            "parse_elapsed_s": 0.01,
                            "backoff_elapsed_s": 0.0,
                            "elapsed_s": 2.21,
                            "attempt_count": 1,
                            "invalid_response_count": 0,
                        },
                        "multi_turn_reasoning": {
                            "status": "success",
                            "queue_elapsed_s": 0.3,
                            "payload_prep_elapsed_s": 0.08,
                            "http_elapsed_s": 3.0,
                            "parse_elapsed_s": 0.02,
                            "backoff_elapsed_s": 0.0,
                            "elapsed_s": 3.32,
                            "attempt_count": 1,
                            "invalid_response_count": 0,
                        },
                    },
                },
            }
        },
    )
    events = sample.metadata["agentic_trace"]["events"]
    for key in list(events):
        events[f"{key}__clock_host"] = "host-a"
    return sample


def test_reward_latency_metrics_include_percentiles_and_group_tail() -> None:
    metrics = _aggregate_rollout_timing_from_agentic_trace(
        samples=[
            _sample(index=0, arrive=100.0, start=101.0, end=105.0),
            _sample(index=1, arrive=100.5, start=101.5, end=106.0),
        ],
        get_samples_times=[],
    )

    assert metrics["perf_detail/reward/global_queue_time/p95"] == pytest.approx(1.0)
    assert metrics["perf_detail/reward/sample_service_time/p50"] == pytest.approx(4.25)
    assert metrics["perf_detail/reward/group_sojourn_time/p50"] == pytest.approx(6.0)
    assert metrics["perf_detail/reward/group_completion_spread_time/p50"] == pytest.approx(1.0)
    assert metrics["perf_detail/reward/group_finalize_delay_time/p50"] == pytest.approx(1.0)
    assert metrics["perf_detail/transfer/reward_ready_to_release_time/p50"] == pytest.approx(1.0)
    assert metrics["perf_detail/reward/answer_accuracy/http_time/p99"] == pytest.approx(2.0)
    assert metrics["perf_detail/reward/multi_turn_reasoning/payload_prep_time/p95"] == pytest.approx(0.08)
    assert metrics["perf_detail/reward/multi_turn_reasoning/queue_time/p90"] == pytest.approx(0.3)
    assert metrics["perf_detail/reward/answer_accuracy/status/success_count"] == 2.0


def test_rejected_trace_snapshots_are_included_without_duplicate_committed_samples() -> None:
    committed = _sample(index=0, arrive=100.0, start=101.0, end=105.0)
    duplicate = committed.metadata.copy()
    rejected = _sample(index=1, arrive=110.0, start=112.0, end=120.0).metadata
    rejected_reward = rejected["agentic_trace"]["reward"]
    rejected_reward["pipeline_status"] = "error"
    rejected_reward["terminal_outcome"] = "group_rejected"
    rejected_reward["judges"]["answer_accuracy"]["status"] = "rejected"

    metrics = _aggregate_rollout_timing_from_agentic_trace(
        samples=[committed],
        get_samples_times=[],
        reward_trace_snapshots=[duplicate, rejected],
    )

    assert metrics["perf_detail/reward/sample_service_time/mean"] == pytest.approx(6.0)
    assert metrics["perf_detail/reward/terminal/status/group_rejected_count"] == 1.0
    assert metrics["perf_detail/reward/answer_accuracy/status/success_count"] == 1.0
    assert metrics["perf_detail/reward/answer_accuracy/status/rejected_count"] == 1.0


def test_agentic_timeline_contains_generation_reward_and_transfer_intervals() -> None:
    sample = _sample(index=0, arrive=100.0, start=101.0, end=105.0)
    sample.metadata["agentic_trace"]["turns"] = [
        {
            "events": {
                "chat_request_arrive_at": 88.0,
                "generation_queue_enter_at": 89.0,
                "generation_start_at": 90.0,
                "generation_end_at": 99.0,
                "chat_end_at": 99.5,
            }
        }
    ]
    turn_events = sample.metadata["agentic_trace"]["turns"][0]["events"]
    for key in list(turn_events):
        turn_events[f"{key}__clock_host"] = "host-a"

    events = _build_agentic_critical_path_timeline_events(rollout_id=3, samples=[sample])

    by_name = {event["name"]: event for event in events}
    assert by_name["critical_path.rollout_generation"]["dur"] == 9_000_000
    assert by_name["critical_path.rollout_pre_generation"]["dur"] == 1_000_000
    assert by_name["critical_path.rollout_post_generation"]["dur"] == 500_000
    assert by_name["critical_path.rollout_queue"]["dur"] == 1_000_000
    assert by_name["critical_path.rollout_finalize"]["dur"] == 500_000
    assert by_name["critical_path.reward_context_build"]["dur"] == 300_000
    assert by_name["critical_path.reward"]["dur"] == 5_000_000
    assert by_name["critical_path.reward_group_finalize"]["dur"] == 1_000_000
    assert by_name["critical_path.transfer_buffer_wait"]["dur"] == 1_000_000
    assert by_name["critical_path.transfer"]["dur"] == 1_000_000
    assert all(event["args"]["step"] == 3 for event in events)
    assert all(event["args"]["context_hash"] == "context-0" for event in events)
    assert all(event["args"]["expected_trainer_components"] == ["actor", "critic"] for event in events)
    assert all(event["args"]["clock_host"] == "host-a" for event in events)


def test_agentic_timeline_uses_actual_train_partition_step() -> None:
    sample = _sample(index=0, arrive=100.0, start=101.0, end=105.0)
    sample.metadata["agentic_trace"]["reward"]["train_partition_step"] = 7

    events = _build_agentic_critical_path_timeline_events(rollout_id=9, samples=[sample])

    assert events
    assert all(event["args"]["step"] == 7 for event in events)


def test_per_turn_judge_has_an_independent_timeline_span_and_metrics() -> None:
    sample = _sample(index=0, arrive=100.0, start=101.0, end=105.0)
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    reward_trace["reasoning_trigger"] = "per_turn"
    reward_trace["reasoning_execution_trigger"] = "per_turn"
    reward_trace["per_turn_judge_count"] = 1
    reward_trace["judges"] = {
        "answer_accuracy": reward_trace["judges"]["answer_accuracy"],
    }
    judge_events = {
        "turn_judge_trigger_at": 91.0,
        "turn_judge_projection_start_at": 91.1,
        "turn_judge_projection_end_at": 91.3,
        "turn_judge_end_at": 94.0,
    }
    for key in list(judge_events):
        judge_events[f"{key}__clock_host"] = "host-a"
    sample.metadata["agentic_trace"]["turns"] = [
        {
            "events": {},
            "judge": {
                "status": "success",
                "role": "judge_multiturn_vlm",
                "response_state_hash": "response-0",
                "observation_state_hash": "observation-0",
                "projection_elapsed_s": 0.2,
                "events": judge_events,
                "judge": {
                    "status": "success",
                    "queue_elapsed_s": 0.4,
                    "http_elapsed_s": 2.0,
                    "elapsed_s": 2.5,
                    "server": {"server_queue_elapsed_s": 0.3, "request_total_elapsed_s": 1.7},
                },
            },
        }
    ]

    events = _build_agentic_critical_path_timeline_events(rollout_id=3, samples=[sample])
    by_name = {event["name"]: event for event in events}
    assert by_name["critical_path.turn_judge"]["dur"] == 3_000_000
    assert by_name["critical_path.turn_judge"]["args"]["judge_status"] == "success"
    assert by_name["critical_path.turn_judge"]["args"]["reasoning_trigger"] == "per_turn"
    assert "critical_path.reward.multi_turn_reasoning" not in by_name

    metrics = _aggregate_rollout_timing_from_agentic_trace(samples=[sample], get_samples_times=[])

    assert metrics["perf_detail/turn_judge/sidecar_elapsed_time/p50"] == pytest.approx(3.0)
    assert metrics["perf_detail/turn_judge/projection_time/p50"] == pytest.approx(0.2)
    assert metrics["perf_detail/turn_judge/multi_turn_reasoning/queue_time/p50"] == pytest.approx(0.4)
    assert metrics["perf_detail/turn_judge/multi_turn_reasoning/server/server_queue_elapsed_s/p50"] == pytest.approx(
        0.3
    )
    assert metrics["perf_detail/turn_judge/status/success_count"] == 1.0
