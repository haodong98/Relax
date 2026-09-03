# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.agentic.profile import agentic_span_clock_host, mark_agentic_event, merge_agentic_trace


def test_merge_agentic_trace_recursively_preserves_reward_judge_fields():
    merged = merge_agentic_trace(
        {
            "reward": {
                "pipeline_status": "running",
                "judges": {"answer_accuracy": {"queue_elapsed_s": 1.0, "status": "running"}},
            }
        },
        {
            "reward": {
                "pipeline_status": "success",
                "judges": {"answer_accuracy": {"status": "success", "http_elapsed_s": 2.0}},
            }
        },
    )

    assert merged["reward"]["pipeline_status"] == "success"
    assert merged["reward"]["judges"]["answer_accuracy"] == {
        "queue_elapsed_s": 1.0,
        "status": "success",
        "http_elapsed_s": 2.0,
    }


def test_agentic_event_endpoints_record_their_actual_clock_host():
    events = {}
    mark_agentic_event(events, "start")
    mark_agentic_event(events, "end")

    assert agentic_span_clock_host(events, "start", "end")
    events["end__clock_host"] = "different-host"
    assert agentic_span_clock_host(events, "start", "end") is None
