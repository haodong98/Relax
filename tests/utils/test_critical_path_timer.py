# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.timer import TimelineEvent, Timer, span_timer, timeline_step


def test_span_timer_allows_nested_repeated_names_and_uses_complete_events():
    timer = Timer()
    timer.reset()
    timer.records.clear()

    with span_timer("critical_path.data_wait"):
        with span_timer("critical_path.data_wait"):
            pass

    assert timer.log_dict()["critical_path.data_wait"] >= 0
    assert [record.name for record in timer.records] == [
        "critical_path.data_wait",
        "critical_path.data_wait",
    ]


def test_timeline_event_can_place_by_wall_clock_and_measure_by_monotonic_duration():
    event = TimelineEvent(
        name="critical_path.reward",
        start_ts=10.0,
        end_ts=9.0,
        duration_s=0.25,
        attributes={"optimizer_step_id": 2},
    )

    assert event.to_trace_event()["dur"] == 250_000
    assert event.to_trace_event()["args"]["clock_host"]
    assert event.to_trace_event()["args"]["optimizer_step_id"] == 2


def test_critical_spans_keep_bound_step_across_intermediate_drains():
    timer = Timer()
    timer.records.clear()

    with timeline_step(7):
        with span_timer("critical_path.training"):
            pass
        with timer.context("ordinary"):
            pass

    ordinary = timer.log_record_and_clear(step=99, include_critical_path=False)
    critical = timer.log_record_and_clear(step=99, critical_path_only=True)

    assert [record.step for record in ordinary] == [7]
    assert [record.step for record in critical] == [7]
