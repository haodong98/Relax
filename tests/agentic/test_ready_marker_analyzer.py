# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.agentic_dual_judge.analyze_ready_markers import _trace_calibration_evidence, analyze_markers


def _markers(*timestamps: int, timeline_enabled: bool = False) -> list[dict]:
    ready = [
        {
            "schema_version": 1,
            "event": "weight_serving_ready",
            "step": step,
            "monotonic_ns": timestamp,
            "clock_host": "host",
            "pid": 7,
            "benchmark_mode": "dual_shadow",
            "benchmark_invariant_hash": "invariant",
            "timeline_enabled": timeline_enabled,
        }
        for step, timestamp in enumerate(timestamps)
    ]
    workloads = [
        {
            "schema_version": 1,
            "event": "reward_workload_fragment",
            "step": step,
            "collection_rollout_id": step,
            "identity_count": 1,
            "identity_digest_sum": f"{step + 1:064x}",
            "identity_digest_xor": f"{step + 1:064x}",
            "invalid_identity_count": 0,
            "invalid_recorded_reward_hash_count": 0,
            "group_sample_counts": [[0, 1]],
            "reward_outcome_counts": {'["success","success",null]': 1},
            "benchmark_mode_counts": {"dual_shadow": 1},
            "judge_branch_signature_counts": {'[["answer_accuracy","success"],["multi_turn_reasoning","success"]]': 1},
            "terminal_snapshot_count": 0,
            "benchmark_invariant_hash": "invariant",
            "clock_host": "rollout-host",
            "pid": 9,
        }
        for step, _timestamp in enumerate(timestamps)
    ]
    return [*ready, *workloads]


def test_ready_marker_analyzer_uses_monotonic_fixed_k_window():
    report = analyze_markers(
        _markers(1_000_000_000, 3_000_000_000, 6_000_000_000),
        warmup_rounds=1,
        measure_rounds=2,
        expected_step_stride=1,
        expected_mode="dual_shadow",
        expected_groups_per_round=1,
        expected_samples_per_group=1,
    )

    assert report["fixed_k_ready_to_ready_makespan_s"] == pytest.approx(5.0)
    assert report["per_publication_round_interval"]["p50_s"] == pytest.approx(2.5)


def test_ready_marker_analyzer_rejects_process_restart():
    markers = _markers(1_000_000_000, 2_000_000_000)
    markers[1]["pid"] = 8

    with pytest.raises(ValueError, match="cross actor process domains"):
        analyze_markers(
            markers,
            warmup_rounds=1,
            measure_rounds=1,
            expected_step_stride=1,
            expected_groups_per_round=1,
            expected_samples_per_group=1,
        )


def test_ready_marker_analyzer_rejects_restart_during_discarded_warmup():
    markers = _markers(1_000_000_000, 2_000_000_000, 3_000_000_000)
    markers[0]["pid"] = 8

    with pytest.raises(ValueError, match="cross actor process domains"):
        analyze_markers(
            markers,
            warmup_rounds=2,
            measure_rounds=1,
            expected_step_stride=1,
            expected_groups_per_round=1,
            expected_samples_per_group=1,
        )


def test_ready_marker_analyzer_rejects_duplicate_step():
    markers = _markers(1_000_000_000, 2_000_000_000)
    markers[1]["step"] = 0

    with pytest.raises(ValueError, match="duplicate ready marker step"):
        analyze_markers(
            markers,
            warmup_rounds=1,
            measure_rounds=1,
            expected_step_stride=1,
            expected_groups_per_round=1,
            expected_samples_per_group=1,
        )


def test_ready_marker_analyzer_rejects_terminal_reward_failure():
    markers = _markers(1_000_000_000, 2_000_000_000)
    workload = next(
        record for record in markers if record["event"] == "reward_workload_fragment" and record["step"] == 1
    )
    workload["terminal_snapshot_count"] = 1

    with pytest.raises(ValueError, match="terminal reward failures"):
        analyze_markers(
            markers,
            warmup_rounds=1,
            measure_rounds=1,
            expected_step_stride=1,
            expected_groups_per_round=1,
            expected_samples_per_group=1,
        )


def test_ready_marker_analyzer_rejects_terminal_failure_outside_measured_window():
    markers = _markers(1_000_000_000, 2_000_000_000, 3_000_000_000)
    workload = next(
        record for record in markers if record["event"] == "reward_workload_fragment" and record["step"] == 0
    )
    workload["terminal_snapshot_count"] = 1

    with pytest.raises(ValueError, match="outside or inside the measured window"):
        analyze_markers(
            markers,
            warmup_rounds=2,
            measure_rounds=1,
            expected_step_stride=1,
            expected_groups_per_round=1,
            expected_samples_per_group=1,
        )


def _variant_report(*, mode: str = "dual_shadow", timeline_enabled: bool) -> dict:
    return {
        "benchmark_mode": mode,
        "timeline_enabled": timeline_enabled,
        "ready_boundary_step": 1,
        "steps": [2, 3],
    }


def _write_timeline(path, *steps: int) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": "critical_path.weight_serving_ready",
                    "ph": "X",
                    "ts": step * 1_000_000,
                    "dur": 1_000,
                    "pid": 1,
                    "tid": 1,
                    "args": {"step": step},
                }
                for step in steps
            ]
        ),
        encoding="utf-8",
    )


def test_trace_calibration_requires_same_benchmark_mode(tmp_path):
    timeline_path = tmp_path / "timeline_step_3.json"
    _write_timeline(timeline_path, 1, 2, 3)

    with pytest.raises(ValueError, match="same benchmark mode"):
        _trace_calibration_evidence(
            {
                "trace_off": _variant_report(timeline_enabled=False),
                "trace_on": _variant_report(mode="accuracy_shadow", timeline_enabled=True),
            },
            baseline_name="trace_off",
            timeline_paths={"trace_on": timeline_path},
        )


def test_trace_calibration_requires_real_ready_events_and_reports_evidence(tmp_path):
    incomplete_path = tmp_path / "incomplete" / "timeline_step_2.json"
    incomplete_path.parent.mkdir()
    _write_timeline(incomplete_path, 1, 2)
    variants = {
        "trace_off": _variant_report(timeline_enabled=False),
        "trace_on": _variant_report(timeline_enabled=True),
    }

    with pytest.raises(ValueError, match="selected steps.*3"):
        _trace_calibration_evidence(
            variants,
            baseline_name="trace_off",
            timeline_paths={"trace_on": incomplete_path.parent},
        )

    timeline_dir = tmp_path / "complete"
    timeline_dir.mkdir()
    _write_timeline(timeline_dir / "timeline_step_3.json", 1, 2, 3)
    evidence = _trace_calibration_evidence(
        variants,
        baseline_name="trace_off",
        timeline_paths={"trace_on": timeline_dir},
    )

    assert evidence["required_ready_steps"] == [1, 2, 3]
    assert evidence["serving_ready_event_count_by_step"] == {"1": 1, "2": 1, "3": 1}
    assert evidence["overhead_scope"] == "timeline_trace_export_aggregation_and_dump"
