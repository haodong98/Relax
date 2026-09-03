# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.agentic_dual_judge.analyze_latency import (
    _exposure_ratio,
    _judge_gpu_efficiency_report,
    _load_judge_gpu_samples,
    _stage_occupancy_seconds,
    analyze_events,
)


def _event(name: str, stage: str, start: float, end: float, step: int = 0) -> dict:
    return {"name": name, "stage": stage, "start_s": start, "end_s": end, "step": step}


def _manifest(role: str, *, interval_s: float = 1.0, base_gpu_id: int = 4) -> dict:
    return {
        "schema_version": 1,
        "record_type": "manifest",
        "role": role,
        "clock_host": "node-0",
        "engine_rank": 0,
        "base_gpu_id": base_gpu_id,
        "num_gpus_per_engine": 1,
        "gpu_uuids": ["GPU-a"],
        "model_path": "/models/fake",
        "nvml_enabled": True,
        "scrape_configured": True,
        "interval_s": interval_s,
    }


def _sample(
    role: str,
    ts: float,
    *,
    clock_host: str = "node-0",
    util_percent: float = 50.0,
    running_reqs: float | None = 1.0,
) -> dict:
    record = {
        "schema_version": 1,
        "record_type": "sample",
        "ts": ts,
        "clock_host": clock_host,
        "role": role,
        "engine_rank": 0,
        "gpu": [
            {
                "index": 4,
                "uuid": "GPU-a",
                "util_percent": util_percent,
                "mem_util_percent": 10.0,
                "mem_used_bytes": 1000,
                "mem_total_bytes": 2000,
            }
        ],
        "sglang": None,
    }
    if running_reqs is not None:
        record["sglang"] = {
            "sglang:num_running_reqs": running_reqs,
            "sglang:num_queue_reqs": 0.0,
            "sglang:token_usage": 0.1,
        }
    return record


# ---------------------------------------------------------------------------
# _load_judge_gpu_samples
# ---------------------------------------------------------------------------


def test_load_judge_gpu_samples_splits_manifest_and_sample_records(tmp_path):
    path = tmp_path / "judge_accuracy_node-0_rank0.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_manifest("judge_accuracy")) + "\n")
        fh.write(json.dumps(_sample("judge_accuracy", 1.0)) + "\n")
        fh.write("not json\n")
        fh.write(json.dumps(_sample("judge_accuracy", 2.0)) + "\n")

    manifests, samples, issues = _load_judge_gpu_samples(tmp_path)

    assert len(manifests) == 1
    assert manifests[0]["role"] == "judge_accuracy"
    assert len(samples) == 2
    assert [s["ts"] for s in samples] == [1.0, 2.0]
    assert issues == ["malformed_gpu_sample_lines:1"]


def test_load_judge_gpu_samples_reads_every_jsonl_file_in_directory(tmp_path):
    (tmp_path / "judge_accuracy_node-0_rank0.jsonl").write_text(
        json.dumps(_sample("judge_accuracy", 1.0)) + "\n", encoding="utf-8"
    )
    (tmp_path / "judge_multiturn_vlm_node-0_rank0.jsonl").write_text(
        json.dumps(_sample("judge_multiturn_vlm", 1.0)) + "\n", encoding="utf-8"
    )

    _manifests, samples, issues = _load_judge_gpu_samples(tmp_path)

    assert {s["role"] for s in samples} == {"judge_accuracy", "judge_multiturn_vlm"}
    assert issues == []


def test_load_judge_gpu_samples_requires_a_directory(tmp_path):
    not_a_dir = tmp_path / "missing"
    with pytest.raises(ValueError, match="directory"):
        _load_judge_gpu_samples(not_a_dir)


# ---------------------------------------------------------------------------
# _judge_gpu_efficiency_report
# ---------------------------------------------------------------------------


def test_judge_gpu_efficiency_report_computes_stats_for_samples_in_window():
    manifests = [_manifest("judge_accuracy", interval_s=1.0)]
    samples = [
        _sample("judge_accuracy", 1.0, util_percent=100.0, running_reqs=2.0),
        _sample("judge_accuracy", 2.0, util_percent=0.0, running_reqs=0.0),
    ]

    report, issues = _judge_gpu_efficiency_report(
        manifests, samples, measurement_window=(0.0, 10.0), training_sample_count=64
    )

    assert issues == ["low_gpu_sample_coverage:judge_accuracy"]  # 2 ticks / 10 expected
    entry = report["judge_accuracy"]
    assert entry["sample_tick_count"] == 2
    assert entry["gpu_count"] == 1
    assert entry["nonidle_fraction"] == pytest.approx(0.5)
    assert entry["util_percent"]["mean_percent"] == pytest.approx(50.0)
    assert entry["zero_request_time_fraction"] == pytest.approx(0.5)
    assert entry["running_reqs"]["mean"] == pytest.approx(1.0)
    assert entry["gpu_hours_in_window"] == pytest.approx(1 * 10.0 / 3600.0)
    assert entry["allocated_gpu_seconds_per_training_sample"] == pytest.approx(1 * 10.0 / 64)
    assert entry["sample_coverage_fraction"] == pytest.approx(0.2)  # 2 / (10s / 1s)


def test_judge_gpu_efficiency_report_excludes_samples_outside_window():
    manifests = [_manifest("judge_accuracy")]
    samples = [
        _sample("judge_accuracy", -1.0),  # before window
        _sample("judge_accuracy", 5.0),  # inside window
        _sample("judge_accuracy", 999.0),  # after window
    ]

    report, _issues = _judge_gpu_efficiency_report(
        manifests, samples, measurement_window=(0.0, 10.0), training_sample_count=None
    )

    assert report["judge_accuracy"]["sample_tick_count"] == 1


def test_judge_gpu_efficiency_report_returns_none_without_any_samples():
    report, issues = _judge_gpu_efficiency_report([], [], measurement_window=(0.0, 10.0), training_sample_count=None)
    assert report is None
    assert issues == []


def test_judge_gpu_efficiency_report_flags_no_samples_in_window_distinctly():
    """Samples exist for the run but none overlap this measurement window -- a
    stronger signal than 'the channel was never used' (empty issues)."""
    manifests = [_manifest("judge_accuracy")]
    samples = [_sample("judge_accuracy", 999.0)]

    report, issues = _judge_gpu_efficiency_report(
        manifests, samples, measurement_window=(0.0, 10.0), training_sample_count=None
    )

    assert report is None
    assert issues == ["no_gpu_samples_in_window"]


def test_judge_gpu_efficiency_report_without_manifest_leaves_coverage_none():
    samples = [_sample("judge_accuracy", 1.0)]

    report, issues = _judge_gpu_efficiency_report(
        [], samples, measurement_window=(0.0, 10.0), training_sample_count=None
    )

    assert report["judge_accuracy"]["sample_coverage_fraction"] is None
    assert issues == []  # cannot assess coverage without a declared interval; not a low-coverage claim


def test_judge_gpu_efficiency_report_full_coverage_raises_no_issue():
    manifests = [_manifest("judge_accuracy", interval_s=1.0)]
    samples = [_sample("judge_accuracy", float(t)) for t in range(10)]

    report, issues = _judge_gpu_efficiency_report(
        manifests, samples, measurement_window=(0.0, 10.0), training_sample_count=None
    )

    assert report["judge_accuracy"]["sample_coverage_fraction"] == pytest.approx(1.0)
    assert issues == []


def test_judge_gpu_efficiency_report_separates_roles():
    manifests = [_manifest("judge_accuracy"), _manifest("judge_multiturn_vlm")]
    samples = [
        _sample("judge_accuracy", 1.0, util_percent=10.0),
        _sample("judge_multiturn_vlm", 1.0, util_percent=90.0),
    ]

    report, _issues = _judge_gpu_efficiency_report(
        manifests, samples, measurement_window=(0.0, 10.0), training_sample_count=None
    )

    assert set(report) == {"judge_accuracy", "judge_multiturn_vlm"}
    assert report["judge_accuracy"]["util_percent"]["mean_percent"] == pytest.approx(10.0)
    assert report["judge_multiturn_vlm"]["util_percent"]["mean_percent"] == pytest.approx(90.0)


def test_judge_gpu_efficiency_report_marks_nvml_unavailable_instead_of_zero_utilization():
    manifest = _manifest("judge_accuracy")
    manifest["nvml_enabled"] = False
    sample = _sample("judge_accuracy", 1.0)
    sample["gpu"] = []

    report, issues = _judge_gpu_efficiency_report(
        [manifest], [sample], measurement_window=(0.0, 10.0), training_sample_count=64
    )

    entry = report["judge_accuracy"]
    assert entry["nvml_available"] is False
    assert entry["gpu_sample_count"] == 0
    assert entry["nonidle_fraction"] is None
    assert entry["util_percent"] is None
    assert entry["sample_coverage_fraction"] is None
    assert "nvml_unavailable:judge_accuracy" in issues


# ---------------------------------------------------------------------------
# analyze_events integration: the judge_gpu_efficiency section end-to-end
# ---------------------------------------------------------------------------


def test_analyze_events_includes_judge_gpu_efficiency_for_non_fixed_k_variant():
    events = [_event("critical_path.reward", "reward", 0.0, 5.0, step=0)]
    manifests = [_manifest("judge_accuracy", interval_s=1.0)]
    samples = [_sample("judge_accuracy", 2.0)]

    report = analyze_events(events, gpu_sample_manifests=manifests, gpu_sample_records=samples)

    assert report["judge_gpu_efficiency"]["judge_accuracy"]["sample_tick_count"] == 1
    assert report["judge_gpu_efficiency_issues"] == ["low_gpu_sample_coverage:judge_accuracy"]


def test_analyze_events_judge_gpu_efficiency_is_none_without_gpu_samples():
    events = [_event("critical_path.reward", "reward", 0.0, 5.0, step=0)]

    report = analyze_events(events)

    assert report["judge_gpu_efficiency"] is None
    assert report["judge_gpu_efficiency_issues"] == []


def test_analyze_events_preserves_gpu_sidecar_parse_issues():
    events = [_event("critical_path.reward", "reward", 0.0, 5.0, step=0)]

    report = analyze_events(events, gpu_sample_load_issues=["malformed_gpu_sample_lines:2"])

    assert report["judge_gpu_efficiency_issues"] == ["malformed_gpu_sample_lines:2"]


def test_rollout_only_analysis_preserves_gpu_sidecar_parse_issues():
    events = [_event("critical_path.reward", "reward", 0.0, 5.0, step=0)]

    report = analyze_events(
        events,
        source="rollout",
        gpu_sample_load_issues=["malformed_gpu_sample_lines:2"],
    )

    assert report["judge_gpu_efficiency_issues"] == ["malformed_gpu_sample_lines:2"]


# ---------------------------------------------------------------------------
# _stage_occupancy_seconds / _exposure_ratio
# ---------------------------------------------------------------------------


def test_stage_occupancy_seconds_converts_percent_to_absolute_seconds():
    report = {"inclusive_occupancy_percent": {"data_wait": 10.0, "reward": 5.0}, "total_window_s": 100.0}
    assert _stage_occupancy_seconds(report, "data_wait") == pytest.approx(10.0)
    assert _stage_occupancy_seconds(report, "reward") == pytest.approx(5.0)


def test_stage_occupancy_seconds_returns_none_when_fields_missing():
    assert _stage_occupancy_seconds({}, "data_wait") is None
    assert _stage_occupancy_seconds({"inclusive_occupancy_percent": {}}, "data_wait") is None


def test_exposure_ratio_computes_delta_and_ratio():
    baseline = {"inclusive_occupancy_percent": {"data_wait": 10.0, "reward": 5.0}, "total_window_s": 100.0}
    candidate = {"inclusive_occupancy_percent": {"data_wait": 15.0, "reward": 20.0}, "total_window_s": 100.0}

    entry = _exposure_ratio(baseline, candidate)

    assert entry["delta_data_wait_s"] == pytest.approx(5.0)
    assert entry["delta_reward_occupancy_s"] == pytest.approx(15.0)
    assert entry["exposure_ratio"] == pytest.approx(5.0 / 15.0)


def test_exposure_ratio_near_zero_means_growth_was_hidden():
    """Reward occupancy grew a lot but data_wait barely moved -- the async
    pipeline absorbed it; exposure_ratio should be small."""
    baseline = {"inclusive_occupancy_percent": {"data_wait": 10.0, "reward": 5.0}, "total_window_s": 100.0}
    candidate = {"inclusive_occupancy_percent": {"data_wait": 10.5, "reward": 25.0}, "total_window_s": 100.0}

    entry = _exposure_ratio(baseline, candidate)

    assert entry["exposure_ratio"] == pytest.approx(0.5 / 20.0)
    assert entry["exposure_ratio"] < 0.1


def test_exposure_ratio_is_none_when_reward_delta_is_zero():
    baseline = {"inclusive_occupancy_percent": {"data_wait": 10.0, "reward": 5.0}, "total_window_s": 100.0}
    candidate = {"inclusive_occupancy_percent": {"data_wait": 15.0, "reward": 5.0}, "total_window_s": 100.0}

    entry = _exposure_ratio(baseline, candidate)

    assert entry["delta_reward_occupancy_s"] == pytest.approx(0.0)
    assert entry["exposure_ratio"] is None


def test_exposure_ratio_handles_missing_fields_gracefully():
    entry = _exposure_ratio({}, {"inclusive_occupancy_percent": {"data_wait": 1.0}, "total_window_s": 10.0})
    assert entry == {
        "delta_data_wait_s": None,
        "delta_reward_occupancy_s": None,
        "delta_turn_judge_occupancy_s": None,
        "delta_total_judge_occupancy_s": None,
        "exposure_ratio": None,
    }
