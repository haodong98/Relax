# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from examples.agentic_dual_judge.aggregate_latency_replicates import aggregate_reports


def _report(
    delta: float,
    *,
    invariant_hash: str = "invariant",
    training_reward_paired: bool = True,
    overlapping_workload_equal: bool = True,
) -> dict:
    baseline = 10.0
    return {
        "measurement_unit": "weight_publication_round",
        "paired_makespan_delta_vs_baseline": {
            "dual_shadow": {
                "paired_steps": [1, 2],
                "baseline_benchmark_mode": "recorded",
                "candidate_benchmark_mode": "dual_shadow",
                "benchmark_invariant_hash": invariant_hash,
                "training_reward_paired": training_reward_paired,
                "overlapping_reward_workload": {"equal": overlapping_workload_equal},
                "baseline_makespan_s": baseline,
                "candidate_makespan_s": baseline + delta,
                "global_delta_s": delta,
                "global_delta_percent": 100.0 * delta / baseline,
            }
        },
    }


def test_replicate_aggregation_uses_one_delta_per_independent_pair():
    report = aggregate_reports(
        [("pair-a", _report(1.0)), ("pair-b", _report(2.0)), ("pair-c", _report(3.0))],
        resamples=200,
        seed=7,
    )

    assert report["independent_pair_count"] == 3
    assert report["global_delta_s"]["raw"] == [1.0, 2.0, 3.0]
    assert report["global_delta_s"]["mean"] == pytest.approx(2.0)
    assert report["global_delta_s"]["median"] == pytest.approx(2.0)
    assert report["global_delta_s"]["sample_stdev"] == pytest.approx(1.0)
    assert report["inference_unit"] == "independent paired run"
    assert report["valid_for_operational_aggregation"] is True
    assert report["valid_for_causal_latency_inference"] is True


def test_replicate_aggregation_allows_independent_seed_manifests():
    report = aggregate_reports(
        [
            ("pair-a", _report(1.0, invariant_hash="seed-a")),
            ("pair-b", _report(2.0, invariant_hash="seed-b")),
        ],
        resamples=20,
    )

    assert report["benchmark_invariant_hash"] is None
    assert report["benchmark_invariant_hash_by_pair"] == {"pair-a": "seed-a", "pair-b": "seed-b"}


def test_replicate_aggregation_rejects_duplicate_pair_ids():
    with pytest.raises(ValueError, match="duplicate independent pair IDs"):
        aggregate_reports([("same", _report(1.0)), ("same", _report(2.0))], resamples=10)


def test_replicate_aggregation_rejects_missing_candidate_pair():
    with pytest.raises(ValueError, match="has no candidate"):
        aggregate_reports([("pair-a", _report(1.0))], candidate="accuracy_shadow", resamples=10)


@pytest.mark.parametrize(
    ("report", "message"),
    [
        (_report(1.0, training_reward_paired=False), "training_reward_not_paired"),
        (_report(1.0, overlapping_workload_equal=False), "overlapping_reward_workload_mismatch"),
    ],
)
def test_replicate_aggregation_rejects_invalid_pair_reports_by_default(report, message):
    with pytest.raises(ValueError, match=message):
        aggregate_reports([("pair-a", report)], resamples=10)


def test_replicate_aggregation_can_label_exploratory_invalid_reports():
    report = aggregate_reports(
        [
            (
                "pair-a",
                _report(
                    1.0,
                    training_reward_paired=False,
                    overlapping_workload_equal=False,
                ),
            )
        ],
        resamples=10,
        allow_invalid=True,
    )

    assert report["valid_for_causal_latency_inference"] is False
    assert report["invalid_pair_ids"] == ["pair-a"]
    assert report["pairs"][0]["validity_issues"] == [
        "training_reward_not_paired_or_unreported",
        "overlapping_reward_workload_mismatch_or_unreported",
    ]


def test_replicate_aggregation_accepts_low_volume_channel_as_operational_only():
    low_volume_report = _report(1.0)
    low_volume_report["channel"] = "low_volume_ready_marker"
    del low_volume_report["paired_makespan_delta_vs_baseline"]["dual_shadow"]["overlapping_reward_workload"]

    report = aggregate_reports([("pair-a", low_volume_report)], resamples=10)

    assert report["valid_for_operational_aggregation"] is True
    assert report["valid_for_causal_latency_inference"] is False
    assert report["operationally_invalid_pair_ids"] == []
    assert report["causally_invalid_pair_ids"] == ["pair-a"]
    assert report["pairs"][0]["overlapping_reward_workload_status"] == "unobserved"
    assert report["pairs"][0]["validity_issues"] == ["overlapping_reward_workload_unobserved"]
