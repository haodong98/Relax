# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Aggregate independent paired latency runs without treating updates as
replicates."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def _parse_report(value: str) -> tuple[str, Path]:
    pair_id, separator, raw_path = value.partition("=")
    if not separator or not pair_id or not raw_path:
        raise argparse.ArgumentTypeError("report must use PAIR_ID=PATH")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"report does not exist: {path}")
    return pair_id, path


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile from no values")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = random.Random(seed)
    draws = [statistic([values[rng.randrange(len(values))] for _ in values]) for _ in range(resamples)]
    tail = (1.0 - confidence) * 50.0
    return {
        "confidence": confidence,
        "low": _percentile(draws, tail),
        "high": _percentile(draws, 100.0 - tail),
        "resamples": float(resamples),
    }


def aggregate_reports(
    reports: list[tuple[str, dict[str, Any]]],
    *,
    candidate: str | None = None,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    allow_invalid: bool = False,
) -> dict[str, Any]:
    pair_counts = Counter(pair_id for pair_id, _report in reports)
    duplicates = sorted(pair_id for pair_id, count in pair_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate independent pair IDs are not allowed: {duplicates}")
    if not reports:
        raise ValueError("at least one independent paired report is required")

    rows = []
    selected_candidate = candidate
    for pair_id, report in reports:
        if report.get("measurement_unit") != "weight_publication_round":
            raise ValueError(f"report {pair_id!r} is not a weight-publication-round analysis")
        comparisons = report.get("paired_makespan_delta_vs_baseline")
        if not isinstance(comparisons, dict) or not comparisons:
            raise ValueError(f"report {pair_id!r} has no paired makespan comparison")
        if selected_candidate is None:
            if len(comparisons) != 1:
                raise ValueError("--candidate is required when a report contains multiple candidates")
            selected_candidate = next(iter(comparisons))
        comparison = comparisons.get(selected_candidate)
        if not isinstance(comparison, dict):
            raise ValueError(f"report {pair_id!r} has no candidate {selected_candidate!r}")
        channel = report.get("channel", "detailed_timeline")
        training_reward_paired = comparison.get("training_reward_paired") is True
        overlapping_workload = comparison.get("overlapping_reward_workload")
        overlap_is_reported = isinstance(overlapping_workload, dict) and isinstance(
            overlapping_workload.get("equal"), bool
        )
        reported_overlapping_workload_equal = overlapping_workload.get("equal") if overlap_is_reported else None
        overlapping_workload_equal = reported_overlapping_workload_equal if channel == "detailed_timeline" else None
        blocking_invalid_reasons = []
        if not training_reward_paired:
            blocking_invalid_reasons.append("training_reward_not_paired_or_unreported")
        if channel == "detailed_timeline" and overlapping_workload_equal is not True:
            blocking_invalid_reasons.append("overlapping_reward_workload_mismatch_or_unreported")
        elif reported_overlapping_workload_equal is False:
            blocking_invalid_reasons.append("overlapping_reward_workload_mismatch")
        causal_validity_issues = list(blocking_invalid_reasons)
        if overlapping_workload_equal is None:
            causal_validity_issues.append("overlapping_reward_workload_unobserved")
        if blocking_invalid_reasons and not allow_invalid:
            raise ValueError(
                f"report {pair_id!r} is invalid for replicate aggregation: {blocking_invalid_reasons}; "
                "use --allow-invalid only for exploratory summaries"
            )
        rows.append(
            {
                "pair_id": pair_id,
                "channel": channel,
                "baseline_mode": comparison.get("baseline_benchmark_mode"),
                "candidate_mode": comparison.get("candidate_benchmark_mode"),
                "benchmark_invariant_hash": comparison.get("benchmark_invariant_hash"),
                "training_reward_paired": training_reward_paired,
                "reported_overlapping_reward_workload_equal": reported_overlapping_workload_equal,
                "overlapping_reward_workload_equal": overlapping_workload_equal,
                "overlapping_reward_workload_status": (
                    "equal"
                    if overlapping_workload_equal is True
                    else "mismatch"
                    if overlapping_workload_equal is False
                    else "unobserved"
                ),
                "valid_for_operational_aggregation": not blocking_invalid_reasons,
                "valid_for_causal_latency_inference": not causal_validity_issues,
                "validity_issues": causal_validity_issues,
                "publication_round_count": len(comparison.get("paired_steps", [])),
                "baseline_makespan_s": float(comparison["baseline_makespan_s"]),
                "candidate_makespan_s": float(comparison["candidate_makespan_s"]),
                "global_delta_s": float(comparison["global_delta_s"]),
                "global_delta_percent": float(comparison["global_delta_percent"]),
            }
        )

    mode_pairs = {(row["baseline_mode"], row["candidate_mode"]) for row in rows}
    if len(mode_pairs) != 1:
        raise ValueError(f"replicates compare different benchmark modes: {sorted(mode_pairs, key=str)}")
    channels = {row["channel"] for row in rows}
    if len(channels) != 1:
        raise ValueError(f"replicates mix measurement channels: {sorted(channels)}")
    if not all(isinstance(row["benchmark_invariant_hash"], str) and row["benchmark_invariant_hash"] for row in rows):
        raise ValueError("every paired replicate requires a nonempty benchmark invariant hash")
    invariant_hashes = {row["benchmark_invariant_hash"] for row in rows}
    publication_round_counts = {row["publication_round_count"] for row in rows}
    if len(publication_round_counts) != 1 or 0 in publication_round_counts:
        raise ValueError(f"replicates use different or missing publication-round counts: {publication_round_counts}")
    deltas = [row["global_delta_s"] for row in rows]
    delta_percentages = [row["global_delta_percent"] for row in rows]

    def mean(values: list[float]) -> float:
        return statistics.fmean(values)

    def median(values: list[float]) -> float:
        return statistics.median(values)

    return {
        "candidate": selected_candidate,
        "baseline_benchmark_mode": rows[0]["baseline_mode"],
        "candidate_benchmark_mode": rows[0]["candidate_mode"],
        "benchmark_invariant_hash": next(iter(invariant_hashes)) if len(invariant_hashes) == 1 else None,
        "benchmark_invariant_hash_by_pair": {row["pair_id"]: row["benchmark_invariant_hash"] for row in rows},
        "channel": rows[0]["channel"],
        "publication_round_count_per_pair": rows[0]["publication_round_count"],
        "independent_pair_count": len(rows),
        "valid_for_operational_aggregation": all(row["valid_for_operational_aggregation"] for row in rows),
        "valid_for_causal_latency_inference": all(row["valid_for_causal_latency_inference"] for row in rows),
        "operationally_invalid_pair_ids": [
            row["pair_id"] for row in rows if not row["valid_for_operational_aggregation"]
        ],
        "causally_invalid_pair_ids": [row["pair_id"] for row in rows if not row["valid_for_causal_latency_inference"]],
        "invalid_pair_ids": [row["pair_id"] for row in rows if not row["valid_for_causal_latency_inference"]],
        "pairs": rows,
        "global_delta_s": {
            "raw": deltas,
            "mean": mean(deltas),
            "median": median(deltas),
            "sample_stdev": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "mean_bootstrap_ci": _bootstrap_interval(
                deltas, mean, confidence=confidence, resamples=resamples, seed=seed
            ),
            "median_bootstrap_ci": _bootstrap_interval(
                deltas, median, confidence=confidence, resamples=resamples, seed=seed + 1
            ),
        },
        "global_delta_percent": {
            "raw": delta_percentages,
            "mean": mean(delta_percentages),
            "median": median(delta_percentages),
            "sample_stdev": statistics.stdev(delta_percentages) if len(delta_percentages) > 1 else None,
        },
        "inference_unit": "independent paired run",
        "warning": (
            "Publication-round observations within one run are autocorrelated and are not used as independent "
            "replicates. With fewer than three pairs, bootstrap intervals are descriptive only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=_parse_report)
    parser.add_argument("--candidate")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="aggregate reward-changing or overlapping-workload-mismatched reports for exploration only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payloads = [(pair_id, json.loads(path.read_text(encoding="utf-8"))) for pair_id, path in args.report]
    try:
        report = aggregate_reports(
            payloads,
            candidate=args.candidate,
            confidence=args.confidence,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
            allow_invalid=args.allow_invalid,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        sys.stdout.write(rendered + "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
