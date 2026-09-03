# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare low-volume weight-serving-ready markers with detailed tracing on or
off."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


try:
    from .analyze_latency import (
        BENCHMARK_MODE_BRANCHES,
        BENCHMARK_MODE_STATUS,
        _parse_expected_mode,
        _reject_duplicate_names,
        _statistics,
        load_variant_events,
    )
except ImportError:
    from analyze_latency import (  # type: ignore[no-redef]
        BENCHMARK_MODE_BRANCHES,
        BENCHMARK_MODE_STATUS,
        _parse_expected_mode,
        _reject_duplicate_names,
        _statistics,
        load_variant_events,
    )


_DIGEST_MODULUS = 1 << 256


def _parse_variant(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("variant must use NAME=PATH")
    path = Path(raw_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"variant path does not exist: {path}")
    return name, path


def _resolve_marker_files(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        ready_file = path
    else:
        direct_candidates = (
            path / "weight_serving_ready.jsonl",
            path / "critical_path" / "weight_serving_ready.jsonl",
        )
        ready_file = next((candidate for candidate in direct_candidates if candidate.is_file()), None)
        if ready_file is None:
            matches = list(path.rglob("weight_serving_ready.jsonl"))
            if len(matches) != 1:
                raise ValueError(f"expected exactly one weight_serving_ready.jsonl under {path}, found {len(matches)}")
            ready_file = matches[0]
    workload_file = ready_file.with_name("reward_workload.jsonl")
    if not workload_file.is_file():
        raise ValueError(f"missing workload marker companion: {workload_file}")
    return ready_file, workload_file


def load_markers(path: Path) -> list[dict[str, Any]]:
    records = []
    ready_file, workload_file = _resolve_marker_files(path)
    required_by_event = {
        "weight_serving_ready": {
            "step",
            "monotonic_ns",
            "clock_host",
            "pid",
            "benchmark_mode",
            "benchmark_invariant_hash",
            "timeline_enabled",
        },
        "reward_workload_fragment": {
            "step",
            "collection_rollout_id",
            "identity_count",
            "identity_digest_sum",
            "identity_digest_xor",
            "invalid_identity_count",
            "invalid_recorded_reward_hash_count",
            "group_sample_counts",
            "reward_outcome_counts",
            "benchmark_mode_counts",
            "judge_branch_signature_counts",
            "terminal_snapshot_count",
            "benchmark_invariant_hash",
            "clock_host",
            "pid",
        },
    }
    for marker_file, expected_event in (
        (ready_file, "weight_serving_ready"),
        (workload_file, "reward_workload_fragment"),
    ):
        with marker_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != 1
                    or record.get("event") != expected_event
                ):
                    raise ValueError(f"invalid {expected_event} marker at {marker_file}:{line_number}")
                missing = required_by_event[expected_event] - set(record)
                if missing:
                    raise ValueError(
                        f"{expected_event} marker missing {sorted(missing)} at {marker_file}:{line_number}"
                    )
                records.append(record)
    if not any(record["event"] == "weight_serving_ready" for record in records):
        raise ValueError(f"no ready markers found in {ready_file}")
    if not any(record["event"] == "reward_workload_fragment" for record in records):
        raise ValueError(f"no workload markers found in {workload_file}")
    return records


def _merge_int_counter(target: Counter[str], value: Any, *, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"workload marker {field} must be an object")
    for key, count in value.items():
        if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid workload marker {field} entry: {key!r}={count!r}")
        target[key] += count


def _aggregate_workload_markers(
    records: list[dict[str, Any]],
    *,
    measured_steps: list[int],
    benchmark_mode: str,
    benchmark_invariant_hash: str,
    expected_groups_per_round: int | None,
    expected_samples_per_group: int | None,
) -> dict[int, dict[str, Any]]:
    if (expected_groups_per_round is None) != (expected_samples_per_group is None):
        raise ValueError("expected group count and samples per group must be specified together")
    fragments_by_step: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("event") != "reward_workload_fragment":
            continue
        terminal_snapshot_count = record.get("terminal_snapshot_count")
        if (
            isinstance(terminal_snapshot_count, bool)
            or not isinstance(terminal_snapshot_count, int)
            or terminal_snapshot_count < 0
        ):
            raise ValueError(f"invalid terminal_snapshot_count in workload marker: {terminal_snapshot_count!r}")
        if terminal_snapshot_count:
            raise ValueError(
                "fresh workload marker files contain "
                f"{terminal_snapshot_count} terminal reward failures outside or inside the measured window"
            )
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"workload marker step must be an integer, got {step!r}")
        if step in measured_steps:
            fragments_by_step[step].append(record)
    missing_steps = [step for step in measured_steps if not fragments_by_step[step]]
    if missing_steps:
        raise ValueError(f"workload markers are missing measured publication rounds: {missing_steps}")

    expected_pipeline, expected_executor = BENCHMARK_MODE_STATUS[benchmark_mode]
    expected_outcome = json.dumps(
        [expected_pipeline, expected_executor, None],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    expected_branches = json.dumps(
        sorted([component, "success"] for component in BENCHMARK_MODE_BRANCHES[benchmark_mode]),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output = {}
    for step in measured_steps:
        identity_count = 0
        identity_sum = 0
        identity_xor = 0
        invalid_identity_count = 0
        invalid_reward_hash_count = 0
        terminal_snapshot_count = 0
        group_counts: Counter[int] = Counter()
        outcome_counts: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()
        branch_counts: Counter[str] = Counter()
        for fragment in fragments_by_step[step]:
            if fragment.get("benchmark_invariant_hash") != benchmark_invariant_hash:
                raise ValueError(f"workload marker invariant hash differs at publication round {step}")
            count = fragment.get("identity_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"invalid identity_count at publication round {step}: {count!r}")
            identity_count += count
            for field, operation in (
                ("identity_digest_sum", "sum"),
                ("identity_digest_xor", "xor"),
            ):
                raw_digest = fragment.get(field)
                if not isinstance(raw_digest, str) or len(raw_digest) != 64:
                    raise ValueError(f"invalid {field} at publication round {step}")
                try:
                    digest = int(raw_digest, 16)
                except ValueError as exc:
                    raise ValueError(f"invalid {field} at publication round {step}") from exc
                if operation == "sum":
                    identity_sum = (identity_sum + digest) % _DIGEST_MODULUS
                else:
                    identity_xor ^= digest
            for field, target_name in (
                ("invalid_identity_count", "identity"),
                ("invalid_recorded_reward_hash_count", "reward_hash"),
                ("terminal_snapshot_count", "terminal"),
            ):
                value = fragment.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid {field} at publication round {step}: {value!r}")
                if target_name == "identity":
                    invalid_identity_count += value
                elif target_name == "reward_hash":
                    invalid_reward_hash_count += value
                else:
                    terminal_snapshot_count += value
            raw_group_counts = fragment.get("group_sample_counts")
            if not isinstance(raw_group_counts, list):
                raise ValueError(f"group_sample_counts must be a list at publication round {step}")
            for item in raw_group_counts:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or isinstance(item[0], bool)
                    or not isinstance(item[0], int)
                    or isinstance(item[1], bool)
                    or not isinstance(item[1], int)
                    or item[1] < 0
                ):
                    raise ValueError(f"invalid group_sample_counts entry at publication round {step}: {item!r}")
                group_counts[item[0]] += item[1]
            _merge_int_counter(outcome_counts, fragment.get("reward_outcome_counts"), field="reward_outcome_counts")
            _merge_int_counter(mode_counts, fragment.get("benchmark_mode_counts"), field="benchmark_mode_counts")
            _merge_int_counter(
                branch_counts,
                fragment.get("judge_branch_signature_counts"),
                field="judge_branch_signature_counts",
            )

        if invalid_identity_count:
            raise ValueError(f"workload marker has {invalid_identity_count} invalid identities at round {step}")
        if invalid_reward_hash_count:
            raise ValueError(
                "recorded/shadow workload has "
                f"{invalid_reward_hash_count} missing training-reward hashes at round {step}"
            )
        if terminal_snapshot_count:
            raise ValueError(
                f"workload marker captured {terminal_snapshot_count} terminal reward failures at round {step}"
            )
        if mode_counts != Counter({benchmark_mode: identity_count}):
            raise ValueError(f"workload benchmark modes are invalid at round {step}: {dict(mode_counts)}")
        if outcome_counts != Counter({expected_outcome: identity_count}):
            raise ValueError(f"workload reward outcomes are invalid at round {step}: {dict(outcome_counts)}")
        if branch_counts != Counter({expected_branches: identity_count}):
            raise ValueError(f"workload Judge branches are invalid at round {step}: {dict(branch_counts)}")
        if expected_groups_per_round is not None and expected_samples_per_group is not None:
            if len(group_counts) != expected_groups_per_round or any(
                count != expected_samples_per_group for count in group_counts.values()
            ):
                raise ValueError(
                    f"workload cardinality mismatch at round {step}: groups={dict(group_counts)}, "
                    f"expected={expected_groups_per_round}x{expected_samples_per_group}"
                )
            expected_count = expected_groups_per_round * expected_samples_per_group
            if identity_count != expected_count:
                raise ValueError(
                    f"workload identity count mismatch at round {step}: {identity_count} != {expected_count}"
                )
        output[step] = {
            "identity_count": identity_count,
            "identity_digest_sum": f"{identity_sum:064x}",
            "identity_digest_xor": f"{identity_xor:064x}",
            "group_sample_counts": dict(sorted(group_counts.items())),
        }
    return output


def analyze_markers(
    records: list[dict[str, Any]],
    *,
    warmup_rounds: int,
    measure_rounds: int,
    expected_step_stride: int,
    required_steps: list[int] | None = None,
    expected_mode: str | None = None,
    expected_groups_per_round: int | None = None,
    expected_samples_per_group: int | None = None,
) -> dict[str, Any]:
    if warmup_rounds < 1 or measure_rounds < 1 or expected_step_stride < 1:
        raise ValueError("warmup, measurement length, and expected step stride must be positive")
    by_step: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "weight_serving_ready":
            continue
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"ready marker step must be an integer, got {step!r}")
        if step in by_step:
            raise ValueError(f"duplicate ready marker step {step}; use a fresh marker directory per run")
        by_step[step] = record
    ordered_steps = sorted(by_step)
    if required_steps is None:
        required_count = warmup_rounds + measure_rounds
        if len(ordered_steps) < required_count:
            raise ValueError(f"need {required_count} ready markers, found {len(ordered_steps)}")
        selected_steps = ordered_steps[:required_count]
        required_steps = selected_steps[warmup_rounds - 1 :]
    else:
        if len(required_steps) != measure_rounds + 1:
            raise ValueError("required ready-marker sequence must contain measure_rounds + 1 steps")
        missing = [step for step in required_steps if step not in by_step]
        if missing:
            raise ValueError(f"ready-marker run is missing steps: {missing}")
        start_index = ordered_steps.index(required_steps[0])
        observed = ordered_steps[start_index : start_index + len(required_steps)]
        if observed != required_steps:
            raise ValueError(f"ready-marker step sequence differs from baseline: {observed} != {required_steps}")
    gaps = [
        (previous, current)
        for previous, current in zip(required_steps, required_steps[1:])
        if current - previous != expected_step_stride
    ]
    if gaps:
        raise ValueError(f"ready-marker steps are not contiguous at stride {expected_step_stride}: {gaps}")

    selected = [by_step[step] for step in required_steps]
    process_validation_steps = [step for step in ordered_steps if step <= required_steps[-1]]
    process_validation_records = [by_step[step] for step in process_validation_steps]
    invalid_monotonic_values = [
        record.get("monotonic_ns")
        for record in process_validation_records
        if isinstance(record.get("monotonic_ns"), bool) or not isinstance(record.get("monotonic_ns"), int)
    ]
    if invalid_monotonic_values:
        raise ValueError(f"ready-marker monotonic_ns values must be integers: {invalid_monotonic_values}")
    process_domains = {(record["clock_host"], record["pid"]) for record in process_validation_records}
    if len(process_domains) != 1:
        raise ValueError(f"monotonic ready markers cross actor process domains: {sorted(process_domains)}")
    process_validation_timestamps = [int(record["monotonic_ns"]) for record in process_validation_records]
    if any(
        current <= previous
        for previous, current in zip(process_validation_timestamps, process_validation_timestamps[1:])
    ):
        raise ValueError("ready-marker monotonic clock did not increase throughout warmup and measurement")
    modes = {record.get("benchmark_mode") for record in selected}
    if len(modes) != 1:
        raise ValueError(f"ready-marker run contains mixed benchmark modes: {sorted(str(mode) for mode in modes)}")
    mode = next(iter(modes))
    if mode not in BENCHMARK_MODE_STATUS:
        raise ValueError(f"unknown ready-marker benchmark mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"ready-marker benchmark mode mismatch: expected {expected_mode!r}, observed {mode!r}")
    invariant_hashes = {record.get("benchmark_invariant_hash") for record in selected}
    if len(invariant_hashes) != 1 or not all(isinstance(value, str) and value for value in invariant_hashes):
        raise ValueError("ready markers require exactly one nonempty benchmark invariant hash")
    timeline_values = {bool(record.get("timeline_enabled")) for record in selected}
    if len(timeline_values) != 1:
        raise ValueError("timeline_enabled changed within one ready-marker run")

    timestamps = [int(record["monotonic_ns"]) for record in selected]
    intervals = [(current - previous) / 1e9 for previous, current in zip(timestamps, timestamps[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError(f"ready-marker intervals must be positive: {intervals}")
    workload_signature = _aggregate_workload_markers(
        records,
        measured_steps=required_steps[1:],
        benchmark_mode=mode,
        benchmark_invariant_hash=next(iter(invariant_hashes)),
        expected_groups_per_round=expected_groups_per_round,
        expected_samples_per_group=expected_samples_per_group,
    )
    return {
        "steps": required_steps[1:],
        "ready_boundary_step": required_steps[0],
        "benchmark_mode": mode,
        "benchmark_invariant_hash": next(iter(invariant_hashes)),
        "timeline_enabled": next(iter(timeline_values)),
        "clock_host": selected[0]["clock_host"],
        "pid": selected[0]["pid"],
        "process_domain_validated_steps": process_validation_steps,
        "workload_signature_by_step": workload_signature,
        "fixed_k_ready_to_ready_makespan_s": sum(intervals),
        "per_publication_round_interval": _statistics(intervals),
    }


def _trace_calibration_evidence(
    variants: dict[str, dict[str, Any]],
    *,
    baseline_name: str,
    timeline_paths: dict[str, Path],
) -> dict[str, Any]:
    if len(variants) != 2:
        raise ValueError("trace calibration requires exactly two variants")
    candidate_names = [name for name in variants if name != baseline_name]
    candidate_name = candidate_names[0]
    baseline = variants[baseline_name]
    candidate = variants[candidate_name]
    if baseline["benchmark_mode"] != candidate["benchmark_mode"]:
        raise ValueError(
            "trace calibration requires the trace-off and trace-on variants to use the same benchmark mode"
        )
    if baseline["timeline_enabled"] is not False or candidate["timeline_enabled"] is not True:
        raise ValueError("trace calibration requires the first variant to have timeline tracing off and the second on")
    if set(timeline_paths) != {candidate_name}:
        raise ValueError(
            "trace calibration requires exactly one --trace-timeline NAME=PATH mapping for the trace-on variant "
            f"{candidate_name!r}"
        )

    timeline_path = timeline_paths[candidate_name]
    timeline_events, source = load_variant_events(timeline_path)
    if source != "timeline":
        raise ValueError(f"trace-on artifact must contain timeline_step_*.json files: {timeline_path}")
    required_steps = [candidate["ready_boundary_step"], *candidate["steps"]]
    ready_counts: Counter[int] = Counter(
        event["step"]
        for event in timeline_events
        if event.get("name") == "critical_path.weight_serving_ready"
        and isinstance(event.get("step"), int)
        and not isinstance(event.get("step"), bool)
    )
    missing_steps = [step for step in required_steps if not ready_counts[step]]
    if missing_steps:
        raise ValueError(
            f"trace-on timeline has no critical_path.weight_serving_ready events for selected steps: {missing_steps}"
        )
    return {
        "trace_off_variant": baseline_name,
        "trace_on_variant": candidate_name,
        "benchmark_mode": baseline["benchmark_mode"],
        "timeline_enabled_by_variant": {baseline_name: False, candidate_name: True},
        "trace_on_timeline_path": str(timeline_path),
        "trace_on_timeline_source": source,
        "trace_on_timeline_event_count": len(timeline_events),
        "required_ready_steps": required_steps,
        "serving_ready_event_count_by_step": {str(step): ready_counts[step] for step in required_steps},
        "overhead_scope": "timeline_trace_export_aggregation_and_dump",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, type=_parse_variant)
    parser.add_argument("--expected-mode", action="append", default=[], type=_parse_expected_mode)
    parser.add_argument("--warmup-publication-rounds", type=int, default=1)
    parser.add_argument("--measure-publication-rounds", type=int, required=True)
    parser.add_argument("--expected-step-stride", type=int, default=1)
    parser.add_argument("--expected-groups-per-round", type=int, required=True)
    parser.add_argument("--expected-samples-per-group", type=int, required=True)
    parser.add_argument("--require-trace-calibration", action="store_true")
    parser.add_argument("--trace-timeline", action="append", default=[], type=_parse_variant)
    parser.add_argument("--allow-reward-changing-pair", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        _reject_duplicate_names(args.variant, label="variant")
        _reject_duplicate_names(args.expected_mode, label="expected-mode")
        _reject_duplicate_names(args.trace_timeline, label="trace-timeline")
        expected_modes = {name: name for name, _path in args.variant if name in BENCHMARK_MODE_STATUS}
        expected_modes.update(dict(args.expected_mode))
        unknown_mode_names = sorted(set(expected_modes) - {name for name, _path in args.variant})
        if unknown_mode_names:
            raise ValueError(f"--expected-mode names do not match any variant: {unknown_mode_names}")
        missing_modes = sorted({name for name, _path in args.variant} - set(expected_modes))
        if missing_modes:
            raise ValueError(f"expected benchmark mode is required for variants: {missing_modes}")
        timeline_paths = dict(args.trace_timeline)
        unknown_timeline_names = sorted(set(timeline_paths) - {name for name, _path in args.variant})
        if unknown_timeline_names:
            raise ValueError(f"--trace-timeline names do not match any variant: {unknown_timeline_names}")
        if timeline_paths and not args.require_trace_calibration:
            raise ValueError("--trace-timeline is only valid with --require-trace-calibration")
        loaded = {name: load_markers(path) for name, path in args.variant}
        baseline_name = args.variant[0][0]
        baseline = analyze_markers(
            loaded[baseline_name],
            warmup_rounds=args.warmup_publication_rounds,
            measure_rounds=args.measure_publication_rounds,
            expected_step_stride=args.expected_step_stride,
            expected_mode=expected_modes[baseline_name],
            expected_groups_per_round=args.expected_groups_per_round,
            expected_samples_per_group=args.expected_samples_per_group,
        )
        required_steps = [baseline["ready_boundary_step"], *baseline["steps"]]
        variants = {
            name: analyze_markers(
                records,
                warmup_rounds=args.warmup_publication_rounds,
                measure_rounds=args.measure_publication_rounds,
                expected_step_stride=args.expected_step_stride,
                required_steps=required_steps,
                expected_mode=expected_modes[name],
                expected_groups_per_round=args.expected_groups_per_round,
                expected_samples_per_group=args.expected_samples_per_group,
            )
            for name, records in loaded.items()
        }
        trace_calibration = None
        if args.require_trace_calibration:
            trace_calibration = _trace_calibration_evidence(
                variants,
                baseline_name=baseline_name,
                timeline_paths=timeline_paths,
            )
        comparisons = {}
        for name, candidate in variants.items():
            if name == baseline_name:
                continue
            reward_preserving_modes = {"recorded", "accuracy_shadow", "dual_shadow"}
            training_reward_paired = (
                baseline["benchmark_mode"] in reward_preserving_modes
                and candidate["benchmark_mode"] in reward_preserving_modes
            )
            if not training_reward_paired and not args.allow_reward_changing_pair:
                raise ValueError(
                    "paired E2E latency requires recorded/shadow modes so policy updates consume the same reward"
                )
            if candidate["benchmark_invariant_hash"] != baseline["benchmark_invariant_hash"]:
                raise ValueError(f"variant {name!r} changed benchmark invariants")
            if candidate["workload_signature_by_step"] != baseline["workload_signature_by_step"]:
                raise ValueError(f"variant {name!r} changed reward workload identities or training rewards")
            baseline_makespan = baseline["fixed_k_ready_to_ready_makespan_s"]
            candidate_makespan = candidate["fixed_k_ready_to_ready_makespan_s"]
            delta = candidate_makespan - baseline_makespan
            comparisons[name] = {
                "paired_steps": candidate["steps"],
                "baseline_benchmark_mode": baseline["benchmark_mode"],
                "candidate_benchmark_mode": candidate["benchmark_mode"],
                "benchmark_invariant_hash": baseline["benchmark_invariant_hash"],
                "training_reward_paired": training_reward_paired,
                "baseline_makespan_s": baseline_makespan,
                "candidate_makespan_s": candidate_makespan,
                "global_delta_s": delta,
                "global_delta_percent": 100.0 * delta / baseline_makespan,
            }
        report = {
            "baseline": baseline_name,
            "measurement_unit": "weight_publication_round",
            "channel": "low_volume_ready_marker",
            "variants": variants,
            "paired_makespan_delta_vs_baseline": comparisons,
            "trace_calibration": trace_calibration,
            "caveat": (
                "Ready markers measure exposed E2E makespan and validate mergeable workload/reward digests, "
                "but contain no stage spans and cannot identify successful future-step work that was still "
                "in flight; retain deterministic admission and use the detailed analyzer for attribution."
            ),
        }
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
