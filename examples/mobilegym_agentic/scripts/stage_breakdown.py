#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Turn an `analyze_latency.py` non-direct stage report into the three tables
the "Critical-path latency distribution" measurement needs:

1. A stall decomposition that sums to exactly 100%: while the trainer was blocked on
   `critical_path.data_wait`, which upstream stage family (reward / rollout generation /
   transfer) was still the only thing observed active. This marginalizes
   `active_set_percent`, which `analyze_latency.py` already computes as an exact partition
   of the measured window (analyze_latency.py:1298) -- no new interval math.
2. An ORM-vs-PRM wall-clock split within the reward stage. This IS new: the stock report
   has no per-component union, only per-call latency stats
   (`event_clipped_duration_by_name`). Reuses `analyze_latency._union_duration`
   (analyze_latency.py:1283) on the same normalized events analyze_latency.py itself loads,
   clipped to the identical measurement window (`analyze_latency._fixed_k_window`).
3. A reward-GPU idle/utilization table, read straight from `judge_gpu_efficiency`.

This script does not replace `analyze_latency.py` and does not weaken any of its
validation -- it consumes a report already produced by (and a timeline dir already
validated by) that tool. See examples/agentic_dual_judge/README.md and the
"Critical-path latency distribution" plan for the precise semantics and caveats of every
field read here, in particular:

  - The stall decomposition is coincidence-based (interval intersection with
    `data_wait`), not proof that the named stage was the *binding* constraint.
  - NVML `util_percent` means "at least one kernel ran in the sampling window", not SM
    occupancy -- always read it next to `running_reqs`/`token_usage`.
  - `sample_coverage_fraction` is only meaningful for single-shard engines (the judges);
    it is structurally wrong for the 12-shard rollout role, so this script does not
    attempt to report rollout GPU efficiency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_AGENTIC_DUAL_JUDGE_DIR = Path(__file__).resolve().parents[2] / "agentic_dual_judge"
sys.path.insert(0, str(_AGENTIC_DUAL_JUDGE_DIR))
import analyze_latency  # noqa: E402  (sys.path must be set up first)


REWARD_STAGES = {"reward", "turn_judge", "request"}
# rollout_admission_wait is the per-session IR admission-gate wait split out of the old
# monolithic rollout_pre_generation span (see relax/agentic/rollout.py); it is genuine
# rollout-side backpressure, not generation compute, but it belongs in the same bucket as
# generation/rollout_queue rather than disappearing into "nothing_observed".
ROLLOUT_GEN_STAGES = {"generation", "rollout_queue", "rollout_admission_wait"}
TRANSFER_STAGES = {"transfer"}


def _parse_named_path(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def stall_decomposition(active_set_percent: dict[str, float]) -> dict[str, float]:
    """Marginalize an exact overlap-state partition into a decision-relevant
    table that still sums to 100%: trainer-not-stalled vs.

    trainer-stalled-by-{reward, rollout generation, transfer, some combination,
    or nothing observed}.
    """
    buckets: dict[str, float] = {}
    trainer_not_stalled = 0.0
    for label, pct in active_set_percent.items():
        stages = set(label.split("+"))
        if "data_wait" not in stages:
            trainer_not_stalled += pct
            continue
        present = []
        if stages & REWARD_STAGES:
            present.append("reward")
        if stages & ROLLOUT_GEN_STAGES:
            present.append("rollout_gen")
        if stages & TRANSFER_STAGES:
            present.append("transfer")
        key = "stall:" + ("+".join(present) if present else "nothing_observed")
        buckets[key] = buckets.get(key, 0.0) + pct
    buckets["trainer_not_stalled"] = trainer_not_stalled
    return buckets


def reward_component_split(events: list[dict[str, Any]], window: tuple[float, float]) -> dict[str, Any]:
    """Wall-clock union coverage of the reward stage as a whole and of its
    ORM/PRM components, clipped to `window`.

    A run is either terminal_once or per_turn, so exactly one of the two PRM
    components will be non-zero -- reporting both is a cheap cross-check that
    the inactive mode contributed nothing.
    """

    def union_for(names: set[str] | None, stages: set[str] | None) -> float:
        intervals = []
        for event in events:
            if names is not None and event["name"] not in names:
                continue
            if stages is not None and event["stage"] not in stages:
                continue
            start = max(window[0], event["start_s"])
            end = min(window[1], event["end_s"])
            if end > start:
                intervals.append((start, end))
        return analyze_latency._union_duration(intervals)

    window_s = window[1] - window[0]
    reward_total_s = union_for(None, REWARD_STAGES)
    orm_s = union_for({"critical_path.reward.answer_accuracy"}, None)
    prm_final_s = union_for({"critical_path.reward.multi_turn_reasoning"}, None)
    prm_round_s = union_for({"critical_path.turn_judge", "critical_path.turn_judge_barrier"}, None)

    def share_of_reward(component_s: float) -> float | None:
        return 100.0 * component_s / reward_total_s if reward_total_s > 0 else None

    return {
        "reward_stage_total_pct_of_window": 100.0 * reward_total_s / window_s,
        "orm_answer_accuracy": {
            "pct_of_window": 100.0 * orm_s / window_s,
            "pct_of_reward_stage": share_of_reward(orm_s),
        },
        "prm_terminal_once": {
            "pct_of_window": 100.0 * prm_final_s / window_s,
            "pct_of_reward_stage": share_of_reward(prm_final_s),
        },
        "prm_per_turn": {
            "pct_of_window": 100.0 * prm_round_s / window_s,
            "pct_of_reward_stage": share_of_reward(prm_round_s),
        },
    }


def gpu_idle_table(
    judge_gpu_efficiency: dict[str, Any] | None, judge_gpu_efficiency_issues: list[str] | None
) -> dict[str, Any]:
    """Idle/utilization summary for the two dedicated reward-model roles only.

    There is no `idle_fraction` field in the source data -- both idle-rate
    numbers here are explicitly derived and labelled per the plan's caveats
    section.
    """
    issues = judge_gpu_efficiency_issues or []
    out: dict[str, Any] = {}
    for role in ("judge_accuracy", "judge_multiturn_vlm"):
        role_issues = [issue for issue in issues if issue.endswith(f":{role}") or issue == role]
        report = (judge_gpu_efficiency or {}).get(role)
        if report is None:
            out[role] = {"status": "unavailable", "issues": role_issues}
            continue
        coverage = report.get("sample_coverage_fraction")
        if role_issues or coverage is None or coverage <= 0.5:
            out[role] = {"status": "unreliable", "issues": role_issues, "sample_coverage_fraction": coverage}
            continue
        nonidle = report.get("nonidle_fraction")
        util = report.get("util_percent") or {}
        running_reqs = report.get("running_reqs") or {}
        out[role] = {
            "status": "ok",
            "nvml_idle_rate_pct": None if nonidle is None else 100.0 * (1.0 - nonidle),
            "engine_idle_rate_pct_zero_request_ticks": 100.0 * report.get("zero_request_time_fraction", 0.0),
            "util_percent_mean": util.get("mean_percent"),
            "util_percent_p50": util.get("p50_percent"),
            "util_percent_p90": util.get("p90_percent"),
            "running_reqs_mean": running_reqs.get("mean"),
            "running_reqs_max": running_reqs.get("max"),
            "token_usage_mean": (report.get("token_usage") or {}).get("mean"),
            "gpu_hours_in_window": report.get("gpu_hours_in_window"),
            "allocated_gpu_seconds_per_training_sample": report.get("allocated_gpu_seconds_per_training_sample"),
            "sample_coverage_fraction": coverage,
        }
    return out


def build_variant_breakdown(
    report_path: Path, timeline_path: Path, warmup_rounds: int, measure_rounds: int
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    variant_names = list(report.get("variants", {}).keys())
    if len(variant_names) != 1:
        raise ValueError(f"{report_path} must contain exactly one variant, found {variant_names}")
    variant = report["variants"][variant_names[0]]

    active_set_percent = variant.get("active_set_percent")
    if not active_set_percent:
        raise ValueError(f"{report_path}: report has no active_set_percent (was it produced with --direct?)")
    stall = stall_decomposition(active_set_percent)
    stall_sum = sum(stall.values())
    if abs(stall_sum - 100.0) > 0.5:
        raise ValueError(f"{report_path}: stall decomposition sums to {stall_sum:.2f}%, expected ~100%")

    events, source = analyze_latency.load_variant_events(timeline_path, include_rollout_with_timeline=False)
    if source != "timeline":
        raise ValueError(f"{timeline_path}: expected a timeline directory, got source={source!r}")
    window, _measured_steps, _boundary_step = analyze_latency._fixed_k_window(
        events, warmup_updates=warmup_rounds, measure_updates=measure_rounds
    )
    reward_split = reward_component_split(events, window)

    gpu = gpu_idle_table(variant.get("judge_gpu_efficiency"), variant.get("judge_gpu_efficiency_issues"))

    return {
        "reasoning_trigger": variant.get("observed_reasoning_trigger"),
        "benchmark_invariant_hash": variant.get("benchmark_invariant_hash"),
        "window_s": window[1] - window[0],
        "stall_decomposition_pct": dict(sorted(stall.items(), key=lambda kv: -kv[1])),
        "reward_component_split": reward_split,
        "reward_gpu_efficiency": gpu,
    }


def _print_markdown(name: str, breakdown: dict[str, Any]) -> None:
    print(f"\n## {name}  (trigger={breakdown['reasoning_trigger']}, window={breakdown['window_s']:.1f}s)\n")
    print("### Stall decomposition (sums to 100%)\n")
    for key, pct in breakdown["stall_decomposition_pct"].items():
        print(f"  {key:38s} {pct:6.2f}%")
    print("\n### Reward stage: ORM vs PRM wall-clock share\n")
    rc = breakdown["reward_component_split"]
    print(f"  reward stage total: {rc['reward_stage_total_pct_of_window']:.2f}% of window")
    for comp in ("orm_answer_accuracy", "prm_terminal_once", "prm_per_turn"):
        c = rc[comp]
        share = "n/a" if c["pct_of_reward_stage"] is None else f"{c['pct_of_reward_stage']:.2f}%"
        print(f"  {comp:22s} {c['pct_of_window']:6.2f}% of window   ({share} of reward stage)")
    print("\n### Reward-GPU idle / utilization\n")
    for role, r in breakdown["reward_gpu_efficiency"].items():
        if r["status"] != "ok":
            print(f"  {role:22s} {r['status']} (issues={r.get('issues')})")
            continue
        print(
            f"  {role:22s} nvml_idle={r['nvml_idle_rate_pct']:.2f}%  "
            f"engine_idle(zero-req-ticks)={r['engine_idle_rate_pct_zero_request_ticks']:.2f}%  "
            f"util_mean={r['util_percent_mean']:.2f}%  running_reqs_mean={r['running_reqs_mean']:.3f}  "
            f"coverage={r['sample_coverage_fraction']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        type=_parse_named_path,
        required=True,
        metavar="NAME=PATH",
        help="analyze_latency.py --output JSON (non-direct mode) for one variant.",
    )
    parser.add_argument(
        "--timeline",
        action="append",
        default=[],
        type=_parse_named_path,
        required=True,
        metavar="NAME=DIR",
        help="Matching timeline directory (same NAME as --report), for the ORM/PRM union recompute.",
    )
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--measure-rounds", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reports = dict(args.report)
    timelines = dict(args.timeline)
    if set(reports) != set(timelines):
        raise SystemExit(f"--report and --timeline names must match: {sorted(reports)} vs {sorted(timelines)}")

    result: dict[str, Any] = {}
    for name in reports:
        result[name] = build_variant_breakdown(reports[name], timelines[name], args.warmup_rounds, args.measure_rounds)
        _print_markdown(name, result[name])

    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
