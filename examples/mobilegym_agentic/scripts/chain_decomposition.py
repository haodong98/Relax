#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-sample dependency-chain decomposition of the post-generation tail.

`stage_breakdown.py`'s stall table answers "while the trainer was blocked, which stage was
active" via interval intersection -- coincidence-based, not causal, and its `active_set`
partition is over the *whole measurement window*, not the path a single sample actually
took. This script answers a narrower, additive question instead: for each sample, walk the
ordered timestamps already in its `latency_trace.events` from "last turn's generation
finished" to "released into the training queue", and attribute every second of that tail to
exactly one named segment. Because every boundary in the chain is `events.get(key)` for
consecutive keys, the segments sum to the tail by construction -- the only way a residual
appears is a broken/missing timestamp on some sample, so a non-zero residual is itself a
diagnostic (which this script reports, not hides).

Segments (see examples/mobilegym_agentic/LATENCY_FINDINGS.md section 1 for the mechanism
behind each one):

  A  trajectory generation      first turn's chat_request_arrive_at -> last chat_end_at
  B  finalize                   last chat_end_at -> finalize_end_at
  C  wait to enter reward       finalize_end_at -> reward_arrive_at
  D  reward compute             reward_arrive_at -> reward_end_at
  E  round barrier (straggler)  reward_end_at -> transfer_release_start_at
  F  transfer release           transfer_release_start_at -> transfer_release_end_at
  G  post-generation tail       B + C + D + E + F  (= last chat_end_at -> transfer_end)
  H  full chain                 A + G              (= first request -> transfer_end)

`transfer_batch_group_count` (relax/agentic/pipeline/transfer.py:113) means F only starts
once the whole round's samples are buffered, so E is a round barrier, not a GRPO-group
barrier: don't read it as measuring one group of 8 in isolation.

Usage:
    python chain_decomposition.py --rollout-result-dir terminal_once=/path/to/rollout_result \\
        --rollout-result-dir per_turn=/path/to/other/rollout_result \\
        --warmup-rounds 2 --output /tmp/chain-report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


_SEGMENT_KEYS: tuple[tuple[str, str, str], ...] = (
    ("B", "last_chat_end_at", "finalize_end_at"),
    ("C", "finalize_end_at", "reward_arrive_at"),
    ("D", "reward_arrive_at", "reward_end_at"),
    ("E", "reward_end_at", "transfer_release_start_at"),
    ("F", "transfer_release_start_at", "transfer_release_end_at"),
)
_RESIDUAL_EPS_S = 1e-6


def _parse_named_path(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def _load_rows(rollout_result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(rollout_result_dir / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _select_rounds(
    rows: list[dict[str, Any]], *, warmup_rounds: int, measure_rounds: int | None
) -> list[dict[str, Any]]:
    round_ids = sorted({row.get("rollout_id") for row in rows if isinstance(row.get("rollout_id"), int)})
    selected = round_ids[warmup_rounds:]
    if measure_rounds is not None:
        selected = selected[:measure_rounds]
    selected_set = set(selected)
    return [row for row in rows if row.get("rollout_id") in selected_set]


def _sample_chain(row: dict[str, Any]) -> dict[str, float] | None:
    """Compute one sample's segment durations, or None if any timestamp on its
    chain is missing (e.g. a right-censored or failed sample)."""
    trace = row.get("latency_trace")
    if not isinstance(trace, dict):
        return None
    events = trace.get("events")
    if not isinstance(events, dict):
        return None
    turns = trace.get("turns")
    if not isinstance(turns, list) or not turns:
        return None

    turn_starts = []
    turn_ends = []
    for turn in turns:
        turn_events = turn.get("events") if isinstance(turn, dict) else None
        if not isinstance(turn_events, dict):
            continue
        start = turn_events.get("chat_request_arrive_at")
        end = turn_events.get("chat_end_at")
        if isinstance(start, (int, float)):
            turn_starts.append(start)
        if isinstance(end, (int, float)):
            turn_ends.append(end)
    if not turn_starts or not turn_ends:
        return None

    marks: dict[str, float] = {
        "first_request_at": min(turn_starts),
        "last_chat_end_at": max(turn_ends),
        "finalize_end_at": events.get("finalize_end_at"),
        "reward_arrive_at": events.get("reward_arrive_at"),
        "reward_end_at": events.get("reward_end_at"),
        "transfer_release_start_at": events.get("transfer_release_start_at"),
        "transfer_release_end_at": events.get("transfer_release_end_at"),
    }
    if any(not isinstance(value, (int, float)) for value in marks.values()):
        return None

    durations: dict[str, float] = {"A": marks["last_chat_end_at"] - marks["first_request_at"]}
    for label, start_key, end_key in _SEGMENT_KEYS:
        durations[label] = marks[end_key] - marks[start_key]
    durations["G"] = sum(durations[label] for label, _, _ in _SEGMENT_KEYS)
    durations["H"] = durations["A"] + durations["G"]
    # By construction G == last_chat_end_at -> transfer_release_end_at; recomputing it
    # independently from the marks (rather than only summing the segments) is what turns
    # "segments sum to G" into an actual residual check instead of a tautology.
    durations["_residual"] = durations["G"] - (marks["transfer_release_end_at"] - marks["last_chat_end_at"])
    return durations


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "mean": statistics.mean(ordered) if n else 0.0,
        "p50": ordered[n // 2] if n else 0.0,
        "p90": ordered[int(0.9 * n)] if n else 0.0,
        "max": ordered[-1] if n else 0.0,
    }


def build_variant_chain(rollout_result_dir: Path, *, warmup_rounds: int, measure_rounds: int | None) -> dict[str, Any]:
    rows = _load_rows(rollout_result_dir)
    if not rows:
        raise SystemExit(f"no rollout-result JSONL rows found under {rollout_result_dir}")
    selected = _select_rounds(rows, warmup_rounds=warmup_rounds, measure_rounds=measure_rounds)

    per_sample = [_sample_chain(row) for row in selected]
    skipped = sum(1 for chain in per_sample if chain is None)
    chains = [chain for chain in per_sample if chain is not None]
    if not chains:
        raise SystemExit(
            f"{rollout_result_dir}: every selected sample is missing a chain timestamp "
            f"({skipped}/{len(selected)}); nothing to report"
        )

    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    stats_by_label = {label: _stats([chain[label] for chain in chains]) for label in labels}
    max_residual = max(abs(chain["_residual"]) for chain in chains)

    return {
        "rounds_available": sorted({row.get("rollout_id") for row in rows if isinstance(row.get("rollout_id"), int)}),
        "rounds_selected": sorted(
            {row.get("rollout_id") for row in selected if isinstance(row.get("rollout_id"), int)}
        ),
        "sample_count": len(selected),
        "sample_count_with_complete_chain": len(chains),
        "sample_count_skipped": skipped,
        "max_abs_residual_s": max_residual,
        "residual_valid": max_residual < _RESIDUAL_EPS_S,
        "segments": stats_by_label,
        "tail_share_of_G_percent_mean": {
            label: 100.0 * stats_by_label[label]["mean"] / stats_by_label["G"]["mean"]
            for label in ("B", "C", "D", "E", "F")
            if stats_by_label["G"]["mean"] > 0
        },
    }


def _print_markdown(name: str, report: dict[str, Any]) -> None:
    print(f"\n### {name}")
    print(
        f"rounds selected: {report['rounds_selected']} "
        f"(of {report['rounds_available']} available) -- "
        f"{report['sample_count_with_complete_chain']}/{report['sample_count']} samples had a complete chain"
    )
    residual_flag = (
        "OK" if report["residual_valid"] else "*** NON-ZERO -- instrumentation gap, see skipped/residual ***"
    )
    print(f"max |residual| = {report['max_abs_residual_s']:.6f}s [{residual_flag}]")
    print("| segment | mean (s) | p50 | p90 | max | % of tail (G) |")
    print("|---|---|---|---|---|---|")
    tail_share = report["tail_share_of_G_percent_mean"]
    for label in ("A", "B", "C", "D", "E", "F", "G", "H"):
        s = report["segments"][label]
        share = f"{tail_share[label]:.1f}%" if label in tail_share else "--"
        print(f"| {label} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p90']:.1f} | {s['max']:.1f} | {share} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rollout-result-dir",
        action="append",
        default=[],
        type=_parse_named_path,
        required=True,
        metavar="NAME=PATH",
        help="rollout_result/train directory for one variant (the same --rollout-result-dir "
        "path passed to training). Repeatable.",
    )
    parser.add_argument("--warmup-rounds", type=int, default=0, help="skip the first N rollout_id values.")
    parser.add_argument(
        "--measure-rounds",
        type=int,
        default=None,
        help="keep at most N rollout_id values after warmup; default is all remaining rounds.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    named_dirs = dict(args.rollout_result_dir)
    result: dict[str, Any] = {}
    for name, path in named_dirs.items():
        result[name] = build_variant_chain(path, warmup_rounds=args.warmup_rounds, measure_rounds=args.measure_rounds)
        _print_markdown(name, result[name])

    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
