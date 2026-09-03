# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Latency/throughput breakdown for a MobileGym agentic run.

Reads two sources that a run already produces -- no extra instrumentation:

* ``<exp>/rollout_result/train/*.jsonl`` -- per-sample agentic trace, which
  carries a per-turn ``events`` dict of absolute timestamps.
* the Slurm ``.out`` log -- SGLang's own ``Prefill batch`` / ``Decode batch``
  lines, which report ``#cached-token`` (prefix-cache reuse) and decode
  throughput against the live context size.

Environment time is not a span in the timeline: it is the gap between one
turn's ``chat_end_at`` and the next turn's ``chat_request_arrive_at``, and is
derived here.

Turn 0 of each session is reported separately: it waits for the rollout step to
be admitted (hundreds of seconds while services start), which is a one-off
warmup and swamps the mean if pooled with steady-state turns.

Usage:
    python analyze_perf.py <job_id> [--exp-root DIR] [--log-root DIR]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics as st
from typing import Any


_DEFAULT_EXP_ROOT = os.environ.get("MOBILEGYM_EXP_ROOT", "")
_DEFAULT_LOG_ROOT = os.environ.get("MOBILEGYM_LOG_ROOT", "")

_PREFILL_RE = re.compile(r"pid=(\d+)\).*?#new-token: (\d+), #cached-token: (\d+)")
_DECODE_RE = re.compile(r"#running-req: (\d+), #token: (\d+).*?gen throughput \(token/s\): ([\d.]+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _turns_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate the per-turn list inside a sample's nested latency trace."""

    def find(node: Any) -> list[dict[str, Any]] | None:
        if isinstance(node, dict):
            turns = node.get("turns")
            if isinstance(turns, list):
                return turns
            for value in node.values():
                hit = find(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for value in node:
                hit = find(value)
                if hit:
                    return hit
        return None

    return find(row) or []


def _stage_durations(turn: dict[str, Any], previous_chat_end: float | None) -> dict[str, float]:
    events = turn.get("events") or {}
    stages: dict[str, float] = {}

    def span(start_key: str, end_key: str) -> float | None:
        start, end = events.get(start_key), events.get(end_key)
        return None if start is None or end is None else end - start

    pairs = {
        # request accepted -> the step admitted it and its IR began running
        "admit_wait": ("ir_created_at", "ir_activated_at"),
        "queue": ("generation_queue_enter_at", "generation_start_at"),
        "generate": ("generation_start_at", "generation_end_at"),
        "post": ("generation_end_at", "chat_end_at"),
    }
    for name, (start_key, end_key) in pairs.items():
        value = span(start_key, end_key)
        if value is not None:
            stages[name] = value

    # tokenizer + chat template + vision preprocessing + media encode
    if events.get("total_elapsed_s") is not None:
        stages["preproc"] = float(events["total_elapsed_s"])
    # the environment acts between turns; it owns no span of its own
    arrive = events.get("chat_request_arrive_at")
    if previous_chat_end is not None and arrive is not None:
        stages["env_gap"] = arrive - previous_chat_end
    return stages


def summarize_turns(exp_dir: str) -> tuple[dict, dict, dict]:
    """Return (warmup_stages, steady_stages, generate_by_turn_index)."""
    warmup: dict[str, list[float]] = collections.defaultdict(list)
    steady: dict[str, list[float]] = collections.defaultdict(list)
    by_index: dict[int, list[float]] = collections.defaultdict(list)

    for path in sorted(glob.glob(os.path.join(exp_dir, "rollout_result", "train", "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                previous_chat_end: float | None = None
                for index, turn in enumerate(_turns_of(json.loads(line))):
                    stages = _stage_durations(turn, previous_chat_end)
                    target = warmup if index == 0 else steady
                    for name, value in stages.items():
                        target[name].append(value)
                    if "generate" in stages:
                        by_index[index].append(stages["generate"])
                    previous_chat_end = (turn.get("events") or {}).get("chat_end_at")
    return warmup, steady, by_index


def summarize_engine_batches(log_path: str) -> tuple[dict, list]:
    """Per-engine prefix-cache reuse and decode throughput vs context size."""
    prefill: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    decode: list[tuple[int, int, float]] = []
    if not os.path.exists(log_path):
        return prefill, decode
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = _ANSI_RE.sub("", raw)
            match = _PREFILL_RE.search(line)
            if match:
                pid, new_tokens, cached = match.group(1), int(match.group(2)), int(match.group(3))
                entry = prefill[pid]
                entry[0] += new_tokens
                entry[1] += cached
                entry[2] += 1
                entry[3] += 1 if cached else 0
                continue
            match = _DECODE_RE.search(line)
            if match and "Decode batch" in line:
                decode.append((int(match.group(1)), int(match.group(2)), float(match.group(3))))
    return prefill, decode


def _report_stages(label: str, stages: dict[str, list[float]]) -> None:
    if not stages:
        return
    print(f"  -- {label} --")
    total = sum(sum(v) for v in stages.values())
    order = sorted(stages.items(), key=lambda kv: -sum(kv[1]))
    for name, values in order:
        share = 100.0 * sum(values) / total if total else 0.0
        print(
            f"    {name:11} n={len(values):4d}  mean={st.mean(values):8.2f}s  "
            f"median={st.median(values):8.2f}s  max={max(values):8.2f}s  "
            f"total={sum(values):9.1f}s  share={share:5.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job_id")
    parser.add_argument("--exp-root", default=_DEFAULT_EXP_ROOT, help="directory holding <job_id>/ experiment dirs")
    parser.add_argument("--log-root", default=_DEFAULT_LOG_ROOT, help="directory holding mobilegym-e2e-<job_id>.out")
    args = parser.parse_args()

    if not args.exp_root or not args.log_root:
        parser.error("set --exp-root/--log-root (or MOBILEGYM_EXP_ROOT/MOBILEGYM_LOG_ROOT)")

    exp_dir = os.path.join(args.exp_root, args.job_id)
    log_path = os.path.join(args.log_root, f"mobilegym-e2e-{args.job_id}.out")

    print(f"=== job {args.job_id} ===")
    warmup, steady, by_index = summarize_turns(exp_dir)
    _report_stages("turn 0 (warmup: waits for step admission)", warmup)
    _report_stages("turns 1..N (steady state)", steady)

    if by_index:
        print("  -- generation vs turn index (prompt grows one screenshot per turn) --")
        for index in sorted(by_index):
            values = by_index[index]
            print(f"    turn {index}: n={len(values):3d}  mean={st.mean(values):6.2f}s  max={max(values):6.2f}s")

    prefill, decode = summarize_engine_batches(log_path)
    if prefill:
        print("  -- SGLang prefix cache, per engine (rollout engine = the busiest) --")
        for pid, (new_tokens, cached, batches, hits) in sorted(prefill.items(), key=lambda kv: -kv[1][2]):
            denominator = new_tokens + cached
            rate = 100.0 * cached / denominator if denominator else 0.0
            print(
                f"    pid={pid:8} batches={batches:4d}  new={new_tokens:10,}  cached={cached:10,}  "
                f"hit_rate={rate:5.1f}%  batches_with_hit={hits}/{batches}"
            )
    if decode:
        buckets: dict[int, list[float]] = collections.defaultdict(list)
        for _running, tokens, throughput in decode:
            if throughput > 1.0:  # drop the first batch after a prefill; it is not steady decode
                buckets[tokens // 20000].append(throughput)
        if buckets:
            print("  -- decode throughput vs live context (batch-wide #token) --")
            for bucket in sorted(buckets):
                values = buckets[bucket]
                print(
                    f"    #token {bucket * 20000:>7,}-{(bucket + 1) * 20000:>7,}: "
                    f"n={len(values):3d}  mean={st.mean(values):8.1f} tok/s"
                )


if __name__ == "__main__":
    main()
