#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed artifact gate for the two-node G3 actor TP4 x DP2 smoke."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _checkpoint_evidence(root: Path) -> tuple[Path, int]:
    checkpoint_dir = root / "checkpoint"
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(f"checkpoint tracker is missing: {tracker}")
    iteration = int(tracker.read_text(encoding="utf-8").strip())
    if iteration != 1:
        raise RuntimeError(f"expected checkpoint iteration 1, got {iteration}")
    iteration_dir = checkpoint_dir / "iter_0000001"
    files = [path for path in iteration_dir.rglob("*") if path.is_file() and path.stat().st_size]
    if not files:
        raise RuntimeError(f"checkpoint iteration is missing or empty: {iteration_dir}")
    return iteration_dir, len(files)


def _timeline_evidence(root: Path) -> dict[str, Any]:
    timeline_dir = root / "train_timeline"
    paths = sorted(timeline_dir.glob("*.json"))
    if not paths:
        raise RuntimeError(f"training timeline is empty: {timeline_dir}")

    ranks_by_event_step: dict[tuple[str, int], set[int]] = defaultdict(set)
    host_by_rank: dict[int, set[str]] = defaultdict(set)
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            name = row.get("name")
            if name not in {"critical_path.training_schedule", "critical_path.optimizer_step"}:
                continue
            args = row.get("args") or {}
            rank = int(args["global_rank"])
            step = int(args["step"])
            host = str(args["clock_host"])
            ranks_by_event_step[(name, step)].add(rank)
            host_by_rank[rank].add(host)

    expected_ranks = set(range(8))
    for name in ("critical_path.training_schedule", "critical_path.optimizer_step"):
        for step in (0, 1):
            ranks = ranks_by_event_step.get((name, step), set())
            if ranks != expected_ranks:
                raise RuntimeError(f"{name} step {step} ranks mismatch: {sorted(ranks)}")
    if set(host_by_rank) != expected_ranks or any(len(hosts) != 1 for hosts in host_by_rank.values()):
        raise RuntimeError(f"actor rank-to-host provenance is incomplete: {dict(host_by_rank)}")
    host_counts: dict[str, int] = defaultdict(int)
    for hosts in host_by_rank.values():
        host_counts[next(iter(hosts))] += 1
    if sorted(host_counts.values()) != [4, 4]:
        raise RuntimeError(f"actor ranks are not split 4+4 across two nodes: {dict(host_counts)}")

    return {
        "timeline_file_count": len(paths),
        "training_steps": [0, 1],
        "actor_ranks": sorted(host_by_rank),
        "actor_hosts": dict(host_counts),
    }


def validate(root: Path, mode: str, driver_log: Path) -> dict[str, Any]:
    if mode not in {"train", "reload"}:
        raise ValueError(f"unsupported mode: {mode}")
    log_text = driver_log.read_text(encoding="utf-8", errors="replace")
    forbidden = [
        marker
        for marker in (
            "CUDNN_STATUS_",
            "TCPStore client has failed",
            "TCPStore timed out",
            "ProcessGroupNCCL.*timed out",
            "Ray job failed",
        )
        if marker in log_text
    ]
    if forbidden:
        raise RuntimeError(f"actor log contains distributed/runtime failures: {forbidden}")

    iteration_dir, checkpoint_files = _checkpoint_evidence(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "mode": mode,
        "checkpoint_iteration": 1,
        "checkpoint_dir": str(iteration_dir),
        "checkpoint_file_count": checkpoint_files,
    }
    if mode == "train":
        report.update(_timeline_evidence(root))
    else:
        required = (
            "loading distributed checkpoint from",
            "successfully loaded checkpoint from",
            "at iteration 1",
            "Actor initialized with starting step 2",
        )
        missing = [marker for marker in required if marker not in log_text]
        if missing:
            raise RuntimeError(f"reload log lacks checkpoint restoration evidence: {missing}")
        report["loaded_iteration"] = 1
        report["starting_step"] = 2
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "reload"), required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    args = parser.parse_args()

    report = validate(args.root, args.mode, args.driver_log)
    output_path = args.root / f"{args.mode}_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
