#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-closed artifact gate for the two-node G3 rollout8 smoke."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


_BUNDLE_RE = re.compile(r"bundle\s+(\d+).*node:\s*([^,\s]+),\s*gpu:\s*(\d+)")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dist_init_host(value: str) -> str:
    host, _port = value.rsplit(":", 1)
    return host.strip("[]")


def validate(exp_dir: Path, driver_log: Path, expected_samples: int, expected_engines: int = 8) -> dict[str, Any]:
    log_text = driver_log.read_text(encoding="utf-8", errors="replace")
    forbidden = [
        marker
        for marker in (
            "TCPStore client has failed",
            "TCPStore timed out",
            "DistNetworkError",
            "Traceback (most recent call last)",
            "Scheduler hit an exception",
            "No locks available",
            "unexpectedly completed without",
        )
        if marker in log_text
    ]
    if forbidden:
        raise RuntimeError(f"cross-node rollout log contains rendezvous/fault failures: {forbidden}")

    bundle_rows: dict[int, tuple[str, int]] = {}
    for match in _BUNDLE_RE.finditer(log_text):
        bundle_rows[int(match.group(1))] = (match.group(2), int(match.group(3)))
    if expected_engines % 4:
        raise RuntimeError(f"expected engine count must be divisible by four, got {expected_engines}")
    if sorted(bundle_rows) != list(range(expected_engines)):
        raise RuntimeError(f"expected {expected_engines} rollout placement bundles, got {bundle_rows}")
    host_counts = Counter(host for host, _gpu in bundle_rows.values())
    expected_host_counts = [4] * (expected_engines // 4)
    if sorted(host_counts.values()) != expected_host_counts:
        raise RuntimeError(f"rollout placement is not a four-engine-per-node split: {host_counts}")
    for host in host_counts:
        gpu_ids = sorted(gpu for row_host, gpu in bundle_rows.values() if row_host == host)
        if gpu_ids != [0, 1, 2, 3]:
            raise RuntimeError(f"node {host} does not own physical GPU IDs 0..3: {gpu_ids}")

    sampler_paths = sorted((exp_dir / "gpu_samples").glob("rollout_*_rank*.jsonl"))
    workspace_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((exp_dir / "flashinfer_workspace").glob("*.json"))
    ]
    expected_nodes = expected_engines // 4
    if (
        len(workspace_rows) != expected_nodes
        or len({str(row.get("hostname")) for row in workspace_rows}) != expected_nodes
    ):
        raise RuntimeError(f"FlashInfer workspace probes do not cover {expected_nodes} nodes: {workspace_rows}")
    expected_workspaces = {str(row.get("workspace")) for row in workspace_rows}
    if len(expected_workspaces) != 1 or not next(iter(expected_workspaces)).startswith("/tmp/relax-flashinfer/"):
        raise RuntimeError(f"FlashInfer workspace is not run-specific node-local storage: {workspace_rows}")
    manifests: dict[int, dict[str, Any]] = {}
    active_ranks: set[int] = set()
    for path in sampler_paths:
        for row in _read_jsonl(path):
            rank = int(row["engine_rank"])
            if row.get("record_type") == "manifest":
                manifests[rank] = row
            elif row.get("record_type") == "sample":
                metrics = row.get("sglang") or {}
                if float(metrics.get("sglang:gen_throughput", 0.0) or 0.0) > 0.0:
                    active_ranks.add(rank)
    if sorted(manifests) != list(range(expected_engines)):
        raise RuntimeError(f"expected {expected_engines} rollout sampler manifests, got {sorted(manifests)}")
    manifest_workspaces = {str(manifest.get("flashinfer_workspace_base")) for manifest in manifests.values()}
    if manifest_workspaces != expected_workspaces:
        raise RuntimeError(
            f"rollout actor FlashInfer workspaces disagree with node probes: "
            f"{manifest_workspaces} != {expected_workspaces}"
        )
    for rank, manifest in manifests.items():
        expected_host = bundle_rows[rank][0]
        actual_host = str(manifest.get("server_host") or "").strip("[]")
        dist_init_addr = str(manifest.get("dist_init_addr") or "")
        if not actual_host or not dist_init_addr:
            raise RuntimeError(f"engine {rank} sampler manifest lacks server/dist-init provenance: {manifest}")
        dist_host = _dist_init_host(dist_init_addr)
        if actual_host != expected_host or dist_host != expected_host:
            raise RuntimeError(
                f"engine {rank} rendezvous host mismatch: bundle={expected_host}, "
                f"host={actual_host}, dist_init={dist_host}"
            )
    if any(not manifest.get("nvml_enabled") for manifest in manifests.values()):
        raise RuntimeError("at least one rollout sampler did not initialize NVML")
    if active_ranks != set(range(expected_engines)):
        raise RuntimeError(f"not every rollout engine served measured decode work: active={sorted(active_ranks)}")

    result_paths = sorted((exp_dir / "rollout_result" / "train").glob("*.jsonl"))
    rollout_rows = [row for path in result_paths for row in _read_jsonl(path)]
    status_counts = Counter(str(row.get("status")) for row in rollout_rows)
    committed_rows = [
        row
        for row in rollout_rows
        if row.get("status") in {"completed", "truncated"}
        and isinstance(row.get("response"), str)
        and row["response"]
        and int(row.get("response_token_count", 0)) > 0
    ]
    if len(committed_rows) != expected_samples or len(rollout_rows) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} committed completed/truncated rollout rows, "
            f"got committed={len(committed_rows)} total={len(rollout_rows)} statuses={dict(status_counts)}"
        )
    debug_paths = sorted((exp_dir / "debug_rollout").glob("*.pt"))
    if len(debug_paths) != 1 or debug_paths[0].stat().st_size == 0:
        raise RuntimeError(f"expected one non-empty debug rollout capture, got {debug_paths}")
    timeline_paths = [path for path in (exp_dir / "timeline").rglob("*") if path.is_file() and path.stat().st_size]
    if not timeline_paths:
        raise RuntimeError("G3 rollout8 timeline is empty")

    return {
        "schema_version": 1,
        "status": "passed",
        "placement_hosts": dict(host_counts),
        "engine_ranks": sorted(manifests),
        "active_sampler_ranks": sorted(active_ranks),
        "flashinfer_workspace": next(iter(expected_workspaces)),
        "committed_samples": len(committed_rows),
        "rollout_status_counts": dict(status_counts),
        "debug_capture": str(debug_paths[0]),
        "timeline_file_count": len(timeline_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=8)
    parser.add_argument("--expected-engines", type=int, default=8)
    args = parser.parse_args()

    report = validate(args.exp_dir, args.driver_log, args.expected_samples, args.expected_engines)
    output_path = args.exp_dir / "g3_rollout8_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
