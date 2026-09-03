#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
from pathlib import Path


def _seconds(output: str, label: str) -> float:
    match = re.search(rf"^{re.escape(label)}\s*:\s*([+-]?[0-9.eE+-]+)\s+seconds", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"chronyc tracking output lacks {label!r}: {output!r}")
    return float(match.group(1))


def sample() -> dict:
    completed = subprocess.run(
        ["chronyc", "tracking"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    system_time_s = abs(_seconds(completed.stdout, "System time"))
    last_offset_s = abs(_seconds(completed.stdout, "Last offset"))
    rms_offset_s = abs(_seconds(completed.stdout, "RMS offset"))
    root_dispersion_s = abs(_seconds(completed.stdout, "Root dispersion"))
    node_bound_s = system_time_s + max(last_offset_s, rms_offset_s) + root_dispersion_s
    return {
        "schema_version": 1,
        "clock_host": socket.gethostname(),
        "source": "chronyc_tracking",
        "system_time_s": system_time_s,
        "last_offset_s": last_offset_s,
        "rms_offset_s": rms_offset_s,
        "root_dispersion_s": root_dispersion_s,
        "node_offset_bound_s": node_bound_s,
    }


def summarize(path: Path, expected_hosts: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hosts = {row.get("clock_host") for row in rows}
    if len(rows) != expected_hosts or len(hosts) != expected_hosts or None in hosts:
        raise RuntimeError(
            f"clock audit requires one row per host: expected={expected_hosts}, rows={len(rows)}, hosts={hosts}"
        )
    bounds = sorted((float(row["node_offset_bound_s"]) for row in rows), reverse=True)
    pairwise_bound_s = sum(bounds[:2]) if len(bounds) > 1 else bounds[0]
    return {
        "schema_version": 1,
        "source": "chronyc_tracking_conservative_pairwise_bound",
        "hosts": sorted(hosts),
        "max_pairwise_offset_ms": pairwise_bound_s * 1000.0,
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sample")
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("path", type=Path)
    summary_parser.add_argument("--expected-hosts", type=int, required=True)
    args = parser.parse_args()
    result = sample() if args.command == "sample" else summarize(args.path, args.expected_hosts)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
