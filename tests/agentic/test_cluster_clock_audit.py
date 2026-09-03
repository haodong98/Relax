# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.mobilegym_agentic.scripts.audit_cluster_clock import _seconds, summarize


def test_clock_audit_uses_conservative_pairwise_bound(tmp_path):
    rows = [
        {"clock_host": "node-a", "node_offset_bound_s": 0.001},
        {"clock_host": "node-b", "node_offset_bound_s": 0.002},
        {"clock_host": "node-c", "node_offset_bound_s": 0.003},
    ]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = summarize(path, expected_hosts=3)

    assert report["max_pairwise_offset_ms"] == pytest.approx(5.0)
    assert report["hosts"] == ["node-a", "node-b", "node-c"]


def test_clock_audit_parses_chrony_tracking_seconds():
    output = "System time     : 0.000004 seconds slow of NTP time\n"
    assert _seconds(output, "System time") == pytest.approx(0.000004)
