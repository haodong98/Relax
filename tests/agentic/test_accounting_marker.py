# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from argparse import Namespace

from relax.utils.training.train_dump_utils import append_agentic_accounting_marker


def test_agentic_accounting_marker_persists_drain_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_DUAL_JUDGE_MARKER_DIR", str(tmp_path))
    args = Namespace(rm_type="dual-agentic-judge", judge_services=object())
    snapshot = {
        "reward_inflight_sample_rewards": 0,
        "transfer_tasks": 0,
        "runtime_slots": 0,
    }

    append_agentic_accounting_marker(args, step=2, snapshot=snapshot)

    row = json.loads((tmp_path / "agentic_accounting_end.jsonl").read_text(encoding="utf-8"))
    assert row["event"] == "agentic_accounting_end"
    assert row["step"] == 2
    assert row["snapshot"] == snapshot
