# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

from relax.utils.metrics.timeline_trace import TimelineTraceAdapter


def test_timeline_dump_does_not_exhaust_quota_on_repeated_step(tmp_path):
    adapter = TimelineTraceAdapter(str(tmp_path), max_dump=2)
    adapter.add_event_dicts([{"name": "first", "ts": 1}])

    for _ in range(32):
        adapter.dump(0)

    adapter.add_event_dicts([{"name": "second", "ts": 2}])
    adapter.dump(1)
    payload = json.loads((tmp_path / "timeline_step_1.json").read_text())

    assert {event["name"] for event in payload} == {"second"}
    assert adapter._dump_cnt == 2


def test_timeline_repeated_step_writes_incremental_shards(tmp_path):
    adapter = TimelineTraceAdapter(str(tmp_path))
    adapter.add_event_dicts([{"name": "rollout", "ts": 1}])
    adapter.dump(3)
    adapter.add_event_dicts([{"name": "trainer", "ts": 2}])
    adapter.dump(3)

    first = json.loads((tmp_path / "timeline_step_3.json").read_text())
    second = json.loads((tmp_path / "timeline_step_3_part_1.json").read_text())

    assert [event["name"] for event in first] == ["rollout"]
    assert [event["name"] for event in second] == ["trainer"]


def test_timeline_quota_drops_rejected_step_events(tmp_path):
    adapter = TimelineTraceAdapter(str(tmp_path), max_dump=1)
    adapter.add_event_dicts([{"name": "kept", "ts": 1}])
    adapter.dump(0)
    adapter.add_event_dicts([{"name": "rejected", "ts": 2}])
    adapter.dump(1)
    adapter.add_event_dicts([{"name": "later", "ts": 3}])
    adapter.dump(0)

    second = json.loads((tmp_path / "timeline_step_0_part_1.json").read_text())

    assert [event["name"] for event in second] == ["later"]
    assert not (tmp_path / "timeline_step_1.json").exists()
