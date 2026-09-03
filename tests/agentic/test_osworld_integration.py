# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

import pytest

from examples.osworld_agentic.app import node_broker
from examples.osworld_agentic.app.protocol import parse_action, render_action
from examples.osworld_agentic.scripts.build_dataset import build_dataset


OSWORLD_REPO = Path("/iopsstor/scratch/cscs/${USER}/Multimodality-RL/osworldv2/external/OSWorld")


@pytest.mark.skipif(not OSWORLD_REPO.exists(), reason="OSWorld checkout is unavailable")
def test_osworld_dataset_is_deterministic_and_policy_safe(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    build_dataset(OSWORLD_REPO, first)
    build_dataset(OSWORLD_REPO, second)
    assert (first / "dataset_manifest.json").read_bytes() == (second / "dataset_manifest.json").read_bytes()
    row = json.loads((first / "smoke_train.jsonl").read_text(encoding="utf-8"))
    assert set(row) == {"input", "metadata"}
    assert "task_config" not in row["metadata"]
    registry = json.loads((first / "trusted_registry.json").read_text(encoding="utf-8"))
    assert len(registry["tasks"]) == 369


def test_osworld_protocol_rejects_prose_and_maps_coordinates() -> None:
    with pytest.raises(ValueError):
        parse_action('click <action>{"type":"done"}</action>')
    assert render_action(parse_action('<action>{"type":"click","x":0.5,"y":0.25}</action>'), 1920, 1080) == {
        "type": "click",
        "x": 960,
        "y": 270,
    }


def test_osworld_protocol_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValueError, match="invalid_coordinate"):
        parse_action('<action>{"type":"click","x":1.1,"y":0.5}</action>')


def test_osworld_protocol_accepts_pyautogui_key_names() -> None:
    # pyautogui uses pagedown/pageup (no space); these must not be rejected.
    assert parse_action('<action>{"type":"press","key":"pagedown"}</action>') == {"type": "press", "key": "pagedown"}
    with pytest.raises(ValueError, match="invalid_key"):
        parse_action('<action>{"type":"press","key":"page down"}</action>')


def test_osworld_protocol_rejects_zero_and_out_of_range_scroll() -> None:
    with pytest.raises(ValueError, match="invalid_scroll"):
        parse_action('<action>{"type":"scroll","x":0.5,"y":0.5,"dx":0,"dy":0}</action>')
    with pytest.raises(ValueError, match="invalid_scroll"):
        parse_action('<action>{"type":"scroll","x":0.5,"y":0.5,"dx":99,"dy":1}</action>')


PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def _make_broker(tmp_path: Path) -> node_broker.Broker:
    registry = tmp_path / "trusted_registry.json"
    registry.write_text(
        json.dumps(
            {
                "tasks": {
                    "task-1": {
                        "task_manifest_digest": "digest",
                        "task_config": {"id": "task-1", "instruction": "do it", "evaluator": {}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    token_file = tmp_path / "broker.token"
    token_file.write_text("token", encoding="utf-8")
    args = argparse.Namespace(
        registry=registry,
        token_file=token_file,
        manifest_dir=tmp_path / "brokers",
        lease_root=tmp_path / "leases",
        start_command=["true"],
        stop_command=["true"],
        evaluate_command=["true"],
        width=1920,
        height=1080,
    )
    return node_broker.Broker(args)


def test_osworld_broker_terminal_action_skips_screenshot_and_release_stops_vm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = _make_broker(tmp_path)
    stops: list[str] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        if command == ["true"] and kwargs.get("env", {}).get("OSWORLD_TASK_CONFIG"):
            return subprocess.CompletedProcess(command, 0, stdout='{"server_url":"http://vm"}', stderr="")
        if kwargs.get("env", {}).get("OSWORLD_LEASE_ID"):
            stops.append(kwargs["env"]["OSWORLD_LEASE_ID"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(broker, "_request", lambda *a, **k: base64.b64decode(PNG_B64))

    lease = broker.lease_task({"request_id": "r", "task_id": "task-1", "task_manifest_digest": "digest"})
    fields = {"lease_id": lease["lease_id"], "generation": lease["generation"]}
    done = broker.action({**fields, "action": {"type": "done"}})
    assert done["terminal"] is True and done["screenshot"] is None
    assert broker.release(fields) == {"released": True}
    assert stops == [lease["lease_id"]]
    with pytest.raises(ValueError, match="stale_or_unknown_lease"):
        broker.action({**fields, "action": {"type": "wait"}})


def test_osworld_broker_generation_increments_across_leases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _make_broker(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout='{"server_url":"http://vm"}', stderr=""),
    )
    monkeypatch.setattr(broker, "_request", lambda *a, **k: base64.b64decode(PNG_B64))
    first = broker.lease_task({"request_id": "r", "task_id": "task-1", "task_manifest_digest": "digest"})
    broker.release({"lease_id": first["lease_id"], "generation": first["generation"]})
    second = broker.lease_task({"request_id": "r", "task_id": "task-1", "task_manifest_digest": "digest"})
    assert second["generation"] == first["generation"] + 1


def test_osworld_broker_start_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _make_broker(tmp_path)
    stops: list[str] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        env = kwargs.get("env", {})
        if env.get("OSWORLD_TASK_CONFIG"):
            raise subprocess.CalledProcessError(1, command)
        if env.get("OSWORLD_LEASE_ID"):
            stops.append(env["OSWORLD_LEASE_ID"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="lifecycle start failed"):
        broker.lease_task({"request_id": "r", "task_id": "task-1", "task_manifest_digest": "digest"})
    assert stops != []  # stop command invoked to clean up the half-started lease
    assert broker.lease is None
