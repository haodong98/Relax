# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path

import pytest

from examples.androidlab_agentic.app import client, node_broker
from examples.androidlab_agentic.app.evaluator import EvaluationInfraError, EvaluationOutcome, outcome_from_result
from examples.androidlab_agentic.app.protocol import (
    ActionKind,
    ActionValidationError,
    normalized_to_pixel,
    parse_action,
)
from examples.androidlab_agentic.scripts.build_dataset import build_dataset


ANDROIDLAB_REPO = (
    Path(os.environ["ANDROIDLAB_REPO_DIR"])
    if os.environ.get("ANDROIDLAB_REPO_DIR")
    else Path("/iopsstor/scratch/cscs/${USER}/Multimodality-RL/AndroidLab/Android-Lab")
)
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def test_androidlab_dataset_is_exact_deterministic_and_policy_safe(tmp_path: Path) -> None:
    if not ANDROIDLAB_REPO.exists():
        pytest.skip("AndroidLab checkout is unavailable")
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build_dataset(ANDROIDLAB_REPO, first)
    build_dataset(ANDROIDLAB_REPO, second)
    assert manifest["counts"] == {"benchmark_all": 138, "operation": 93, "query": 45}
    assert manifest["source_metric_type_anomalies"] == [{"source_metric_type": "operations", "task_id": "map_14"}]
    for filename in ("benchmark_all.jsonl", "operation.jsonl", "query.jsonl", "trusted_registry.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    for row in map(json.loads, (first / "benchmark_all.jsonl").read_text(encoding="utf-8").splitlines()):
        serialized = json.dumps(row)
        assert "metric_module" not in serialized
        assert "adb_query" not in serialized
        assert "package" not in serialized


@pytest.mark.parametrize(
    "text,kind",
    [
        ('<action>{"type":"tap","x":0.5,"y":1}</action>', ActionKind.TAP),
        ('<action>{"type":"long_press","x":0,"y":1,"duration_ms":800}</action>', ActionKind.LONG_PRESS),
        ('<action>{"type":"swipe","x1":0,"y1":0,"x2":1,"y2":1,"duration_ms":400}</action>', ActionKind.SWIPE),
        ('<action>{"type":"type","text":"hello"}</action>', ActionKind.TYPE),
        ('<action>{"type":"wait","seconds":1}</action>', ActionKind.WAIT),
        ('<action>{"type":"launch"}</action>', ActionKind.LAUNCH),
        ('<action>{"type":"done","answer":"42"}</action>', ActionKind.DONE),
        ('<action>{"type":"fail"}</action>', ActionKind.FAIL),
    ],
)
def test_androidlab_action_protocol_accepts_typed_actions(text: str, kind: ActionKind) -> None:
    assert parse_action(text).kind is kind


@pytest.mark.parametrize(
    "text",
    [
        'prose <action>{"type":"done"}</action>',
        '<action>{"type":"tap","x":NaN,"y":0}</action>',
        '<action>{"type":"tap","x":0,"x":1,"y":0}</action>',
        '<action>{"type":"tap","x":-0.1,"y":0}</action>',
        '<action>{"type":"launch","package":"com.evil"}</action>',
        '<action>{"type":"shell","command":"adb root"}</action>',
        '<action>{"type":"wait","seconds":0}</action>',
        '<action>{"type":"done","answer":true}</action>',
    ],
)
def test_androidlab_action_protocol_rejects_ambiguous_or_unsafe_actions(text: str) -> None:
    with pytest.raises(ActionValidationError):
        parse_action(text)


def test_androidlab_reward_preserves_valid_zero_and_rejects_invalid_metric() -> None:
    outcome = outcome_from_result({"complete": False, "judge_page": True, "subgoal": False})
    assert outcome.score == 0.0
    assert outcome.partial_subgoals == {"judge_page": True, "subgoal": False}
    with pytest.raises(EvaluationInfraError, match="metric_missing_complete"):
        outcome_from_result({"judge_page": True})


def test_androidlab_client_preserves_append_only_history_and_scalar_reward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("x" * 64, encoding="utf-8")
    monkeypatch.setenv("ANDROIDLAB_BROKER_MANIFEST_DIR", str(tmp_path))
    monkeypatch.setenv("ANDROIDLAB_BROKER_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("RELAX_SESSION_ID", "session-token")
    monkeypatch.setenv("RELAX_BASE_URL", "http://relax/agentic_api")
    monkeypatch.setenv("ANDROIDLAB_MAX_STEPS", "2")
    monkeypatch.setattr(
        client,
        "acquire_broker",
        lambda *args, **kwargs: (
            "http://broker",
            {"lease_id": "lease", "generation": 1, "screenshot": PNG_B64, "width": 1, "height": 1},
        ),
    )
    histories: list[list[dict]] = []
    actions: list[str] = []

    def fake_post(url: str, payload: dict, *, token: str, timeout: float) -> dict:
        if url.endswith("/chat/completions"):
            histories.append(copy.deepcopy(payload["messages"]))
            content = (
                '<action>{"type":"wait","seconds":1}</action>'
                if len(histories) == 1
                else '<action>{"type":"done"}</action>'
            )
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}
        if url.endswith("/v1/action"):
            actions.append(payload["action"]["type"])
            return {"screenshot": PNG_B64}
        if url.endswith("/v1/evaluate"):
            return {"score": 0.0, "reason": "native_complete", "partial_subgoals": {}}
        if url.endswith("/v1/release"):
            return {"released": True}
        raise AssertionError(url)

    monkeypatch.setattr(client, "_post_json", fake_post)
    output = client.run_episode(
        {
            "messages": [{"role": "user", "content": "Open settings"}],
            "metadata": {
                "app": "Settings",
                "environment": "androidlab",
                "task_id": "setting_21",
                "task_manifest_digest": "digest",
            },
        }
    )
    assert output["reward"] == 0.0
    assert output["metadata"]["termination_reason"] == "done"
    assert actions == ["wait", "done"]
    assert len(histories[0]) == 2
    assert len(histories[1]) == 4
    assert histories[1][:2] == histories[0]
    assert "digest" not in json.dumps(histories)


def test_androidlab_broker_keeps_valid_zero_distinct_from_lifecycle_failure(tmp_path: Path) -> None:
    class FakeLifecycle:
        def __init__(self) -> None:
            self.cleanups = 0

        def start(self, _lease_id: str) -> str:
            return "emulator-5554"

        def cleanup(self) -> None:
            self.cleanups += 1

    class FakeEnvironment:
        def __init__(self, **kwargs) -> None:
            self.width = 1
            self.height = 1
            self.actions: list[ActionKind] = []

        def screenshot(self) -> bytes:
            return base64.b64decode(PNG_B64)

        def execute(self, action) -> bytes:
            self.actions.append(action.kind)
            return self.screenshot()

        def evaluate(self, *, terminal_answer: str | None, judge) -> EvaluationOutcome:
            assert terminal_answer is None
            assert judge is None
            return EvaluationOutcome(score=0.0, reason="native_complete", partial_subgoals={})

        def write_trace(self) -> Path:
            path = tmp_path / "trace.json"
            path.write_text("[]", encoding="utf-8")
            return path

    lifecycle = FakeLifecycle()
    state = node_broker.BrokerState(
        lifecycle=lifecycle,
        registry={"setting_21": {"task_manifest_digest": "digest", "instruction": "Open settings", "package": "pkg"}},
        adb="adb",
        androidlab_repo=ANDROIDLAB_REPO,
        work_root=tmp_path / "work",
        lease_ttl=60,
        query_judge=None,
        environment_factory=FakeEnvironment,
    )
    lease = state.lease({"request_id": "request", "task_id": "setting_21", "task_manifest_digest": "digest"})
    fields = {"lease_id": lease["lease_id"], "generation": lease["generation"]}
    state.action({**fields, "action": {"type": "done"}})
    assert state.evaluate(fields)["score"] == 0.0
    assert state.release(fields) == {"released": True}
    assert lifecycle.cleanups == 1
    with pytest.raises(ValueError, match="unknown_or_stale_task"):
        state.lease({"request_id": "request", "task_id": "setting_21", "task_manifest_digest": "other"})


def test_androidlab_coordinate_mapping_is_deterministic() -> None:
    assert [normalized_to_pixel(value, 1080) for value in (0.0, 0.5, 1.0)] == [0, 540, 1079]


def test_androidlab_four_node_recipe_preserves_async_and_budget_contract() -> None:
    run_script = Path("scripts/training/multimodal/run-qwen3-vl-8b-androidlab-4node-async.sh").read_text(
        encoding="utf-8"
    )
    wrapper = Path("examples/androidlab_agentic/submit_androidlab_4node.sh").read_text(encoding="utf-8")
    assert "--reward-key" not in run_script
    assert '--multimodal-keys \'{"image":"images"}\'' in run_script
    assert "--fully-async --max-staleness 0 --num-iters-per-train-update 1" in run_script
    assert '"actor":[1,4],"rollout":[1,12]' in run_script
    assert "ANDROIDLAB_BROKER_MANIFEST_DIR" in run_script
    assert "#SBATCH --nodes=4" in wrapper
    assert "#SBATCH --gpus-per-node=4" in wrapper
    assert "#SBATCH --time=02:00:00" in wrapper
    assert "requires exactly four nodes" in wrapper
    assert "start_node_broker.sh" in wrapper
