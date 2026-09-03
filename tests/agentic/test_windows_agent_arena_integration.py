# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import base64
import copy
import json
import os
import threading
from argparse import Namespace
from pathlib import Path

import pytest

from examples.windows_agent_arena_agentic.app import client, node_broker, notepad_smoke_env
from examples.windows_agent_arena_agentic.app.evaluator import (
    EvaluationInfraError,
    StrictTerminalEvaluator,
)
from examples.windows_agent_arena_agentic.app.protocol import (
    ActionKind,
    ActionValidationError,
    normalized_to_pixel,
    parse_action,
    render_pyautogui,
)
from examples.windows_agent_arena_agentic.scripts import verify_formal_run
from examples.windows_agent_arena_agentic.scripts.build_dataset import build_dataset
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data


WAA_REPO = Path(os.environ["WAA_REPO_DIR"]) if os.environ.get("WAA_REPO_DIR") else None
EXPECTED_ASSIGNMENT_DIGEST = "5cf5a23d34c071720161b4181e735378a20de4b9e2cfacd4a8cce6ea155088b9"
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


class _FakeResponse:
    def __init__(self, status_code: int, *, content: bytes = b"", text: str = "", payload: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.text = text
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _FakeWAARequests:
    def __init__(self, file_response: _FakeResponse):
        self.file_response = file_response
        self.posts: list[tuple[str, dict | None, dict | None]] = []

    def get(self, url: str, *, timeout: float) -> _FakeResponse:
        assert url.endswith("/screenshot")
        assert timeout == 30
        return _FakeResponse(200, content=base64.b64decode(PNG_B64))

    def post(
        self,
        url: str,
        *,
        json: dict | None = None,
        data: dict | None = None,
        timeout: float,
    ) -> _FakeResponse:
        self.posts.append((url, json, data))
        if url.endswith("/execute"):
            return _FakeResponse(200, payload={"status": "success", "returncode": 0})
        if url.endswith("/setup/open_file") or url.endswith("/setup/activate_window"):
            return _FakeResponse(200)
        if url.endswith("/file"):
            return self.file_response
        raise AssertionError(url)


def _pinned_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_response: _FakeResponse,
) -> tuple[notepad_smoke_env.PinnedNotepadEnvironment, _FakeWAARequests]:
    asset_root = tmp_path / "assets"
    task_assets = asset_root / notepad_smoke_env.PINNED_NOTEPAD_TASK_ID
    task_assets.mkdir(parents=True)
    (task_assets / "draft_gold.txt").write_text("This is a draft.", encoding="utf-8")
    monkeypatch.setenv("WAA_ASSET_CACHE", str(asset_root))
    monkeypatch.setattr(notepad_smoke_env.time, "sleep", lambda _: None)
    requester = _FakeWAARequests(file_response)
    task_config = {
        "id": notepad_smoke_env.PINNED_NOTEPAD_TASK_ID,
        "config": [],
        "evaluator": copy.deepcopy(notepad_smoke_env._EXPECTED_EVALUATOR),
    }
    env = notepad_smoke_env.PinnedNotepadEnvironment(
        "http://waa",
        task_config,
        tmp_path / "lease-cache",
        requester=requester,
    )
    return env, requester


def test_waa_dataset_is_exact_deterministic_and_policy_safe(tmp_path: Path) -> None:
    if WAA_REPO is None or not WAA_REPO.exists():
        pytest.skip("set WAA_REPO_DIR to run the 154-task dataset contract")
    first, second = tmp_path / "first", tmp_path / "second"
    first_manifest = build_dataset(WAA_REPO, first)
    build_dataset(WAA_REPO, second)

    assert first_manifest["counts"] == {"train": 108, "dev": 23, "test": 23}
    assert first_manifest["assignment_digest"] == EXPECTED_ASSIGNMENT_DIGEST
    assert len(first_manifest["source_id_anomalies"]) == 6
    for filename in (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "smoke_train.jsonl",
        "trusted_registry.json",
        "capability_manifest.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    for row in (first / "train.jsonl").read_text(encoding="utf-8").splitlines():
        policy_row = json.loads(row)
        serialized = json.dumps(policy_row)
        assert "evaluator" not in serialized
        assert "expected" not in serialized
        assert "postconfig" not in serialized

    capability = json.loads((first / "capability_manifest.json").read_text(encoding="utf-8"))
    assert capability["domains"]["notepad"]["status"] == "hardware_gate_required"
    assert capability["domains"]["chrome"]["required_guest_forward_ports"] == [5000, 9222]
    assert all(status == "ingested" for status in capability["task_status"].values())


@pytest.mark.parametrize(
    "text,kind",
    [
        ('<action>{"type":"move","x":0.5,"y":1}</action>', ActionKind.MOVE),
        ('<action>{"type":"click","x":0,"y":1,"button":"right","count":2}</action>', ActionKind.CLICK),
        ('<action>{"type":"drag","x1":0,"y1":0,"x2":1,"y2":1}</action>', ActionKind.DRAG),
        ('<action>{"type":"scroll","x":0.5,"y":0.5,"dx":0,"dy":-3}</action>', ActionKind.SCROLL),
        ('<action>{"type":"type","text":"draft"}</action>', ActionKind.TYPE),
        ('<action>{"type":"press","key":"enter"}</action>', ActionKind.PRESS),
        ('<action>{"type":"hotkey","keys":["ctrl","s"]}</action>', ActionKind.HOTKEY),
        ('<action>{"type":"wait"}</action>', ActionKind.WAIT),
        ('<action>{"type":"done"}</action>', ActionKind.DONE),
        ('<action>{"type":"fail"}</action>', ActionKind.FAIL),
    ],
)
def test_action_protocol_accepts_only_typed_variants(text: str, kind: ActionKind) -> None:
    assert parse_action(text).kind is kind


@pytest.mark.parametrize(
    "text",
    [
        'prose <action>{"type":"done"}</action>',
        '<action>{"type":"done"}</action><action>{"type":"done"}</action>',
        '<action>{"type":"move","x":NaN,"y":0}</action>',
        '<action>{"type":"move","x":true,"y":0}</action>',
        '<action>{"type":"move","x":0,"x":1,"y":0}</action>',
        '<action>{"type":"wait","code":"pyautogui.click()"}</action>',
        '<action>{"type":"click","x":0,"y":0,"count":1.0}</action>',
        '<action>{"type":"COMPUTER_CODE","code":"import os"}</action>',
        '<action>{"type":"scroll","x":0,"y":0,"dx":0,"dy":0}</action>',
    ],
)
def test_action_protocol_rejects_code_ambiguity_and_nonfinite_values(text: str) -> None:
    with pytest.raises(ActionValidationError):
        parse_action(text)


def test_action_coordinate_mapping_and_render_are_deterministic() -> None:
    assert [normalized_to_pixel(value, 1440) for value in (0.0, 0.5, 1.0)] == [0, 720, 1439]
    action = parse_action('<action>{"type":"click","x":0.5,"y":0.5}</action>')
    rendered = render_pyautogui(action, width=1440, height=900)
    assert rendered == "pyautogui.click(720, 450, clicks=1, button='left', interval=0.1)"
    assert "import" not in rendered


def _strict_evaluator(config: dict, *, values: dict, postconfig=lambda _: None) -> StrictTerminalEvaluator:
    getters = {
        "expected": lambda _env, getter_config: getter_config["value"],
        "result": lambda _env, getter_config: values[getter_config["key"]],
        "missing": lambda _env, _config: (_ for _ in ()).throw(FileNotFoundError()),
        "broken": lambda _env, _config: (_ for _ in ()).throw(ConnectionError()),
    }
    metrics = {
        "identity": lambda result: result,
        "equal": lambda result, expected: float(result == expected),
        "nan": lambda result: float("nan"),
    }
    return StrictTerminalEvaluator(
        config,
        getter_resolver=lambda name: getters[name],
        metric_resolver=lambda name: metrics[name],
        postconfig=postconfig,
    )


def test_strict_evaluator_preserves_valid_zero_and_native_aggregation() -> None:
    missing = _strict_evaluator({"func": "identity", "result": {"type": "missing"}}, values={})
    missing.preflight_expected(object())
    assert missing.evaluate(object(), last_action="DONE").score == 0.0

    conjunction = _strict_evaluator(
        {
            "func": ["identity", "identity"],
            "result": [{"type": "result", "key": "one"}, {"type": "result", "key": "half"}],
            "conj": "and",
        },
        values={"one": 1.0, "half": 0.5},
    )
    conjunction.preflight_expected(object())
    assert conjunction.evaluate(object(), last_action="DONE").score == 0.75

    disjunction = _strict_evaluator(
        {
            "func": ["identity", "identity"],
            "result": [{"type": "result", "key": "zero"}, {"type": "result", "key": "half"}],
            "conj": "or",
        },
        values={"zero": 0.0, "half": 0.5},
    )
    disjunction.preflight_expected(object())
    assert disjunction.evaluate(object(), last_action="DONE").score == 0.5


def test_strict_evaluator_fail_and_infra_contracts() -> None:
    infeasible = _strict_evaluator({"func": "infeasible"}, values={})
    infeasible.preflight_expected(object())
    assert infeasible.evaluate(object(), last_action="FAIL").score == 1.0
    assert infeasible.evaluate(object(), last_action="DONE").score == 0.0

    broken = _strict_evaluator({"func": "identity", "result": {"type": "broken"}}, values={})
    broken.preflight_expected(object())
    with pytest.raises(EvaluationInfraError, match="result_unavailable"):
        broken.evaluate(object(), last_action="DONE")

    invalid_metric = _strict_evaluator(
        {"func": "nan", "result": {"type": "result", "key": "one"}}, values={"one": 1.0}
    )
    invalid_metric.preflight_expected(object())
    with pytest.raises(EvaluationInfraError, match="metric_out_of_range"):
        invalid_metric.evaluate(object(), last_action="DONE")


@pytest.mark.parametrize(
    "file_response,expected_score",
    [
        (_FakeResponse(404, text="File not found"), 0.0),
        (_FakeResponse(200, content=b"wrong"), 0.0),
        (_FakeResponse(200, content=b"This is a draft."), 1.0),
    ],
)
def test_pinned_notepad_preserves_native_reward_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    file_response: _FakeResponse,
    expected_score: float,
) -> None:
    env, _ = _pinned_env(monkeypatch, tmp_path, file_response)
    assert env.reset().startswith(b"\x89PNG")
    assert env.evaluate().score == expected_score


def test_pinned_notepad_renders_only_validated_actions_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env, requester = _pinned_env(monkeypatch, tmp_path, _FakeResponse(500, text="server error"))
    env.execute(parse_action('<action>{"type":"wait"}</action>'))
    execute_payload = next(payload for url, payload, _ in requester.posts if url.endswith("/execute"))
    assert execute_payload is not None
    assert execute_payload["shell"] is False
    assert execute_payload["command"] == [
        "python",
        "-c",
        "import time,pyautogui; pyautogui.FAILSAFE=False; time.sleep(1.0)",
    ]
    with pytest.raises(EvaluationInfraError, match="result_unavailable"):
        env.evaluate()


def test_agent_uses_full_append_only_history_and_terminal_reward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("x" * 64, encoding="utf-8")
    monkeypatch.setenv("WAA_BROKER_MANIFEST_DIR", str(tmp_path))
    monkeypatch.setenv("WAA_BROKER_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("RELAX_SESSION_ID", "session-token")
    monkeypatch.setenv("RELAX_BASE_URL", "http://relax/agentic_api")
    monkeypatch.setenv("WAA_MAX_STEPS", "2")
    monkeypatch.setattr(
        client,
        "acquire_broker",
        lambda *args, **kwargs: (
            "http://broker",
            {"lease_id": "lease", "generation": 1, "screenshot": PNG_B64, "width": 1, "height": 1},
        ),
    )
    chat_histories: list[list[dict]] = []
    action_count = 0

    def fake_post(url: str, payload: dict, *, token: str, timeout: float) -> dict:
        nonlocal action_count
        if url.endswith("/chat/completions"):
            chat_histories.append(copy.deepcopy(payload["messages"]))
            content = (
                '<action>{"type":"wait"}</action>' if len(chat_histories) == 1 else '<action>{"type":"done"}</action>'
            )
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}
        if url.endswith("/v1/action"):
            action_count += 1
            return {"screenshot": PNG_B64, "terminal": payload["action"]["type"] == "done"}
        if url.endswith("/v1/evaluate"):
            return {"score": 0.0, "reason": "and_short_circuit"}
        if url.endswith("/v1/release"):
            return {"released": True}
        raise AssertionError(url)

    monkeypatch.setattr(client, "_post_json", fake_post)
    session_input = {
        "messages": [{"role": "user", "content": "Create draft.txt"}],
        "metadata": {"domain": "notepad", "task_id": "task", "task_manifest_digest": "digest"},
    }
    output = client.run_episode(session_input)
    assert output["reward"] == 0.0
    assert output["metadata"]["termination_reason"] == "done"
    assert action_count == 2
    assert len(chat_histories[0]) == 2
    assert len(chat_histories[1]) == 4
    assert chat_histories[1][:2] == chat_histories[0]
    assert "digest" not in json.dumps(chat_histories)


def test_waa_training_script_transfers_scalar_reward_and_multimodal_inputs() -> None:
    script = Path("scripts/training/multimodal/run-qwen3-vl-2b-waa-4node-async.sh").read_text(encoding="utf-8")
    assert "--reward-key" not in script
    assert '--multimodal-keys \'{"image":"images"}\'' in script
    assert "--fully-async" in script
    assert "--max-staleness 0" in script
    assert 'if [ "${num_rollout}" -ne 3 ]' in script

    args = Namespace(
        custom_reward_post_process_path=None,
        reward_key=None,
        agentic_custom_advantage_path=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        multimodal_keys={"image": "images"},
        use_opd=False,
        debug_train_only=True,
    )
    sample = Sample(
        index=0,
        tokens=[1, 2],
        response_length=1,
        reward=0.5,
        multimodal_train_inputs={"pixel_values": "sentinel"},
    )
    train_data = convert_samples_to_train_data(args, [sample])
    assert train_data["rewards"] == [0.5]
    assert train_data["multimodal_train_inputs"] == [{"pixel_values": "sentinel"}]

    model_config = Path("scripts/models/qwen3-vl-2B-instruct.sh").read_text(encoding="utf-8")
    assert "--rotary-base 5000000" in model_config
    assert "--untie-embeddings-and-output-weights" not in model_config

    wrapper = Path("examples/windows_agent_arena_agentic/submit_waa_4node.sh").read_text(encoding="utf-8")
    assert "unset RAY_NO_WAIT" in wrapper
    assert '--cpus-per-task="${train_cpus_per_task}"' in wrapper
    assert "EXP_DIR must be a fresh path" in wrapper
    assert "preflight_training_env.py" in wrapper
    assert "enter_training_env.sh" in wrapper


@pytest.mark.parametrize("output,port", [("127.0.0.1:49152\n", 49152), ("[::1]:60000\n", 60000)])
def test_podman_dynamic_port_parser(output: str, port: int) -> None:
    assert node_broker.parse_podman_port(output) == port


def test_broker_renames_itself_before_spmd_python_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    class FakeLibc:
        def prctl(self, *args):
            calls.append(args)
            return 0

    monkeypatch.setattr(node_broker.ctypes, "CDLL", lambda *args, **kwargs: FakeLibc())
    node_broker.set_process_name()
    assert calls == [(node_broker.PR_SET_NAME, b"waa-broker", 0, 0, 0)]


def test_broker_cold_start_admission_is_nonblocking_and_cancellable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    start_entered = threading.Event()
    allow_start = threading.Event()

    class FakeLifecycle:
        def __init__(self):
            self.cleanup_count = 0

        def start(self, _lease_id: str) -> str:
            start_entered.set()
            assert allow_start.wait(timeout=2)
            return "http://waa"

        def cleanup(self) -> None:
            self.cleanup_count += 1

    class FakeEnvironment:
        def __init__(self, *_args):
            pass

        def reset(self) -> bytes:
            return base64.b64decode(PNG_B64)

    monkeypatch.setattr(node_broker, "PinnedNotepadEnvironment", FakeEnvironment)
    lifecycle = FakeLifecycle()
    digest = "digest"
    registry = {
        "tasks": {
            notepad_smoke_env.PINNED_NOTEPAD_TASK_ID: {
                "task_config": {"id": notepad_smoke_env.PINNED_NOTEPAD_TASK_ID},
                "task_manifest_digest": digest,
            }
        }
    }
    state = node_broker.BrokerState(
        lifecycle=lifecycle,
        registry=registry,
        cache_root=tmp_path / "cache",
        lease_ttl=60,
    )
    first_request = "a" * 32
    first_error: list[Exception] = []

    def first_acquire() -> None:
        try:
            state.acquire(
                {
                    "request_id": first_request,
                    "task_id": notepad_smoke_env.PINNED_NOTEPAD_TASK_ID,
                    "task_manifest_digest": digest,
                }
            )
        except Exception as exc:
            first_error.append(exc)

    first_thread = threading.Thread(target=first_acquire)
    first_thread.start()
    assert start_entered.wait(timeout=1)

    with pytest.raises(BlockingIOError, match="broker_busy"):
        state.acquire(
            {
                "request_id": "b" * 32,
                "task_id": notepad_smoke_env.PINNED_NOTEPAD_TASK_ID,
                "task_manifest_digest": digest,
            }
        )
    assert state.cancel({"request_id": first_request}) == {"cancelled": True, "cleanup_pending": True}
    allow_start.set()
    first_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert len(first_error) == 1 and "lease_cancelled" in str(first_error[0])
    assert lifecycle.cleanup_count == 1
    assert state.lease_id is None


def test_broker_cancel_during_action_defers_cleanup_without_blocking(tmp_path: Path) -> None:
    execute_entered = threading.Event()
    allow_execute = threading.Event()

    class FakeLifecycle:
        def __init__(self):
            self.cleanup_count = 0

        def cleanup(self) -> None:
            self.cleanup_count += 1

    class FakeEnvironment:
        def execute(self, _action, *, width: int, height: int) -> bytes:
            assert (width, height) == (1, 1)
            execute_entered.set()
            assert allow_execute.wait(timeout=2)
            return base64.b64decode(PNG_B64)

    lifecycle = FakeLifecycle()
    state = node_broker.BrokerState(lifecycle=lifecycle, registry={}, cache_root=tmp_path / "cache", lease_ttl=60)
    state.lease_id = "lease"
    state.request_id = "a" * 32
    state.generation = 1
    state.screen_width = 1
    state.screen_height = 1
    state.env = FakeEnvironment()
    action_error: list[Exception] = []

    def run_action() -> None:
        try:
            state.action({"lease_id": "lease", "generation": 1, "action": {"type": "wait"}})
        except Exception as exc:
            action_error.append(exc)

    action_thread = threading.Thread(target=run_action)
    action_thread.start()
    assert execute_entered.wait(timeout=1)
    assert state.cancel({"request_id": "a" * 32}) == {"cancelled": True, "cleanup_pending": True}
    assert state.lease_id == "lease"
    allow_execute.set()
    action_thread.join(timeout=2)

    assert not action_thread.is_alive()
    assert not action_error
    assert lifecycle.cleanup_count == 1
    assert state.lease_id is None


def test_formal_verifier_requires_full_four_node_three_round_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exp_dir = tmp_path / "experiment"
    checkpoint_dir = tmp_path / "checkpoint"
    hosts = [f"nid{i:06d}" for i in range(4)]
    exp_dir.mkdir()
    (exp_dir / "hosts").write_text("\n".join(hosts) + "\n", encoding="utf-8")

    preflight_dir = exp_dir / "training_env_preflight"
    cleanup_dir = exp_dir / "cleanup_audit"
    broker_dir = exp_dir / "broker_events"
    rollout_dir = exp_dir / "rollout_result" / "train"
    log_dir = exp_dir / "logs"
    timeline_dir = exp_dir / "timeline"
    for directory in (preflight_dir, cleanup_dir, broker_dir, rollout_dir, log_dir, timeline_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for host in hosts:
        (preflight_dir / f"{host}.json").write_text(
            json.dumps(
                {
                    "contract": verify_formal_run.EXPECTED_PROVIDER_CONTRACT,
                    "hostname": host,
                    "ray_version": "2.0",
                    "schema_version": "waa.training_env_preflight.v1",
                    "sglang_version": "1.0",
                }
            ),
            encoding="utf-8",
        )
        (cleanup_dir / f"{host}.json").write_text(
            json.dumps(
                {
                    "broker_manifests": [],
                    "containers": [],
                    "hostname": host,
                    "node_root_absent": True,
                    "schema_version": "waa.cleanup_audit.v1",
                    "token_absent": True,
                }
            ),
            encoding="utf-8",
        )
        events = []
        for index in range(3):
            lease_id = f"{host}-{index}"
            common = {"hostname": host, "lease_id": lease_id, "schema_version": "waa.broker_event.v1"}
            events.extend(
                [
                    {**common, "event": "lease_ready", "acquire_seconds": 30.0, "width": 800, "height": 600},
                    {**common, "event": "lease_evaluated", "score": 0.0},
                    {**common, "event": "lease_released"},
                ]
            )
        (broker_dir / f"broker-{host}.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )

    for step in range(3):
        rows = [
            {
                "group_index": step,
                "image_count": 1,
                "multimodal_train_inputs": {"image_grid_thw": "Tensor", "pixel_values": "Tensor"},
                "reward": 0.0,
                "status": "completed",
                "weight_versions": [str(step)],
            }
            for _ in range(4)
        ]
        (rollout_dir / f"{step}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    events = [
        {"name": name, "args": {"step": step}}
        for step in range(3)
        for name in (
            "critical_path.optimizer_step",
            "critical_path.weight_update",
            "critical_path.weight_serving_ready",
        )
    ]
    (timeline_dir / "timeline.json").write_text(json.dumps(events), encoding="utf-8")
    (log_dir / "driver-test.log").write_text(
        "python train.py --fully-async --max-staleness 0 --num-rollout 3\n", encoding="utf-8"
    )
    (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text("2\n", encoding="utf-8")
    monkeypatch.setattr(
        verify_formal_run,
        "_slurm_accounting",
        lambda _job_id: {
            "alloc_tres": "gres/gpu=16",
            "elapsed_seconds": 60,
            "formal_gpu_hours": 16 / 60,
            "gpu_count": 16,
            "slurm_state": "COMPLETED",
        },
    )

    report = verify_formal_run.validate(
        exp_dir,
        checkpoint_dir,
        "job",
        preliminary_gpu_hours=0.1,
        gpu_hour_budget=80,
    )
    assert report["status"] == "passed"
    assert report["optimizer_steps"] == [0, 1, 2]
    assert report["lease_count"] == 12

    events = [
        event
        for event in events
        if not (event["name"].endswith("weight_serving_ready") and event["args"]["step"] == 2)
    ]
    (timeline_dir / "timeline.json").write_text(json.dumps(events), encoding="utf-8")
    with pytest.raises(RuntimeError, match="training timeline is incomplete"):
        verify_formal_run.validate(
            exp_dir,
            checkpoint_dir,
            "job",
            preliminary_gpu_hours=0.1,
            gpu_hour_budget=80,
        )


@pytest.mark.skip(reason="requires a 4-node GH200 allocation, KVM, Podman, Windows guest, Ray, and SGLang")
def test_waa_qwen3_vl_2b_four_node_gpu_smoke() -> None:
    pass
