# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Relax managed-command client for one AndroidLab environment episode."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import signal
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .node_broker import atomic_write_json
from .protocol import ACTION_SYSTEM_PROMPT, ActionValidationError, parse_action


PNG_SIGNATURE_B64 = "iVBORw0KGgo"


class RemoteError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _post_json(url: str, payload: dict[str, Any], *, token: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
            if response.status != 200 or not isinstance(result, dict):
                raise RemoteError(f"unexpected HTTP status {response.status}")
            return result
    except urllib.error.HTTPError as exc:
        raise RemoteError(f"HTTP {exc.code} from AndroidLab broker", status=exc.code) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RemoteError("AndroidLab broker unavailable") from exc


def _instruction(session_input: dict[str, Any]) -> str:
    messages = session_input.get("messages")
    if not isinstance(messages, list):
        messages = session_input.get("input")
    if not isinstance(messages, list):
        raise ValueError("session input must contain messages/input")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
            if message["content"].strip():
                return message["content"].strip()
    raise ValueError("session input has no user instruction")


def _metadata(session_input: dict[str, Any]) -> dict[str, Any]:
    metadata = session_input.get("metadata")
    required = ("environment", "task_id", "task_manifest_digest")
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(key), str) or not metadata[key] for key in required
    ):
        raise ValueError("session input metadata is incomplete")
    if metadata["environment"] != "androidlab":
        raise ValueError("session input is not an AndroidLab task")
    return metadata


def _png_dimensions(screenshot: str) -> tuple[int, int]:
    if not isinstance(screenshot, str) or not screenshot.startswith(PNG_SIGNATURE_B64):
        raise RuntimeError("broker returned invalid PNG screenshot")
    try:
        payload = base64.b64decode(screenshot, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("broker returned invalid PNG screenshot") from exc
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise RuntimeError("broker returned invalid PNG screenshot")
    width, height = int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")
    if width <= 0 or height <= 0:
        raise RuntimeError("broker returned invalid screenshot dimensions")
    return width, height


def _image_part(screenshot: str) -> dict[str, Any]:
    _png_dimensions(screenshot)
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}


def initial_messages(instruction: str, screenshot: str, *, width: int, height: int) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Task:\n{instruction}\n\nObservation 0. Screen: {width}x{height}."},
                _image_part(screenshot),
            ],
        },
    ]


def observation_message(screenshot: str, *, turn: int, status: str, width: int, height: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Observation {turn}. Previous action: {status}. Screen: {width}x{height}."},
            _image_part(screenshot),
        ],
    }


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("chat response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("chat response has no assistant message")
    return {key: value for key, value in message.items() if value is not None}


def _load_brokers(manifest_dir: Path) -> list[str]:
    urls: list[str] = []
    for path in sorted(manifest_dir.glob("broker-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "androidlab.broker_manifest.v1" and isinstance(
            payload.get("broker_url"), str
        ):
            urls.append(payload["broker_url"].rstrip("/"))
    return urls


def acquire_broker(
    manifest_dir: Path,
    *,
    token: str,
    task_id: str,
    task_manifest_digest: str,
    request_id: str,
    timeout: float,
    on_attempt: Callable[[str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    randomizer = random.Random(os.environ.get("RELAX_SESSION_ID", ""))
    while time.monotonic() < deadline:
        brokers = _load_brokers(manifest_dir)
        randomizer.shuffle(brokers)
        for broker in brokers:
            if on_attempt is not None:
                on_attempt(broker)
            try:
                lease = _post_json(
                    broker + "/v1/lease",
                    {"request_id": request_id, "task_id": task_id, "task_manifest_digest": task_manifest_digest},
                    token=token,
                    timeout=min(360.0, max(1.0, deadline - time.monotonic())),
                )
                return broker, lease
            except RemoteError as exc:
                if exc.status in (400, 401, 403):
                    raise
                continue
        time.sleep(1.0)
    raise TimeoutError("timed out waiting for an AndroidLab broker lease")


def run_episode(session_input: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(session_input)
    instruction = _instruction(session_input)
    manifest_dir = Path(os.environ["ANDROIDLAB_BROKER_MANIFEST_DIR"])
    token = Path(os.environ["ANDROIDLAB_BROKER_TOKEN_FILE"]).read_text(encoding="utf-8").strip()
    session_id = os.environ["RELAX_SESSION_ID"]
    relax_base_url = os.environ["RELAX_BASE_URL"].rstrip("/")
    max_steps = int(os.environ.get("ANDROIDLAB_MAX_STEPS", "12"))
    lease_wait = float(os.environ.get("ANDROIDLAB_LEASE_WAIT_S", "1200"))
    chat_timeout = float(os.environ.get("ANDROIDLAB_CHAT_TIMEOUT_S", "1800"))
    if not 1 <= max_steps <= 32:
        raise ValueError("ANDROIDLAB_MAX_STEPS must be in [1,32]")

    broker: str | None = None
    broker_attempt: str | None = None
    lease: dict[str, Any] | None = None
    request_id = uuid.uuid4().hex
    steps = 0
    termination_reason = "max_steps"

    def set_broker_attempt(url: str) -> None:
        nonlocal broker_attempt
        broker_attempt = url

    try:
        broker, lease = acquire_broker(
            manifest_dir,
            token=token,
            task_id=metadata["task_id"],
            task_manifest_digest=metadata["task_manifest_digest"],
            request_id=request_id,
            timeout=lease_wait,
            on_attempt=set_broker_attempt,
        )
        width, height = int(lease["width"]), int(lease["height"])
        screenshot = lease["screenshot"]
        if _png_dimensions(screenshot) != (width, height):
            raise RuntimeError("broker screenshot dimensions do not match lease")
        messages = initial_messages(instruction, screenshot, width=width, height=height)
        lease_fields = {"generation": lease["generation"], "lease_id": lease["lease_id"]}
        for turn in range(1, max_steps + 1):
            response = _post_json(
                relax_base_url + "/chat/completions",
                {"messages": messages, "model": "relax-policy", "n": 1, "stream": False},
                token=session_id,
                timeout=chat_timeout,
            )
            assistant = _assistant_message(response)
            messages.append(assistant)
            steps = turn
            try:
                action = parse_action(assistant.get("content"))
            except ActionValidationError as exc:
                if turn < max_steps:
                    messages.append(
                        observation_message(
                            screenshot, turn=turn, status=f"rejected:{exc.code}", width=width, height=height
                        )
                    )
                continue
            result = _post_json(
                broker + "/v1/action",
                {**lease_fields, "action": {"type": action.kind.value, **action.arguments}},
                token=token,
                timeout=180.0,
            )
            screenshot = result["screenshot"]
            if _png_dimensions(screenshot) != (width, height):
                raise RuntimeError("broker screenshot dimensions changed during episode")
            if action.terminal:
                termination_reason = action.kind.value
                break
            if turn < max_steps:
                messages.append(observation_message(screenshot, turn=turn, status="ok", width=width, height=height))
        evaluation = _post_json(broker + "/v1/evaluate", lease_fields, token=token, timeout=300.0)
        score = evaluation.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RuntimeError("broker returned non-numeric reward")
        reward = float(score)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise RuntimeError("broker returned invalid reward")
        return {
            "metadata": {
                "app": metadata.get("app"),
                "environment": "androidlab",
                "evaluator_status": "valid",
                "partial_subgoals": evaluation.get("partial_subgoals", {}),
                "reward_reason": evaluation.get("reason"),
                "steps": steps,
                "task_id": metadata["task_id"],
                "termination_reason": termination_reason,
            },
            "reward": reward,
        }
    finally:
        if broker is not None and lease is not None:
            _post_json(
                broker + "/v1/release",
                {"generation": lease["generation"], "lease_id": lease["lease_id"]},
                token=token,
                timeout=240.0,
            )
        elif broker_attempt is not None:
            try:
                _post_json(broker_attempt + "/v1/cancel", {"request_id": request_id}, token=token, timeout=360.0)
            except RemoteError:
                pass


def main() -> None:
    def terminate(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"managed AndroidLab client received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output = run_episode(json.loads(args.input_json.read_text(encoding="utf-8")))
    atomic_write_json(args.output_json, output)


if __name__ == "__main__":
    main()
