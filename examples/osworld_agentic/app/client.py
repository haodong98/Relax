# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Managed-command OSWorld episode driver."""

from __future__ import annotations

import json
import os
import random
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .protocol import ACTION_SYSTEM_PROMPT, parse_action


def _post(url: str, payload: dict[str, Any], token: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError("broker response is not an object")
    return result


def _assistant(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("chat response must contain one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("chat response has no assistant content")
    return message["content"]


def run_episode(session_input: dict[str, Any]) -> dict[str, Any]:
    metadata = session_input.get("metadata")
    messages = session_input.get("messages", session_input.get("input"))
    if not isinstance(metadata, dict) or metadata.get("environment") != "osworld":
        raise ValueError("invalid OSWorld metadata")
    if not isinstance(messages, list):
        raise ValueError("session input has no messages")
    instruction = next((m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("session input has no instruction")
    manifest_dir = Path(os.environ["OSWORLD_BROKER_MANIFEST_DIR"])
    token = Path(os.environ["OSWORLD_BROKER_TOKEN_FILE"]).read_text(encoding="utf-8").strip()
    broker_files = sorted(manifest_dir.glob("broker-*.json"))
    if not broker_files:
        raise RuntimeError("no OSWorld broker manifest")
    brokers = [json.loads(path.read_text())["broker_url"].rstrip("/") for path in broker_files]
    random.Random(os.environ.get("RELAX_SESSION_ID", "")).shuffle(brokers)
    lease = None
    broker = None
    for candidate in brokers:
        try:
            broker = candidate
            lease = _post(
                candidate + "/v1/lease",
                {
                    "request_id": uuid.uuid4().hex,
                    "task_id": metadata["task_id"],
                    "task_manifest_digest": metadata["task_manifest_digest"],
                },
                token,
                600,
            )
            break
        except Exception:
            continue
    if lease is None or broker is None:
        raise RuntimeError("unable to acquire OSWorld lease")
    fields = {"generation": lease["generation"], "lease_id": lease["lease_id"]}
    try:
        screenshot = lease["screenshot"]
        history = [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}},
                ],
            },
        ]
        for step in range(1, int(os.environ.get("OSWORLD_MAX_STEPS", "8")) + 1):
            response = _post(
                os.environ["RELAX_BASE_URL"].rstrip("/") + "/chat/completions",
                {"messages": history, "model": "relax-policy", "n": 1, "stream": False},
                os.environ["RELAX_SESSION_ID"],
                900,
            )
            action_text = _assistant(response)
            action = parse_action(action_text)
            result = _post(broker + "/v1/action", {**fields, "action": action}, token, 300)
            history.append({"role": "assistant", "content": action_text})
            if result.get("terminal"):
                break
            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Observation {step}: {result.get('status', 'ok')}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{result['screenshot']}"},
                        },
                    ],
                }
            )
        else:
            # Model exhausted max_steps without done/fail: mark the terminal state so the
            # native evaluator sees a deliberate end-of-episode rather than an open VM.
            _post(broker + "/v1/action", {**fields, "action": {"type": "fail"}}, token, 300)
        outcome = _post(broker + "/v1/evaluate", fields, token, 600)
        return {
            "metadata": {"environment": "osworld", "task_id": metadata["task_id"], "steps": step},
            "reward": float(outcome["score"]),
        }
    finally:
        _post(broker + "/v1/release", fields, token, 300)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output = run_episode(json.loads(args.input_json.read_text(encoding="utf-8")))
    args.output_json.write_text(json.dumps(output, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
