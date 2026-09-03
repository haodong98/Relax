# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Authenticated OSWorld broker with a trusted external VM lifecycle."""

from __future__ import annotations

import argparse
import atexit
import base64
import hmac
import json
import os
import subprocess
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class Broker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.token = args.token_file.read_text(encoding="utf-8").strip()
        self.registry = _read_json(args.registry)["tasks"]
        self.lock = threading.RLock()
        self.lease: dict[str, Any] | None = None
        self._generation = 0

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> bytes:
        request = Request(
            self.lease["server_url"] + path,
            method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        with urlopen(request, timeout=300) as response:
            return response.read()

    def _start(self, task: dict[str, Any], lease_id: str) -> dict[str, Any]:
        env = dict(
            os.environ,
            OSWORLD_LEASE_ID=lease_id,
            OSWORLD_LEASE_ROOT=str(self.args.lease_root / lease_id),
            OSWORLD_TASK_CONFIG=json.dumps(task["task_config"], separators=(",", ":")),
        )
        try:
            result = subprocess.run(
                self.args.start_command, env=env, text=True, capture_output=True, check=True, timeout=900
            )
            value = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            self._stop_lease(lease_id)
            raise RuntimeError("OSWorld lifecycle start failed") from exc
        if not isinstance(value, dict) or not isinstance(value.get("server_url"), str):
            self._stop_lease(lease_id)
            raise RuntimeError('OSWorld start command must print {"server_url": ...}')
        return value

    def _stop_lease(self, lease_id: str) -> None:
        """Stop the VM for a lease without validating the active lease (safe
        for cleanup)."""
        if not self.args.stop_command:
            return
        try:
            subprocess.run(
                self.args.stop_command,
                env=dict(
                    os.environ,
                    OSWORLD_LEASE_ID=lease_id,
                    OSWORLD_LEASE_ROOT=str(self.args.lease_root / lease_id),
                ),
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def lease_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.lease is not None:
                raise RuntimeError("broker_busy")
            task_id, digest = payload.get("task_id"), payload.get("task_manifest_digest")
            task = self.registry.get(task_id)
            if not isinstance(task, dict) or task.get("task_manifest_digest") != digest:
                raise ValueError("unknown_or_stale_task")
            lease_id = uuid.uuid4().hex
            self.args.lease_root.mkdir(parents=True, exist_ok=True)
            server = self._start(task, lease_id)
            self._generation += 1
            generation = self._generation
            self.lease = {"lease_id": lease_id, "generation": generation, "task": task, **server}
            screenshot = base64.b64encode(self._request("GET", "/screenshot")).decode()
            return {
                "lease_id": lease_id,
                "generation": generation,
                "width": self.args.width,
                "height": self.args.height,
                "screenshot": screenshot,
                "terminal": False,
            }

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._check(payload)
            action = payload.get("action")
            if not isinstance(action, dict):
                raise ValueError("invalid_action")
            kind = action["type"]
            terminal = kind in {"done", "fail"}
            if not terminal and kind != "wait":
                x, y = action.get("x", 0), action.get("y", 0)
                x, y = round(float(x) * self.args.width), round(float(y) * self.args.height)
                if kind == "click":
                    command = f"import pyautogui; pyautogui.click({x}, {y})"
                elif kind == "move":
                    command = f"import pyautogui; pyautogui.moveTo({x}, {y})"
                elif kind == "type":
                    command = f"import pyautogui; pyautogui.write({action['text']!r})"
                elif kind == "press":
                    command = f"import pyautogui; pyautogui.press({action['key']!r})"
                elif kind == "scroll":
                    command = f"import pyautogui; pyautogui.scroll({action['dy']}, {x}, {y})"
                else:
                    raise ValueError("unsupported_action")
                result = json.loads(self._request("POST", "/execute", {"command": command, "shell": False}))
                if result.get("returncode", 1) != 0:
                    raise RuntimeError("OSWorld action execution failed")
            # Terminal actions skip the screenshot: the VM may already be tearing down.
            if terminal:
                return {"screenshot": None, "terminal": True, "status": kind}
            screenshot = base64.b64encode(self._request("GET", "/screenshot")).decode()
            return {"screenshot": screenshot, "terminal": False, "status": kind}

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._check(payload)
            command = self.args.evaluate_command
            if command is None:
                raise RuntimeError("OSWORLD_EVALUATE_COMMAND_JSON is required for native evaluation")
            env = dict(
                os.environ,
                OSWORLD_TASK_CONFIG=json.dumps(self.lease["task"]["task_config"]),
                OSWORLD_SERVER_URL=self.lease["server_url"],
            )
            value = json.loads(
                subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=900).stdout
            )
            score = value.get("score") if isinstance(value, dict) else None
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                raise RuntimeError("native evaluator returned invalid score")
            return {"score": float(score), "reason": "native_osworld_evaluator"}

    def _check(self, payload: dict[str, Any]) -> None:
        if (
            self.lease is None
            or payload.get("lease_id") != self.lease["lease_id"]
            or payload.get("generation") != self.lease["generation"]
        ):
            raise ValueError("stale_or_unknown_lease")

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._check(payload)
            lease_id = self.lease["lease_id"]
            self._stop_lease(lease_id)
            self.lease = None
            return {"released": True}

    def shutdown(self) -> None:
        """Best-effort cleanup at process exit; does not validate the lease."""
        lease = self.lease
        if lease is not None:
            self._stop_lease(lease["lease_id"])
            self.lease = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--start-command-json", required=True)
    parser.add_argument("--stop-command-json")
    parser.add_argument("--evaluate-command-json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    args.start_command = json.loads(args.start_command_json)
    args.stop_command = json.loads(args.stop_command_json) if args.stop_command_json else None
    args.evaluate_command = json.loads(args.evaluate_command_json) if args.evaluate_command_json else None
    broker = Broker(args)
    atexit.register(broker.shutdown)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = b'{"status":"ready"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {broker.token}"):
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = {
                    "/v1/lease": broker.lease_task,
                    "/v1/action": broker.action,
                    "/v1/evaluate": broker.evaluate,
                    "/v1/release": broker.release,
                }[self.path](payload)
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, *_: Any) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest_dir / f"broker-{os.uname().nodename}.json"
    advertise_host = args.advertise_host or os.uname().nodename
    manifest.write_text(
        json.dumps(
            {
                "broker_url": f"http://{advertise_host}:{server.server_port}",
                "schema_version": "osworld.broker_manifest.v1",
            }
        )
        + "\n"
    )
    try:
        server.serve_forever()
    finally:
        manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
