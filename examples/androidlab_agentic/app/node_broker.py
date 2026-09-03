# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Authenticated, capacity-one AndroidLab broker running outside Ray."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .evaluator import EvaluationInfraError, LocalAnswerJudge
from .protocol import ActionValidationError, parse_action
from .trusted_env import AndroidLabEnvironment


def atomic_write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)


class CommandLifecycle:
    """Run trusted start/stop commands; policy text never reaches this
    boundary."""

    def __init__(self, *, start_command: list[str], stop_command: list[str] | None, lease_root: Path) -> None:
        if not start_command:
            raise ValueError("AndroidLab start command is required")
        self._start_command = start_command
        self._stop_command = stop_command
        self._lease_root = lease_root
        self._lease_dir: Path | None = None

    @staticmethod
    def _run(command: list[str], *, env: dict[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=timeout)

    def start(self, lease_id: str) -> str:
        self.cleanup()
        self._lease_dir = self._lease_root / f"lease-{lease_id}"
        self._lease_dir.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ)
        env["ANDROIDLAB_LEASE_ID"] = lease_id
        env["ANDROIDLAB_LEASE_ROOT"] = str(self._lease_dir)
        try:
            result = self._run(self._start_command, env=env, timeout=900.0)
            payload = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            self.cleanup()
            raise EvaluationInfraError("lifecycle_start_failed", "lease") from exc
        serial = payload.get("serial") if isinstance(payload, dict) else None
        if not isinstance(serial, str) or not serial:
            self.cleanup()
            raise EvaluationInfraError("lifecycle_missing_serial", "lease")
        return serial

    def cleanup(self) -> None:
        lease_dir = self._lease_dir
        self._lease_dir = None
        if lease_dir is None:
            return
        env = dict(os.environ)
        env["ANDROIDLAB_LEASE_ROOT"] = str(lease_dir)
        try:
            if self._stop_command:
                self._run(self._stop_command, env=env, timeout=300.0)
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            if lease_dir.exists():
                for child in sorted(lease_dir.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                lease_dir.rmdir()


class BrokerState:
    """Serialize Android device leases and retain trusted evaluator state."""

    def __init__(
        self,
        *,
        lifecycle: CommandLifecycle,
        registry: dict[str, Any],
        adb: str,
        androidlab_repo: Path,
        work_root: Path,
        lease_ttl: float,
        query_judge: LocalAnswerJudge | None,
        event_path: Path | None = None,
        environment_factory: Any = AndroidLabEnvironment,
    ) -> None:
        self._lifecycle = lifecycle
        self._registry = registry
        self._adb = adb
        self._androidlab_repo = androidlab_repo.resolve()
        self._work_root = work_root
        self._lease_ttl = lease_ttl
        self._query_judge = query_judge
        self._event_path = event_path
        self._environment_factory = environment_factory
        self._lock = threading.RLock()
        self._env: AndroidLabEnvironment | None = None
        self._lease_id: str | None = None
        self._request_id: str | None = None
        self._generation = 0
        self._lease_deadline = 0.0
        self._terminal_answer: str | None = None

    def _event(self, event: str, **fields: Any) -> None:
        if self._event_path is None:
            return
        self._event_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": event,
            "hostname": socket.gethostname(),
            "lease_id": self._lease_id,
            "schema_version": "androidlab.broker_event.v1",
            "timestamp": time.time(),
            **fields,
        }
        with self._event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _require_lease(self, payload: dict[str, Any]) -> AndroidLabEnvironment:
        if payload.get("lease_id") != self._lease_id or payload.get("generation") != self._generation:
            raise ValueError("stale_or_unknown_lease")
        if self._env is None:
            raise ValueError("lease_not_ready")
        if time.monotonic() > self._lease_deadline:
            self._cleanup(reason="lease_ttl_expired")
            raise TimeoutError("lease_expired")
        return self._env

    def _cleanup(self, *, reason: str) -> None:
        env = self._env
        self._env = None
        if env is not None:
            try:
                trace_path = env.write_trace()
                self._event("trace_written", trace_path=str(trace_path))
            except Exception:
                self._event("trace_write_failed")
        self._lifecycle.cleanup()
        self._event(reason)
        self._lease_id = None
        self._request_id = None
        self._terminal_answer = None
        self._lease_deadline = 0.0

    def lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("request_id")
        task_id = payload.get("task_id")
        digest = payload.get("task_manifest_digest")
        if not all(isinstance(value, str) and value for value in (request_id, task_id, digest)):
            raise ValueError("invalid_lease_request")
        with self._lock:
            if self._env is not None:
                raise RuntimeError("broker_busy")
            task = self._registry.get(task_id)
            if not isinstance(task, dict) or task.get("task_manifest_digest") != digest:
                raise ValueError("unknown_or_stale_task")
            lease_id = uuid.uuid4().hex
            self._lease_id = lease_id
            self._request_id = request_id
            self._generation += 1
            started = time.monotonic()
            try:
                serial = self._lifecycle.start(lease_id)
                env = self._environment_factory(
                    adb=self._adb,
                    androidlab_repo=self._androidlab_repo,
                    serial=serial,
                    task=task,
                    work_dir=self._work_root / lease_id,
                )
                screenshot = env.screenshot()
            except Exception:
                self._cleanup(reason="lease_start_failed")
                raise
            self._env = env
            self._lease_deadline = time.monotonic() + self._lease_ttl
            self._event(
                "lease_ready",
                acquire_seconds=time.monotonic() - started,
                task_id=task_id,
                width=env.width,
                height=env.height,
            )
            return {
                "generation": self._generation,
                "height": env.height,
                "lease_id": lease_id,
                "screenshot": base64.b64encode(screenshot).decode("ascii"),
                "width": env.width,
            }

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            env = self._require_lease(payload)
            raw_action = payload.get("action")
            if not isinstance(raw_action, dict):
                raise ValueError("invalid_action_request")
            try:
                action = parse_action(f"<action>{json.dumps(raw_action, separators=(',', ':'))}</action>")
            except ActionValidationError as exc:
                raise ValueError(f"invalid_action:{exc.code}") from exc
            try:
                screenshot = env.execute(action)
            except Exception:
                self._event("lease_action_failed", action=action.kind.value)
                raise
            if action.kind.value == "done":
                self._terminal_answer = action.arguments.get("answer")
            self._event("lease_action", action=action.kind.value)
            return {"screenshot": base64.b64encode(screenshot).decode("ascii")}

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            env = self._require_lease(payload)
            outcome = env.evaluate(terminal_answer=self._terminal_answer, judge=self._query_judge)
            self._event("lease_evaluated", reason=outcome.reason, score=outcome.score)
            return {
                "partial_subgoals": outcome.partial_subgoals,
                "reason": outcome.reason,
                "score": outcome.score,
            }

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._require_lease(payload)
            self._cleanup(reason="lease_released")
        return {"released": True}

    def cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("invalid_cancel_request")
        with self._lock:
            if request_id != self._request_id:
                return {"cancelled": False}
            self._cleanup(reason="lease_cancelled")
            return {"cancelled": True}

    def close(self) -> None:
        with self._lock:
            if self._env is not None:
                self._cleanup(reason="broker_shutdown")


def _handler(state: BrokerState, token: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _authorized(self) -> bool:
            value = self.headers.get("Authorization", "")
            return hmac.compare_digest(value, f"Bearer {token}")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(HTTPStatus.OK, {"ok": True, "schema_version": "androidlab.broker.v1"})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("invalid_content_length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("invalid_json")
                method = {
                    "/v1/lease": state.lease,
                    "/v1/action": state.action,
                    "/v1/evaluate": state.evaluate,
                    "/v1/release": state.release,
                    "/v1/cancel": state.cancel,
                }.get(self.path)
                if method is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                self._json(HTTPStatus.OK, method(payload))
            except RuntimeError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (EvaluationInfraError, TimeoutError) as exc:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "broker_internal_error"})

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def _load_command(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    command = json.loads(raw)
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("lifecycle command must be a non-empty JSON string array")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--lease-ttl", type=float, default=1800.0)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--androidlab-repo", type=Path, required=True)
    parser.add_argument("--start-command-json", required=True)
    parser.add_argument("--stop-command-json")
    parser.add_argument("--query-judge-url")
    args = parser.parse_args()
    registry_payload = json.loads(args.registry.read_text(encoding="utf-8"))
    registry = registry_payload.get("tasks") if isinstance(registry_payload, dict) else None
    if not isinstance(registry, dict):
        raise ValueError("invalid AndroidLab trusted registry")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("broker token is empty")
    lifecycle = CommandLifecycle(
        start_command=_load_command(args.start_command_json) or [],
        stop_command=_load_command(args.stop_command_json),
        lease_root=args.lease_root,
    )
    state = BrokerState(
        lifecycle=lifecycle,
        registry=registry,
        adb=args.adb,
        androidlab_repo=args.androidlab_repo,
        work_root=args.work_root,
        lease_ttl=args.lease_ttl,
        query_judge=LocalAnswerJudge(args.query_judge_url) if args.query_judge_url else None,
        event_path=args.event_path,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _handler(state, token))
    host, port = server.server_address[:2]
    manifest = args.manifest_dir / f"broker-{socket.gethostname()}.json"
    atomic_write_json(
        manifest,
        {
            "broker_url": f"http://{socket.gethostname()}:{port}",
            "hostname": socket.gethostname(),
            "schema_version": "androidlab.broker_manifest.v1",
        },
    )

    def shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        state.close()
        manifest.unlink(missing_ok=True)
        server.server_close()


if __name__ == "__main__":
    main()
