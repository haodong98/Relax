# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-slot node-local WAA VM broker with authenticated structured actions."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hmac
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .notepad_smoke_env import PINNED_NOTEPAD_TASK_ID, PinnedNotepadEnvironment
from .protocol import parse_action
from .trusted_env import TrustedWAAEnvironment


IMAGE = "docker.io/dockurr/windows@sha256:25f95472f8370d8e48cc7f86d7dbf222114cf84b671f054ea74b80a5350b44ef"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PR_SET_NAME = 15


def set_process_name() -> None:
    """Keep SPMD's stale-Python cleanup from killing the live host broker."""

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NAME, b"waa-broker", 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def atomic_write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)


def parse_podman_port(output: str) -> int:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one podman port mapping, got {len(lines)}")
    try:
        port = int(lines[0].rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("invalid podman port output") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("podman port is out of range")
    return port


def png_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != PNG_SIGNATURE or payload[12:16] != b"IHDR":
        raise RuntimeError("invalid WAA PNG screenshot")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid WAA PNG dimensions")
    return width, height


class PodmanWaaLifecycle:
    """Create a fresh qcow2 overlay and WAA VM for every lease."""

    def __init__(self, *, waa_repo: Path, golden_storage: Path, node_root: Path, ready_timeout: float) -> None:
        self.waa_repo = waa_repo.resolve()
        self.golden_storage = golden_storage.resolve()
        self.node_root = node_root.resolve()
        self.ready_timeout = ready_timeout
        self.container_name: str | None = None
        self.storage: Path | None = None
        self.base_url: str | None = None
        self.node_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(command: list[str], *, timeout: float = 240, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=check, text=True, capture_output=True, timeout=timeout)

    def _prepare_overlay(self, lease_id: str) -> Path:
        storage = self.node_root / f"lease-{lease_id}"
        if storage.exists():
            raise RuntimeError(f"lease storage already exists: {storage}")
        storage.mkdir()
        self.storage = storage
        for source in self.golden_storage.iterdir():
            if source.name == "data.img":
                continue
            destination = storage / source.name
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True)
            else:
                shutil.copy2(source, destination, follow_symlinks=False)
        self._run(
            [
                "podman",
                "run",
                "--rm",
                "--entrypoint",
                "/usr/bin/qemu-img",
                "-v",
                f"{storage}:/storage",
                "-v",
                f"{self.golden_storage}:/base:ro",
                IMAGE,
                "create",
                "-f",
                "qcow2",
                "-F",
                "raw",
                "-b",
                "/base/data.img",
                "/storage/data.qcow2",
            ],
            timeout=120,
        )
        return storage

    def start(self, lease_id: str) -> str:
        self.cleanup()
        self.storage = self._prepare_overlay(lease_id)
        self.container_name = (
            f"relax-waa-{os.environ.get('SLURM_JOB_ID', 'local')}-{socket.gethostname()}-{lease_id[:8]}"
        )
        command = [
            "podman",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--platform",
            "linux/arm64",
            "--device=/dev/kvm",
            "--device=/dev/net/tun",
            "--cap-add",
            "NET_ADMIN",
            "--stop-timeout",
            "120",
            "-p",
            "127.0.0.1::5000",
            "-p",
            "127.0.0.1::8006",
            "-e",
            "VERSION=11",
            "-e",
            "RAM_SIZE=16G",
            "-e",
            "CPU_CORES=8",
            "-e",
            "DISK_SIZE=40G",
            "-e",
            "WIDTH=1440",
            "-e",
            "HEIGHT=900",
            "-e",
            "USERNAME=Docker",
            "-e",
            "PASSWORD=admin",
            "-e",
            "NETWORK=user",
            "-e",
            "USER_PORTS=5000",
            "-e",
            "DISK_FMT=qcow2",
            "-e",
            "RAM_CHECK=N",
            "-v",
            f"{self.storage}:/storage",
            "-v",
            f"{self.golden_storage}:/base:ro",
            "-v",
            f"{self.waa_repo / 'src/win-arena-container/vm/setup'}:/shared:ro",
            IMAGE,
        ]
        self._run(command)
        port_result = self._run(["podman", "port", self.container_name, "5000/tcp"], timeout=30)
        self.base_url = f"http://127.0.0.1:{parse_podman_port(port_result.stdout)}"
        self._wait_ready()
        return self.base_url

    def _wait_ready(self) -> None:
        if self.base_url is None:
            raise RuntimeError("WAA base URL not initialized")
        deadline = time.monotonic() + self.ready_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self.base_url + "/probe", timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError(f"probe status {response.status}")
                with urllib.request.urlopen(self.base_url + "/screenshot", timeout=20) as response:
                    body = response.read()
                    if response.status != 200 or not body.startswith(PNG_SIGNATURE):
                        raise RuntimeError("screenshot not ready")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"WAA readiness timeout: {last_error}")

    def cleanup(self) -> None:
        if self.container_name:
            self._run(["podman", "rm", "--force", self.container_name], timeout=180, check=False)
            self.container_name = None
        if self.storage and self.storage.exists():
            root = self.node_root.resolve()
            target = self.storage.resolve()
            if target.parent != root or not target.name.startswith("lease-"):
                raise RuntimeError(f"refusing unsafe WAA storage removal: {target}")
            shutil.rmtree(target)
        self.storage = None
        self.base_url = None

    def cleanup_node_root(self) -> None:
        root = self.node_root.resolve()
        if root == Path("/") or root.parent == root:
            raise RuntimeError(f"refusing unsafe node-root cleanup: {root}")
        cache = root / "cache"
        if cache.exists():
            if cache.resolve().parent != root:
                raise RuntimeError(f"refusing unsafe broker-cache cleanup: {cache}")
            shutil.rmtree(cache)
        if root.exists():
            root.rmdir()


class BrokerState:
    def __init__(
        self, *, lifecycle: PodmanWaaLifecycle, registry: dict[str, Any], cache_root: Path, lease_ttl: float
    ) -> None:
        self.lifecycle = lifecycle
        self.registry = registry
        self.cache_root = cache_root
        self.lock = threading.RLock()
        self.lease_id: str | None = None
        self.request_id: str | None = None
        self.starting = False
        self.cancel_requested = False
        self.active_operations = 0
        self.screen_width: int | None = None
        self.screen_height: int | None = None
        self.generation = 0
        self.env: TrustedWAAEnvironment | PinnedNotepadEnvironment | None = None
        self.evaluated = False
        self.stopping = False
        self.lease_ttl = lease_ttl
        self.lease_deadline: float | None = None
        self.lease_cache: Path | None = None
        event_dir = os.environ.get("WAA_BROKER_EVENT_DIR")
        self.event_path = Path(event_dir) / f"broker-{socket.gethostname()}.jsonl" if event_dir else None
        self.event_lock = threading.Lock()

    def _record_event(self, event: str, **fields: Any) -> None:
        if self.event_path is None:
            return
        payload = {
            "event": event,
            "hostname": socket.gethostname(),
            "schema_version": "waa.broker_event.v1",
            "timestamp_unix": time.time(),
            **fields,
        }
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_lock:
            descriptor = os.open(self.event_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def acquire(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.stopping:
                raise RuntimeError("broker_stopping")
            if self.lease_id is not None:
                raise BlockingIOError("broker_busy")
            task_id = payload.get("task_id")
            task = self.registry.get("tasks", {}).get(task_id)
            if task is None:
                raise KeyError("unknown_task")
            if not hmac.compare_digest(str(payload.get("task_manifest_digest", "")), task["task_manifest_digest"]):
                raise PermissionError("task_digest_mismatch")
            request_id = payload.get("request_id")
            try:
                normalized_request_id = uuid.UUID(str(request_id)).hex
            except (ValueError, AttributeError) as exc:
                raise ValueError("invalid_request_id") from exc
            if request_id != normalized_request_id:
                raise ValueError("invalid_request_id")
            lease_id = uuid.uuid4().hex
            lease_cache = self.cache_root / lease_id
            self.request_id = request_id
            self.lease_id = lease_id
            self.lease_cache = lease_cache
            self.starting = True
            self.cancel_requested = False

        self._record_event("lease_reserved", lease_id=lease_id, request_id=request_id, task_id=task_id)
        start_time = time.monotonic()

        try:
            base_url = self.lifecycle.start(lease_id)
            environment_type = PinnedNotepadEnvironment if task_id == PINNED_NOTEPAD_TASK_ID else TrustedWAAEnvironment
            env = environment_type(base_url, task["task_config"], lease_cache)
            screenshot = env.reset()
            screen_width, screen_height = png_dimensions(screenshot)
            with self.lock:
                if self.request_id != request_id or self.cancel_requested or self.stopping:
                    raise RuntimeError("lease_cancelled")
                self.generation += 1
                generation = self.generation
                self.env = env
                self.screen_width = screen_width
                self.screen_height = screen_height
                self.starting = False
                self.evaluated = False
                self.lease_deadline = time.monotonic() + self.lease_ttl
            self._record_event(
                "lease_ready",
                acquire_seconds=time.monotonic() - start_time,
                height=screen_height,
                lease_id=lease_id,
                request_id=request_id,
                task_id=task_id,
                width=screen_width,
            )
        except Exception as error:
            cleanup_error: Exception | None = None
            try:
                self.lifecycle.cleanup()
                if lease_cache.exists():
                    shutil.rmtree(lease_cache)
            except Exception as exc:
                cleanup_error = exc
            finally:
                with self.lock:
                    if self.request_id == request_id:
                        self.env = None
                        self.screen_width = None
                        self.screen_height = None
                        self.lease_cache = None
                        self.lease_id = None
                        self.request_id = None
                        self.starting = False
                        self.cancel_requested = False
                        self.evaluated = False
                        self.lease_deadline = None
            self._record_event(
                "lease_failed",
                error_type=type(error).__name__,
                lease_id=lease_id,
                request_id=request_id,
                task_id=task_id,
            )
            if cleanup_error is not None:
                raise RuntimeError("failed to clean cancelled WAA lease") from cleanup_error
            raise

        with self.lock:
            if self.request_id != request_id or self.env is not env:
                raise RuntimeError("lease_cancelled")
            self.evaluated = False
            return {
                "generation": generation,
                "lease_id": lease_id,
                "request_id": request_id,
                "screenshot": base64.b64encode(screenshot).decode(),
                "width": screen_width,
                "height": screen_height,
            }

    def _require(self, payload: dict[str, Any]) -> TrustedWAAEnvironment | PinnedNotepadEnvironment:
        if self.lease_deadline is not None and time.monotonic() >= self.lease_deadline:
            self._expire_locked()
            raise PermissionError("expired_lease")
        if not hmac.compare_digest(str(payload.get("lease_id", "")), self.lease_id or ""):
            raise PermissionError("invalid_lease")
        if payload.get("generation") != self.generation or self.env is None:
            raise PermissionError("stale_generation")
        return self.env

    def _expire_locked(self) -> None:
        self.lifecycle.cleanup()
        if self.lease_cache is not None and self.lease_cache.exists():
            shutil.rmtree(self.lease_cache)
        self.env = None
        self.lease_cache = None
        self.lease_id = None
        self.request_id = None
        self.starting = False
        self.cancel_requested = False
        self.active_operations = 0
        self.screen_width = None
        self.screen_height = None
        self.evaluated = False
        self.lease_deadline = None

    def expire_if_needed(self) -> None:
        with self.lock:
            if self.lease_deadline is not None and time.monotonic() >= self.lease_deadline:
                if self.active_operations:
                    self.cancel_requested = True
                else:
                    self._expire_locked()

    def _finish_operation(self) -> None:
        with self.lock:
            self.active_operations -= 1
            if self.active_operations < 0:
                raise RuntimeError("negative active WAA operation count")
            if self.active_operations == 0 and self.cancel_requested:
                lease_id = self.lease_id
                self._expire_locked()
                self._record_event("lease_cleanup_after_operation", lease_id=lease_id)

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            env = self._require(payload)
            if self.active_operations:
                raise BlockingIOError("operation_busy")
            if self.screen_width is None or self.screen_height is None:
                raise RuntimeError("missing_screen_dimensions")
            action_json = json.dumps(payload.get("action"), separators=(",", ":"), allow_nan=False)
            action = parse_action(f"<action>{action_json}</action>")
            self.active_operations += 1
        try:
            screenshot = env.execute(action, width=self.screen_width, height=self.screen_height)
            return {
                "screenshot": base64.b64encode(screenshot).decode(),
                "terminal": action.terminal,
            }
        finally:
            self._finish_operation()

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            env = self._require(payload)
            if self.active_operations:
                raise BlockingIOError("operation_busy")
            if self.evaluated:
                raise RuntimeError("already_evaluated")
            self.active_operations += 1
        try:
            outcome = env.evaluate()
            with self.lock:
                if self.env is env:
                    self.evaluated = True
            self._record_event(
                "lease_evaluated",
                lease_id=str(payload.get("lease_id", "")),
                reason=outcome.reason,
                score=outcome.score,
            )
            return {"reason": outcome.reason, "score": outcome.score}
        finally:
            self._finish_operation()

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._require(payload)
            lease_id = self.lease_id
            if self.active_operations:
                self.cancel_requested = True
                self._record_event("lease_release_pending", lease_id=lease_id)
                return {"released": True, "cleanup_pending": True}
            self._expire_locked()
            self._record_event("lease_released", lease_id=lease_id)
            return {"released": True, "cleanup_pending": False}

    def cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            supplied = str(payload.get("request_id", ""))
            if self.request_id is None:
                return {"cancelled": False}
            if not hmac.compare_digest(supplied, self.request_id):
                raise PermissionError("invalid_request_id")
            if self.starting:
                self.cancel_requested = True
                self._record_event("lease_cancel_pending", request_id=supplied)
                return {"cancelled": True, "cleanup_pending": True}
            if self.active_operations:
                self.cancel_requested = True
                self._record_event("lease_cancel_pending", request_id=supplied)
                return {"cancelled": True, "cleanup_pending": True}
            self._expire_locked()
            self._record_event("lease_cancelled", request_id=supplied)
            return {"cancelled": True, "cleanup_pending": False}

    def shutdown(self) -> None:
        with self.lock:
            self.stopping = True
            if self.starting:
                self.cancel_requested = True
                return
            if self.active_operations:
                self.cancel_requested = True
                return
            self._expire_locked()


def make_handler(state: BrokerState, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RelaxWAABroker/1"

        def log_message(self, format_string: str, *args: Any) -> None:
            return

        def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, f"Bearer {token}")

        def _payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid_content_length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("invalid_payload")
            return payload

        def do_GET(self) -> None:
            if self.path != "/health":
                self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._reply(HTTPStatus.OK, {"busy": state.lease_id is not None, "status": "ok"})

        def do_POST(self) -> None:
            if not self._authorized():
                self._reply(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                payload = self._payload()
                routes = {
                    "/v1/lease": state.acquire,
                    "/v1/cancel": state.cancel,
                    "/v1/action": state.action,
                    "/v1/evaluate": state.evaluate,
                    "/v1/release": state.release,
                }
                if self.path not in routes:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                self._reply(HTTPStatus.OK, routes[self.path](payload))
            except BlockingIOError as exc:
                self._reply(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (KeyError, PermissionError, ValueError) as exc:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "broker_internal"})

    return Handler


def _default_advertise_host() -> str:
    return socket.gethostbyname(socket.gethostname())


def main() -> None:
    set_process_name()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waa-repo", type=Path, required=True)
    parser.add_argument("--golden-storage", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--advertise-host", default=None)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--lease-ttl", type=float, default=2700.0)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("broker token must contain at least 32 characters")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    lifecycle = PodmanWaaLifecycle(
        waa_repo=args.waa_repo,
        golden_storage=args.golden_storage,
        node_root=args.node_root,
        ready_timeout=args.ready_timeout,
    )
    state = BrokerState(
        lifecycle=lifecycle,
        registry=registry,
        cache_root=args.node_root / "cache",
        lease_ttl=args.lease_ttl,
    )
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(state, token))
    manifest_path = args.manifest_dir / f"broker-{socket.gethostname()}.json"
    atomic_write_json(
        manifest_path,
        {
            "broker_url": f"http://{args.advertise_host or _default_advertise_host()}:{server.server_port}",
            "hostname": socket.gethostname(),
            "schema_version": "waa.broker_manifest.v1",
        },
        mode=0o644,
    )

    def stop_server(_: int, __: Any) -> None:
        state.stopping = True

        def drain() -> None:
            server.shutdown()
            state.shutdown()

        threading.Thread(target=drain, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)

    def lease_reaper() -> None:
        while not state.stopping:
            state.expire_if_needed()
            time.sleep(5.0)

    threading.Thread(target=lease_reaper, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        state.shutdown()
        manifest_path.unlink(missing_ok=True)
        server.server_close()
        lifecycle.cleanup_node_root()


if __name__ == "__main__":
    main()
