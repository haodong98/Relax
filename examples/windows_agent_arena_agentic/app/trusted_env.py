# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Trusted WAA adapter with lazy imports and strict terminal evaluation."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .evaluator import EvaluationOutcome, StrictTerminalEvaluator
from .protocol import ActionKind, ComputerAction, render_pyautogui


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _cloud_destinations(value: Any) -> list[str]:
    destinations: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "cloud_file":
            dest = value.get("dest")
            destinations.extend(dest if isinstance(dest, list) else [dest])
        else:
            for nested in value.values():
                destinations.extend(_cloud_destinations(nested))
    elif isinstance(value, list):
        for nested in value:
            destinations.extend(_cloud_destinations(nested))
    return destinations


class TrustedWAAEnvironment:
    """Compose stock WAA controllers without exposing their raw code API."""

    def __init__(self, base_url: str, task_config: dict[str, Any], cache_root: Path) -> None:
        # These dependencies belong to the external WAA environment, not the
        # Relax training image. The node broker is launched with WAA_PYTHON.
        from desktop_env.controllers.python import PythonController
        from desktop_env.controllers.setup import SetupController
        from desktop_env.evaluators import getters, metrics

        self.vm_ip = "127.0.0.1"
        self.vm_platform = "docker"
        self.task_id = str(task_config["id"])
        self.action_history: list[str] = []
        self.cache_dir = str(cache_root / self.task_id)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        asset_cache = os.environ.get("WAA_ASSET_CACHE")
        if asset_cache:
            task_assets = Path(asset_cache) / self.task_id
            if task_assets.is_dir():
                for source in task_assets.iterdir():
                    if source.is_file():
                        shutil.copy2(source, Path(self.cache_dir) / source.name)
        for destination in _cloud_destinations(task_config.get("evaluator", {}).get("expected")):
            if not isinstance(destination, str) or Path(destination).name != destination:
                raise RuntimeError("invalid trusted cloud-file destination")
            if not (Path(self.cache_dir) / destination).is_file():
                raise RuntimeError(f"trusted evaluator asset is not cached: {destination}")
        self.controller = PythonController(self.vm_ip)
        self.controller.http_server = base_url.rstrip("/")
        self.controller.get_file = self._get_file_strict
        self.controller.get_screenshot = self._get_screenshot_strict
        self.setup_controller = SetupController(self.vm_ip, self.cache_dir)
        self.setup_controller.http_server = base_url.rstrip("/")
        self.setup_controller.http_server_setup_root = base_url.rstrip("/") + "/setup"
        self._getters = getters
        self._metrics = metrics
        self._task_config = task_config
        self._evaluator = StrictTerminalEvaluator(
            task_config["evaluator"],
            getter_resolver=lambda name: getattr(self._getters, f"get_{name}"),
            metric_resolver=lambda name: getattr(self._metrics, name),
            postconfig=self._run_postconfig,
        )

    def _get_screenshot_strict(self) -> bytes:
        import requests

        response = requests.get(self.controller.http_server + "/screenshot", timeout=30)
        if response.status_code != 200 or not response.content.startswith(PNG_SIGNATURE):
            raise RuntimeError(f"WAA screenshot returned HTTP {response.status_code}")
        return response.content

    def _get_file_strict(self, file_path: str) -> bytes:
        import requests

        response = requests.post(self.controller.http_server + "/file", data={"file_path": file_path}, timeout=90)
        if response.status_code == 200:
            return response.content
        if response.status_code == 404 and "not found" in response.text.lower():
            raise FileNotFoundError(file_path)
        raise RuntimeError(f"WAA /file returned HTTP {response.status_code}")

    def _run_postconfig(self, configs: list[dict[str, Any]]) -> None:
        import requests

        strict_routes = {"open": "open_file", "activate_window": "activate_window"}
        for config in configs:
            config_type = config.get("type")
            if config_type not in strict_routes:
                self.setup_controller.setup([config])
                continue
            response = requests.post(
                f"{self.controller.http_server}/setup/{strict_routes[config_type]}",
                json=config.get("parameters", {}),
                timeout=90,
            )
            if response.status_code == 200:
                continue
            if response.status_code == 404 and "not found" in response.text.lower():
                raise FileNotFoundError(response.text)
            raise RuntimeError(f"WAA postconfig {config_type} returned HTTP {response.status_code}")

    def reset(self) -> bytes:
        self.setup_controller.reset_cache_dir(self.cache_dir)
        self.setup_controller.setup(self._task_config.get("config", []))
        self._evaluator.preflight_expected(self)
        time.sleep(1.0)
        return self.screenshot()

    def screenshot(self) -> bytes:
        screenshot = self.controller.get_screenshot()
        if not isinstance(screenshot, bytes) or not screenshot.startswith(PNG_SIGNATURE):
            raise RuntimeError("invalid WAA screenshot")
        return screenshot

    def execute(self, action: ComputerAction, *, width: int = 1440, height: int = 900) -> bytes:
        if action.terminal:
            self.action_history.append("DONE" if action.kind is ActionKind.DONE else "FAIL")
            return self.screenshot()
        command = render_pyautogui(action, width=width, height=height)
        result = self.controller.execute_python_command(command)
        if (
            not isinstance(result, dict)
            or result.get("status") != "success"
            or result.get("returncode") not in (None, 0)
        ):
            raise RuntimeError("WAA action endpoint rejected the structured action")
        self.action_history.append(action.kind.value)
        time.sleep(0.5)
        return self.screenshot()

    def evaluate(self) -> EvaluationOutcome:
        last_action = self.action_history[-1] if self.action_history else None
        return self._evaluator.evaluate(self, last_action=last_action)


def temporary_cache_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="waa-broker-cache-"))
