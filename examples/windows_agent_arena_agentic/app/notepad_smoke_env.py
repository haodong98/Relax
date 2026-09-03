# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dependency-light native-compatible adapter for the pinned Notepad smoke
task."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .evaluator import EvaluationInfraError, EvaluationOutcome
from .protocol import ActionKind, ComputerAction, render_pyautogui


PINNED_NOTEPAD_TASK_ID = "366de66e-cbae-4d72-b042-26390db2b145-WOS"
PINNED_FILE_PATH = r"C:\Users\Docker\Documents\draft.txt"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_EXPECTED_EVALUATOR = {
    "postconfig": [
        {"type": "open", "parameters": {"path": PINNED_FILE_PATH}},
        {"type": "activate_window", "parameters": {"window_name": "draft.txt - Notepad"}},
        {"type": "sleep", "parameters": {"seconds": 0.5}},
    ],
    "func": ["exact_match", "compare_text_file"],
    "result": [
        {
            "type": "vm_file_exists_in_vm_folder",
            "folder_name": r"C:\Users\Docker\Documents",
            "file_name": "draft.txt",
        },
        {"type": "vm_file", "path": PINNED_FILE_PATH, "dest": "draft.txt"},
    ],
    "expected": [
        {"type": "rule", "rules": {"expected": 1.0}},
        {
            "type": "cloud_file",
            "path": (
                "https://raw.githubusercontent.com/rogeriobonatti/winarenafiles/main/task_files/notepad/"
                f"{PINNED_NOTEPAD_TASK_ID}/eval/draft.txt"
            ),
            "dest": "draft_gold.txt",
        },
    ],
}


class PinnedNotepadEnvironment:
    """Run one audited WAA task without importing the stock optional controller
    stack."""

    def __init__(
        self,
        base_url: str,
        task_config: dict[str, Any],
        cache_root: Path,
        *,
        requester: Any | None = None,
    ) -> None:
        if task_config.get("id") != PINNED_NOTEPAD_TASK_ID:
            raise RuntimeError("pinned Notepad adapter received a different task")
        if task_config.get("config") != [] or task_config.get("evaluator") != _EXPECTED_EVALUATOR:
            raise RuntimeError("pinned Notepad task contract differs from the audited upstream config")
        if requester is None:
            import requests

            requester = requests
        self._requests = requester
        self._base_url = base_url.rstrip("/")
        self._cache_dir = cache_root / PINNED_NOTEPAD_TASK_ID
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._gold_path = self._cache_dir / "draft_gold.txt"
        self._result_path = self._cache_dir / "draft.txt"
        self._action_history: list[str] = []

        asset_root = Path(os.environ.get("WAA_ASSET_CACHE", ""))
        source = asset_root / PINNED_NOTEPAD_TASK_ID / "draft_gold.txt"
        if not source.is_file():
            raise EvaluationInfraError("expected_unavailable", "preflight")
        self._gold_path.write_bytes(source.read_bytes())
        if not self._gold_path.read_bytes():
            raise EvaluationInfraError("expected_unavailable", "preflight")

    @staticmethod
    def _is_explicit_not_found(response: Any) -> bool:
        return response.status_code == 404 and "not found" in response.text.lower()

    def _post_json(self, route: str, payload: dict[str, Any], *, timeout: float = 90) -> Any:
        try:
            return self._requests.post(f"{self._base_url}{route}", json=payload, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(f"WAA {route} request failed") from exc

    def reset(self) -> bytes:
        # The pinned task deliberately has no setup actions. Gold availability
        # was checked in __init__, before the first policy call.
        time.sleep(1.0)
        return self.screenshot()

    def screenshot(self) -> bytes:
        try:
            response = self._requests.get(f"{self._base_url}/screenshot", timeout=30)
        except Exception as exc:
            raise RuntimeError("WAA screenshot request failed") from exc
        if response.status_code != 200 or not response.content.startswith(PNG_SIGNATURE):
            raise RuntimeError(f"WAA screenshot returned HTTP {response.status_code}")
        return response.content

    def execute(self, action: ComputerAction, *, width: int = 1440, height: int = 900) -> bytes:
        if action.terminal:
            self._action_history.append("DONE" if action.kind is ActionKind.DONE else "FAIL")
            return self.screenshot()
        rendered = render_pyautogui(action, width=width, height=height)
        command = ["python", "-c", f"import time,pyautogui; pyautogui.FAILSAFE=False; {rendered}"]
        response = self._post_json("/execute", {"command": command, "shell": False})
        if response.status_code != 200:
            raise RuntimeError(f"WAA action returned HTTP {response.status_code}")
        try:
            result = response.json()
        except Exception as exc:
            raise RuntimeError("WAA action returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("status") != "success" or result.get("returncode") != 0:
            raise RuntimeError("WAA action endpoint rejected the structured action")
        self._action_history.append(action.kind.value)
        time.sleep(0.5)
        return self.screenshot()

    def _run_postconfig(self) -> None:
        routes = [
            ("/setup/open_file", {"path": PINNED_FILE_PATH}),
            ("/setup/activate_window", {"window_name": "draft.txt - Notepad"}),
        ]
        for route, payload in routes:
            response = self._post_json(route, payload)
            if response.status_code == 200 or self._is_explicit_not_found(response):
                continue
            raise EvaluationInfraError("postconfig_failed", "postconfig")
        time.sleep(0.5)

    def _fetch_result(self) -> bytes | None:
        try:
            response = self._requests.post(f"{self._base_url}/file", data={"file_path": PINNED_FILE_PATH}, timeout=90)
        except Exception as exc:
            raise EvaluationInfraError("result_unavailable", "result") from exc
        if response.status_code == 200:
            return response.content
        if self._is_explicit_not_found(response):
            return None
        raise EvaluationInfraError("result_unavailable", "result")

    def evaluate(self) -> EvaluationOutcome:
        self._run_postconfig()
        last_action = self._action_history[-1] if self._action_history else None
        if last_action == "FAIL":
            return EvaluationOutcome(0.0, "feasible_declared_failed")
        content = self._fetch_result()
        if content is None:
            return EvaluationOutcome(0.0, "and_short_circuit")
        self._result_path.write_bytes(content)
        try:
            actual = self._result_path.read_text(encoding="utf-8")
            expected = self._gold_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvaluationInfraError("metric_failed", "metric") from exc
        content_score = float(actual == expected)
        if content_score == 0.0:
            return EvaluationOutcome(0.0, "and_short_circuit")
        # Native WAA averages component scores only after every "and" metric
        # is nonzero. This task's existence and exact-text metrics are both 1.
        return EvaluationOutcome((1.0 + content_score) / 2.0, "and")
