# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Trusted AndroidLab ADB execution and evaluator integration."""

from __future__ import annotations

import base64
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .evaluator import EvaluationInfraError, EvaluationOutcome, LocalAnswerJudge, outcome_from_result
from .protocol import ActionKind, AndroidAction, normalized_to_pixel


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class AndroidLabEnvironment:
    """Execute only validated actions against one broker-owned ADB device."""

    def __init__(self, *, adb: str, serial: str, task: dict[str, Any], work_dir: Path, androidlab_repo: Path) -> None:
        self._adb = adb
        self._serial = serial
        self._task = dict(task)
        self._work_dir = work_dir
        self._androidlab_repo = androidlab_repo.resolve()
        self._trace: list[dict[str, Any]] = []
        self._width, self._height = self._screen_size()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _run(self, argv: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(argv, check=True, capture_output=True, timeout=timeout)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise EvaluationInfraError("adb_command_failed", "environment") from exc

    def _adb_argv(self, *args: str) -> list[str]:
        return [self._adb, "-s", self._serial, *args]

    def _screen_size(self) -> tuple[int, int]:
        output = self._run(self._adb_argv("shell", "wm", "size")).stdout.decode("utf-8", errors="replace")
        for line in output.splitlines():
            if "Physical size:" not in line:
                continue
            width, height = line.rsplit(":", 1)[1].strip().split("x", 1)
            if width.isdigit() and height.isdigit() and int(width) > 0 and int(height) > 0:
                return int(width), int(height)
        raise EvaluationInfraError("invalid_screen_size", "environment")

    def screenshot(self) -> bytes:
        payload = self._run(self._adb_argv("exec-out", "screencap", "-p")).stdout
        if not payload.startswith(PNG_SIGNATURE):
            raise EvaluationInfraError("invalid_screenshot", "environment")
        return payload

    def _xml(self) -> str:
        remote_path = f"/sdcard/relax-agentic-{os.getpid()}.xml"
        self._run(self._adb_argv("shell", "uiautomator", "dump", remote_path))
        xml = self._run(self._adb_argv("exec-out", "cat", remote_path)).stdout.decode("utf-8", errors="replace")
        if not xml.lstrip().startswith("<?xml"):
            raise EvaluationInfraError("invalid_xml", "environment")
        return xml

    def _record(self, action: AndroidAction, *, instruction: str) -> None:
        self._trace.append(
            {
                "parsed_action": {"action": action.kind.value, "kwargs": dict(action.arguments)},
                "target": instruction,
                "xml": self._xml(),
            }
        )

    def execute(self, action: AndroidAction) -> bytes:
        instruction = self._task["instruction"]
        self._record(action, instruction=instruction)
        x = lambda value: str(normalized_to_pixel(value, self._width))
        y = lambda value: str(normalized_to_pixel(value, self._height))
        if action.kind is ActionKind.TAP:
            self._run(self._adb_argv("shell", "input", "tap", x(action.arguments["x"]), y(action.arguments["y"])))
        elif action.kind is ActionKind.LONG_PRESS:
            px, py = x(action.arguments["x"]), y(action.arguments["y"])
            self._run(self._adb_argv("shell", "input", "swipe", px, py, px, py, str(action.arguments["duration_ms"])))
        elif action.kind is ActionKind.SWIPE:
            self._run(
                self._adb_argv(
                    "shell",
                    "input",
                    "swipe",
                    x(action.arguments["x1"]),
                    y(action.arguments["y1"]),
                    x(action.arguments["x2"]),
                    y(action.arguments["y2"]),
                    str(action.arguments["duration_ms"]),
                )
            )
        elif action.kind is ActionKind.TYPE:
            encoded = base64.b64encode(action.arguments["text"].encode("utf-8")).decode("ascii")
            self._run(self._adb_argv("shell", "am", "broadcast", "-a", "ADB_INPUT_B64", "--es", "msg", encoded))
        elif action.kind in {ActionKind.BACK, ActionKind.HOME, ActionKind.ENTER}:
            key = {
                ActionKind.BACK: "KEYCODE_BACK",
                ActionKind.HOME: "KEYCODE_HOME",
                ActionKind.ENTER: "KEYCODE_ENTER",
            }[action.kind]
            self._run(self._adb_argv("shell", "input", "keyevent", key))
        elif action.kind is ActionKind.LAUNCH:
            self._run(
                self._adb_argv(
                    "shell", "monkey", "-p", self._task["package"], "-c", "android.intent.category.LAUNCHER", "1"
                )
            )
        elif action.kind is ActionKind.WAIT:
            time.sleep(action.arguments["seconds"])
        elif not action.terminal:
            raise EvaluationInfraError("unsupported_validated_action", "environment")
        return self.screenshot()

    def _metric_instance(self) -> Any:
        module_name = self._task.get("metric_module")
        task_id = self._task.get("task_id")
        if not isinstance(module_name, str) or not isinstance(task_id, str):
            raise EvaluationInfraError("invalid_registry_metric", "metric")
        try:
            repo_text = str(self._androidlab_repo)
            if repo_text not in sys.path:
                sys.path.insert(0, repo_text)
            module = importlib.import_module(module_name)
            metric_type = module.function_map[task_id]
            try:
                return metric_type(SimpleNamespace(judge_model="", api_key=""))
            except TypeError:
                return metric_type()
        except Exception as exc:
            raise EvaluationInfraError("metric_import_failed", "metric") from exc

    @staticmethod
    def _compress_xml(xml: str) -> Any:
        try:
            from utils_mobile.xml_tool import UIXMLTree

            return json.loads(UIXMLTree().process(xml, level=1, str_type="json").strip())
        except Exception as exc:
            raise EvaluationInfraError("xml_compression_failed", "metric") from exc

    def evaluate(self, *, terminal_answer: str | None, judge: LocalAnswerJudge | None) -> EvaluationOutcome:
        metric = self._metric_instance()
        result: dict[str, Any] | None = None
        for record in self._trace:
            try:
                candidate = metric.judge(self._compress_xml(record["xml"]), record)
            except Exception as exc:
                raise EvaluationInfraError("metric_failed", "metric") from exc
            if isinstance(candidate, dict) and candidate.get("judge_page") is not False:
                result = candidate
        if result is None:
            raise EvaluationInfraError("no_evaluable_page", "metric")
        if self._task["metric_type"] == "operation":
            return outcome_from_result(result)
        reference = getattr(metric, "final_ground_truth", None)
        if not isinstance(reference, str) or judge is None:
            raise EvaluationInfraError("query_judge_unavailable", "query")
        if terminal_answer is None:
            return EvaluationOutcome(score=0.0, reason="query_missing_answer", partial_subgoals={})
        score = judge.score(question=self._task["instruction"], reference_answer=reference, answer=terminal_answer)
        return EvaluationOutcome(score=score, reason="local_query_judge", partial_subgoals={})

    def write_trace(self) -> Path:
        self._work_dir.mkdir(parents=True, exist_ok=True)
        path = self._work_dir / "trace.json"
        path.write_text(json.dumps(self._trace, ensure_ascii=False), encoding="utf-8")
        return path
