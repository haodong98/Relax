# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict, code-free computer-action protocol for WindowsAgentArena."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


MAX_ACTION_CHARS = 4096
ACTION_RE = re.compile(r"<action>(\{.*\})</action>", re.DOTALL)
SAFE_KEYS = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + [
        "alt",
        "backspace",
        "ctrl",
        "delete",
        "down",
        "end",
        "enter",
        "esc",
        "home",
        "insert",
        "left",
        "pagedown",
        "pageup",
        "playpause",
        "right",
        "shift",
        "space",
        "tab",
        "up",
        "volumedown",
        "volumemute",
        "volumeup",
        "win",
    ]
    + [f"f{index}" for index in range(1, 13)]
)


class ActionValidationError(ValueError):
    """A policy-generated action failed the public protocol contract."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ActionKind(str, Enum):
    MOVE = "move"
    CLICK = "click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    PRESS = "press"
    HOTKEY = "hotkey"
    WAIT = "wait"
    DONE = "done"
    FAIL = "fail"


@dataclass(frozen=True)
class ComputerAction:
    kind: ActionKind
    arguments: dict[str, Any]

    @property
    def terminal(self) -> bool:
        return self.kind in (ActionKind.DONE, ActionKind.FAIL)


def _reject_constant(_: str) -> None:
    raise ActionValidationError("non_finite_number")


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionValidationError("duplicate_key")
        result[key] = value
    return result


def _expect_fields(payload: dict[str, Any], required: set[str], optional: set[str] | None = None) -> None:
    optional = optional or set()
    if set(payload) != required and not (required <= set(payload) <= required | optional):
        raise ActionValidationError("invalid_fields")


def _coord(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionValidationError("invalid_coordinate")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ActionValidationError("invalid_coordinate")
    return number


def _small_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -10 <= value <= 10:
        raise ActionValidationError("invalid_scroll")
    return value


def _key(value: Any) -> str:
    if not isinstance(value, str) or value.lower() not in SAFE_KEYS:
        raise ActionValidationError("invalid_key")
    return value.lower()


def _parse_payload(payload: dict[str, Any]) -> ComputerAction:
    if not isinstance(payload.get("type"), str):
        raise ActionValidationError("missing_type")
    try:
        kind = ActionKind(payload["type"])
    except ValueError as exc:
        raise ActionValidationError("unsupported_action") from exc

    if kind in (ActionKind.WAIT, ActionKind.DONE, ActionKind.FAIL):
        _expect_fields(payload, {"type"})
        return ComputerAction(kind, {})
    if kind is ActionKind.MOVE:
        _expect_fields(payload, {"type", "x", "y"})
        return ComputerAction(kind, {"x": _coord(payload["x"]), "y": _coord(payload["y"])})
    if kind is ActionKind.CLICK:
        _expect_fields(payload, {"type", "x", "y"}, {"button", "count"})
        button = payload.get("button", "left")
        count = payload.get("count", 1)
        if (
            button not in ("left", "middle", "right")
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count not in (1, 2)
        ):
            raise ActionValidationError("invalid_click")
        return ComputerAction(
            kind,
            {"x": _coord(payload["x"]), "y": _coord(payload["y"]), "button": button, "count": count},
        )
    if kind is ActionKind.DRAG:
        _expect_fields(payload, {"type", "x1", "y1", "x2", "y2"}, {"button"})
        if payload.get("button", "left") != "left":
            raise ActionValidationError("invalid_drag")
        return ComputerAction(kind, {name: _coord(payload[name]) for name in ("x1", "y1", "x2", "y2")})
    if kind is ActionKind.SCROLL:
        _expect_fields(payload, {"type", "x", "y", "dx", "dy"})
        dx, dy = _small_int(payload["dx"]), _small_int(payload["dy"])
        if dx == dy == 0:
            raise ActionValidationError("invalid_scroll")
        return ComputerAction(kind, {"x": _coord(payload["x"]), "y": _coord(payload["y"]), "dx": dx, "dy": dy})
    if kind is ActionKind.TYPE:
        _expect_fields(payload, {"type", "text"})
        value = payload["text"]
        if not isinstance(value, str) or not 1 <= len(value) <= 2048:
            raise ActionValidationError("invalid_text")
        if any(ord(char) < 32 and char not in "\n\t" for char in value):
            raise ActionValidationError("invalid_text")
        return ComputerAction(kind, {"text": value})
    if kind is ActionKind.PRESS:
        _expect_fields(payload, {"type", "key"})
        return ComputerAction(kind, {"key": _key(payload["key"])})
    if kind is ActionKind.HOTKEY:
        _expect_fields(payload, {"type", "keys"})
        keys = payload["keys"]
        if not isinstance(keys, list) or not 2 <= len(keys) <= 4:
            raise ActionValidationError("invalid_hotkey")
        safe_keys = [_key(value) for value in keys]
        if len(set(safe_keys)) != len(safe_keys):
            raise ActionValidationError("invalid_hotkey")
        return ComputerAction(kind, {"keys": safe_keys})
    raise AssertionError(f"unhandled action kind: {kind}")


def parse_action(text: str) -> ComputerAction:
    """Parse exactly one action tag with strict JSON and no trailing prose."""

    if not isinstance(text, str) or len(text) > MAX_ACTION_CHARS:
        raise ActionValidationError("invalid_envelope")
    match = ACTION_RE.fullmatch(text.strip())
    if match is None:
        raise ActionValidationError("invalid_envelope")
    try:
        payload = json.loads(
            match.group(1),
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
        )
    except ActionValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ActionValidationError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise ActionValidationError("invalid_json_type")
    return _parse_payload(payload)


def normalized_to_pixel(value: float, size: int) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    return min(size - 1, int(value * (size - 1) + 0.5))


def render_pyautogui(action: ComputerAction, *, width: int, height: int) -> str:
    """Render only validated structured actions to deterministic pyautogui."""

    args = action.arguments

    def px(value: float) -> int:
        return normalized_to_pixel(value, width)

    def py(value: float) -> int:
        return normalized_to_pixel(value, height)

    if action.kind is ActionKind.MOVE:
        return f"pyautogui.moveTo({px(args['x'])}, {py(args['y'])}, duration=0.2)"
    if action.kind is ActionKind.CLICK:
        return (
            f"pyautogui.click({px(args['x'])}, {py(args['y'])}, "
            f"clicks={args['count']}, button={args['button']!r}, interval=0.1)"
        )
    if action.kind is ActionKind.DRAG:
        return (
            f"pyautogui.moveTo({px(args['x1'])}, {py(args['y1'])}); "
            f"pyautogui.dragTo({px(args['x2'])}, {py(args['y2'])}, duration=0.5, button='left')"
        )
    if action.kind is ActionKind.SCROLL:
        return (
            f"pyautogui.moveTo({px(args['x'])}, {py(args['y'])}); "
            f"pyautogui.hscroll({args['dx']}); pyautogui.scroll({args['dy']})"
        )
    if action.kind is ActionKind.TYPE:
        return f"pyautogui.write({args['text']!r}, interval=0.01)"
    if action.kind is ActionKind.PRESS:
        return f"pyautogui.press({args['key']!r})"
    if action.kind is ActionKind.HOTKEY:
        return f"pyautogui.hotkey({', '.join(repr(key) for key in args['keys'])})"
    if action.kind is ActionKind.WAIT:
        return "time.sleep(1.0)"
    if action.terminal:
        raise ValueError("terminal actions are not executable")
    raise AssertionError(f"unhandled action kind: {action.kind}")


ACTION_SYSTEM_PROMPT = """You control a Windows desktop from screenshots. Reply with exactly one action and no prose:
<action>{\"type\":\"...\",...}</action>
Allowed actions: move(x,y), click(x,y,button=left|middle|right,count=1|2), drag(x1,y1,x2,y2),
scroll(x,y,dx,dy), type(text), press(key), hotkey(keys), wait, done, fail.
Coordinates are normalized to [0,1]. Use done only when the task is complete and fail only when the task is infeasible.
Never emit Python, shell commands, code fences, or more than one action."""
