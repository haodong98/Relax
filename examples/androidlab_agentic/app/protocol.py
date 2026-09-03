# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict, code-free Android action protocol used by the trusted broker."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


MAX_ACTION_CHARS = 4096
MAX_TEXT_CHARS = 2048
ACTION_RE = re.compile(r"<action>(\{.*\})</action>", re.DOTALL)


class ActionValidationError(ValueError):
    """A policy action did not satisfy the public Android action contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ActionKind(str, Enum):
    TAP = "tap"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    TYPE = "type"
    BACK = "back"
    HOME = "home"
    ENTER = "enter"
    WAIT = "wait"
    LAUNCH = "launch"
    DONE = "done"
    FAIL = "fail"


@dataclass(frozen=True)
class AndroidAction:
    kind: ActionKind
    arguments: dict[str, Any]

    @property
    def terminal(self) -> bool:
        return self.kind in {ActionKind.DONE, ActionKind.FAIL}


def _reject_constant(_: str) -> None:
    raise ActionValidationError("non_finite_number")


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionValidationError("duplicate_key")
        result[key] = value
    return result


def _require_fields(payload: dict[str, Any], fields: set[str]) -> None:
    if set(payload) != fields:
        raise ActionValidationError("invalid_fields")


def _coordinate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionValidationError("invalid_coordinate")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ActionValidationError("invalid_coordinate")
    return number


def _duration(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 2000:
        raise ActionValidationError("invalid_duration")
    return value


def _wait_seconds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ActionValidationError("invalid_wait")
    return value


def _parse_payload(payload: dict[str, Any]) -> AndroidAction:
    action_name = payload.get("type")
    if not isinstance(action_name, str):
        raise ActionValidationError("missing_type")
    try:
        kind = ActionKind(action_name)
    except ValueError as exc:
        raise ActionValidationError("unsupported_action") from exc

    if kind in {ActionKind.BACK, ActionKind.HOME, ActionKind.ENTER, ActionKind.LAUNCH, ActionKind.FAIL}:
        _require_fields(payload, {"type"})
        return AndroidAction(kind, {})
    if kind is ActionKind.WAIT:
        _require_fields(payload, {"type", "seconds"})
        return AndroidAction(kind, {"seconds": _wait_seconds(payload["seconds"])})
    if kind is ActionKind.TAP:
        _require_fields(payload, {"type", "x", "y"})
        return AndroidAction(kind, {"x": _coordinate(payload["x"]), "y": _coordinate(payload["y"])})
    if kind is ActionKind.LONG_PRESS:
        _require_fields(payload, {"type", "x", "y", "duration_ms"})
        return AndroidAction(
            kind,
            {
                "x": _coordinate(payload["x"]),
                "y": _coordinate(payload["y"]),
                "duration_ms": _duration(payload["duration_ms"]),
            },
        )
    if kind is ActionKind.SWIPE:
        _require_fields(payload, {"type", "x1", "y1", "x2", "y2", "duration_ms"})
        coordinates = {name: _coordinate(payload[name]) for name in ("x1", "y1", "x2", "y2")}
        coordinates["duration_ms"] = _duration(payload["duration_ms"])
        return AndroidAction(
            kind,
            coordinates,
        )
    if kind is ActionKind.TYPE:
        _require_fields(payload, {"type", "text"})
        text = payload["text"]
        if not isinstance(text, str) or not 1 <= len(text) <= MAX_TEXT_CHARS:
            raise ActionValidationError("invalid_text")
        if any(ord(character) < 32 and character not in "\n\t" for character in text):
            raise ActionValidationError("invalid_text")
        return AndroidAction(kind, {"text": text})
    if kind is ActionKind.DONE:
        if set(payload) not in ({"type"}, {"type", "answer"}):
            raise ActionValidationError("invalid_fields")
        answer = payload.get("answer")
        if answer is not None and (not isinstance(answer, str) or len(answer) > MAX_TEXT_CHARS):
            raise ActionValidationError("invalid_answer")
        return AndroidAction(kind, {"answer": answer} if answer is not None else {})
    raise AssertionError(f"unhandled action kind: {kind}")


def parse_action(text: str) -> AndroidAction:
    """Parse exactly one ``<action>{...}</action>`` envelope."""

    if not isinstance(text, str) or len(text) > MAX_ACTION_CHARS:
        raise ActionValidationError("invalid_envelope")
    match = ACTION_RE.fullmatch(text.strip())
    if match is None:
        raise ActionValidationError("invalid_envelope")
    try:
        payload = json.loads(match.group(1), object_pairs_hook=_reject_duplicate, parse_constant=_reject_constant)
    except ActionValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ActionValidationError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise ActionValidationError("invalid_json_type")
    return _parse_payload(payload)


def normalized_to_pixel(value: float, size: int) -> int:
    if size <= 0:
        raise ValueError("screen size must be positive")
    return min(size - 1, int(value * (size - 1) + 0.5))


ACTION_SYSTEM_PROMPT = """You control an Android phone from screenshots. Reply with exactly one action and no prose:
<action>{\"type\":\"...\",...}</action>
Allowed actions: tap(x,y), long_press(x,y,duration_ms), swipe(x1,y1,x2,y2,duration_ms), type(text),
back, home, enter, wait(seconds), launch, done(answer optional), fail. Coordinates are normalized to [0,1].
`launch` only launches the current task's approved app. Never emit Python, shell commands, ADB commands, code fences,
or more than one action."""
