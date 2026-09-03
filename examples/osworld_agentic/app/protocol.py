# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict structured action protocol for OSWorld."""

from __future__ import annotations

import json
import math
import re
from typing import Any


ACTION_RE = re.compile(r"^<action>(\{.*\})</action>$", re.DOTALL)
SAFE_KEYS = {
    "alt",
    "backspace",
    "ctrl",
    "delete",
    "down",
    "end",
    "enter",
    "esc",
    "home",
    "left",
    "pagedown",
    "pageup",
    "right",
    "shift",
    "space",
    "tab",
    "up",
}
ACTION_SYSTEM_PROMPT = (
    "You control a desktop through exactly one action per turn. "
    "Return only <action>{JSON}</action>. Supported actions are move, click, "
    "type, press, scroll, wait, done, and fail. Coordinates are normalized to [0,1]."
)


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("invalid_coordinate")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("invalid_coordinate")
    return value


def parse_action(text: str) -> dict[str, Any]:
    match = ACTION_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError("invalid_action_envelope")
    payload = json.loads(match.group(1), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite")))
    if not isinstance(payload, dict) or set(payload) - {"type", "x", "y", "text", "key", "dx", "dy"}:
        raise ValueError("invalid_action_fields")
    kind = payload.get("type")
    if kind in {"wait", "done", "fail"}:
        if set(payload) != {"type"}:
            raise ValueError("invalid_action_fields")
        return payload
    if kind in {"move", "click"}:
        if set(payload) != {"type", "x", "y"}:
            raise ValueError("invalid_action_fields")
        return {"type": kind, "x": _finite(payload["x"]), "y": _finite(payload["y"])}
    if kind == "type":
        if set(payload) != {"type", "text"} or not isinstance(payload["text"], str) or not payload["text"]:
            raise ValueError("invalid_text")
        return payload
    if kind == "press":
        if set(payload) != {"type", "key"} or payload.get("key", "").lower() not in SAFE_KEYS:
            raise ValueError("invalid_key")
        return {"type": "press", "key": payload["key"].lower()}
    if kind == "scroll":
        if set(payload) != {"type", "x", "y", "dx", "dy"}:
            raise ValueError("invalid_scroll")
        if (
            any(
                isinstance(payload[name], bool) or not isinstance(payload[name], int) or not -10 <= payload[name] <= 10
                for name in ("dx", "dy")
            )
            or payload["dx"] == payload["dy"] == 0
        ):
            raise ValueError("invalid_scroll")
        return {**payload, "x": _finite(payload["x"]), "y": _finite(payload["y"])}
    raise ValueError("unsupported_action")


def render_action(action: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    result = dict(action)
    if "x" in result:
        result["x"] = round(float(result["x"]) * width)
    if "y" in result:
        result["y"] = round(float(result["y"]) * height)
    return result
