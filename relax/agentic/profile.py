# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import socket
import time
from typing import Any


TRACE_KEY = "agentic_trace"
EVENTS_KEY = "events"
REWARD_TRACE_KEY = "reward"
REWARD_TRACE_SCHEMA_VERSION = 1
EVENT_CLOCK_HOST_SUFFIX = "__clock_host"
_clock_host = socket.gethostname()


def ensure_agentic_trace(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        metadata = {}
    trace = metadata.get(TRACE_KEY)
    if trace is None:
        trace = {}
        metadata[TRACE_KEY] = trace
    if not isinstance(trace, dict):
        raise TypeError(f"{TRACE_KEY} must be a dict, got {type(trace)}")
    return trace


def agentic_trace_events(metadata: dict[str, Any] | None) -> dict[str, Any]:
    trace = ensure_agentic_trace(metadata)
    events = trace.get(EVENTS_KEY)
    if events is None:
        events = {}
        trace[EVENTS_KEY] = events
    if not isinstance(events, dict):
        raise TypeError(f"{TRACE_KEY}.{EVENTS_KEY} must be a dict, got {type(events)}")
    return events


def ensure_reward_trace(metadata: dict[str, Any] | None) -> dict[str, Any]:
    trace = ensure_agentic_trace(metadata)
    reward_trace = trace.get(REWARD_TRACE_KEY)
    if reward_trace is None:
        reward_trace = {
            "schema_version": REWARD_TRACE_SCHEMA_VERSION,
            "judges": {},
        }
        trace[REWARD_TRACE_KEY] = reward_trace
    if not isinstance(reward_trace, dict):
        raise TypeError(f"{TRACE_KEY}.{REWARD_TRACE_KEY} must be a dict, got {type(reward_trace)}")
    judges = reward_trace.get("judges")
    if judges is None:
        judges = {}
        reward_trace["judges"] = judges
    if not isinstance(judges, dict):
        raise TypeError(f"{TRACE_KEY}.{REWARD_TRACE_KEY}.judges must be a dict, got {type(judges)}")
    return reward_trace


def ensure_reward_judge_trace(metadata: dict[str, Any] | None, *, component: str, role: str) -> dict[str, Any]:
    reward_trace = ensure_reward_trace(metadata)
    judges = reward_trace["judges"]
    judge_trace = judges.get(component)
    if judge_trace is None:
        judge_trace = {"role": role}
        judges[component] = judge_trace
    if not isinstance(judge_trace, dict):
        raise TypeError(f"{TRACE_KEY}.{REWARD_TRACE_KEY}.judges.{component} must be a dict, got {type(judge_trace)}")
    judge_trace["role"] = role
    return judge_trace


def sample_agentic_trace_events(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    return agentic_trace_events(metadata)


def mark_agentic_event(events: dict[str, Any], key: str, timestamp: float | None = None) -> None:
    events[key] = time.time() if timestamp is None else timestamp
    events[f"{key}{EVENT_CLOCK_HOST_SUFFIX}"] = _clock_host


def mark_agentic_event_once(events: dict[str, Any], key: str, timestamp: float) -> None:
    events.setdefault(key, timestamp)
    events.setdefault(f"{key}{EVENT_CLOCK_HOST_SUFFIX}", _clock_host)


def agentic_span_clock_host(events: dict[str, Any], start_key: str, end_key: str) -> str | None:
    """Return one clock host only when both endpoints share explicit
    provenance."""
    start_host = events.get(f"{start_key}{EVENT_CLOCK_HOST_SUFFIX}")
    end_host = events.get(f"{end_key}{EVENT_CLOCK_HOST_SUFFIX}")
    if isinstance(start_host, str) and start_host and start_host == end_host:
        return start_host
    return None


def mark_metadata_agentic_event(metadata: dict[str, Any] | None, key: str, timestamp: float | None = None) -> None:
    mark_agentic_event(agentic_trace_events(metadata), key, timestamp)


def mark_sample_agentic_event(sample: Any, key: str, timestamp: float | None = None) -> None:
    mark_agentic_event(sample_agentic_trace_events(sample), key, timestamp)


def mark_sample_agentic_event_once(sample: Any, key: str, timestamp: float) -> None:
    mark_agentic_event_once(sample_agentic_trace_events(sample), key, timestamp)


def _merge_trace_mapping(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_trace_mapping(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_agentic_trace(base_trace: dict[str, Any] | None, overlay_trace: dict[str, Any] | None) -> dict[str, Any]:
    if base_trace is not None and not isinstance(base_trace, dict):
        raise TypeError(f"{TRACE_KEY} must be a dict, got {type(base_trace)}")
    if overlay_trace is not None and not isinstance(overlay_trace, dict):
        raise TypeError(f"{TRACE_KEY} overlay must be a dict, got {type(overlay_trace)}")
    merged = copy.deepcopy(base_trace) if isinstance(base_trace, dict) else {}
    if not overlay_trace:
        return merged
    for key, value in overlay_trace.items():
        if key == EVENTS_KEY:
            if not isinstance(value, dict):
                raise TypeError(f"{TRACE_KEY}.{EVENTS_KEY} overlay must be a dict, got {type(value)}")
            base_events = merged.get(EVENTS_KEY)
            if base_events is not None and not isinstance(base_events, dict):
                raise TypeError(f"{TRACE_KEY}.{EVENTS_KEY} must be a dict, got {type(base_events)}")
            events = copy.deepcopy(base_events) if isinstance(base_events, dict) else {}
            events.update(copy.deepcopy(value))
            merged[EVENTS_KEY] = events
            continue
        if key == REWARD_TRACE_KEY:
            if not isinstance(value, dict):
                raise TypeError(f"{TRACE_KEY}.{REWARD_TRACE_KEY} overlay must be a dict, got {type(value)}")
            base_reward = merged.get(REWARD_TRACE_KEY)
            if base_reward is not None and not isinstance(base_reward, dict):
                raise TypeError(f"{TRACE_KEY}.{REWARD_TRACE_KEY} must be a dict, got {type(base_reward)}")
            merged[REWARD_TRACE_KEY] = _merge_trace_mapping(base_reward or {}, value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged
