# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Canonical, transient reward context for terminal agentic trajectories."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable


REWARD_CONTEXT_SCHEMA = "relax.reward_context.v1"
MOBILEGYM_OUTCOME_SCHEMA = "mobilegym.outcome.v1"


class RewardContextError(ValueError):
    """Base class for sample-local reward-context construction errors."""

    code = "invalid_context"


class InvalidMediaError(RewardContextError):
    code = "invalid_media"


class UnsupportedMediaError(RewardContextError):
    code = "unsupported_media"


def canonical_json(value: Any) -> str:
    """Serialize a JSON value deterministically and reject non-JSON/NaN
    data."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _decode_image_data(value: str) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value:
        raise InvalidMediaError("image payload must be a non-empty string")
    mime_type = "image/png"
    encoded = value
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
        except ValueError as exc:
            raise InvalidMediaError("malformed image data URI") from exc
        if ";base64" not in header:
            raise InvalidMediaError("image data URI must use base64 encoding")
        mime_type = header[5:].split(";", 1)[0]
        if not mime_type.startswith("image/"):
            raise UnsupportedMediaError(f"unsupported media MIME type: {mime_type}")
    try:
        return base64.b64decode(encoded, validate=True), mime_type
    except (ValueError, TypeError) as exc:
        raise InvalidMediaError("image payload is not valid base64") from exc


def _media_part_type(part: dict[str, Any]) -> str | None:
    part_type = part.get("type")
    if part_type in {"image", "image_url", "input_image"}:
        return "image"
    if part_type in {"video", "video_url", "input_video"}:
        return "video"
    if part_type in {"audio", "input_audio"}:
        return "audio"
    return None


def _replace_node_media(
    messages: list[dict[str, Any]],
    image_payloads: list[str],
    *,
    blobs: dict[str, "MediaBlob"],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered = copy.deepcopy(messages)
    image_index = 0
    for message in rendered:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        replaced: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                replaced.append(part)
                continue
            modality = _media_part_type(part)
            if modality in {"video", "audio"}:
                raise UnsupportedMediaError(f"RewardContextV1 does not support {modality}")
            if modality != "image":
                replaced.append(part)
                continue
            if image_index >= len(image_payloads):
                raise InvalidMediaError("message image count exceeds backend_image_data_delta")
            raw, mime_type = _decode_image_data(image_payloads[image_index])
            digest = hashlib.sha256(raw).hexdigest()
            media_id = f"sha256:{digest}"
            blob = MediaBlob(media_id=media_id, mime_type=mime_type, data=raw)
            prior = blobs.get(media_id)
            if prior is not None and (prior.mime_type != mime_type or prior.data != raw):
                raise InvalidMediaError(f"digest collision for {media_id}")
            blobs[media_id] = blob
            manifest.append(
                {
                    "media_id": media_id,
                    "mime_type": mime_type,
                    "size_bytes": len(raw),
                    "occurrence": len(manifest),
                }
            )
            replaced.append({"type": "image", "media_id": media_id})
            image_index += 1
        message["content"] = replaced
    if image_index != len(image_payloads):
        raise InvalidMediaError(
            "backend_image_data_delta count exceeds message image count: "
            f"payloads={len(image_payloads)}, message_images={image_index}"
        )
    return rendered


@dataclass(frozen=True)
class MediaBlob:
    media_id: str
    mime_type: str
    data: bytes = field(repr=False)

    def data_uri(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass
class RewardContextV1:
    """Complete reward envelope.

    ``media_blobs`` is never serialized to training.
    """

    identity: dict[str, Any]
    task: dict[str, Any]
    turns: list[dict[str, Any]]
    terminal: dict[str, Any]
    media_manifest: list[dict[str, Any]]
    content_hash: str
    outcome_evidence: dict[str, Any] | None = None
    media_blobs: dict[str, MediaBlob] = field(default_factory=dict, repr=False)
    schema_version: str = REWARD_CONTEXT_SCHEMA
    # Per-turn judge outcomes are transient execution artifacts. They are
    # deliberately excluded from ``semantic_payload``: the context hash must
    # identify what the judge read, not a later model response to it.
    per_turn_judgements: list[dict[str, Any]] = field(default_factory=list)
    # ``per_turn`` has no VLM component for a trajectory with no completed
    # interaction. The terminal executor explicitly falls back to the legacy
    # full-trajectory projection in that exceptional case.
    per_turn_fallback_terminal_once: bool = False

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": {key: value for key, value in self.identity.items() if key != "context_id"},
            "task": self.task,
            "turns": self.turns,
            "terminal": self.terminal,
            "media_manifest": self.media_manifest,
            "outcome_evidence": self.outcome_evidence,
        }

    def release(self) -> None:
        self.media_blobs.clear()
        self.task.clear()
        self.turns.clear()
        self.terminal.clear()
        self.media_manifest.clear()
        if self.outcome_evidence is not None:
            self.outcome_evidence.clear()
        self.per_turn_judgements.clear()


def _terminal_content(messages: Iterable[dict[str, Any]]) -> Any:
    for message in reversed(list(messages)):
        if message.get("role") == "assistant":
            return copy.deepcopy(message.get("content"))
    return None


def _mobilegym_outcome_evidence(static_metadata: dict[str, Any]) -> dict[str, Any] | None:
    evidence = static_metadata.get("mobilegym_outcome_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != MOBILEGYM_OUTCOME_SCHEMA:
        return None
    forbidden = {"passed", "success", "progress", "is_success", "clean"}

    def validate(value: Any) -> Any:
        if isinstance(value, list):
            return [validate(item) for item in value]
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise RewardContextError("MobileGym outcome evidence contains a verdict field")
            return {key: validate(item) for key, item in value.items()}
        return copy.deepcopy(value)

    return validate(evidence)


def build_reward_context(
    *,
    session_id: str,
    group_index: int | None,
    sample_index: int | None,
    leaf_state_hash: str,
    lineage: list[Any],
    reference_answer: Any,
    static_metadata: dict[str, Any],
    tools: list[dict[str, Any]],
    terminal_status: str,
    remove_sample: bool,
) -> RewardContextV1:
    """Build a context directly from ordered lineage deltas before
    flattening."""
    blobs: dict[str, MediaBlob] = {}
    manifest: list[dict[str, Any]] = []
    initial_messages: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    active_turn: dict[str, Any] | None = None
    terminal_response_messages: list[dict[str, Any]] | None = None

    for node in lineage:
        if getattr(node, "backend_video_data_delta", None):
            raise UnsupportedMediaError("RewardContextV1 does not support video")
        if getattr(node, "backend_audio_data_delta", None):
            raise UnsupportedMediaError("RewardContextV1 does not support audio")
        node_messages = _replace_node_media(
            list(getattr(node, "messages_delta", []) or []),
            list(getattr(node, "backend_image_data_delta", []) or []),
            blobs=blobs,
            manifest=manifest,
        )
        if getattr(node, "kind", None) == "resp":
            terminal_response_messages = node_messages
            active_turn = {
                "assistant_messages": node_messages,
                "response_state_hash": node.state_hash,
                "rollout_id": node.rollout_id,
                "status": str(node.status or ""),
                "observations": [],
            }
            turns.append(active_turn)
        elif active_turn is None:
            initial_messages.extend(node_messages)
        else:
            active_turn["observations"].extend(node_messages)

    task = {
        "initial_messages": initial_messages,
        "reference_answer": copy.deepcopy(reference_answer),
        "rubric": copy.deepcopy(static_metadata.get("rubric")),
        "data_source": copy.deepcopy(static_metadata.get("data_source")),
        "tool_schemas": copy.deepcopy(tools),
    }
    terminal_content = (
        _terminal_content(terminal_response_messages or []) if getattr(lineage[-1], "kind", None) == "resp" else None
    )
    terminal = {
        "final_assistant_content": terminal_content,
        "status": terminal_status,
        "remove_sample": bool(remove_sample),
    }
    identity = {
        "session_id": session_id,
        "group_index": group_index,
        "sample_index": sample_index,
        "leaf_state_hash": leaf_state_hash,
        "rollout_id": getattr(lineage[-1], "rollout_id", None),
    }
    context = RewardContextV1(
        identity=identity,
        task=task,
        turns=turns,
        terminal=terminal,
        media_manifest=manifest,
        content_hash="",
        media_blobs=blobs,
        outcome_evidence=_mobilegym_outcome_evidence(static_metadata),
    )
    try:
        context.content_hash = canonical_hash(context.semantic_payload())
    except (TypeError, ValueError) as exc:
        raise RewardContextError(f"reward context is not canonical JSON: {exc}") from exc
    context.identity["context_id"] = f"ctx:{context.content_hash}"
    return context


def build_accuracy_reward_context(
    *,
    session_id: str,
    group_index: int | None,
    sample_index: int | None,
    leaf_state_hash: str,
    lineage: list[Any],
    reference_answer: Any,
    static_metadata: dict[str, Any],
    terminal_status: str,
    remove_sample: bool,
) -> RewardContextV1:
    """Build the terminal Answer-Accuracy context without decoding trajectory
    media.

    The Answer-Accuracy projection replaces images with a marker and never
    reads tool observations. Keeping this context scoped avoids a full
    trajectory/media reconstruction in ``per_turn`` mode merely to issue the
    final text Judge request.
    """
    initial_messages: list[dict[str, Any]] = []
    terminal_response_messages: list[dict[str, Any]] | None = None
    saw_response = False
    for node in lineage:
        node_messages = copy.deepcopy(list(getattr(node, "messages_delta", []) or []))
        if getattr(node, "kind", None) == "resp":
            saw_response = True
            terminal_response_messages = node_messages
        elif not saw_response:
            initial_messages.extend(node_messages)

    terminal_content = (
        _terminal_content(terminal_response_messages or []) if getattr(lineage[-1], "kind", None) == "resp" else None
    )
    context = RewardContextV1(
        identity={
            "session_id": session_id,
            "group_index": group_index,
            "sample_index": sample_index,
            "leaf_state_hash": leaf_state_hash,
            "rollout_id": getattr(lineage[-1], "rollout_id", None),
        },
        task={
            "initial_messages": initial_messages,
            "reference_answer": copy.deepcopy(reference_answer),
            "rubric": copy.deepcopy(static_metadata.get("rubric")),
            "data_source": copy.deepcopy(static_metadata.get("data_source")),
            "tool_schemas": [],
        },
        turns=[],
        terminal={
            "final_assistant_content": terminal_content,
            "status": terminal_status,
            "remove_sample": bool(remove_sample),
        },
        media_manifest=[],
        content_hash="",
        outcome_evidence=_mobilegym_outcome_evidence(static_metadata),
    )
    try:
        context.content_hash = canonical_hash(context.semantic_payload())
    except (TypeError, ValueError) as exc:
        raise RewardContextError(f"reward context is not canonical JSON: {exc}") from exc
    context.identity["context_id"] = f"ctx:{context.content_hash}"
    return context


def _replace_turn_node_media(
    node: Any,
    *,
    blobs: dict[str, MediaBlob],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if getattr(node, "backend_video_data_delta", None):
        raise UnsupportedMediaError("RewardContextV1 does not support video")
    if getattr(node, "backend_audio_data_delta", None):
        raise UnsupportedMediaError("RewardContextV1 does not support audio")
    return _replace_node_media(
        list(getattr(node, "messages_delta", []) or []),
        list(getattr(node, "backend_image_data_delta", []) or []),
        blobs=blobs,
        manifest=manifest,
    )


def build_turn_judge_context(
    *,
    session_id: str,
    group_index: int | None,
    sample_index: int | None,
    initial_node: Any,
    response_node: Any,
    observation_node: Any,
    turn_index: int,
    static_metadata: dict[str, Any],
    tools: list[dict[str, Any]],
) -> RewardContextV1:
    """Build the bounded ``resp -> observation`` context for one VLM call.

    Only the immutable task-prefix node and the newly completed interaction are
    traversed. In particular, this intentionally does *not* rescan or decode
    media from earlier turns, which would make a T-turn run incur O(T^2)
    projection-side work.
    """
    if getattr(response_node, "kind", None) != "resp" or getattr(observation_node, "kind", None) != "obs":
        raise RewardContextError("turn judge context requires a response followed by an observation")
    if getattr(observation_node, "parent_state_hash", None) != getattr(response_node, "state_hash", None):
        raise RewardContextError("turn judge observation is not the response child")

    blobs: dict[str, MediaBlob] = {}
    manifest: list[dict[str, Any]] = []
    initial_messages = _replace_turn_node_media(initial_node, blobs=blobs, manifest=manifest)
    assistant_messages = _replace_turn_node_media(response_node, blobs=blobs, manifest=manifest)
    observations = _replace_turn_node_media(observation_node, blobs=blobs, manifest=manifest)
    context = RewardContextV1(
        identity={
            "session_id": session_id,
            "group_index": group_index,
            "sample_index": sample_index,
            "leaf_state_hash": observation_node.state_hash,
            "response_state_hash": response_node.state_hash,
            "observation_state_hash": observation_node.state_hash,
            "turn_index": int(turn_index),
            "rollout_id": getattr(observation_node, "rollout_id", None),
        },
        task={
            "initial_messages": initial_messages,
            "reference_answer": None,
            "rubric": copy.deepcopy(static_metadata.get("rubric")),
            "data_source": copy.deepcopy(static_metadata.get("data_source")),
            "tool_schemas": copy.deepcopy(tools),
        },
        turns=[
            {
                "assistant_messages": assistant_messages,
                "response_state_hash": response_node.state_hash,
                "rollout_id": getattr(response_node, "rollout_id", None),
                "status": str(getattr(response_node, "status", "") or ""),
                "observations": observations,
            }
        ],
        terminal={
            "phase": "completed_interaction",
            "status": "in_progress",
            "remove_sample": False,
        },
        media_manifest=manifest,
        content_hash="",
        media_blobs=blobs,
    )
    try:
        context.content_hash = canonical_hash(context.semantic_payload())
    except (TypeError, ValueError) as exc:
        raise RewardContextError(f"reward context is not canonical JSON: {exc}") from exc
    context.identity["context_id"] = f"ctx:{context.content_hash}"
    return context
