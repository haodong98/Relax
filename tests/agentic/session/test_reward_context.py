# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import copy
from dataclasses import dataclass, field

import pytest

from relax.agentic.session.reward_context import (
    MOBILEGYM_OUTCOME_SCHEMA,
    InvalidMediaError,
    RewardContextError,
    UnsupportedMediaError,
    build_accuracy_reward_context,
    build_reward_context,
    build_turn_judge_context,
    canonical_json,
)
from relax.agentic.session.state import TrainingFieldArtifact
from relax.utils.types import Sample


@dataclass
class _Node:
    kind: str
    state_hash: str
    rollout_id: int
    messages_delta: list[dict]
    backend_image_data_delta: list[str] = field(default_factory=list)
    backend_audio_data_delta: list[str] = field(default_factory=list)
    backend_video_data_delta: list[str] = field(default_factory=list)
    status: str | None = None
    parent_state_hash: str | None = None


def _image_data(value: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(value).decode("ascii")


def test_reward_context_preserves_order_tools_reasoning_and_media_hash():
    lineage = [
        _Node(
            "obs",
            "root",
            0,
            [{"role": "user", "content": [{"type": "text", "text": "task"}, {"type": "image_url"}]}],
            [_image_data(b"original")],
        ),
        _Node(
            "resp",
            "r1",
            0,
            [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "inspect",
                    "tool_calls": [{"id": "call-1", "function": {"name": "crop", "arguments": "{}"}}],
                }
            ],
        ),
        _Node(
            "obs",
            "o1",
            0,
            [{"role": "tool", "tool_call_id": "call-1", "content": [{"type": "image_url"}]}],
            [_image_data(b"crop")],
        ),
        _Node("resp", "final", 0, [{"role": "assistant", "content": "answer"}], status="completed"),
    ]
    context = build_reward_context(
        session_id="s",
        group_index=1,
        sample_index=2,
        leaf_state_hash="final",
        lineage=lineage,
        reference_answer="reference",
        static_metadata={"data_source": "test"},
        tools=[{"type": "function", "function": {"name": "crop"}}],
        terminal_status="completed",
        remove_sample=False,
    )
    assert context.turns[0]["assistant_messages"][0]["reasoning_content"] == "inspect"
    assert context.turns[0]["observations"][0]["tool_call_id"] == "call-1"
    assert context.terminal["final_assistant_content"] == "answer"
    assert len(context.media_blobs) == 2
    assert len(context.content_hash) == 64
    assert canonical_json({"é": 1}) == '{"é":1}'


def test_reward_context_does_not_treat_terminal_observation_as_final_answer():
    lineage = [
        _Node("obs", "root", 0, [{"role": "user", "content": "task"}]),
        _Node("resp", "r1", 0, [{"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}]),
        _Node("obs", "leaf", 0, [{"role": "tool", "tool_call_id": "x", "content": "failed"}]),
    ]
    context = build_reward_context(
        session_id="s",
        group_index=0,
        sample_index=0,
        leaf_state_hash="leaf",
        lineage=lineage,
        reference_answer=None,
        static_metadata={},
        tools=[],
        terminal_status="truncated",
        remove_sample=False,
    )
    assert context.terminal["final_assistant_content"] is None


def test_terminal_media_is_content_addressed_before_projection():
    lineage = [
        _Node("obs", "root", 0, [{"role": "user", "content": "task"}]),
        _Node(
            "resp",
            "final",
            0,
            [{"role": "assistant", "content": [{"type": "image_url"}, {"type": "text", "text": "answer"}]}],
            [_image_data(b"terminal-image")],
            status="completed",
        ),
    ]
    context = build_reward_context(
        session_id="s",
        group_index=0,
        sample_index=0,
        leaf_state_hash="final",
        lineage=lineage,
        reference_answer="reference",
        static_metadata={},
        tools=[],
        terminal_status="completed",
        remove_sample=False,
    )
    final_image = context.terminal["final_assistant_content"][0]
    assert final_image["type"] == "image"
    assert final_image["media_id"] in context.media_blobs


def test_context_hash_changes_with_semantics_and_rejects_media_mismatch():
    lineage = [
        _Node("obs", "root", 0, [{"role": "user", "content": "task"}]),
        _Node("resp", "final", 0, [{"role": "assistant", "content": "answer"}], status="completed"),
    ]
    kwargs = {
        "session_id": "s",
        "group_index": 0,
        "sample_index": 0,
        "leaf_state_hash": "final",
        "lineage": lineage,
        "reference_answer": "reference",
        "static_metadata": {},
        "tools": [],
        "terminal_status": "completed",
        "remove_sample": False,
    }
    first = build_reward_context(**kwargs)
    changed_lineage = copy.deepcopy(lineage)
    changed_lineage[-1].messages_delta[0]["content"] = "different"
    second = build_reward_context(**(kwargs | {"lineage": changed_lineage}))
    assert first.content_hash != second.content_hash

    bad_lineage = copy.deepcopy(lineage)
    bad_lineage[0].messages_delta[0]["content"] = [{"type": "image_url"}]
    with pytest.raises(InvalidMediaError):
        build_reward_context(**(kwargs | {"lineage": bad_lineage}))


def test_mobilegym_outcome_evidence_is_preserved_without_verdict_labels():
    lineage = [
        _Node("obs", "root", 0, [{"role": "user", "content": "task"}]),
        _Node("resp", "final", 0, [{"role": "assistant", "content": "done"}], status="completed"),
    ]
    evidence = {
        "schema_version": MOBILEGYM_OUTCOME_SCHEMA,
        "task_id": "settings.toggle",
        "task_name": "Toggle a setting",
        "execution": {"stop_reason": "COMPLETE", "agent_message": "done", "agent_answer": None, "steps": 2},
        "goal_checks": [{"field": "enabled", "expected": True, "actual": True}],
    }
    context = build_reward_context(
        session_id="s",
        group_index=0,
        sample_index=0,
        leaf_state_hash="final",
        lineage=lineage,
        reference_answer=None,
        static_metadata={"mobilegym_outcome_evidence": evidence},
        tools=[],
        terminal_status="completed",
        remove_sample=False,
    )

    assert context.outcome_evidence == evidence
    assert "is_success" not in canonical_json(context.semantic_payload())


def test_mobilegym_outcome_evidence_rejects_verdict_labels():
    lineage = [
        _Node("obs", "root", 0, [{"role": "user", "content": "task"}]),
        _Node("resp", "final", 0, [{"role": "assistant", "content": "done"}], status="completed"),
    ]
    with pytest.raises(RewardContextError, match="verdict field"):
        build_reward_context(
            session_id="s",
            group_index=0,
            sample_index=0,
            leaf_state_hash="final",
            lineage=lineage,
            reference_answer=None,
            static_metadata={
                "mobilegym_outcome_evidence": {
                    "schema_version": MOBILEGYM_OUTCOME_SCHEMA,
                    "execution": {"success": True},
                }
            },
            tools=[],
            terminal_status="completed",
            remove_sample=False,
        )


def test_scoped_contexts_avoid_full_trajectory_work_for_per_turn_judging():
    initial = _Node("obs", "root", 0, [{"role": "user", "content": "task"}])
    prior_response = _Node("resp", "r0", 0, [{"role": "assistant", "content": "old action"}])
    prior_observation = _Node(
        "obs",
        "o0",
        0,
        [{"role": "tool", "content": "old result"}],
        parent_state_hash="r0",
    )
    response = _Node(
        "resp",
        "r1",
        0,
        [{"role": "assistant", "content": "current action", "tool_calls": [{"id": "call-1"}]}],
    )
    observation = _Node(
        "obs",
        "o1",
        0,
        [{"role": "tool", "tool_call_id": "call-1", "content": "current result"}],
        parent_state_hash="r1",
    )
    final = _Node("resp", "final", 0, [{"role": "assistant", "content": "final answer"}], status="completed")

    turn = build_turn_judge_context(
        session_id="s",
        group_index=0,
        sample_index=0,
        initial_node=initial,
        response_node=response,
        observation_node=observation,
        turn_index=1,
        static_metadata={"rubric": "grounded"},
        tools=[{"type": "function", "function": {"name": "tap"}}],
    )
    accuracy = build_accuracy_reward_context(
        session_id="s",
        group_index=0,
        sample_index=0,
        leaf_state_hash="final",
        lineage=[initial, prior_response, prior_observation, response, observation, final],
        reference_answer="reference",
        static_metadata={},
        terminal_status="completed",
        remove_sample=False,
    )

    assert len(turn.turns) == 1
    assert "old result" not in canonical_json(turn.semantic_payload())
    assert "current result" in canonical_json(turn.semantic_payload())
    assert accuracy.turns == []
    assert accuracy.media_blobs == {}
    assert accuracy.terminal["final_assistant_content"] == "final answer"


@pytest.mark.parametrize("part_type", ["video", "audio"])
def test_reward_context_rejects_unsupported_modalities(part_type: str):
    lineage = [
        _Node("obs", "root", 0, [{"role": "user", "content": [{"type": part_type}]}]),
        _Node("resp", "final", 0, [{"role": "assistant", "content": "answer"}], status="completed"),
    ]
    with pytest.raises(UnsupportedMediaError):
        build_reward_context(
            session_id="s",
            group_index=0,
            sample_index=0,
            leaf_state_hash="final",
            lineage=lineage,
            reference_answer=None,
            static_metadata={},
            tools=[],
            terminal_status="completed",
            remove_sample=False,
        )


def test_reward_context_is_transient_but_crosses_session_artifact():
    marker = {"transient": True}
    sample = Sample(reward_context=marker)
    assert "reward_context" not in sample.to_dict()
    restored = TrainingFieldArtifact.from_sample(sample).to_sample()
    assert restored.reward_context == marker
