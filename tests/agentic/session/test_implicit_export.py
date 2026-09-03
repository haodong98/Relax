# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Implicit-export leaf selection over branching session forests."""

from __future__ import annotations

import pytest

from relax.agentic.session.service import AgenticSessionShard, _NonFinalizableExportError, _SessionRecord
from relax.agentic.session.state import SessionForest, check_messages


# AgenticSessionShard is wrapped into a Ray ActorClass by @serve.deployment;
# reach the undecorated class so the export helper can be exercised as a plain
# function, without a running Ray Serve replica.
_SHARD_CLASS = getattr(getattr(AgenticSessionShard, "__ray_metadata__", None), "modified_class", AgenticSessionShard)


def _chars(text: str) -> list[int]:
    return [ord(char) for char in text]


def _branch(forest: SessionForest, *, prompt: str, response: str) -> str:
    """Append one obs->resp branch off the root and return its leaf hash."""
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": prompt}]}]),
        train_token_delta=_chars(prompt),
        rollout_token_delta=_chars(prompt),
    )
    leaf = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": response}]}],
        train_token_delta=_chars(response),
        rollout_token_delta=_chars(response),
        logprob_delta=[-0.1] * len(response),
        status="completed",
    )
    return leaf.state_hash


def _record(**branches: str) -> tuple[_SessionRecord, list[str]]:
    forest = SessionForest.create_empty(session_id="sess-export")
    leaf_hashes = [_branch(forest, prompt=prompt, response=response) for prompt, response in branches.items()]
    return _SessionRecord(forest=forest), leaf_hashes


def test_implicit_export_returns_every_branch_leaf() -> None:
    """A history-rewriting agent roots each turn as its own branch; all of them
    are exportable trajectories and none may be silently discarded.

    The exported order must be the order the branches were created, since it
    becomes the order of the training samples this session contributes.
    """
    record, leaf_hashes = _record(step_one="tap", step_two="swipe", step_three="done")

    exported = _SHARD_CLASS._implicit_export_leaf_hashes(None, record)

    assert exported == leaf_hashes


def test_implicit_export_returns_single_leaf_for_appending_agent() -> None:
    record, leaf_hashes = _record(only_step="tap")

    assert _SHARD_CLASS._implicit_export_leaf_hashes(None, record) == leaf_hashes


def test_implicit_export_rejects_forest_without_committed_response() -> None:
    forest = SessionForest.create_empty(session_id="sess-no-resp")
    forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )

    with pytest.raises(_NonFinalizableExportError):
        _SHARD_CLASS._implicit_export_leaf_hashes(None, _SessionRecord(forest=forest))


def test_implicit_export_rejects_missing_forest() -> None:
    with pytest.raises(_NonFinalizableExportError):
        _SHARD_CLASS._implicit_export_leaf_hashes(None, _SessionRecord(forest=None))
