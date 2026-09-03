# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from argparse import Namespace

import pytest

import relax.engine.rewards.dual_agentic_judge as dual_judge_module
from relax.engine.rewards.dual_agentic_judge import (
    DualJudgeExecutor,
    JudgeSampleRejected,
    recorded_reward_hash,
)
from relax.utils.judge_config import JudgeServicesConfig
from relax.utils.types import Sample

from .test_reward_projection import _context, _spec


class _FakeClient:
    def __init__(self, response: str, barrier: asyncio.Event, calls: list[str], role: str):
        self.response = response
        self.barrier = barrier
        self.calls = calls
        self.role = role

    async def generate_once(self, payload):
        self.calls.append(self.role)
        if len(self.calls) == 2:
            self.barrier.set()
        await asyncio.wait_for(self.barrier.wait(), 1)
        return self.response


class _ImmediateClient:
    def __init__(self, response: str, calls: list[str], role: str):
        self.response = response
        self.calls = calls
        self.role = role

    async def generate_once(self, payload):
        self.calls.append(self.role)
        return self.response


class _ProfiledClient(_ImmediateClient):
    async def generate_once_profiled(self, payload):
        self.calls.append(self.role)
        return self.response, {"engine_http_elapsed_s": 0.125, "input_tokens": 32}


class _SequencedProfiledClient:
    def __init__(self):
        self.responses = [
            ("invalid", {"request_total_elapsed_s": 0.1, "input_tokens": 8}),
            ('{"score":1,"verdict":"pass","rationale":"x"}', {"request_total_elapsed_s": 0.2, "input_tokens": 8}),
        ]

    async def generate_once_profiled(self, payload):
        return self.responses.pop(0)


def test_recorded_reward_hash_tracks_numeric_training_reward_only():
    assert recorded_reward_hash(0.5) == recorded_reward_hash({"score": 0.5, "diagnostic": "ignored"})
    assert recorded_reward_hash(0.5) != recorded_reward_hash(0.6)


@pytest.mark.asyncio
async def test_dual_judges_fan_out_and_aggregate_zero_as_valid():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning)
    barrier = asyncio.Event()
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _FakeClient('{"score":0,"verdict":"fail","rationale":"x"}', barrier, calls, "a"),
            "judge_multiturn_vlm": _FakeClient('{"score":1.0,"verdict":"pass","rationale":"x"}', barrier, calls, "r"),
        },
    )
    sample = Sample(reward=7.0, reward_context=_context(), metadata={})
    reward = await DualJudgeExecutor(args).score(sample)
    assert set(calls) == {"a", "r"}
    assert reward["score"] == 0.2
    assert sample.metadata["reward_audit"]["environment_reward"] == 7.0
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["executor_status"] == "success"
    assert reward_trace["executor_elapsed_s"] >= 0
    assert reward_trace["answer_accuracy_projection_elapsed_s"] >= 0
    assert reward_trace["multi_turn_reasoning_projection_elapsed_s"] >= 0
    assert set(reward_trace["judges"]) == {"answer_accuracy", "multi_turn_reasoning"}
    for judge_trace in reward_trace["judges"].values():
        assert judge_trace["status"] == "success"
        assert judge_trace["attempt_count"] == 1
        assert judge_trace["queue_elapsed_s"] >= 0
        assert judge_trace["payload_prep_elapsed_s"] >= 0
        assert judge_trace["http_elapsed_s"] >= 0
        assert judge_trace["parse_elapsed_s"] >= 0
        assert judge_trace["elapsed_s"] >= 0
    assert sample.reward_context is None


@pytest.mark.asyncio
async def test_dual_judge_projections_run_concurrently_off_the_event_loop(monkeypatch):
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning)
    calls: list[str] = []
    projection_barrier = threading.Barrier(2)
    original_accuracy = dual_judge_module.build_accuracy_projection
    original_reasoning = dual_judge_module.build_reasoning_projection

    def build_accuracy(*args, **kwargs):
        projection_barrier.wait(timeout=1)
        return original_accuracy(*args, **kwargs)

    def build_reasoning(*args, **kwargs):
        projection_barrier.wait(timeout=1)
        return original_reasoning(*args, **kwargs)

    monkeypatch.setattr(dual_judge_module, "build_accuracy_projection", build_accuracy)
    monkeypatch.setattr(dual_judge_module, "build_reasoning_projection", build_reasoning)
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _ImmediateClient('{"score":1,"verdict":"pass","rationale":"x"}', calls, "accuracy"),
            "judge_multiturn_vlm": _ImmediateClient(
                '{"score":1.0,"verdict":"pass","rationale":"x"}', calls, "reasoning"
            ),
        },
    )

    reward = await DualJudgeExecutor(args).score(Sample(reward_context=_context(), metadata={}))

    assert reward["score"] == 1.0
    assert set(calls) == {"accuracy", "reasoning"}


@pytest.mark.asyncio
async def test_required_invalid_judge_rejects_without_partial_score():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning)
    barrier = asyncio.Event()
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _FakeClient("invalid", barrier, calls, "a"),
            "judge_multiturn_vlm": _FakeClient('{"score":1.0,"verdict":"pass","rationale":"x"}', barrier, calls, "r"),
        },
    )
    sample = Sample(reward_context=_context(), metadata={})
    executor = DualJudgeExecutor(args)
    with pytest.raises(JudgeSampleRejected, match="invalid JSON twice"):
        await executor.score(sample)
    assert sample.reward is None
    assert sample.reward_context is None
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["executor_status"] == "rejected"
    assert reward_trace["executor_error_code"] == "invalid_response"
    accuracy_trace = reward_trace["judges"]["answer_accuracy"]
    assert accuracy_trace["status"] == "rejected"
    assert accuracy_trace["error_code"] == "invalid_response"
    assert accuracy_trace["attempt_count"] == 2
    assert accuracy_trace["invalid_response_count"] == 2
    assert executor.drain_metrics() == {
        "judge_retry/judge_accuracy/invalid_response": 1,
        "judge_error/judge_accuracy/invalid_response": 1,
    }


@pytest.mark.asyncio
async def test_accuracy_benchmark_mode_does_not_call_reasoning_judge():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, benchmark_mode="accuracy")
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _ImmediateClient('{"score":1,"verdict":"pass","rationale":"x"}', calls, "accuracy"),
            "judge_multiturn_vlm": _ImmediateClient(
                '{"score":0.0,"verdict":"fail","rationale":"x"}', calls, "reasoning"
            ),
        },
    )

    sample = Sample(reward_context=_context(), metadata={})
    reward = await DualJudgeExecutor(args).score(sample)

    assert calls == ["accuracy"]
    assert reward["score"] == 1.0
    assert set(sample.metadata["reward_audit"]["judges"]) == {"answer_accuracy"}
    assert set(sample.metadata["agentic_trace"]["reward"]["judges"]) == {"answer_accuracy"}


@pytest.mark.asyncio
async def test_per_turn_trigger_aggregates_round_scores_without_terminal_vlm_call():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, reasoning_trigger="per_turn")
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _ImmediateClient('{"score":1,"verdict":"pass","rationale":"x"}', calls, "accuracy"),
            "judge_multiturn_vlm": _ImmediateClient(
                '{"score":0.0,"verdict":"fail","rationale":"x"}', calls, "reasoning"
            ),
        },
    )
    context = _context()
    context.per_turn_judgements = [
        {"turn_index": 0, "role": "judge_multiturn_vlm", "status": "success", "score": 0.2},
        {"turn_index": 1, "role": "judge_multiturn_vlm", "status": "success", "score": 0.8},
    ]
    sample = Sample(reward_context=context, metadata={})

    reward = await DualJudgeExecutor(args).score(sample)

    assert calls == ["accuracy"]
    assert reward["score"] == pytest.approx(0.9)
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["reasoning_trigger"] == "per_turn"
    assert reward_trace["per_turn_judge_count"] == 2
    assert set(reward_trace["judges"]) == {"answer_accuracy"}


@pytest.mark.asyncio
async def test_turn_reasoning_records_round_trace_and_releases_its_context():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, reasoning_trigger="per_turn")
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_multiturn_vlm": _ImmediateClient(
                '{"score":0.5,"verdict":"mixed","rationale":"x"}', calls, "reasoning"
            )
        },
    )
    context = _context()
    context.identity["turn_index"] = 0

    outcome = await DualJudgeExecutor(args).score_turn_reasoning(
        context=context,
        outcome_base={"turn_index": 0, "role": "judge_multiturn_vlm", "events": {}},
    )

    assert calls == ["reasoning"]
    assert outcome["status"] == "success"
    assert outcome["score"] == pytest.approx(0.5)
    assert "turn_judge_projection_start_at" in outcome["events"]
    assert "turn_judge_queue_enter_at" in outcome["events"]
    assert "turn_judge_end_at" in outcome["events"]
    assert context.turns == []


@pytest.mark.asyncio
async def test_per_turn_zero_rounds_uses_explicit_terminal_fallback():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, reasoning_trigger="per_turn")
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _ImmediateClient('{"score":1,"verdict":"pass","rationale":"x"}', calls, "accuracy"),
            "judge_multiturn_vlm": _ImmediateClient(
                '{"score":0.5,"verdict":"mixed","rationale":"x"}', calls, "reasoning"
            ),
        },
    )
    context = _context()
    context.per_turn_fallback_terminal_once = True
    sample = Sample(reward_context=context, metadata={})

    reward = await DualJudgeExecutor(args).score(sample)

    assert set(calls) == {"accuracy", "reasoning"}
    assert reward["score"] == pytest.approx(0.9)
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["reasoning_trigger"] == "per_turn"
    assert reward_trace["reasoning_execution_trigger"] == "terminal_once_fallback"
    assert reward_trace["per_turn_fallback_terminal_once"] is True


@pytest.mark.asyncio
async def test_judge_trace_preserves_server_timing_decomposition():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, benchmark_mode="accuracy")
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _ProfiledClient('{"score":1,"verdict":"pass","rationale":"x"}', calls, "accuracy")
        },
    )

    sample = Sample(reward_context=_context(), metadata={})
    await DualJudgeExecutor(args).score(sample)

    judge_trace = sample.metadata["agentic_trace"]["reward"]["judges"]["answer_accuracy"]
    assert judge_trace["server"] == {"engine_http_elapsed_s": 0.125, "input_tokens": 32}


@pytest.mark.asyncio
async def test_profiled_invalid_retry_accumulates_all_server_attempts():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, benchmark_mode="accuracy")
    args = Namespace(judge_services=config, _judge_clients={"judge_accuracy": _SequencedProfiledClient()})

    sample = Sample(reward_context=_context(), metadata={})
    await DualJudgeExecutor(args).score(sample)

    judge_trace = sample.metadata["agentic_trace"]["reward"]["judges"]["answer_accuracy"]
    assert len(judge_trace["server_attempts"]) == 2
    assert judge_trace["server"]["request_total_elapsed_s"] == pytest.approx(0.3)
    assert judge_trace["server"]["input_tokens"] == 16


@pytest.mark.asyncio
async def test_dual_shadow_waits_for_both_judges_but_returns_recorded_reward():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, benchmark_mode="dual_shadow")
    barrier = asyncio.Event()
    calls: list[str] = []
    args = Namespace(
        judge_services=config,
        _judge_clients={
            "judge_accuracy": _FakeClient('{"score":1,"verdict":"pass","rationale":"x"}', barrier, calls, "accuracy"),
            "judge_multiturn_vlm": _FakeClient(
                '{"score":0.0,"verdict":"fail","rationale":"x"}', barrier, calls, "reasoning"
            ),
        },
    )

    sample = Sample(reward=0.75, reward_context=_context(), metadata={})
    reward = await DualJudgeExecutor(args).score(sample)

    assert set(calls) == {"accuracy", "reasoning"}
    assert reward["score"] == 0.75
    assert reward["_benchmark_recorded"] is True
    assert sample.metadata["reward_audit"]["benchmark_mode"] == "dual_shadow"


@pytest.mark.asyncio
async def test_shadow_missing_context_records_failure_without_changing_workload():
    accuracy = _spec("answer_accuracy", "judge_accuracy")
    reasoning = _spec("multi_turn_reasoning", "judge_multiturn_vlm", media=True)
    config = JudgeServicesConfig(accuracy, reasoning, benchmark_mode="dual_shadow")
    args = Namespace(judge_services=config, _judge_clients={})
    sample = Sample(reward=0.75, reward_context=None, metadata={})

    reward = await DualJudgeExecutor(args).score(sample)

    assert reward["score"] == 0.75
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["executor_status"] == "shadow_judge_error"
    assert reward_trace["executor_error_code"] == "missing_context"
