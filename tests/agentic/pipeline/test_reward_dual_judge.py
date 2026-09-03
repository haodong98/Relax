# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from argparse import Namespace

import pytest

from relax.agentic.pipeline.reward import JudgeReplacementBudgetExceeded, RewardDomain
from relax.engine.rewards.dual_agentic_judge import JudgeSampleRejected
from relax.utils.judge_config import (
    JudgeServicesConfig,
    dual_judge_benchmark_invariant_hash,
    parse_judge_services_config,
)
from relax.utils.types import Sample


class _Context:
    def __init__(self, content_hash=None):
        self.released = False
        self.content_hash = content_hash

    def release(self):
        self.released = True


def _judge_services(benchmark_mode: str = "dual") -> JudgeServicesConfig:
    def service(model_path: str, port_base: int) -> dict:
        return {
            "model_path": model_path,
            "num_gpus_per_engine": 1,
            "engine_config": {"max_context_len": 1024},
            "sampling_config": {"temperature": 0.0, "max_response_len": 24},
            "max_input_tokens": 1000,
            "max_output_tokens": 24,
            "timeout_s": 30,
            "max_attempts": 3,
            "max_concurrency": 2,
            "port_base": port_base,
        }

    return parse_judge_services_config(
        {
            "schema_version": 1,
            "benchmark_mode": benchmark_mode,
            "max_group_replacements_per_step": 8,
            "answer_accuracy": service("/accuracy", 16000),
            "multi_turn_reasoning": service("/vlm", 17000),
        }
    )


@pytest.mark.asyncio
async def test_sample_rejection_atomically_drops_group_and_releases_sibling_contexts(monkeypatch):
    sibling_started = asyncio.Event()

    async def fake_rm(_args, sample):
        if sample.index == 0:
            await sibling_started.wait()
            raise JudgeSampleRejected("timeout", "required judge failed")
        sibling_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", fake_rm)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=None,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services(),
        dual_judge_benchmark_invariant_hash="test-invariant",
    )
    contexts = [_Context(), _Context()]
    group = [
        Sample(session_id=f"s{index}", index=index, group_index=0, reward=7.0, reward_context=contexts[index])
        for index in range(2)
    ]
    domain = RewardDomain(args=args, group_filter=None)
    await domain.ingest_groups([group])
    await domain.wait_for_next_completion()
    await domain.precompute_once()
    assert domain.drain_rejected_group_keys()
    assert not domain.has_ready_output()
    assert all(context.released for context in contexts)
    assert all(sample.reward_context is None for sample in group)
    assert all(sample.metadata["reward_audit"]["environment_reward"] == 7.0 for sample in group)
    traces = [sample.metadata["agentic_trace"] for sample in group]
    assert all("reward_end_at" in trace["events"] for trace in traces)
    assert {trace["reward"]["pipeline_status"] for trace in traces} == {"error", "cancelled"}
    assert traces[0]["reward"]["pipeline_error_code"] == "timeout"
    snapshots = domain.drain_reward_trace_snapshots()
    assert len(snapshots) == 2
    assert {snapshot["agentic_trace"]["reward"]["terminal_outcome"] for snapshot in snapshots} == {"group_rejected"}
    await domain.shutdown()


@pytest.mark.asyncio
async def test_replacement_budget_fails_after_eight_groups(monkeypatch):
    async def reject(_args, _sample):
        raise JudgeSampleRejected("timeout", "required judge failed")

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", reject)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=None,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services(),
        dual_judge_benchmark_invariant_hash="test-invariant",
    )
    domain = RewardDomain(args=args, group_filter=None)
    for group_index in range(8):
        group = [Sample(session_id=f"s{group_index}", index=0, group_index=group_index)]
        await domain.ingest_groups([group])
        await domain.wait_for_next_completion()
        await domain.precompute_once()
        assert domain.drain_rejected_group_keys()

    ninth = [Sample(session_id="s9", index=0, group_index=9)]
    await domain.ingest_groups([ninth])
    await domain.wait_for_next_completion()
    with pytest.raises(JudgeReplacementBudgetExceeded):
        await domain.precompute_once()
    await domain.shutdown()


@pytest.mark.asyncio
async def test_recorded_benchmark_mode_bypasses_judges_and_preserves_reward(monkeypatch):
    async def unexpected_rm(_args, _sample):
        raise AssertionError("recorded benchmark mode must not call the reward executor")

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", unexpected_rm)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=None,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services("recorded"),
        dual_judge_benchmark_invariant_hash="test-invariant",
        use_critic=True,
    )
    sample = Sample(
        session_id="recorded",
        index=0,
        group_index=0,
        reward=0.75,
        metadata={},
        reward_context=_Context(content_hash="same-trajectory"),
    )
    domain = RewardDomain(args=args, group_filter=None)

    await domain.ingest_groups([[sample]])

    assert domain.has_ready_output()
    assert sample.reward["score"] == 0.75
    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["pipeline_status"] == "bypassed"
    assert reward_trace["executor_status"] == "bypassed"
    assert reward_trace["context_hash"] == "same-trajectory"
    assert isinstance(reward_trace["recorded_reward_hash"], str)
    assert reward_trace["benchmark_invariant_hash"] == dual_judge_benchmark_invariant_hash(args)
    assert reward_trace["benchmark_invariant_hash"] != args.dual_judge_benchmark_invariant_hash
    assert reward_trace["expected_trainer_components"] == ["actor", "critic"]
    assert (
        sample.metadata["agentic_trace"]["events"]["reward_start_at"]
        == sample.metadata["agentic_trace"]["events"]["reward_end_at"]
    )
    await domain.shutdown()


@pytest.mark.asyncio
async def test_preloaded_composite_reward_does_not_bypass_required_judges(monkeypatch):
    calls = 0

    async def fake_rm(_args, sample):
        nonlocal calls
        calls += 1
        assert sample.reward is None
        return {"score": 0.9, "_schema_version": "relax.composite_reward.v1"}

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", fake_rm)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=None,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services("dual"),
    )
    preloaded = {"score": 0.25, "_schema_version": "relax.composite_reward.v1"}
    sample = Sample(
        session_id="preloaded",
        index=0,
        group_index=0,
        reward=preloaded,
        metadata={},
        reward_context=_Context(content_hash="trajectory"),
    )
    domain = RewardDomain(args=args, group_filter=None)

    await domain.ingest_groups([[sample]])
    await domain.wait_for_next_completion()
    await domain.precompute_once()

    assert calls == 1
    assert sample.reward["score"] == pytest.approx(0.9)
    assert sample.metadata["reward_audit"]["environment_reward"] == preloaded
    await domain.shutdown()


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_global_reward_slot_is_traced(monkeypatch):
    async def unexpected_rm(_args, _sample):
        raise AssertionError("queued sample must be cancelled before reward execution")

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", unexpected_rm)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=1,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services(),
        dual_judge_benchmark_invariant_hash="test-invariant",
    )
    sample = Sample(session_id="queued", index=0, group_index=0, reward=1.0, metadata={})
    domain = RewardDomain(args=args, group_filter=None)
    await domain._reward_semaphore.acquire()
    await domain.ingest_groups([[sample]])
    await asyncio.sleep(0)

    await domain.drop_resident_groups()
    domain._reward_semaphore.release()

    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["pipeline_status"] == "cancelled"
    assert reward_trace["pipeline_error_code"] == "cancelled_while_queued"
    assert reward_trace["global_queue_elapsed_s"] >= 0
    assert "reward_end_at" in sample.metadata["agentic_trace"]["events"]


@pytest.mark.asyncio
async def test_immediate_queued_cancellation_is_traced_before_task_starts(monkeypatch):
    async def unexpected_rm(_args, _sample):
        raise AssertionError("queued sample must be cancelled before reward execution")

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", unexpected_rm)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=1,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services(),
        dual_judge_benchmark_invariant_hash="test-invariant",
    )
    sample = Sample(session_id="queued-immediate", index=0, group_index=0, reward=1.0, metadata={})
    domain = RewardDomain(args=args, group_filter=None)
    await domain._reward_semaphore.acquire()
    await domain.ingest_groups([[sample]])

    await domain.drop_resident_groups()
    domain._reward_semaphore.release()

    reward_trace = sample.metadata["agentic_trace"]["reward"]
    assert reward_trace["pipeline_error_code"] == "cancelled_while_queued"
    assert reward_trace["global_queue_elapsed_s"] >= 0


@pytest.mark.asyncio
async def test_shutdown_drains_cancelled_reward_before_releasing_context(monkeypatch):
    task_cancelled = asyncio.Event()

    class ContextReleasedTooEarly(RuntimeError):
        pass

    class OrderedContext(_Context):
        def release(self):
            if not task_cancelled.is_set():
                raise ContextReleasedTooEarly("reward context was released before the reward task drained")
            super().release()

    async def blocking_rm(_args, _sample):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            task_cancelled.set()
            raise

    monkeypatch.setattr("relax.agentic.pipeline.reward._async_rm", blocking_rm)
    args = Namespace(
        group_rm=False,
        reward_max_concurrency=None,
        agentic_custom_advantage_path=None,
        rm_type="dual-agentic-judge",
        judge_services=_judge_services(),
        dual_judge_benchmark_invariant_hash="test-invariant",
    )
    context = OrderedContext()
    sample = Sample(session_id="shutdown-order", index=0, group_index=0, reward=1.0, reward_context=context)
    domain = RewardDomain(args=args, group_filter=None)
    await domain.ingest_groups([[sample]])
    await asyncio.sleep(0)

    await domain.shutdown()

    assert task_cancelled.is_set()
    assert context.released
    assert sample.reward_context is None


@pytest.mark.asyncio
async def test_release_waits_for_all_cancelled_tasks_before_context_cleanup():
    sibling_drained = asyncio.Event()

    class OrderedContext(_Context):
        def release(self):
            assert sibling_drained.is_set()
            super().release()

    async def raises_after_cancel():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise RuntimeError("cleanup failure") from exc

    async def drains_after_cancel():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            sibling_drained.set()
            raise

    args = Namespace(
        group_rm=False,
        reward_max_concurrency=None,
        agentic_custom_advantage_path=None,
        rm_type=None,
    )
    contexts = [OrderedContext(), OrderedContext()]
    group = [
        Sample(session_id=f"drain-{index}", index=index, group_index=0, reward_context=contexts[index])
        for index in range(2)
    ]
    domain = RewardDomain(args=args, group_filter=None)
    for sample, coroutine in zip(group, (raises_after_cancel(), drains_after_cancel())):
        domain._inflight_sample_tasks[(sample.session_id, sample.index, None)] = asyncio.create_task(coroutine)
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="cleanup failure"):
        await domain._release_group_sample_reward_cache(group)

    assert sibling_drained.is_set()
    assert all(context.released for context in contexts)
    assert all(sample.reward_context is None for sample in group)
