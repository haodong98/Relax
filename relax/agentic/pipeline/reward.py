# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from relax.agentic.pipeline import ExportMode, GroupKey, PendingExportUnit, SampleKey, sample_group_key, sample_key
from relax.agentic.profile import ensure_reward_trace, mark_metadata_agentic_event
from relax.utils.misc import load_function


def _sample_needs_reward(sample: Any) -> bool:
    return sample.reward is None


class JudgeReplacementBudgetExceeded(RuntimeError):
    """Raised when required judges reject too many GRPO groups in one step."""


def _effective_reward_concurrency(args, *, explicit_limit: int | None = None) -> int | None:
    if explicit_limit is not None:
        if explicit_limit <= 0:
            raise RuntimeError(f"reward concurrency limit must be positive when set, got {explicit_limit}.")
        return explicit_limit
    configured = args.reward_max_concurrency
    if configured is None:
        return None
    if configured <= 0:
        raise RuntimeError(f"reward_max_concurrency must be positive when set, got {configured}.")
    return configured


async def _async_rm(args, sample):
    from relax.engine.rewards import async_rm

    return await async_rm(args, sample)


async def _batched_async_rm(args, samples):
    from relax.engine.rewards import batched_async_rm

    return await batched_async_rm(args, samples)


@dataclass
class RewardWaitingGroup:
    expected_count: int
    units_by_slot: dict[int, list[PendingExportUnit]] = field(default_factory=dict)

    def add_slot(self, *, slot_idx: int, units: list[PendingExportUnit]) -> None:
        self.units_by_slot[slot_idx] = units

    def is_complete(self) -> bool:
        return len(self.units_by_slot) >= self.expected_count

    def materialized_units(self) -> list[PendingExportUnit]:
        units: list[PendingExportUnit] = []
        for idx in range(self.expected_count):
            units.extend(self.units_by_slot.get(idx, []))
        return units

    def materialized_group(self) -> list[Any]:
        return [unit.sample for unit in self.materialized_units()]

    def metadata_by_slot(self) -> list[dict[Any, dict[str, Any]]]:
        payload: list[dict[Any, dict[str, Any]]] = []
        for idx in range(self.expected_count):
            slot_payload: dict[Any, dict[str, Any]] = {}
            for unit in self.units_by_slot.get(idx, []):
                slot_payload[unit.name] = copy.deepcopy(unit.sample.metadata)
            payload.append(slot_payload)
        return payload

    def materialized_slot_count(self) -> int:
        return len(self.units_by_slot)


class RewardDomain:
    def __init__(
        self,
        *,
        args,
        group_filter: Callable[[list[Any]], bool] | None,
        max_submissions_per_step: int | None = None,
        use_custom_advantage: bool = True,
    ) -> None:
        self.args = args
        self.group_rm = args.group_rm
        self.group_filter = group_filter
        self.max_submissions_per_step = _effective_reward_concurrency(args, explicit_limit=max_submissions_per_step)
        custom_advantage_path = getattr(args, "agentic_custom_advantage_path", None) if use_custom_advantage else None
        self.custom_advantage_func = (
            load_function(custom_advantage_path) if custom_advantage_path is not None else None
        )
        self._waiting_groups: dict[GroupKey, RewardWaitingGroup] = {}
        self._inflight_sample_tasks: dict[SampleKey, asyncio.Task] = {}
        self._inflight_group_tasks: dict[GroupKey, asyncio.Task] = {}
        self._completed_samples: dict[SampleKey, Any] = {}
        self._completed_group_results: dict[GroupKey, list[Any]] = {}
        self._ready_dispatches: deque[tuple[GroupKey, list[Any]]] = deque()
        self._ready_scored_sample_count = 0
        self._progress_counted_samples: set[SampleKey] = set()
        self._sample_group_keys: dict[SampleKey, GroupKey] = {}
        self._prepared_dual_samples: set[SampleKey] = set()
        self._reward_arrival_ns: dict[SampleKey, int] = {}
        self._reward_trace_snapshots: dict[SampleKey, dict[str, Any]] = {}
        self._rejected_group_keys: set[GroupKey] = set()
        judge_services = getattr(args, "judge_services", None)
        self._max_judge_replacements = (
            judge_services.max_group_replacements_per_step if judge_services is not None else 0
        )
        self._judge_replacements_this_step = 0
        self._judge_metrics: dict[str, int] = {}
        self._benchmark_invariant_hash = None
        self._expected_trainer_components = ["actor"]
        if getattr(args, "use_critic", False):
            self._expected_trainer_components.append("critic")
        if getattr(args, "rm_type", None) == "dual-agentic-judge":
            from relax.utils.judge_config import dual_judge_benchmark_invariant_hash

            self._benchmark_invariant_hash = dual_judge_benchmark_invariant_hash(args)
        self._reward_semaphore = (
            asyncio.Semaphore(self.max_submissions_per_step) if self.max_submissions_per_step is not None else None
        )

    def rebind_step(
        self,
        *,
        group_filter: Callable[[list[Any]], bool] | None,
    ) -> None:
        self.group_filter = group_filter
        self._ready_scored_sample_count = 0
        self._progress_counted_samples.clear()
        self._rejected_group_keys.clear()
        self._judge_replacements_this_step = 0
        self._judge_metrics.clear()

    async def drop_waiting_groups_by_key(self, group_keys: set[GroupKey]) -> int:
        if not group_keys:
            return 0
        dropped_group_keys: set[GroupKey] = set()
        orphaned_group_tasks: list[asyncio.Task] = []
        dropped_groups: list[list[Any]] = []
        for key in group_keys:
            if key in self._completed_group_results:
                raise RuntimeError(f"RewardDomain cannot drop completed group after runtime discarded it: {key!r}")
        for key, _group in self._ready_dispatches:
            if key in group_keys:
                raise RuntimeError(f"RewardDomain cannot drop ready group after runtime discarded it: {key!r}")
        for key in group_keys:
            waiting_group = self._waiting_groups.pop(key, None)
            if waiting_group is None:
                group_task = self._inflight_group_tasks.pop(key, None)
                if group_task is not None:
                    group_task.cancel()
                    orphaned_group_tasks.append(group_task)
                    dropped_group_keys.add(key)
                continue
            materialized_group = waiting_group.materialized_group()
            dropped_groups.append(materialized_group)
            self._drop_scored_progress_for_group(materialized_group)
            await self._release_group_sample_reward_cache(materialized_group)
            dropped_group_keys.add(key)
        await self._drain_cancelled_tasks(orphaned_group_tasks)
        for group in dropped_groups:
            self._capture_group_reward_trace_snapshots(group, terminal_outcome="runtime_discarded")
        return len(dropped_group_keys)

    def _cancel_inflight_sample_tasks(self) -> list[asyncio.Task]:
        inflight_sample_tasks = list(self._inflight_sample_tasks.values())
        self._inflight_sample_tasks.clear()
        for task in inflight_sample_tasks:
            task.cancel()
        return inflight_sample_tasks

    def _cancel_inflight_group_tasks(self) -> list[asyncio.Task]:
        inflight_group_tasks = list(self._inflight_group_tasks.values())
        self._inflight_group_tasks.clear()
        for task in inflight_group_tasks:
            task.cancel()
        return inflight_group_tasks

    async def _drain_cancelled_tasks(self, tasks: list[asyncio.Task]) -> None:
        from relax.engine.rewards.dual_agentic_judge import JudgeSampleRejected

        first_error: BaseException | None = None
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, JudgeSampleRejected):
                pass
            except BaseException as exc:
                # Do not let one already-failed task skip draining a sibling
                # that may still own a shielded projection thread.  Callers
                # release RewardContext only after this method returns.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _clear_resident_state(self) -> None:
        self._waiting_groups.clear()
        self._completed_samples.clear()
        self._completed_group_results.clear()
        self._ready_dispatches.clear()
        self._ready_scored_sample_count = 0
        self._progress_counted_samples.clear()
        self._sample_group_keys.clear()
        self._prepared_dual_samples.clear()
        self._reward_arrival_ns.clear()
        self._rejected_group_keys.clear()
        self._judge_metrics.clear()

    def _capture_reward_trace_snapshot(self, sample: Any, *, terminal_outcome: str | None = None) -> None:
        if getattr(self.args, "rm_type", None) != "dual-agentic-judge":
            return
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else None
        if metadata is None:
            return
        agentic_trace = metadata.get("agentic_trace")
        reward_trace = agentic_trace.get("reward") if isinstance(agentic_trace, dict) else None
        if not isinstance(reward_trace, dict):
            return
        pipeline_status = reward_trace.get("pipeline_status")
        executor_status = reward_trace.get("executor_status")
        if (
            terminal_outcome is None
            and pipeline_status not in {"error", "cancelled"}
            and executor_status
            not in {
                "error",
                "cancelled",
                "rejected",
            }
        ):
            return
        snapshot = copy.deepcopy(metadata)
        if terminal_outcome is not None:
            snapshot["agentic_trace"]["reward"]["terminal_outcome"] = terminal_outcome
        self._reward_trace_snapshots[sample_key(sample)] = snapshot

    def _capture_group_reward_trace_snapshots(self, group: list[Any], *, terminal_outcome: str) -> None:
        for sample in group:
            self._capture_reward_trace_snapshot(sample, terminal_outcome=terminal_outcome)

    def drain_reward_trace_snapshots(self) -> list[dict[str, Any]]:
        snapshots = list(self._reward_trace_snapshots.values())
        self._reward_trace_snapshots.clear()
        return snapshots

    def _prepare_dual_judge_sample(self, sample: Any) -> None:
        if getattr(self.args, "rm_type", None) != "dual-agentic-judge":
            return
        key = sample_key(sample)
        if key in self._prepared_dual_samples:
            return
        self._prepared_dual_samples.add(key)
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        reward_trace = ensure_reward_trace(metadata)
        group_key = self._sample_group_keys.get(key)
        reward_trace.update(
            {
                "sample_key": list(key),
                "group_key": [list(item) for item in group_key] if group_key is not None else None,
                "group_index": getattr(sample, "group_index", None),
                "sample_index": getattr(sample, "index", None),
                "weight_versions": list(getattr(sample, "weight_versions", []) or []),
                "benchmark_invariant_hash": self._benchmark_invariant_hash,
                "expected_trainer_components": list(self._expected_trainer_components),
            }
        )
        reward_context = getattr(sample, "reward_context", None)
        context_hash = getattr(reward_context, "content_hash", None)
        if isinstance(context_hash, str) and context_hash:
            # This hash is computed while constructing RewardContextV1, so it
            # adds no benchmark-only projection/tokenization work to the
            # recorded baseline. The analyzer uses it to reject unpaired
            # trajectory workloads across otherwise matching sample slots.
            reward_trace["context_hash"] = context_hash
        context_identity = getattr(reward_context, "identity", None)
        trajectory_hash = context_identity.get("leaf_state_hash") if isinstance(context_identity, dict) else None
        if isinstance(trajectory_hash, str) and trajectory_hash:
            # Context hashes intentionally differ between terminal_once
            # (full trajectory) and per_turn (accuracy-only terminal scope).
            # The leaf state hash remains the stable whole-trajectory pairing
            # identity across those two experimental implementations.
            reward_trace["trajectory_hash"] = trajectory_hash
        benchmark_mode = getattr(self.args.judge_services, "benchmark_mode", "dual")
        reward_trace["benchmark_mode"] = benchmark_mode
        reward_trace["reasoning_trigger"] = getattr(self.args.judge_services, "reasoning_trigger", "terminal_once")
        reward = getattr(sample, "reward", None)
        if benchmark_mode in {"recorded", "accuracy_shadow", "dual_shadow"} and reward is not None:
            from relax.engine.rewards.dual_agentic_judge import recorded_reward_hash

            reward_trace["recorded_reward_hash"] = recorded_reward_hash(reward)
        if benchmark_mode == "recorded":
            if reward is None:
                raise RuntimeError("dual-agentic-judge recorded benchmark mode requires a precomputed reward")
            from relax.engine.rewards.dual_agentic_judge import recorded_reward_payload

            metadata.setdefault("reward_audit", {}).setdefault("environment_reward", copy.deepcopy(reward))
            sample.reward = recorded_reward_payload(reward)
            reward_trace["pipeline_status"] = "bypassed"
            reward_trace["executor_status"] = "bypassed"
            reward_trace["global_queue_elapsed_s"] = 0.0
            reward_trace["pipeline_elapsed_s"] = 0.0
            return
        if reward is not None:
            metadata.setdefault("reward_audit", {}).setdefault("environment_reward", copy.deepcopy(reward))
        sample.reward = None

    def _mark_reward_arrival(self, sample: Any) -> None:
        key = sample_key(sample)
        arrived_at = time.time()
        mark_metadata_agentic_event(sample.metadata, "reward_arrive_at", arrived_at)
        if _sample_needs_reward(sample):
            self._reward_arrival_ns.setdefault(key, time.perf_counter_ns())
            return
        if getattr(self.args, "rm_type", None) == "dual-agentic-judge":
            mark_metadata_agentic_event(sample.metadata, "reward_start_at", arrived_at)
            mark_metadata_agentic_event(sample.metadata, "reward_end_at", arrived_at)

    @staticmethod
    def _unit_from_sample(sample: Any) -> PendingExportUnit:
        return PendingExportUnit(
            name=None,
            sample=sample,
            assistant_token_spans=[],
            mode=ExportMode.IMPLICIT,
        )

    async def shutdown(self) -> None:
        # Release RewardContext only after every task that may still read it
        # has observed cancellation and exited.  In particular, dual-judge
        # projection uses a worker thread that cannot be stopped directly.
        shutdown_error: BaseException | None = None
        try:
            await self.drop_resident_groups()
        except BaseException as exc:
            shutdown_error = exc
        inflight_sample_tasks = self._cancel_inflight_sample_tasks()
        inflight_group_tasks = self._cancel_inflight_group_tasks()
        try:
            await self._drain_cancelled_tasks(inflight_sample_tasks + inflight_group_tasks)
        except BaseException as exc:
            if shutdown_error is None:
                shutdown_error = exc
        finally:
            self._clear_resident_state()
        if getattr(self.args, "rm_type", None) == "dual-agentic-judge":
            from relax.engine.rewards.dual_agentic_judge import close_dual_judge_executor_for_current_loop

            try:
                await close_dual_judge_executor_for_current_loop(self.args)
            except BaseException as exc:
                if shutdown_error is None:
                    shutdown_error = exc
        if shutdown_error is not None:
            raise shutdown_error

    def _mark_queued_reward_cancelled(self, sample: Any) -> None:
        """Close a queued trace before cancellation can prevent the coroutine
        from starting."""
        if getattr(self.args, "rm_type", None) != "dual-agentic-judge":
            return
        events = sample.metadata.get("agentic_trace", {}).get("events", {})
        if "reward_start_at" in events:
            return
        key = sample_key(sample)
        ended_ns = time.perf_counter_ns()
        arrival_ns = self._reward_arrival_ns.pop(key, None)
        reward_trace = ensure_reward_trace(sample.metadata)
        reward_trace["pipeline_status"] = "cancelled"
        reward_trace["pipeline_error_code"] = "cancelled_while_queued"
        reward_trace["pipeline_elapsed_s"] = 0.0
        if arrival_ns is not None:
            reward_trace["global_queue_elapsed_s"] = (ended_ns - arrival_ns) / 1e9
        ended_at = time.time()
        mark_metadata_agentic_event(sample.metadata, "reward_start_at", ended_at)
        mark_metadata_agentic_event(sample.metadata, "reward_end_at", ended_at)

    async def _release_group_sample_reward_cache(self, group) -> None:
        """Detach group reward state, then release contexts after task drain.

        ``RewardContextV1.release`` clears media blobs in place.  A cancelled
        reward coroutine can still be waiting for a shielded executor thread,
        so releasing before awaiting the cancelled task lets that thread read
        cleared media.  Keep the context attached until every associated task
        has completed its cancellation path.
        """
        cancelled_tasks: list[asyncio.Task] = []
        contexts: list[tuple[Any, Any]] = []
        group_task = self._inflight_group_tasks.pop(sample_group_key(group), None)
        if group_task is not None:
            group_task.cancel()
            cancelled_tasks.append(group_task)
        for sample in group:
            context = getattr(sample, "reward_context", None)
            if context is not None:
                contexts.append((sample, context))
            key = sample_key(sample)
            task = self._inflight_sample_tasks.pop(key, None)
            if task is not None:
                self._mark_queued_reward_cancelled(sample)
                task.cancel()
                cancelled_tasks.append(task)
            self._completed_samples.pop(key, None)
            self._sample_group_keys.pop(key, None)
            self._prepared_dual_samples.discard(key)
            self._reward_arrival_ns.pop(key, None)
        try:
            await self._drain_cancelled_tasks(cancelled_tasks)
        finally:
            for sample, context in contexts:
                # The dual executor normally owns its successful/cancelled
                # context release and clears this field in its own ``finally``.
                # Only release here when the task did not already do so.
                if getattr(sample, "reward_context", None) is not context:
                    continue
                release = getattr(context, "release", None)
                if callable(release):
                    release()
                if hasattr(sample, "reward_context"):
                    sample.reward_context = None

    def _drop_scored_progress_for_group(self, group) -> None:
        if self.group_rm:
            return
        for sample in group:
            key = sample_key(sample)
            if key in self._progress_counted_samples:
                self._progress_counted_samples.remove(key)
                self._ready_scored_sample_count -= 1

    def accept_session_materializations(self, outputs) -> None:
        for batch in outputs:
            for record in batch:
                group_wait_key = record["group_key"]
                waiting_group = self._waiting_groups.get(group_wait_key)
                if waiting_group is None:
                    waiting_group = RewardWaitingGroup(
                        expected_count=record["expected_count"],
                    )
                    self._waiting_groups[group_wait_key] = waiting_group
                pending_units = record["pending_units"]
                waiting_group.add_slot(
                    slot_idx=record["slot_idx"],
                    units=pending_units,
                )
                for sample in (unit.sample for unit in pending_units):
                    self._sample_group_keys[sample_key(sample)] = group_wait_key
                    self._prepare_dual_judge_sample(sample)
                    self._mark_reward_arrival(sample)
                    self._note_scored_sample(sample)
                if self.custom_advantage_func is None:
                    self._start_missing_sample_rewards([unit.sample for unit in pending_units])

    async def ingest_groups(self, groups) -> None:
        for group in groups:
            await self._enqueue_group(group=group)

    async def _enqueue_group(self, *, group) -> None:
        group_wait_key = sample_group_key(group)
        for sample in group:
            self._sample_group_keys[sample_key(sample)] = group_wait_key
            self._prepare_dual_judge_sample(sample)
            self._mark_reward_arrival(sample)
        if self.group_rm:
            key = sample_group_key(group)
            if key not in self._completed_group_results and key not in self._inflight_group_tasks:
                self._inflight_group_tasks[key] = asyncio.create_task(self._run_group_reward(copy.deepcopy(group)))
            return
        self._start_missing_sample_rewards(group)
        if self._all_group_rewards_ready(group):
            await self._finalize_group(key=group_wait_key, group=group)
            return
        self._waiting_groups[group_wait_key] = RewardWaitingGroup(
            expected_count=len(group),
            units_by_slot={idx: [self._unit_from_sample(sample)] for idx, sample in enumerate(group)},
        )

    def _start_missing_sample_rewards(self, group) -> int:
        started = 0
        for sample in group:
            key = sample_key(sample)
            if (
                key in self._completed_samples
                or key in self._inflight_sample_tasks
                or not _sample_needs_reward(sample)
            ):
                continue
            self._inflight_sample_tasks[key] = asyncio.create_task(self._run_sample_reward(sample))
            started += 1
        return started

    def _note_scored_sample(self, sample) -> None:
        if self.group_rm:
            return
        if _sample_needs_reward(sample):
            return
        key = sample_key(sample)
        if key in self._progress_counted_samples:
            return
        self._progress_counted_samples.add(key)
        self._ready_scored_sample_count += 1

    async def _run_sample_reward(self, sample):
        async def score_sample():
            key = sample_key(sample)
            started_ns = time.perf_counter_ns()
            arrival_ns = self._reward_arrival_ns.pop(key, None)
            reward_trace = (
                ensure_reward_trace(sample.metadata)
                if getattr(self.args, "rm_type", None) == "dual-agentic-judge"
                else None
            )
            if reward_trace is not None:
                reward_trace["pipeline_status"] = "running"
                if arrival_ns is not None:
                    reward_trace["global_queue_elapsed_s"] = (started_ns - arrival_ns) / 1e9
            mark_metadata_agentic_event(sample.metadata, "reward_start_at")
            try:
                sample.reward = await _async_rm(self.args, sample)
                if reward_trace is not None:
                    reward_trace["pipeline_status"] = "success"
            except asyncio.CancelledError:
                if reward_trace is not None:
                    reward_trace["pipeline_status"] = "cancelled"
                    reward_trace["pipeline_error_code"] = "cancelled"
                raise
            except Exception as exc:
                if reward_trace is not None:
                    reward_trace["pipeline_status"] = "error"
                    reward_trace["pipeline_error_code"] = getattr(exc, "code", type(exc).__name__)
                raise
            finally:
                if reward_trace is not None:
                    reward_trace["pipeline_elapsed_s"] = (time.perf_counter_ns() - started_ns) / 1e9
                mark_metadata_agentic_event(sample.metadata, "reward_end_at")
            self._note_scored_sample(sample)
            return sample

        if self._reward_semaphore is None:
            return await score_sample()
        try:
            async with self._reward_semaphore:
                return await score_sample()
        except asyncio.CancelledError:
            self._mark_queued_reward_cancelled(sample)
            raise

    async def _run_group_reward(self, group):
        if self._reward_semaphore is None:
            group_started_at = time.time()
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "reward_start_at", group_started_at)
            rewards = await _batched_async_rm(self.args, group)
            for sample, reward in zip(group, rewards, strict=True):
                sample.reward = reward
                mark_metadata_agentic_event(sample.metadata, "reward_end_at")
            return group
        async with self._reward_semaphore:
            group_started_at = time.time()
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "reward_start_at", group_started_at)
            rewards = await _batched_async_rm(self.args, group)
            for sample, reward in zip(group, rewards, strict=True):
                sample.reward = reward
                mark_metadata_agentic_event(sample.metadata, "reward_end_at")
        return group

    def _rewarded_sample(self, sample):
        key = sample_key(sample)
        task = self._inflight_sample_tasks.get(key)
        if task is not None and task.done():
            self._inflight_sample_tasks.pop(key, None)
            self._completed_samples[key] = task.result()
        rewarded = self._completed_samples.get(key)
        if rewarded is not None:
            sample.reward = copy.deepcopy(getattr(rewarded, "reward", None))
            if getattr(rewarded, "metadata", None):
                sample.metadata.update(copy.deepcopy(rewarded.metadata))
            self._note_scored_sample(sample)
        return rewarded

    def _all_group_rewards_ready(self, group) -> bool:
        for sample in group:
            self._rewarded_sample(sample)
            if _sample_needs_reward(sample):
                return False
        return True

    @staticmethod
    def _apply_custom_advantage(value: Any, unit: PendingExportUnit) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            unit.sample.custom_advantage = float(value)
            return
        # TODO(agentic): support assistant-turn-level advantages from list[float].
        # TODO(agentic): support assistant-token-level advantages from list[list[float]].
        raise TypeError("custom advantage output must be a number in this version.")

    async def _run_custom_advantage(self, waiting_group: RewardWaitingGroup) -> list[Any] | None:
        result = self.custom_advantage_func(waiting_group.metadata_by_slot())
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            # None is the group-level drop signal from custom advantage.
            return None
        for slot_idx in range(waiting_group.expected_count):
            slot_values = result[slot_idx]
            for unit in waiting_group.units_by_slot.get(slot_idx, []):
                self._apply_custom_advantage(slot_values[unit.name], unit)
                self._note_scored_sample(unit.sample)
        return waiting_group.materialized_group()

    def _harvest_group_reward_completions(self) -> int:
        done_keys = [key for key, task in self._inflight_group_tasks.items() if task.done()]
        for key in done_keys:
            task = self._inflight_group_tasks.pop(key)
            self._completed_group_results[key] = task.result()
        return len(done_keys)

    async def _harvest_sample_reward_completions(self) -> int:
        done_keys = [key for key, task in self._inflight_sample_tasks.items() if task.done()]
        rejected_groups: list[list[Any]] = []
        replacement_budget_error: tuple[JudgeReplacementBudgetExceeded, Exception] | None = None
        for key in done_keys:
            task = self._inflight_sample_tasks.pop(key, None)
            if task is None:
                continue
            try:
                self._completed_samples[key] = task.result()
            except Exception as exc:
                from relax.engine.rewards.dual_agentic_judge import JudgeSampleRejected

                if not isinstance(exc, JudgeSampleRejected):
                    raise
                mapped_group_key = self._sample_group_keys.get(key)
                matching_groups = {mapped_group_key} if mapped_group_key in self._waiting_groups else set()
                if not matching_groups:
                    raise RuntimeError(f"Rejected judge sample has no resident group: {key!r}") from exc
                for group_key in matching_groups:
                    waiting_group = self._waiting_groups.pop(group_key)
                    group = waiting_group.materialized_group()
                    rejected_groups.append(group)
                    self._drop_scored_progress_for_group(group)
                    await self._release_group_sample_reward_cache(group)
                    if group_key not in self._rejected_group_keys:
                        self._judge_replacements_this_step += 1
                    self._rejected_group_keys.add(group_key)
                    if self._judge_replacements_this_step > self._max_judge_replacements:
                        replacement_budget_error = (
                            JudgeReplacementBudgetExceeded(
                                "dual-agentic-judge replacement budget exceeded: "
                                f"{self._judge_replacements_this_step} > {self._max_judge_replacements}"
                            ),
                            exc,
                        )
        for group in rejected_groups:
            self._capture_group_reward_trace_snapshots(group, terminal_outcome="group_rejected")
        if replacement_budget_error is not None:
            error, cause = replacement_budget_error
            raise error from cause
        for waiting_group in self._waiting_groups.values():
            if waiting_group.is_complete() and self.custom_advantage_func is None:
                self._start_missing_sample_rewards(waiting_group.materialized_group())
        ready_groups: list[tuple[GroupKey, list[Any]]] = []
        dropped_keys: list[GroupKey] = []
        for key, waiting_group in self._waiting_groups.items():
            if key in self._completed_group_results:
                continue
            if not waiting_group.is_complete():
                continue
            if self.custom_advantage_func is not None:
                group = await self._run_custom_advantage(waiting_group)
                if group is None:
                    group = waiting_group.materialized_group()
                    self._drop_scored_progress_for_group(group)
                    await self._release_group_sample_reward_cache(group)
                    dropped_keys.append(key)
                    continue
                ready_groups.append((key, group))
                continue
            group = waiting_group.materialized_group()
            if not self._all_group_rewards_ready(group):
                continue
            ready_groups.append((key, group))
        for key, group in ready_groups:
            self._waiting_groups.pop(key, None)
            self._completed_group_results[key] = group
        for key in dropped_keys:
            self._waiting_groups.pop(key, None)
        return len(done_keys) + len(ready_groups) + len(dropped_keys)

    def drain_rejected_group_keys(self) -> set[GroupKey]:
        rejected = set(self._rejected_group_keys)
        self._rejected_group_keys.clear()
        return rejected

    async def precompute_once(self) -> bool:
        try:
            progressed = False
            if await self._harvest_sample_reward_completions():
                progressed = True
            if self._harvest_group_reward_completions():
                progressed = True
            return progressed
        finally:
            if getattr(self.args, "rm_type", None) == "dual-agentic-judge":
                from relax.engine.rewards.dual_agentic_judge import drain_dual_judge_metrics

                for key, count in drain_dual_judge_metrics(self.args).items():
                    self._judge_metrics[key] = self._judge_metrics.get(key, 0) + count

    async def finalize_ready_for_step(self) -> bool:
        progressed = False
        for key in list(self._completed_group_results.keys()):
            completed_group = self._completed_group_results.pop(key, None)
            if completed_group is None:
                continue
            await self._finalize_group(key=key, group=completed_group)
            progressed = True
        return progressed

    async def _finalize_group(self, *, key: GroupKey, group) -> None:
        group_finalize_started_at = time.time()
        for sample in group:
            reward_trace = (
                ensure_reward_trace(sample.metadata)
                if getattr(self.args, "rm_type", None) == "dual-agentic-judge"
                else None
            )
            if reward_trace is not None:
                reward_trace["group_key"] = [list(item) for item in key]
            mark_metadata_agentic_event(sample.metadata, "group_reward_ready_at", group_finalize_started_at)
            mark_metadata_agentic_event(sample.metadata, "group_finalize_start_at", group_finalize_started_at)
        if self.group_filter is not None and not self.group_filter(group):
            for sample in group:
                mark_metadata_agentic_event(sample.metadata, "group_finalize_end_at")
            self._capture_group_reward_trace_snapshots(group, terminal_outcome="filter_dropped")
            self._drop_scored_progress_for_group(group)
            await self._release_group_sample_reward_cache(group)
            return
        await self._release_group_sample_reward_cache(group)
        self._ready_dispatches.append((key, group))
        group_finalize_ended_at = time.time()
        for sample in group:
            mark_metadata_agentic_event(sample.metadata, "group_finalize_end_at", group_finalize_ended_at)

    async def step_once(self) -> bool:
        progressed = False
        if await self.precompute_once():
            progressed = True
        if await self.finalize_ready_for_step():
            progressed = True
        return progressed

    async def wait_for_next_completion(self) -> bool:
        wait_set = set(self._inflight_sample_tasks.values()) | set(self._inflight_group_tasks.values())
        if wait_set:
            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
            return bool(done)
        return False

    def drain_ready_dispatch(self, *, max_groups: int | None = None) -> list[list[Any]]:
        ready_groups: list[list[Any]] = []
        retained_groups: list[list[Any]] = []
        if max_groups is None:
            remaining_groups = None
        else:
            remaining_groups = max_groups
        while self._ready_dispatches:
            key, group = self._ready_dispatches.popleft()
            if remaining_groups is None or remaining_groups > 0:
                ready_groups.append(group)
                if remaining_groups is not None:
                    remaining_groups -= 1
                continue
            retained_groups.append((key, group))
        for item in reversed(retained_groups):
            self._ready_dispatches.appendleft(item)
        return ready_groups

    async def drop_completed_groups(self) -> int:
        dropped_groups = 0
        first_error: BaseException | None = None
        while self._ready_dispatches:
            _key, group = self._ready_dispatches.popleft()
            self._capture_group_reward_trace_snapshots(group, terminal_outcome="output_dropped")
            try:
                await self._release_group_sample_reward_cache(group)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            dropped_groups += 1
        for key, completed_group in list(self._completed_group_results.items()):
            self._completed_group_results.pop(key, None)
            self._capture_group_reward_trace_snapshots(completed_group, terminal_outcome="output_dropped")
            try:
                await self._release_group_sample_reward_cache(completed_group)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            dropped_groups += 1
        if first_error is not None:
            raise first_error
        return dropped_groups

    async def drop_resident_groups(self) -> int:
        first_error: BaseException | None = None
        try:
            dropped_groups = await self.drop_completed_groups()
        except BaseException as exc:
            # ``drop_completed_groups`` has already attempted every group;
            # continue with waiting groups so shutdown cannot strand a context.
            dropped_groups = 0
            first_error = exc
        resident_groups: list[list[Any]] = []
        for key, waiting_group in list(self._waiting_groups.items()):
            self._waiting_groups.pop(key, None)
            materialized_group = waiting_group.materialized_group()
            resident_groups.append(materialized_group)
            self._drop_scored_progress_for_group(materialized_group)
            try:
                await self._release_group_sample_reward_cache(materialized_group)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            dropped_groups += 1
        orphaned_group_tasks = self._cancel_inflight_group_tasks()
        try:
            await self._drain_cancelled_tasks(orphaned_group_tasks)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        dropped_groups += len(orphaned_group_tasks)
        for group in resident_groups:
            self._capture_group_reward_trace_snapshots(group, terminal_outcome="resident_discarded")
        if first_error is not None:
            raise first_error
        return dropped_groups

    def accounting_snapshot(self) -> dict[str, int]:
        waiting_groups = len(self._waiting_groups)
        waiting_records = sum(
            waiting_group.materialized_slot_count() for waiting_group in self._waiting_groups.values()
        )
        ready_groups = len(self._ready_dispatches)
        completed_groups = len(self._completed_group_results)
        inflight_sample_rewards = len(self._inflight_sample_tasks)
        inflight_group_rewards = len(self._inflight_group_tasks)
        return {
            "waiting_groups": waiting_groups,
            "waiting_records": waiting_records,
            "ready_groups": ready_groups,
            "completed_groups": completed_groups,
            "inflight_sample_rewards": inflight_sample_rewards,
            "inflight_group_rewards": inflight_group_rewards,
            "scored_samples_ready": self._ready_scored_sample_count,
            "judge_group_replacements": self._judge_replacements_this_step,
            **self._judge_metrics,
        }

    def resident_group_keys(self) -> set[GroupKey]:
        keys = set(self._waiting_groups)
        keys.update(self._completed_group_results)
        keys.update(self._inflight_group_tasks)
        keys.update(key for key, _group in self._ready_dispatches)
        return keys

    def has_inflight_work(self) -> bool:
        return bool(self._inflight_group_tasks or self._inflight_sample_tasks)

    def has_pending_submission_work(self) -> bool:
        return bool(self._waiting_groups or self._inflight_group_tasks)

    def has_ready_output(self) -> bool:
        return bool(self._ready_dispatches)
