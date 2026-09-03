# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import contextlib
import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from relax.agentic import format_agentic_event
from relax.agentic.pipeline.runtime import (
    RuntimeDomain,
    get_agentic_runtime_resources,
)
from relax.agentic.profile import TRACE_KEY, agentic_span_clock_host
from relax.engine.filters.base_types import MetricGatherer, call_dynamic_filter
from relax.engine.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import (
    compute_rollout_explicit_reward_metrics,
    compute_rollout_step,
    compute_statistics,
    finalize_rollout_explicit_metric_values,
    has_repetition,
)
from relax.utils.misc import group_by
from relax.utils.profile_utils import start_sglang_profile, stop_sglang_profile
from relax.utils.training.eval_config import EvalDatasetConfig
from relax.utils.training.train_dump_utils import (
    append_agentic_accounting_marker,
    append_reward_workload_markers,
    save_debug_rollout_data,
    save_reward_trace_snapshots_jsonl,
    save_rollout_result_jsonl,
)
from relax.utils.types import Sample

from .pipeline.prepare import PrepareDomain
from .pipeline.reward import RewardDomain
from .pipeline.transfer import TransferDomain


logger = get_logger(__name__)

# How long the resident dataflow may make zero progress before it logs why.
_DATAFLOW_STALL_REPORT_AFTER_S = 120.0
_AGENT_METADATA_INTERNAL_KEYS = {TRACE_KEY}
_IDLE_HEARTBEAT_INTERVAL_S = 30.0
_BACKGROUND_POLL_INTERVAL_S = 0.05


_RESIDENT_PIPELINE: "AgenticResidentPipeline | None" = None
_RESIDENT_PIPELINE_LOCK = threading.Lock()
_RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID: int | None = None
_RESIDENT_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_RESIDENT_ASYNC_THREAD: threading.Thread | None = None
_RESIDENT_ASYNC_LOCK = threading.Lock()


def _sample_log_summary(sample: Sample) -> dict[str, Any]:
    """Return bounded diagnostics without formatting the trajectory or image
    tensors."""
    reward = sample.reward
    if isinstance(reward, dict):
        reward_summary: Any = {
            str(key): value for key, value in reward.items() if isinstance(value, (bool, int, float))
        }
    elif isinstance(reward, (bool, int, float)):
        reward_summary = reward
    else:
        reward_summary = type(reward).__name__ if reward is not None else None

    multimodal_shapes = {}
    multimodal_train_inputs = sample.multimodal_train_inputs
    if isinstance(multimodal_train_inputs, dict):
        for key, value in multimodal_train_inputs.items():
            shape = getattr(value, "shape", None)
            if shape is not None:
                multimodal_shapes[str(key)] = tuple(shape)

    return {
        "session_id": sample.session_id,
        "index": sample.index,
        "status": str(sample.status),
        "prompt_chars": len(sample.prompt or ""),
        "response_chars": len(sample.response or ""),
        "prompt_tokens": max(0, len(sample.tokens) - int(sample.response_length or 0)),
        "response_tokens": int(sample.response_length or 0),
        "multimodal_shapes": multimodal_shapes,
        "reward": reward_summary,
        "weight_versions": sample.weight_versions,
    }


def _eval_scope_id(*, dataset_name: str, rollout_id: int) -> str:
    return f"eval:{dataset_name}:{rollout_id}"


def _post_train_eval_expected(args, rollout_id: int, data_source) -> bool:
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval is None or getattr(args, "eval_prompt_data", None) is None:
        return False
    step = rollout_id + 1
    if step % eval_interval == 0:
        return True
    if not getattr(args, "rollout_global_dataset", False):
        return False

    import ray

    num_rollout_per_epoch = ray.get(data_source.lengths.remote()) // args.rollout_batch_size
    return step % num_rollout_per_epoch == 0


def _resident_async_loop_main(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _shutdown_resident_async_loop() -> None:
    global _RESIDENT_ASYNC_LOOP, _RESIDENT_ASYNC_THREAD
    with _RESIDENT_ASYNC_LOCK:
        loop = _RESIDENT_ASYNC_LOOP
        thread = _RESIDENT_ASYNC_THREAD
        _RESIDENT_ASYNC_LOOP = None
        _RESIDENT_ASYNC_THREAD = None
    if loop is None:
        return
    if not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
    if not loop.is_closed() and (thread is None or thread is not threading.current_thread()):
        loop.close()


def _run_on_resident_async_loop(coro):
    global _RESIDENT_ASYNC_LOOP, _RESIDENT_ASYNC_THREAD
    with _RESIDENT_ASYNC_LOCK:
        loop = _RESIDENT_ASYNC_LOOP
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_resident_async_loop_main, args=(loop,), name="agentic-resident-loop", daemon=True
            )
            thread.start()
            _RESIDENT_ASYNC_LOOP = loop
            _RESIDENT_ASYNC_THREAD = thread
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _shutdown_resident_pipeline_after_deferred_eval(rollout_id: int) -> None:
    global _RESIDENT_PIPELINE, _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID
    with _RESIDENT_PIPELINE_LOCK:
        if _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID != rollout_id:
            return
        _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID = None
        resident_pipeline = _RESIDENT_PIPELINE
        _RESIDENT_PIPELINE = None
    if resident_pipeline is not None:
        _run_on_resident_async_loop(resident_pipeline.shutdown())
    _shutdown_resident_async_loop()


class RolloutProgress:
    def __init__(self, *, total_sessions: int, rollout_id: int) -> None:
        if total_sessions <= 0:
            raise RuntimeError(f"RolloutProgress requires a positive total_sessions, got {total_sessions}.")
        self.total_sessions = total_sessions
        self._materialized_sessions = 0
        self._committed_sessions = 0
        self._scored_samples = 0
        self._bar = tqdm(
            total=self.total_sessions,
            desc=f"Rollout {rollout_id} generation",
            unit="session",
            disable=False,
            mininterval=0.0,
            miniters=1,
        )
        self._set_bar_postfix(refresh=True)

    def _set_bar_postfix(self, *, refresh: bool) -> None:
        self._bar.set_postfix_str(f"scored={self._scored_samples}", refresh=refresh)

    def update_counts(
        self,
        *,
        materialized_sessions: int,
        committed_sessions: int,
        scored_samples: int,
    ) -> None:
        if materialized_sessions < self._materialized_sessions:
            raise RuntimeError(
                "RolloutProgress materialized session count regressed: "
                f"{materialized_sessions} < {self._materialized_sessions}."
            )
        if committed_sessions < self._committed_sessions:
            raise RuntimeError(
                "RolloutProgress committed session count regressed: "
                f"{committed_sessions} < {self._committed_sessions}."
            )
        if (
            materialized_sessions == self._materialized_sessions
            and committed_sessions == self._committed_sessions
            and scored_samples == self._scored_samples
        ):
            return
        old_bar_count = min(self._materialized_sessions, self.total_sessions)
        new_bar_count = min(materialized_sessions, self.total_sessions)
        if new_bar_count > old_bar_count:
            self._bar.update(new_bar_count - old_bar_count)
        self._materialized_sessions = materialized_sessions
        self._committed_sessions = committed_sessions
        self._scored_samples = scored_samples
        self._set_bar_postfix(refresh=True)

    def update_total_sessions(self, total_sessions: int) -> None:
        if total_sessions == self.total_sessions:
            return
        self.total_sessions = total_sessions
        self._bar.total = total_sessions
        self._bar.refresh()

    def close(self) -> None:
        self._bar.close()

    def snapshot(self) -> dict[str, int]:
        return {
            "materialized_sessions": self._materialized_sessions,
            "committed_sessions": self._committed_sessions,
            "scored_samples": self._scored_samples,
        }


class AgenticResidentPipeline:
    def __init__(self) -> None:
        self.prepare_domain: PrepareDomain | None = None
        self.runtime_domain: RuntimeDomain | None = None
        self.transfer_domain: TransferDomain | None = None
        self.reward_domain: RewardDomain | None = None
        self.resident_dataflow_task: asyncio.Task | None = None
        self.resident_dataflow_progress_event: asyncio.Event | None = None
        self.resident_dataflow_lock: asyncio.Lock | None = None
        self.resident_dataflow_error: BaseException | None = None
        self.step_admission_closed = False
        self.prepare_requires_open_step = False
        self._step_get_samples_wait_started_at: float | None = None
        self._step_get_samples_times: list[float] = []
        self._active_step_handle: _AgenticStepHandle | None = None
        # How many groups the most recent step left short of its transfer-queue target
        # (rollout_batch_size - committed_current). The next step backfills exactly this
        # many groups into its previous partition. 0 before the first step / when a step
        # fully met its target. Only meaningful under fully_async.
        self._last_step_current_deficit = 0
        # Stall reporting: the per-domain accounting snapshots are only emitted at step
        # boundaries, so a step that can never complete produces endless prepare churn
        # and no clue about which stage stopped advancing. Track the last time the
        # dataflow actually moved so a wedged pipeline can describe itself.
        self._last_pump_progress_at = time.monotonic()
        self._last_stall_report_at = time.monotonic()
        # Service startup takes minutes during which the pump correctly has nothing
        # to move; only an idle pipeline that previously ran is a stall worth
        # reporting, so hold fire until the dataflow has moved at least once.
        self._dataflow_ever_progressed = False

    def _dataflow_lock(self) -> asyncio.Lock:
        lock = self.resident_dataflow_lock
        if lock is None:
            lock = asyncio.Lock()
            self.resident_dataflow_lock = lock
        return lock

    async def _pump_once(self) -> bool:
        async with self._dataflow_lock():
            progressed = False
            while True:
                tick_progressed = False
                if await self._pump_prepare_once():
                    tick_progressed = True
                if await self._pump_admission_once():
                    tick_progressed = True
                if await self._pump_runtime_to_reward_once():
                    tick_progressed = True
                if await self._pump_reward_to_transfer_once():
                    tick_progressed = True
                if await self._pump_transfer_once():
                    tick_progressed = True
                if not tick_progressed:
                    self._report_dataflow_stall(progressed)
                    return progressed
                progressed = True

    def _report_dataflow_stall(self, progressed: bool) -> None:
        """Log per-domain accounting when the resident dataflow stops
        advancing.

        Every pump stage (prepare -> admission -> runtime -> reward ->
        transfer) reports whether it moved anything. When none of them move for
        a sustained period the step cannot finish, but that is otherwise
        invisible: the domain accounting snapshots are only emitted at step
        boundaries, which a wedged step never reaches. Emitting them here turns
        "the log shows endless prepare churn" into "runtime holds N
        materialized records that reward never saw".
        """
        now = time.monotonic()
        if progressed:
            self._last_pump_progress_at = now
            self._dataflow_ever_progressed = True
            return
        if not self._dataflow_ever_progressed:
            # Still starting up (services deploying); an idle pump is expected.
            self._last_pump_progress_at = now
            return
        if now - self._last_pump_progress_at < _DATAFLOW_STALL_REPORT_AFTER_S:
            return
        if now - self._last_stall_report_at < _DATAFLOW_STALL_REPORT_AFTER_S:
            return
        self._last_stall_report_at = now
        stalled_for = now - self._last_pump_progress_at
        snapshots: dict[str, Any] = {}
        for name, domain in (
            ("runtime", self.runtime_domain),
            ("reward", self.reward_domain),
            ("transfer", self.transfer_domain),
        ):
            if domain is None:
                continue
            try:
                snapshots[name] = dict(domain.accounting_snapshot())
            except Exception as exc:  # noqa: BLE001 -- diagnostics must never break the pump
                snapshots[name] = {"accounting_snapshot_failed": f"{type(exc).__name__}: {exc}"}
        logger.warning(
            "Agentic dataflow made no progress for %.0fs; per-domain accounting: %s",
            stalled_for,
            snapshots,
        )

    @property
    def resident_group_count(self) -> int:
        if (
            self.runtime_domain is None
            or self.reward_domain is None
            or self.transfer_domain is None
            or not hasattr(self.runtime_domain, "runtime_groups_by_key")
        ):
            return 0
        runtime_keys = set(self.runtime_domain.resident_group_keys())
        reward_keys = set(self.reward_domain.resident_group_keys())
        transfer_keys = set(self.transfer_domain.resident_group_keys())
        return len(runtime_keys | reward_keys | transfer_keys)

    async def _pump_prepare_once(self) -> bool:
        prepare_domain = self.prepare_domain
        runtime_domain = self.runtime_domain
        if prepare_domain is None or runtime_domain is None:
            raise RuntimeError("Agentic prepare pump requires initialized prepare and runtime domains.")
        progressed = False
        if prepare_domain.has_ready_output():
            if await prepare_domain.accept_fetched_batch():
                progressed = True
        should_refresh_prepare = (
            prepare_domain.has_pending_prepare()
            or prepare_domain.has_inflight_work()
            or prepare_domain.has_ready_output()
            or prepare_domain.has_warming_groups()
        )
        if should_refresh_prepare:
            # Drop (do not crash on) a prepare group whose managed session
            # finished before producing a chat IR — e.g. an apptainer instance
            # that failed to start. The over-sampling prepare pool + transfer
            # quota gate refill the dropped group, so batch size is preserved;
            # crashing here would instead trigger a full global restart.
            if await prepare_domain.refresh_ready_groups(
                status_fetcher=runtime_domain.prepare_group_status,
                drop_completed_before_ready=True,
            ):
                progressed = True
        if prepare_domain.has_pending_prepare():
            if await prepare_domain.launch_pending():
                progressed = True
        if (not self.prepare_requires_open_step or not self.step_admission_closed) and prepare_domain.start_fetch():
            progressed = True
        return progressed

    async def _pump_admission_once(self) -> bool:
        prepare_domain = self.prepare_domain
        runtime_domain = self.runtime_domain
        transfer_domain = self.transfer_domain
        reward_domain = self.reward_domain
        if prepare_domain is None or runtime_domain is None:
            raise RuntimeError("Agentic admission pump requires initialized prepare and runtime domains.")
        if self.step_admission_closed:
            return False
        if transfer_domain is None or reward_domain is None:
            raise RuntimeError("Agentic admission pump requires initialized transfer and reward domains.")

        progressed = False
        transfer_snapshot = dict(transfer_domain.accounting_snapshot())
        _, _, _, current_window_slack = self._current_window_admission_counts(
            resident_group_count=self.resident_group_count,
            transfer_snapshot=transfer_snapshot,
        )
        if current_window_slack == 0:
            self._step_get_samples_wait_started_at = None
            return progressed
        if self._step_get_samples_wait_started_at is None:
            self._step_get_samples_wait_started_at = time.monotonic()
        if prepare_domain.has_warming_groups():
            # See _pump_prepare_once: drop groups that completed before ready
            # instead of crashing; the over-sampling pool refills them.
            await prepare_domain.refresh_ready_groups(
                status_fetcher=runtime_domain.prepare_group_status,
                drop_completed_before_ready=True,
            )
        if not prepare_domain.has_ready_groups():
            return progressed
        batch_input = await prepare_domain.lease_ready_groups(
            quota_group_count=min(transfer_domain.over_sampling_batch_size, current_window_slack),
            rollout_id=runtime_domain.require_rollout_id(),
        )
        if batch_input is None:
            return progressed
        prepare_domain.start_fetch()
        if self._step_get_samples_wait_started_at is not None:
            self._step_get_samples_times.append(time.monotonic() - self._step_get_samples_wait_started_at)
            self._step_get_samples_wait_started_at = None
        admitted_batch_groups = await runtime_domain.start_batch(batch_input=batch_input)
        if admitted_batch_groups <= 0:
            raise RuntimeError("RuntimeDomain accepted an admission batch without owning any groups.")
        return True

    async def _pump_runtime_to_reward_once(self) -> bool:
        runtime_domain = self.runtime_domain
        reward_domain = self.reward_domain
        if runtime_domain is None or reward_domain is None:
            raise RuntimeError("Agentic runtime-to-reward pump requires initialized runtime and reward domains.")
        progressed = False
        if runtime_domain.has_pending_runtime_work():
            batch_progress = await runtime_domain.wait_for_next_runtime_slot(timeout_s=0.0)
            if batch_progress is not None:
                progressed = True
        discarded_group_keys = runtime_domain.drain_discarded_group_keys()
        if discarded_group_keys:
            await reward_domain.drop_waiting_groups_by_key(discarded_group_keys)
            progressed = True
        runtime_dispatch = runtime_domain.drain_ready_execution()
        if runtime_dispatch.materialized_batches:
            if not reward_domain.group_rm or reward_domain.custom_advantage_func is not None:
                reward_domain.accept_session_materializations(runtime_dispatch.materialized_batches)
            progressed = True
        if reward_domain.custom_advantage_func is None:
            for group in runtime_dispatch.ready_groups:
                await reward_domain.ingest_groups([group])
                progressed = True
        return progressed

    async def _pump_reward_to_transfer_once(self) -> bool:
        reward_domain = self.reward_domain
        transfer_domain = self.transfer_domain
        runtime_domain = self.runtime_domain
        if reward_domain is None:
            raise RuntimeError("Agentic reward-to-transfer pump requires initialized reward domain.")
        progressed = await reward_domain.step_once()
        rejected_group_keys = reward_domain.drain_rejected_group_keys()
        if rejected_group_keys:
            if runtime_domain is None:
                raise RuntimeError("Agentic judge rejection requires initialized runtime domain.")
            await runtime_domain.drop_groups_by_key(rejected_group_keys)
            progressed = True
        if transfer_domain is None:
            return progressed

        ready_reward_groups = reward_domain.drain_ready_dispatch(max_groups=transfer_domain.remaining_ready_capacity())
        if ready_reward_groups:
            for group in ready_reward_groups:
                transfer_domain.enqueue_ready_groups([group])
            progressed = True
        return progressed

    async def _pump_transfer_once(self) -> bool:
        transfer_domain = self.transfer_domain
        if transfer_domain is None:
            raise RuntimeError("Agentic transfer pump requires initialized transfer domain.")
        released_groups, committed_group_count = await transfer_domain.drain_ready_group_payloads()
        if committed_group_count <= 0:
            return False
        self._log_first_rollout_sample_for_active_step(released_groups)
        return True

    def _log_first_rollout_sample_for_active_step(self, released_groups: list[list]) -> None:
        step_handle = self._active_step_handle
        if step_handle is None or step_handle.first_rollout_sample_logged or not released_groups:
            return
        sample = released_groups[0][0][0] if isinstance(released_groups[0][0], list) else released_groups[0][0]
        logger.info("First rollout sample summary: %s", _sample_log_summary(sample))
        step_handle.first_rollout_sample_logged = True

    def _raise_resident_dataflow_error(self) -> None:
        error = self.resident_dataflow_error
        if error is None:
            return
        self.resident_dataflow_error = None
        raise RuntimeError("agentic resident dataflow loop failed") from error

    async def _wait_for_resident_dataflow_progress(self) -> bool:
        self._raise_resident_dataflow_error()
        progress_event = self.resident_dataflow_progress_event
        if progress_event is None:
            await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
            self._raise_resident_dataflow_error()
            return False
        if progress_event.is_set():
            self._raise_resident_dataflow_error()
            progress_event.clear()
            return True
        try:
            await asyncio.wait_for(progress_event.wait(), timeout=_BACKGROUND_POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            self._raise_resident_dataflow_error()
            return False
        self._raise_resident_dataflow_error()
        if not progress_event.is_set():
            return False
        progress_event.clear()
        return True

    async def _wait_for_step_event(self) -> bool:
        task = self.resident_dataflow_task
        if task is not None and not task.done():
            return await self._wait_for_resident_dataflow_progress()
        await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
        return False

    async def _resident_dataflow_loop(self) -> None:
        while True:
            try:
                progressed = await self._pump_once()
                if progressed:
                    if self.resident_dataflow_progress_event is not None:
                        self.resident_dataflow_progress_event.set()
                    continue
                await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.resident_dataflow_error = exc
                if self.resident_dataflow_progress_event is not None:
                    self.resident_dataflow_progress_event.set()
                logger.exception("Agentic resident pipeline dataflow loop failed")
                return

    async def start_resident_dataflow(self) -> None:
        if self.resident_dataflow_task is not None and not self.resident_dataflow_task.done():
            return
        self.resident_dataflow_error = None
        self.resident_dataflow_progress_event = asyncio.Event()
        self.resident_dataflow_task = asyncio.create_task(self._resident_dataflow_loop())

    async def stop_resident_dataflow(self) -> None:
        task = self.resident_dataflow_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.resident_dataflow_task = None
        self.resident_dataflow_progress_event = None
        self.resident_dataflow_error = None

    async def shutdown(self) -> None:
        await self.stop_resident_dataflow()
        close_judge_clients = (
            self.reward_domain is not None
            and getattr(self.reward_domain.args, "rm_type", None) == "dual-agentic-judge"
        )
        domains = (
            ("prepare_domain", self.prepare_domain),
            ("runtime_domain", self.runtime_domain),
            ("transfer_domain", self.transfer_domain),
            ("reward_domain", self.reward_domain),
        )
        for attr_name, domain in domains:
            if domain is None:
                continue
            await domain.shutdown()
            setattr(self, attr_name, None)
        if close_judge_clients:
            from relax.utils.genrm_client import close_all_genrm_clients

            await close_all_genrm_clients()
        self.step_admission_closed = False

    async def init_pipeline(
        self,
        *,
        args,
        data_source,
        data_system_client,
    ) -> None:
        pool_target_group_count = args.agentic_prepare_pool_size or args.over_sampling_batch_size
        async with self._dataflow_lock():
            prepare_domain = self.prepare_domain
            if prepare_domain is None:
                prepare_domain = PrepareDomain(
                    scope_id="train",
                    data_source=data_source,
                    prefetch_concurrency=1,
                    pool_target_group_count=pool_target_group_count,
                )
                self.prepare_domain = prepare_domain
            elif prepare_domain.pool_target_group_count != pool_target_group_count:
                raise RuntimeError(
                    "PrepareDomain configuration changed after initialization: "
                    f"pool_target_group_count {prepare_domain.pool_target_group_count} "
                    f"-> {pool_target_group_count}."
                )
            runtime_domain = self.runtime_domain
            if runtime_domain is None:
                runtime_domain = RuntimeDomain(args=args, scope_id="train")
                self.runtime_domain = runtime_domain
            if self.reward_domain is None:
                self.reward_domain = RewardDomain(
                    args=args,
                    group_filter=None,
                )
            if self.transfer_domain is None:
                self.transfer_domain = TransferDomain(
                    args=args,
                    data_system_client=data_system_client,
                )
            prepare_domain.configure(
                runtime_driver=runtime_domain,
                pool_target_group_count=pool_target_group_count,
            )
            self.prepare_requires_open_step = args.agentic_prepare_pool_size == 0
            self.step_admission_closed = True
            if not self.prepare_requires_open_step:
                prepare_domain.start_fetch()
        await self.start_resident_dataflow()

    async def open_step(
        self,
        *,
        args,
        rollout_id: int,
        group_filter,
        defer_terminal_shutdown_for_eval: bool = False,
    ) -> "_AgenticStepHandle":
        fully_async = args.fully_async
        num_rollout = getattr(args, "num_rollout", None)
        final_backfill_step = fully_async and num_rollout is not None and rollout_id >= num_rollout
        current_partition_quota = 0 if final_backfill_step else args.rollout_batch_size
        terminal_step = num_rollout is not None and rollout_id + 1 >= num_rollout
        async with self._dataflow_lock():
            prepare_domain = self.prepare_domain
            runtime_domain = self.runtime_domain
            reward_domain = self.reward_domain
            transfer_domain = self.transfer_domain
            if None in (prepare_domain, runtime_domain, reward_domain, transfer_domain):
                raise RuntimeError("Agentic resident pipeline must be initialized before opening a rollout step.")
            runtime_domain.rebind_step(
                args=args,
                rollout_id=rollout_id,
            )
            reward_domain.rebind_step(
                group_filter=group_filter,
            )
            transfer_domain.rebind_step(
                rollout_id=rollout_id,
            )
            self._step_get_samples_wait_started_at = None
            self._step_get_samples_times = []
            # Previous-partition backfill debt is exactly how many groups the previous
            # step left short of its transfer-queue target (rollout_batch_size). With
            # over-sampling (over_sampling_batch_size > rollout_batch_size) the previous
            # step may finish completely (deficit 0) yet still hold surplus completed
            # groups in the transfer ready buffer; those surplus groups are NOT debt —
            # they re-commit toward this step's current_partition_quota on their own.
            # Sizing the debt from the recorded current-partition deficit (set at the
            # previous close_step) keeps the ledger self-consistent and lets over-sampling
            # actually run. See _last_step_current_deficit.
            previous_partition_quota = self._last_step_current_deficit if fully_async else 0
            transfer_domain.configure_transfer_quota(
                previous_partition_quota=previous_partition_quota,
                current_partition_quota=current_partition_quota,
            )
            self.step_admission_closed = False
            await runtime_domain.release_partial_resume_gate()
            transfer_start_snapshot = transfer_domain.accounting_snapshot()
        progress = RolloutProgress(
            total_sessions=transfer_domain.target_group_count() * transfer_start_snapshot["group_size"],
            rollout_id=rollout_id,
        )
        step_handle = _AgenticStepHandle(
            rollout_id=rollout_id,
            required_group_count=current_partition_quota,
            terminal_step=terminal_step,
            final_backfill_step=final_backfill_step,
            defer_terminal_shutdown_for_eval=defer_terminal_shutdown_for_eval,
            progress=progress,
        )
        self._active_step_handle = step_handle
        await self.start_resident_dataflow()
        return step_handle

    def _committed_target_count(self) -> int:
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        return transfer_snapshot["committed_current_groups"]

    @property
    def _final_backfill_pending(self) -> bool:
        return self._last_step_current_deficit > 0

    def _interrupted_close_accounting_enabled(self) -> bool:
        return bool(self.runtime_domain.args.fully_async)

    async def _refresh_close_accounting(self) -> None:
        refresher = getattr(self.runtime_domain, "refresh_interrupted_close_accounting", None)
        if callable(refresher):
            await refresher()

    def _remaining_previous_debt(self, transfer_snapshot: dict[str, int]) -> int:
        committed_previous_groups = transfer_snapshot["committed_previous_groups"]
        previous_partition_quota = transfer_snapshot["previous_partition_quota"]
        remaining_previous_debt = previous_partition_quota - committed_previous_groups
        if remaining_previous_debt < 0:
            raise RuntimeError(
                "Agentic previous partition accounting is inconsistent: "
                f"quota={previous_partition_quota}, committed={committed_previous_groups}."
            )
        return remaining_previous_debt

    def _current_window_admission_counts(
        self,
        *,
        resident_group_count: int,
        transfer_snapshot: dict[str, int],
    ) -> tuple[int, int, int, int]:
        current_admission_quota = self._current_window_admission_quota(transfer_snapshot)
        remaining_previous_debt = self._remaining_previous_debt(transfer_snapshot)
        if resident_group_count < remaining_previous_debt:
            raise RuntimeError(
                "Agentic resident accounting cannot cover remaining previous partition debt: "
                f"resident={resident_group_count}, remaining_previous_debt={remaining_previous_debt}."
            )
        resident_current_window_groups = resident_group_count - remaining_previous_debt
        current_window_admitted_groups = transfer_snapshot["committed_current_groups"] + resident_current_window_groups
        current_window_slack = current_admission_quota - current_window_admitted_groups
        if current_window_slack < 0:
            raise RuntimeError(
                "Agentic current window admission exceeded current admission quota: "
                f"admission_accounted={current_window_admitted_groups}, "
                f"current_admission_quota={current_admission_quota}, "
                f"current_partition_quota={transfer_snapshot['current_partition_quota']}."
            )
        return (
            remaining_previous_debt,
            resident_current_window_groups,
            current_window_admitted_groups,
            current_window_slack,
        )

    def _current_window_admission_quota(self, transfer_snapshot: dict[str, int]) -> int:
        if self.transfer_domain is None:
            return transfer_snapshot["current_partition_quota"]
        return int(
            getattr(self.transfer_domain, "over_sampling_batch_size", transfer_snapshot["current_partition_quota"])
        )

    def _close_status(self, step_handle: "_AgenticStepHandle") -> str | None:
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        committed_previous = transfer_snapshot["committed_previous_groups"]
        previous_quota = transfer_snapshot["previous_partition_quota"]
        if step_handle.final_backfill_step:
            # There is no later step to carry another deficit, so final backfill
            # closes only after the previous TQ partition is actually complete.
            return None if committed_previous >= previous_quota else "previous_partition_debt"
        committed_current = transfer_snapshot["committed_current_groups"]
        if self._interrupted_close_accounting_enabled():
            # Match the original fully-async rollout's combined FIFO window:
            # completed and interrupted groups are fungible for soft close.
            interrupted_groups = int(self.runtime_domain.accounting_snapshot()["interrupted_groups"])
            accounted_groups = committed_previous + committed_current + interrupted_groups
            target_groups = previous_quota + step_handle.required_group_count
            return None if accounted_groups >= target_groups else "committed_target"
        if committed_previous < previous_quota:
            return "previous_partition_debt"
        if committed_current < step_handle.required_group_count:
            return "committed_target"
        return None

    def _refresh_progress(self, step_handle: "_AgenticStepHandle") -> None:
        progress = step_handle.progress
        if progress is None:
            return
        reward_snapshot = dict(self.reward_domain.accounting_snapshot())
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        group_size = transfer_snapshot["group_size"]
        previous_partition_quota = transfer_snapshot["previous_partition_quota"]
        progress.update_total_sessions((step_handle.required_group_count + previous_partition_quota) * group_size)
        complete_groups = (
            transfer_snapshot["committed_previous_groups"]
            + transfer_snapshot["committed_current_groups"]
            + transfer_snapshot["ready_groups"]
            + reward_snapshot["ready_groups"]
        )
        complete_group_sessions = min(
            progress.total_sessions,
            complete_groups * group_size,
        )
        progress_snapshot = progress.snapshot()
        committed_sessions = (
            transfer_snapshot["committed_previous_groups"] + transfer_snapshot["committed_current_groups"]
        ) * group_size
        progress.update_counts(
            materialized_sessions=max(progress_snapshot["materialized_sessions"], complete_group_sessions),
            committed_sessions=max(progress_snapshot["committed_sessions"], committed_sessions),
            scored_samples=reward_snapshot["scored_samples_ready"],
        )

    def _compact_accounting_snapshot(self, step_handle: "_AgenticStepHandle", *, phase: str) -> dict[str, object]:
        transfer_snapshot = dict(self.transfer_domain.accounting_snapshot())
        runtime_snapshot = dict(self.runtime_domain.accounting_snapshot())
        prepare_snapshot = dict(self.prepare_domain.accounting_snapshot())
        reward_snapshot = dict(self.reward_domain.accounting_snapshot())
        resident_group_count = self.resident_group_count
        interrupted_close_accounting = self._interrupted_close_accounting_enabled()
        previous_partition_quota = transfer_snapshot["previous_partition_quota"]
        (
            remaining_previous_debt,
            resident_current_window_groups,
            admission_current_window_groups,
            admission_current_slack,
        ) = self._current_window_admission_counts(
            resident_group_count=resident_group_count,
            transfer_snapshot=transfer_snapshot,
        )
        target_data_size = previous_partition_quota + transfer_snapshot["current_partition_quota"]
        admission_current_quota = self._current_window_admission_quota(transfer_snapshot)
        interrupted_groups = runtime_snapshot["interrupted_groups"]
        close_accounted_interrupted_groups = (
            interrupted_groups if interrupted_close_accounting and not step_handle.final_backfill_step else 0
        )
        return {
            "phase": phase,
            "rollout_id": step_handle.rollout_id,
            "final_backfill_step": step_handle.final_backfill_step,
            "train_target_groups": step_handle.required_group_count,
            "target_data_size": target_data_size,
            "resident_group_count": resident_group_count,
            "previous_partition_quota": previous_partition_quota,
            "remaining_previous_debt": remaining_previous_debt,
            "resident_current_window_groups": resident_current_window_groups,
            "admission_current_window_groups": admission_current_window_groups,
            "admission_current_slack": admission_current_slack,
            "admission_current_quota": admission_current_quota,
            "interrupted_groups": interrupted_groups,
            "close_accounted_groups": transfer_snapshot["committed_previous_groups"]
            + transfer_snapshot["committed_current_groups"]
            + close_accounted_interrupted_groups,
            "finish_eligible_mode": (
                "committed_previous"
                if step_handle.final_backfill_step
                else "committed_plus_interrupted"
                if interrupted_close_accounting
                else "committed_current"
            ),
            "prepare_pool_groups": prepare_snapshot["pool_groups"],
            "prepare_pool_target_groups": prepare_snapshot["pool_target_groups"],
            "prepare_pending_groups": prepare_snapshot["pending_prepare_groups"],
            "prepare_ready_groups": prepare_snapshot["ready_groups"],
            "prepare_warming_groups": prepare_snapshot["warming_groups"],
            "reward_waiting_groups": reward_snapshot["waiting_groups"],
            "reward_waiting_records": reward_snapshot["waiting_records"],
            "reward_ready_groups": reward_snapshot["ready_groups"],
            "reward_completed_groups": reward_snapshot["completed_groups"],
            "reward_inflight_sample_rewards": reward_snapshot["inflight_sample_rewards"],
            "reward_inflight_group_rewards": reward_snapshot["inflight_group_rewards"],
            "transfer_committed_current_groups": transfer_snapshot["committed_current_groups"],
            "transfer_committed_previous_groups": transfer_snapshot["committed_previous_groups"],
            "transfer_current_quota": transfer_snapshot["current_partition_quota"],
            "transfer_previous_quota": transfer_snapshot["previous_partition_quota"],
            "transfer_buffer_groups": transfer_snapshot["transfer_buffer_groups"],
            "transfer_tasks": transfer_snapshot["transfer_tasks"],
            "finish_eligible_blocked_by": self._close_status(step_handle),
            "transfer_ready_groups": transfer_snapshot["ready_groups"],
            "runtime_groups": runtime_snapshot["runtime_groups"],
            "runtime_slots": runtime_snapshot["runtime_slots"],
            "runtime_ready_materialized_batch_groups": runtime_snapshot["ready_materialized_batch_groups"],
            "runtime_ready_materialized_groups": runtime_snapshot["ready_materialized_groups"],
            "previous_step_current_deficit": self._last_step_current_deficit,
            **{key: value for key, value in reward_snapshot.items() if key.startswith("judge_")},
        }

    async def _wait_step_target(self, step_handle: "_AgenticStepHandle") -> None:
        while True:
            self._raise_resident_dataflow_error()
            if self._interrupted_close_accounting_enabled():
                await self._refresh_close_accounting()
            self._refresh_progress(step_handle)
            async with self._dataflow_lock():
                if self._close_status(step_handle) is None:
                    self.step_admission_closed = True
                    self._refresh_progress(step_handle)
                    return
            progressed = await self._wait_for_step_event()
            now = time.time()
            if progressed:
                self._refresh_progress(step_handle)
                step_handle.last_progress_at = now
                continue
            if (
                now - step_handle.last_progress_at >= _IDLE_HEARTBEAT_INTERVAL_S
                and now - step_handle.last_idle_heartbeat_at >= _IDLE_HEARTBEAT_INTERVAL_S
            ):
                step_handle.last_idle_heartbeat_at = now
                runtime_snapshot = self.runtime_domain.debug_snapshot()
                logger.info(
                    "Agentic rollout=%s idle_heartbeat idle_for=%.1fs committed_target=%s required=%s "
                    "step_accounting_snapshot=%s runtime=%s",
                    step_handle.rollout_id,
                    now - step_handle.last_progress_at,
                    self._committed_target_count(),
                    step_handle.required_group_count,
                    self._compact_accounting_snapshot(step_handle, phase="idle_heartbeat"),
                    runtime_snapshot,
                )

    async def _gate_step_irs_once(
        self,
        step_handle: "_AgenticStepHandle",
        *,
        shutdown_all_irs: bool = False,
    ) -> int:
        """Apply the step-close IR gate once.

        Partial close uses a partial-resume gate. Non-partial close gates the
        current rollout for discard. Terminal/error cleanup gates all train
        IRs.
        """
        if step_handle.rollout_irs_gated and not shutdown_all_irs:
            return 0
        if shutdown_all_irs:
            locked = await self.runtime_domain.gate_all_irs_for_shutdown()
        elif self.runtime_domain.args.partial_rollout or self.runtime_domain.args.fully_async:
            locked = await self.runtime_domain.gate_rollout_irs_for_partial_resume()
        else:
            locked = await self.runtime_domain.gate_rollout_irs_for_discard()
        step_handle.rollout_irs_gated = True
        return locked

    async def _wait_for_active_request_counts(
        self,
        *,
        require_no_protected: bool,
        require_no_abortable: bool,
    ) -> dict[str, int]:
        while True:
            self._raise_resident_dataflow_error()
            counts = await self.runtime_domain.active_rollout_request_counts()
            protected_active = counts.get("protected_active", 0) or 0
            abortable_active = counts.get("abortable_active", 0) or 0
            evaluating = counts.get("evaluating", 0) or 0
            if evaluating == 0:
                protected_done = (not require_no_protected) or protected_active == 0
                abortable_done = (not require_no_abortable) or abortable_active == 0
                if protected_done and abortable_done:
                    return counts
            await self._wait_for_step_event()

    async def _seal_step(
        self,
        step_handle: "_AgenticStepHandle",
        *,
        shutdown_all_irs: bool = False,
    ) -> None:
        if step_handle.active_backend_requests_sealed:
            return
        tail_carry_enabled = bool(
            (self.runtime_domain.args.partial_rollout or self.runtime_domain.args.fully_async) and not shutdown_all_irs
        )
        if tail_carry_enabled:
            await self._wait_for_active_request_counts(
                require_no_protected=True,
                require_no_abortable=False,
            )
        await self._gate_step_irs_once(step_handle, shutdown_all_irs=shutdown_all_irs)
        abort_result = await self.runtime_domain.abort_rollout_requests()
        logger.info(
            "Agentic rollout=%s abort_all requested_workers=%s failed_workers=%s abortable_active=%s",
            step_handle.rollout_id,
            abort_result.get("abort_requested_workers", 0) or 0,
            abort_result.get("abort_failed_workers", 0) or 0,
            abort_result.get("abortable_active", 0) or 0,
        )
        await self._wait_for_active_request_counts(
            require_no_protected=tail_carry_enabled,
            require_no_abortable=True,
        )
        step_handle.active_backend_requests_sealed = True

    async def _discard_resident_tail(self) -> None:
        async with self._dataflow_lock():
            await self.runtime_domain.drop_resident_results()
            await self.reward_domain.drop_resident_groups()
            dropped_ready_groups = self.transfer_domain.drop_ready_groups()
            dropped_transfer_groups, cancelled_transfer_tasks = await self.transfer_domain.discard_pending_transfers()
            if dropped_ready_groups or dropped_transfer_groups or cancelled_transfer_tasks:
                logger.info(
                    "Agentic discarded resident transfer tail: ready_groups=%s transfer_groups=%s transfer_tasks=%s",
                    dropped_ready_groups,
                    dropped_transfer_groups,
                    cancelled_transfer_tasks,
                )

    async def close_step(
        self,
        *,
        step_handle: "_AgenticStepHandle",
        cleanup_only: bool = False,
    ) -> "_AgenticClosedStep | None":
        terminal_step = step_handle.terminal_step
        if cleanup_only:
            async with self._dataflow_lock():
                self.step_admission_closed = True
            await self._seal_step(step_handle, shutdown_all_irs=True)
            await self.stop_resident_dataflow()
            await self._discard_resident_tail()
            await self.runtime_domain.trim_agentic_session_shards(reason="cleanup")
            if step_handle.progress is not None:
                step_handle.progress.close()
                step_handle.progress = None
            return None
        await self._wait_step_target(step_handle)
        async with self._dataflow_lock():
            self.transfer_domain.close_output_window()
            output = await self.transfer_domain.build_output()
            end_snapshot = self.transfer_domain.accounting_snapshot()
            # Record how many groups this step fell short of its current-partition target
            # (rollout_batch_size). The next step backfills exactly this many into its
            # previous partition. required_group_count == current_partition_quota.
            self._last_step_current_deficit = max(
                step_handle.required_group_count - end_snapshot["committed_current_groups"], 0
            )
            committed_groups = self.transfer_domain.committed_transfer_groups_snapshot()
            progress_snapshot = step_handle.progress.snapshot() if step_handle.progress is not None else {}
            get_samples_times = list(self._step_get_samples_times)
            await self.reward_domain.drop_completed_groups()
            self.transfer_domain.release_step_output_payloads()
            if step_handle.progress is not None:
                step_handle.progress.close()
                step_handle.progress = None
        await self.transfer_domain.wait_for_pending_transfers()
        lifecycle_terminal = terminal_step and not self._final_backfill_pending
        shutdown_all_irs = lifecycle_terminal and not step_handle.defer_terminal_shutdown_for_eval
        await self._seal_step(step_handle, shutdown_all_irs=shutdown_all_irs)
        if lifecycle_terminal or not (
            self.runtime_domain.args.partial_rollout or self.runtime_domain.args.fully_async
        ):
            await self._discard_resident_tail()
        await self.runtime_domain.trim_agentic_session_shards(reason=f"close_step_{step_handle.rollout_id}")
        async with self._dataflow_lock():
            accounting_end_snapshot = self._compact_accounting_snapshot(step_handle, phase="accounting_end")
            reward_trace_snapshots = self.reward_domain.drain_reward_trace_snapshots()
        runtime_snapshot = self.runtime_domain.debug_snapshot()
        return _AgenticClosedStep(
            output=output,
            end_snapshot=end_snapshot,
            runtime_snapshot=runtime_snapshot,
            committed_groups=committed_groups,
            progress_snapshot=progress_snapshot,
            get_samples_times=get_samples_times,
            reward_trace_snapshots=reward_trace_snapshots,
            accounting_end_snapshot=accounting_end_snapshot,
        )

    async def run_step(
        self,
        *,
        args,
        rollout_id,
        defer_terminal_shutdown_for_eval: bool = False,
    ) -> RolloutFnTrainOutput:
        rollout_started_at = time.monotonic()
        await start_sglang_profile(args, rollout_id)
        sglang_profile_stopped = False
        metric_gatherer = MetricGatherer()
        filter_path = args.dynamic_sampling_filter_path
        if filter_path:
            from relax.utils.utils import load_function

            filter_fn = load_function(filter_path)

            def group_filter(group) -> bool:
                output = call_dynamic_filter(filter_fn, args, group)
                keep = output.keep
                if not keep:
                    metric_gatherer.on_dynamic_filter_drop(reason=getattr(output, "reason", None))
                return keep

        else:
            group_filter = None
        step_handle = await self.open_step(
            args=args,
            rollout_id=rollout_id,
            group_filter=group_filter,
            defer_terminal_shutdown_for_eval=defer_terminal_shutdown_for_eval,
        )
        terminal_step = step_handle.terminal_step
        transfer_start_snapshot = self.transfer_domain.accounting_snapshot()
        accounting_start_snapshot = self._compact_accounting_snapshot(step_handle, phase="accounting_start")
        logger.info(
            format_agentic_event(
                "ROLLOUT",
                "accounting_start",
                rollout=rollout_id,
                group_size=transfer_start_snapshot["group_size"],
                target_groups=self.transfer_domain.target_group_count(),
                terminal_step=terminal_step,
                final_backfill_step=step_handle.final_backfill_step,
                ready_groups=transfer_start_snapshot["ready_groups"],
            )
        )
        logger.info(
            format_agentic_event(
                "ROLLOUT",
                "accounting_snapshot",
                **accounting_start_snapshot,
            )
        )
        step_sealed = False
        try:
            # Resident flow: keep moving data until previous/current transfer slots are filled.
            # Protected sessions finish before the IR release gate catches the remaining abortable request tail.
            closed_step = await self.close_step(step_handle=step_handle)
            if closed_step is None:
                raise RuntimeError("Agentic pipeline close_step returned no output outside cleanup mode.")
            step_sealed = True
            output = closed_step.output
            end_snapshot = closed_step.end_snapshot
            runtime_snapshot = closed_step.runtime_snapshot
            committed_groups = closed_step.committed_groups
            progress_snapshot = closed_step.progress_snapshot
            logger.info(
                format_agentic_event(
                    "ROLLOUT",
                    "accounting_end",
                    rollout=rollout_id,
                    group_size=end_snapshot["group_size"],
                    committed_previous_groups=end_snapshot["committed_previous_groups"],
                    committed_current_groups=end_snapshot["committed_current_groups"],
                    materialized_sessions=progress_snapshot["materialized_sessions"],
                    scored_samples=progress_snapshot["scored_samples"],
                    accounting_snapshot=closed_step.accounting_end_snapshot,
                )
            )
            append_agentic_accounting_marker(
                args,
                step=rollout_id,
                snapshot=closed_step.accounting_end_snapshot,
            )
            resident_groups = int(closed_step.accounting_end_snapshot.get("resident_group_count", 0) or 0)
            if resident_groups:
                logger.info(
                    f"Rollout not completed for rollout_id: {rollout_id}, have {resident_groups} samples aborted."
                )
            else:
                logger.info(f"Rollout fully completed for rollout_id: {rollout_id}.")
            logger.info(
                "Agentic runtime volatile resource snapshot rollout=%s emitted=%s drained=%s "
                "waiting_materialized_records=%s prepare_gate_blocked_irs=%s "
                "partial_resume_gate_blocked_irs=%s session_debug_totals=%s",
                rollout_id,
                runtime_snapshot.get("emitted_materialized_session_count_total"),
                runtime_snapshot.get("drained_materialized_session_count_total"),
                runtime_snapshot.get("waiting_materialized_records"),
                runtime_snapshot.get("prepare_gate_blocked_ir_count"),
                runtime_snapshot.get("partial_resume_gate_blocked_ir_count"),
                runtime_snapshot.get("session_debug_totals"),
            )
            committed_samples = (
                _assert_and_flatten_agentic_export_samples(committed_groups) if committed_groups else []
            )
            agentic_metrics = dict(output.metrics or {})
            agentic_metrics.update(
                _aggregate_rollout_timing_from_agentic_trace(
                    samples=committed_samples,
                    get_samples_times=closed_step.get_samples_times,
                    reward_trace_snapshots=closed_step.reward_trace_snapshots,
                )
            )
            if getattr(args, "use_metrics_service", False) and getattr(args, "timeline_dump_dir", None):
                timeline_events = _build_agentic_critical_path_timeline_events(
                    rollout_id=compute_rollout_step(args, rollout_id),
                    samples=committed_samples,
                    reward_trace_snapshots=closed_step.reward_trace_snapshots,
                )
                if timeline_events:
                    agentic_metrics["__timeline_events__"] = timeline_events
            agentic_metrics.update(metric_gatherer.collect())
            agentic_metrics.update(
                {
                    f"agentic/{key}": float(value)
                    for key, value in closed_step.accounting_end_snapshot.items()
                    if key.startswith("judge_")
                }
            )
            append_reward_workload_markers(
                args,
                collection_rollout_id=rollout_id,
                default_step=compute_rollout_step(args, rollout_id),
                committed_samples=committed_samples,
                reward_trace_snapshots=closed_step.reward_trace_snapshots,
            )
            save_reward_trace_snapshots_jsonl(
                args,
                rollout_id,
                closed_step.reward_trace_snapshots,
                committed_samples=committed_samples,
            )
            _merge_non_conflicting_metrics(
                agentic_metrics, compute_rollout_explicit_reward_metrics(args, committed_samples)
            )
            _merge_non_conflicting_metrics(agentic_metrics, collect_agentic_metadata_metrics(committed_samples))
            await stop_sglang_profile(args, rollout_id)
            sglang_profile_stopped = True
            if committed_samples:
                last_sample = committed_samples[-1]
                logger.info("Finish rollout sample summary: %s", _sample_log_summary(last_sample))
                rollout_time = time.monotonic() - rollout_started_at
                if args.save_debug_rollout_data is not None:
                    save_debug_rollout_data(
                        args,
                        committed_samples,
                        rollout_id=rollout_id,
                        evaluation=False,
                        tokenizer=get_agentic_runtime_resources(args).compiler.tokenizer,
                    )
                _log_rollout_data(
                    rollout_id,
                    args,
                    committed_samples,
                    agentic_metrics,
                    rollout_time,
                )
                if args.debug_rollout_only:
                    if self.transfer_domain is None or self.transfer_domain.data_system_client is None:
                        raise RuntimeError("Agentic debug rollout cleanup requires an initialized TransferDomain.")
                    logger.info("Debug rollout only mode - data system cleanup")
                    partition_rollout_id = rollout_id - 1 if step_handle.final_backfill_step else rollout_id
                    await self.transfer_domain.data_system_client.async_clear_partition(
                        partition_id=f"train_{partition_rollout_id}"
                    )
            output.metrics = agentic_metrics
            flat_samples = _assert_and_flatten_agentic_export_samples(output.samples)
            output.samples = flat_samples
            return output
        finally:
            if not sglang_profile_stopped:
                await stop_sglang_profile(args, rollout_id)
            if not step_sealed:
                await self.close_step(step_handle=step_handle, cleanup_only=True)
            lifecycle_terminal = terminal_step and (not step_sealed or not self._final_backfill_pending)
            if lifecycle_terminal and not step_handle.defer_terminal_shutdown_for_eval:
                await self.shutdown()
            if self._active_step_handle is step_handle:
                self._active_step_handle = None


def get_agentic_resident_pipeline() -> AgenticResidentPipeline:
    with _RESIDENT_PIPELINE_LOCK:
        if _RESIDENT_PIPELINE is None:
            raise RuntimeError(
                "Agentic resident pipeline has not been initialized. "
                "Call init_agentic_resident_pipeline before generate_rollout."
            )
        return _RESIDENT_PIPELINE


def init_agentic_resident_pipeline(args, data_source, data_system_client) -> None:
    global _RESIDENT_PIPELINE
    with _RESIDENT_PIPELINE_LOCK:
        if _RESIDENT_PIPELINE is None:
            _RESIDENT_PIPELINE = AgenticResidentPipeline()
        resident_pipeline = _RESIDENT_PIPELINE
    _run_on_resident_async_loop(
        resident_pipeline.init_pipeline(
            args=args,
            data_source=data_source,
            data_system_client=data_system_client,
        )
    )


def _assert_and_flatten_agentic_export_samples(groups: list[list[Sample]]) -> list[Sample]:
    flat_samples: list[Sample] = []
    for group in groups:
        if not isinstance(group, list):
            raise TypeError(f"Agentic export expected list[list[Sample]], got group type {type(group)}")
        for sample in group:
            session_id = sample.session_id
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("Agentic export expects every sample to carry a non-empty session_id.")
            flat_samples.append(sample)
    return flat_samples


@dataclass(frozen=True)
class _AgenticClosedStep:
    output: RolloutFnTrainOutput
    end_snapshot: dict[str, Any]
    runtime_snapshot: dict[str, Any]
    committed_groups: list[list[Sample]]
    progress_snapshot: dict[str, int]
    get_samples_times: list[float]
    reward_trace_snapshots: list[dict[str, Any]]
    accounting_end_snapshot: dict[str, object]


@dataclass
class _AgenticStepHandle:
    """Step boundary handle for a resident agentic pipeline.

    Dataflow and lifecycle control stay in AgenticResidentPipeline; this object
    carries step-scoped domains, progress, and accounting state.
    """

    rollout_id: int
    required_group_count: int
    terminal_step: bool = False
    final_backfill_step: bool = False
    defer_terminal_shutdown_for_eval: bool = False
    progress: RolloutProgress | None = None
    rollout_irs_gated: bool = False
    active_backend_requests_sealed: bool = False
    first_rollout_sample_logged: bool = False
    last_progress_at: float = field(default_factory=time.time)
    last_idle_heartbeat_at: float = 0.0


def _latency_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _append_trace_duration(
    values: dict[str, list[float]],
    *,
    metric_name: str,
    events: dict[str, Any],
    start_key: str,
    end_key: str,
) -> None:
    started_at = events.get(start_key)
    ended_at = events.get(end_key)
    if not isinstance(started_at, (int, float)) or not isinstance(ended_at, (int, float)):
        return
    if ended_at < started_at:
        return
    values.setdefault(metric_name, []).append(float(ended_at) - float(started_at))


def _trace_metadata_with_reward_snapshots(
    samples: list[Sample],
    reward_trace_snapshots: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    trace_metadata: list[dict[str, Any]] = [
        sample.metadata if isinstance(sample.metadata, dict) else {} for sample in samples
    ]
    committed_sample_keys = {
        repr(metadata.get(TRACE_KEY, {}).get("reward", {}).get("sample_key"))
        for metadata in trace_metadata
        if isinstance(metadata.get(TRACE_KEY), dict)
        and isinstance(metadata.get(TRACE_KEY, {}).get("reward"), dict)
        and metadata.get(TRACE_KEY, {}).get("reward", {}).get("sample_key") is not None
    }
    for snapshot in reward_trace_snapshots or []:
        if not isinstance(snapshot, dict):
            continue
        reward_trace = snapshot.get(TRACE_KEY, {}).get("reward", {})
        snapshot_key = repr(reward_trace.get("sample_key")) if isinstance(reward_trace, dict) else None
        if snapshot_key is not None and snapshot_key in committed_sample_keys:
            continue
        trace_metadata.append(snapshot)
    return trace_metadata


def _complete_timeline_event(
    *,
    name: str,
    start_s: Any,
    end_s: Any,
    pid: int,
    tid: int,
    rollout_id: int,
    clock_host: str | None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)) or end_s < start_s:
        return None
    event_attributes = {"step": rollout_id, "clock_host": clock_host}
    if attributes:
        event_attributes.update(copy.deepcopy(attributes))
    return {
        "name": name,
        "ph": "X",
        "ts": int(float(start_s) * 1e6),
        "dur": int((float(end_s) - float(start_s)) * 1e6),
        "pid": pid,
        "tid": tid,
        "args": event_attributes,
    }


def _build_agentic_critical_path_timeline_events(
    *,
    rollout_id: int,
    samples: list[Sample],
    reward_trace_snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    timeline_events: list[dict[str, Any]] = []
    for tid, metadata in enumerate(_trace_metadata_with_reward_snapshots(samples, reward_trace_snapshots), start=1):
        agentic_trace = metadata.get(TRACE_KEY)
        if not isinstance(agentic_trace, dict):
            continue
        reward_trace = agentic_trace.get("reward")
        attributes = {}
        if isinstance(reward_trace, dict):
            attributes = {
                "sample_key": reward_trace.get("sample_key"),
                "group_key": reward_trace.get("group_key"),
                "terminal_outcome": reward_trace.get("terminal_outcome"),
                "pipeline_status": reward_trace.get("pipeline_status"),
                "executor_status": reward_trace.get("executor_status"),
                "executor_error_code": reward_trace.get("executor_error_code"),
                "benchmark_mode": reward_trace.get("benchmark_mode"),
                "group_index": reward_trace.get("group_index"),
                "sample_index": reward_trace.get("sample_index"),
                "context_hash": reward_trace.get("context_hash"),
                "trajectory_hash": reward_trace.get("trajectory_hash"),
                "recorded_reward_hash": reward_trace.get("recorded_reward_hash"),
                "benchmark_invariant_hash": reward_trace.get("benchmark_invariant_hash"),
                "reasoning_trigger": reward_trace.get("reasoning_trigger"),
                "reasoning_execution_trigger": reward_trace.get("reasoning_execution_trigger"),
                "per_turn_judge_count": reward_trace.get("per_turn_judge_count"),
                "expected_trainer_components": reward_trace.get("expected_trainer_components"),
                "collection_rollout_id": reward_trace.get("collection_rollout_id"),
                "train_partition_rollout_id": reward_trace.get("train_partition_rollout_id"),
            }
        event_step = (
            reward_trace.get("train_partition_step", rollout_id) if isinstance(reward_trace, dict) else rollout_id
        )
        trace_events = agentic_trace.get("events")
        if isinstance(trace_events, dict):
            for name, start_key, end_key, pid in (
                ("critical_path.rollout_finalize", "finalize_start_at", "finalize_end_at", 2001),
                (
                    "critical_path.rollout_materialize_wait",
                    "materialize_ready_at",
                    "materialize_harvest_at",
                    2001,
                ),
                (
                    "critical_path.reward_context_build",
                    "reward_context_build_start_at",
                    "reward_context_build_end_at",
                    2002,
                ),
                ("critical_path.reward", "reward_arrive_at", "reward_end_at", 2002),
                (
                    "critical_path.reward_group_finalize",
                    "group_finalize_start_at",
                    "group_finalize_end_at",
                    2002,
                ),
                (
                    "critical_path.transfer_buffer_wait",
                    "transfer_buffer_enter_at",
                    "transfer_release_start_at",
                    2003,
                ),
                ("critical_path.transfer", "transfer_release_start_at", "transfer_release_end_at", 2003),
            ):
                event = _complete_timeline_event(
                    name=name,
                    start_s=trace_events.get(start_key),
                    end_s=trace_events.get(end_key),
                    pid=pid,
                    tid=tid,
                    rollout_id=event_step,
                    clock_host=agentic_span_clock_host(trace_events, start_key, end_key),
                    attributes=attributes,
                )
                if event is not None:
                    timeline_events.append(event)
            turn_judge_barrier_event = _complete_timeline_event(
                name="critical_path.turn_judge_barrier",
                start_s=trace_events.get("turn_judge_barrier_start_at"),
                end_s=trace_events.get("turn_judge_barrier_release_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    trace_events,
                    "turn_judge_barrier_start_at",
                    "turn_judge_barrier_release_at",
                ),
                attributes=attributes,
            )
            if turn_judge_barrier_event is not None:
                timeline_events.append(turn_judge_barrier_event)
            if isinstance(reward_trace, dict):
                judges = reward_trace.get("judges")
                for component, judge_trace in (judges if isinstance(judges, dict) else {}).items():
                    if not isinstance(judge_trace, dict):
                        judge_trace = {}
                    # Sub-timings (queue_elapsed_s, http_elapsed_s, the server-reported
                    # engine_http_elapsed_s inside `server`, ...) previously existed only
                    # in the rollout JSONL, not the timeline -- attach them here so the
                    # dominant cost inside a reward branch (client-side concurrency-limit
                    # queueing, measured separately below as `.queue`) is visible without
                    # cross-referencing the JSONL.
                    component_attributes = {
                        **attributes,
                        "attempt_count": judge_trace.get("attempt_count"),
                        "invalid_response_count": judge_trace.get("invalid_response_count"),
                        "queue_elapsed_s": judge_trace.get("queue_elapsed_s"),
                        "payload_prep_elapsed_s": judge_trace.get("payload_prep_elapsed_s"),
                        "http_elapsed_s": judge_trace.get("http_elapsed_s"),
                        "parse_elapsed_s": judge_trace.get("parse_elapsed_s"),
                        "backoff_elapsed_s": judge_trace.get("backoff_elapsed_s"),
                        "server": judge_trace.get("server"),
                    }
                    event = _complete_timeline_event(
                        name=f"critical_path.reward.{component}",
                        start_s=trace_events.get(f"judge_{component}_queue_enter_at"),
                        end_s=trace_events.get(f"judge_{component}_end_at"),
                        pid=2002,
                        tid=tid,
                        rollout_id=event_step,
                        clock_host=agentic_span_clock_host(
                            trace_events,
                            f"judge_{component}_queue_enter_at",
                            f"judge_{component}_end_at",
                        ),
                        attributes=component_attributes,
                    )
                    if event is not None:
                        timeline_events.append(event)
                    # Client-side concurrency-limit wait: a request that is fully ready
                    # but blocked on `self._semaphores[spec.role]` in
                    # relax/engine/rewards/dual_agentic_judge.py. Measured to be 65% of
                    # the reward branch on the terminal_once reference run.
                    queue_event = _complete_timeline_event(
                        name=f"critical_path.reward.{component}.queue",
                        start_s=trace_events.get(f"judge_{component}_queue_enter_at"),
                        end_s=trace_events.get(f"judge_{component}_queue_acquired_at"),
                        pid=2002,
                        tid=tid,
                        rollout_id=event_step,
                        clock_host=agentic_span_clock_host(
                            trace_events,
                            f"judge_{component}_queue_enter_at",
                            f"judge_{component}_queue_acquired_at",
                        ),
                        attributes={**attributes, "queue_elapsed_s": judge_trace.get("queue_elapsed_s")},
                    )
                    if queue_event is not None:
                        timeline_events.append(queue_event)
                    # http_start_at/http_end_at are overwritten on each retry attempt, so
                    # this span covers only the LAST attempt, not the cumulative
                    # http_elapsed_s (which sums every attempt); attempt_count is carried
                    # as an attribute so a multi-attempt sample is identifiable.
                    http_event = _complete_timeline_event(
                        name=f"critical_path.reward.{component}.http",
                        start_s=trace_events.get(f"judge_{component}_http_start_at"),
                        end_s=trace_events.get(f"judge_{component}_http_end_at"),
                        pid=2002,
                        tid=tid,
                        rollout_id=event_step,
                        clock_host=agentic_span_clock_host(
                            trace_events,
                            f"judge_{component}_http_start_at",
                            f"judge_{component}_http_end_at",
                        ),
                        attributes={
                            **attributes,
                            "attempt_count": judge_trace.get("attempt_count"),
                            "server": judge_trace.get("server"),
                        },
                    )
                    if http_event is not None:
                        timeline_events.append(http_event)
        turns = agentic_trace.get("turns")
        for turn_index, turn in enumerate(turns if isinstance(turns, list) else []):
            if not isinstance(turn, dict):
                continue
            turn_judge = turn.get("judge")
            judge_events = turn_judge.get("events") if isinstance(turn_judge, dict) else None
            if isinstance(judge_events, dict):
                judge_attributes = {
                    **attributes,
                    "turn_index": turn_index,
                    "judge_role": turn_judge.get("role"),
                    "judge_status": turn_judge.get("status"),
                    "response_state_hash": turn_judge.get("response_state_hash"),
                    "observation_state_hash": turn_judge.get("observation_state_hash"),
                }
                turn_judge_event = _complete_timeline_event(
                    name="critical_path.turn_judge",
                    start_s=judge_events.get("turn_judge_trigger_at"),
                    end_s=judge_events.get("turn_judge_end_at"),
                    pid=2002,
                    tid=tid,
                    rollout_id=event_step,
                    clock_host=agentic_span_clock_host(
                        judge_events,
                        "turn_judge_trigger_at",
                        "turn_judge_end_at",
                    ),
                    attributes=judge_attributes,
                )
                if turn_judge_event is not None:
                    timeline_events.append(turn_judge_event)
            turn_events = turn.get("events")
            if not isinstance(turn_events, dict):
                continue
            # `rollout_pre_generation` used to span chat_request_arrive_at ->
            # generation_queue_enter_at as one span and was misclassified as
            # rollout work in ad-hoc analyses (it is the largest span family in
            # the trace: p50 0.12s but p90 107s, max 720s). It is split here
            # along the two admission-control marks that actually exist on this
            # path (see relax/agentic/session/service.py: ir_created_at is set
            # in `_create_inflight_request`, ir_activated_at when
            # `_maybe_start_next_ir_locked` pops the IR off the per-session
            # admission gate, generation_queue_enter_at when `_run_ir` hands it
            # toward the SGLang queue) so the genuine backpressure wait
            # (admission_wait) is distinguishable from setup and dispatch
            # bookkeeping around it.
            admission_setup_event = _complete_timeline_event(
                name="critical_path.rollout_admission_setup",
                start_s=turn_events.get("chat_request_arrive_at"),
                end_s=turn_events.get("ir_created_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "chat_request_arrive_at",
                    "ir_created_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if admission_setup_event is not None:
                timeline_events.append(admission_setup_event)
            admission_wait_event = _complete_timeline_event(
                name="critical_path.rollout_admission_wait",
                start_s=turn_events.get("ir_created_at"),
                end_s=turn_events.get("ir_activated_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "ir_created_at",
                    "ir_activated_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if admission_wait_event is not None:
                timeline_events.append(admission_wait_event)
            dispatch_event = _complete_timeline_event(
                name="critical_path.rollout_dispatch",
                start_s=turn_events.get("ir_activated_at"),
                end_s=turn_events.get("generation_queue_enter_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "ir_activated_at",
                    "generation_queue_enter_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if dispatch_event is not None:
                timeline_events.append(dispatch_event)
            queue_event = _complete_timeline_event(
                name="critical_path.rollout_queue",
                start_s=turn_events.get("generation_queue_enter_at"),
                end_s=turn_events.get("generation_start_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "generation_queue_enter_at",
                    "generation_start_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if queue_event is not None:
                timeline_events.append(queue_event)
            event = _complete_timeline_event(
                name="critical_path.rollout_generation",
                start_s=turn_events.get("generation_start_at"),
                end_s=turn_events.get("generation_end_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "generation_start_at",
                    "generation_end_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if event is not None:
                timeline_events.append(event)
            post_generation_event = _complete_timeline_event(
                name="critical_path.rollout_post_generation",
                start_s=turn_events.get("generation_end_at"),
                end_s=turn_events.get("chat_end_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "generation_end_at",
                    "chat_end_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if post_generation_event is not None:
                timeline_events.append(post_generation_event)
            managed_session_event = _complete_timeline_event(
                name="critical_path.rollout_managed_session",
                start_s=turn_events.get("managed_session_runner_start_at"),
                end_s=turn_events.get("managed_session_runner_end_at"),
                pid=2001,
                tid=tid,
                rollout_id=event_step,
                clock_host=agentic_span_clock_host(
                    turn_events,
                    "managed_session_runner_start_at",
                    "managed_session_runner_end_at",
                ),
                attributes={**attributes, "turn_index": turn_index},
            )
            if managed_session_event is not None:
                timeline_events.append(managed_session_event)
    return timeline_events


def _aggregate_rollout_timing_from_agentic_trace(
    *,
    samples: list[Sample],
    get_samples_times: list[float],
    reward_trace_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    generate_values: list[float] = []
    post_generate_values: list[float] = []
    reward_metric_values: dict[str, list[float]] = {}
    reward_status_counts: dict[tuple[str, str], int] = {}
    turn_judge_status_counts: dict[str, int] = {}
    group_timelines: dict[str, dict[str, list[float]]] = {}
    phase_values: dict[str, list[float]] = {
        "process_vision_info": [],
        "image_processor": [],
        "mm_encode": [],
    }
    for metadata in _trace_metadata_with_reward_snapshots(samples, reward_trace_snapshots):
        agentic_trace = metadata.get(TRACE_KEY)
        if not isinstance(agentic_trace, dict):
            continue
        trace_events = agentic_trace.get("events")
        if not isinstance(trace_events, dict):
            trace_events = {}
        for metric_name, start_key, end_key in (
            ("reward/global_queue_time", "reward_arrive_at", "reward_start_at"),
            ("reward/sample_service_time", "reward_start_at", "reward_end_at"),
            ("reward/sample_sojourn_time", "reward_arrive_at", "reward_end_at"),
            ("transfer/release_time", "transfer_release_start_at", "transfer_release_end_at"),
            ("turn_judge/barrier_time", "turn_judge_barrier_start_at", "turn_judge_barrier_release_at"),
        ):
            _append_trace_duration(
                reward_metric_values,
                metric_name=metric_name,
                events=trace_events,
                start_key=start_key,
                end_key=end_key,
            )

        # A per-turn judge scores one resp->observation pair, so the final
        # response of a trajectory is never covered by the VLM. Report the
        # coverage explicitly rather than leaving it to be inferred.
        judged_turns = agentic_trace.get("per_turn_judge_count")
        assistant_turns = agentic_trace.get("per_turn_assistant_turn_count")
        if isinstance(judged_turns, int) and isinstance(assistant_turns, int) and assistant_turns > 0:
            reward_metric_values.setdefault("turn_judge/judged_turn_count", []).append(float(judged_turns))
            reward_metric_values.setdefault("turn_judge/assistant_turn_count", []).append(float(assistant_turns))
            reward_metric_values.setdefault("turn_judge/turn_coverage_ratio", []).append(
                float(judged_turns) / float(assistant_turns)
            )
        off_lineage_judges = agentic_trace.get("per_turn_off_lineage_judge_count")
        if isinstance(off_lineage_judges, int) and off_lineage_judges > 0:
            reward_metric_values.setdefault("turn_judge/off_lineage_judge_count", []).append(float(off_lineage_judges))

        reward_trace = agentic_trace.get("reward")
        if isinstance(reward_trace, dict):
            for field_name, metric_name in (
                ("global_queue_elapsed_s", "reward/global_queue_monotonic_time"),
                ("pipeline_elapsed_s", "reward/pipeline_time"),
                ("executor_elapsed_s", "reward/executor_time"),
                ("answer_accuracy_projection_elapsed_s", "reward/answer_accuracy/projection_time"),
                ("multi_turn_reasoning_projection_elapsed_s", "reward/multi_turn_reasoning/projection_time"),
            ):
                value = reward_trace.get(field_name)
                if isinstance(value, (int, float)):
                    reward_metric_values.setdefault(metric_name, []).append(float(value))
            for status_field, component in (
                ("pipeline_status", "pipeline"),
                ("executor_status", "executor"),
                ("terminal_outcome", "terminal"),
            ):
                status = reward_trace.get(status_field)
                if isinstance(status, str):
                    status_key = (component, status)
                    reward_status_counts[status_key] = reward_status_counts.get(status_key, 0) + 1
            judges = reward_trace.get("judges")
            if isinstance(judges, dict):
                for component, judge_trace in judges.items():
                    if not isinstance(component, str) or not isinstance(judge_trace, dict):
                        continue
                    for field_name, metric_suffix in (
                        ("queue_elapsed_s", "queue_time"),
                        ("payload_prep_elapsed_s", "payload_prep_time"),
                        ("http_elapsed_s", "http_time"),
                        ("parse_elapsed_s", "parse_time"),
                        ("backoff_elapsed_s", "backoff_time"),
                        ("elapsed_s", "branch_time"),
                        ("attempt_count", "attempt_count"),
                        ("invalid_response_count", "invalid_response_count"),
                    ):
                        value = judge_trace.get(field_name)
                        if isinstance(value, (int, float)):
                            metric_name = f"reward/{component}/{metric_suffix}"
                            reward_metric_values.setdefault(metric_name, []).append(float(value))
                    server_timings = judge_trace.get("server")
                    if isinstance(server_timings, dict):
                        for field_name, value in server_timings.items():
                            if (
                                not isinstance(field_name, str)
                                or isinstance(value, bool)
                                or not isinstance(value, (int, float))
                            ):
                                continue
                            metric_name = f"reward/{component}/server/{field_name}"
                            reward_metric_values.setdefault(metric_name, []).append(float(value))
                        client_http = judge_trace.get("http_elapsed_s")
                        server_total = server_timings.get("request_total_elapsed_s")
                        if (
                            isinstance(client_http, (int, float))
                            and isinstance(server_total, (int, float))
                            and client_http >= server_total
                        ):
                            reward_metric_values.setdefault(
                                f"reward/{component}/client_server_residual_time", []
                            ).append(float(client_http) - float(server_total))
                    status = judge_trace.get("status")
                    if isinstance(status, str):
                        status_key = (component, status)
                        reward_status_counts[status_key] = reward_status_counts.get(status_key, 0) + 1

            group_key = reward_trace.get("group_key")
            group_id = (
                repr(group_key)
                if group_key is not None
                else f"group_index:{reward_trace.get('group_index', 'unknown')}"
            )
            timeline = group_timelines.setdefault(group_id, {"arrive": [], "end": [], "ready": [], "release": []})
            for timeline_key, event_key in (
                ("arrive", "reward_arrive_at"),
                ("end", "reward_end_at"),
                ("ready", "group_reward_ready_at"),
                ("release", "transfer_release_start_at"),
            ):
                value = trace_events.get(event_key)
                if isinstance(value, (int, float)):
                    timeline[timeline_key].append(float(value))

        turns = agentic_trace.get("turns")
        for turn in turns if isinstance(turns, list) else []:
            if not isinstance(turn, dict):
                continue
            turn_judge = turn.get("judge")
            if isinstance(turn_judge, dict):
                turn_judge_events = turn_judge.get("events")
                if isinstance(turn_judge_events, dict):
                    _append_trace_duration(
                        reward_metric_values,
                        metric_name="turn_judge/sidecar_elapsed_time",
                        events=turn_judge_events,
                        start_key="turn_judge_trigger_at",
                        end_key="turn_judge_end_at",
                    )
                    _append_trace_duration(
                        reward_metric_values,
                        metric_name="turn_judge/projection_time",
                        events=turn_judge_events,
                        start_key="turn_judge_projection_start_at",
                        end_key="turn_judge_projection_end_at",
                    )
                projection_elapsed_s = turn_judge.get("projection_elapsed_s")
                if isinstance(projection_elapsed_s, (int, float)) and not isinstance(projection_elapsed_s, bool):
                    reward_metric_values.setdefault("turn_judge/projection_reported_time", []).append(
                        float(projection_elapsed_s)
                    )
                judge_trace = turn_judge.get("judge")
                if isinstance(judge_trace, dict):
                    for field_name, metric_suffix in (
                        ("queue_elapsed_s", "queue_time"),
                        ("payload_prep_elapsed_s", "payload_prep_time"),
                        ("http_elapsed_s", "http_time"),
                        ("parse_elapsed_s", "parse_time"),
                        ("backoff_elapsed_s", "backoff_time"),
                        ("elapsed_s", "branch_time"),
                        ("attempt_count", "attempt_count"),
                        ("invalid_response_count", "invalid_response_count"),
                    ):
                        value = judge_trace.get(field_name)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            reward_metric_values.setdefault(
                                f"turn_judge/multi_turn_reasoning/{metric_suffix}", []
                            ).append(float(value))
                    server_timings = judge_trace.get("server")
                    if isinstance(server_timings, dict):
                        for field_name, value in server_timings.items():
                            if (
                                not isinstance(field_name, str)
                                or isinstance(value, bool)
                                or not isinstance(value, (int, float))
                            ):
                                continue
                            reward_metric_values.setdefault(
                                f"turn_judge/multi_turn_reasoning/server/{field_name}", []
                            ).append(float(value))
                        client_http = judge_trace.get("http_elapsed_s")
                        server_total = server_timings.get("request_total_elapsed_s")
                        if (
                            isinstance(client_http, (int, float))
                            and isinstance(server_total, (int, float))
                            and client_http >= server_total
                        ):
                            reward_metric_values.setdefault(
                                "turn_judge/multi_turn_reasoning/client_server_residual_time", []
                            ).append(float(client_http) - float(server_total))
                turn_judge_status = turn_judge.get("status")
                if isinstance(turn_judge_status, str):
                    turn_judge_status_counts[turn_judge_status] = (
                        turn_judge_status_counts.get(turn_judge_status, 0) + 1
                    )
            generation_elapsed_s = turn.get("generation_elapsed_s")
            wall_elapsed_s = turn.get("wall_elapsed_s")
            events = turn.get("events")
            if isinstance(generation_elapsed_s, (int, float)):
                generate_values.append(float(generation_elapsed_s))
            if (
                isinstance(wall_elapsed_s, (int, float))
                and isinstance(generation_elapsed_s, (int, float))
                and wall_elapsed_s >= generation_elapsed_s
            ):
                post_generate_values.append(float(wall_elapsed_s) - float(generation_elapsed_s))
            if not isinstance(events, dict):
                continue
            _append_trace_duration(
                reward_metric_values,
                metric_name="rollout/generation_queue_time",
                events=events,
                start_key="generation_queue_enter_at",
                end_key="generation_start_at",
            )
            for event_key, phase in (
                ("process_vision_info_elapsed_s", "process_vision_info"),
                ("processor_elapsed_s", "image_processor"),
                ("media_encode_elapsed_s", "mm_encode"),
            ):
                value = events.get(event_key)
                if isinstance(value, (int, float)):
                    phase_values[phase].append(float(value))

    metrics: dict[str, float] = {}
    for phase, values in phase_values.items():
        if values:
            for statistic, value in _latency_statistics(values).items():
                metrics[f"perf_detail/rollout/{phase}_time/{statistic}"] = value
    if generate_values:
        for statistic, value in _latency_statistics(generate_values).items():
            metrics[f"perf_detail/rollout/generate_time/{statistic}"] = value
    if post_generate_values:
        for statistic, value in _latency_statistics(post_generate_values).items():
            metrics[f"perf_detail/rollout/post_generate_time/{statistic}"] = value
    if get_samples_times:
        metrics["perf_detail/rollout/get_samples_time/total"] = sum(get_samples_times)
        metrics["perf_detail/rollout/get_samples_time/mean"] = sum(get_samples_times) / len(get_samples_times)
    for timeline in group_timelines.values():
        arrivals = timeline["arrive"]
        reward_ends = timeline["end"]
        ready_times = timeline["ready"]
        release_times = timeline["release"]
        if arrivals and reward_ends:
            group_sojourn = max(reward_ends) - min(arrivals)
            if group_sojourn >= 0:
                reward_metric_values.setdefault("reward/group_sojourn_time", []).append(group_sojourn)
        if len(reward_ends) > 1:
            reward_metric_values.setdefault("reward/group_completion_spread_time", []).append(
                max(reward_ends) - min(reward_ends)
            )
        if reward_ends and ready_times:
            finalize_delay = min(ready_times) - max(reward_ends)
            if finalize_delay >= 0:
                reward_metric_values.setdefault("reward/group_finalize_delay_time", []).append(finalize_delay)
        if ready_times and release_times:
            release_delay = min(release_times) - max(ready_times)
            if release_delay >= 0:
                reward_metric_values.setdefault("transfer/reward_ready_to_release_time", []).append(release_delay)
    for metric_name, values in reward_metric_values.items():
        if not values:
            continue
        for statistic, value in _latency_statistics(values).items():
            metrics[f"perf_detail/{metric_name}/{statistic}"] = value
    for (component, status), count in reward_status_counts.items():
        metrics[f"perf_detail/reward/{component}/status/{status}_count"] = float(count)
    for status, count in turn_judge_status_counts.items():
        metrics[f"perf_detail/turn_judge/status/{status}_count"] = float(count)
    return metrics


def _merge_non_conflicting_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = value


def collect_agentic_metadata_metrics(samples: list[Sample]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else None
        if not metadata:
            continue
        for key, value in metadata.items():
            if not isinstance(key, str) or not key or key.startswith("_") or key in _AGENT_METADATA_INTERNAL_KEYS:
                continue
            if not isinstance(value, (int, float)):
                continue
            metric_values.setdefault(key, []).append(float(value))
    return finalize_rollout_explicit_metric_values(metric_values)


def _dict_add_prefix(d, prefix):
    return {f"{prefix}{k}": v for k, v in d.items()}


def _compute_zero_std_metrics(args, all_samples: list[Sample]) -> dict[str, float]:
    if args.advantage_estimator == "ppo":
        return {}

    all_sample_groups = group_by(all_samples, lambda sample: sample.group_index)
    reward_groups = [
        [sample.get_reward_value(args) for sample in group if sample.reward is not None]
        for group in all_sample_groups.values()
    ]
    interesting_rewards = [
        str(round(rewards[0], 1))
        for rewards in reward_groups
        if rewards and all(rewards[0] == reward for reward in rewards)
    ]
    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_reward_cat_metrics(args, all_samples: list[Sample]) -> dict[str, float]:
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}
    rewarded_samples = [
        sample for sample in all_samples if isinstance(sample.reward, dict) and reward_cat_key in sample.reward
    ]
    if not rewarded_samples:
        return {}
    samples_of_reward_cat = group_by(rewarded_samples, lambda sample: sample.reward[reward_cat_key])
    return {
        f"error_cat/{reward_cat}": len(samples) / len(rewarded_samples)
        for reward_cat, samples in samples_of_reward_cat.items()
    }


def _compute_prefix_cache_metrics(args, samples: list[Sample]) -> dict[str, float]:
    num_samples = len(samples)
    if num_samples == 0:
        return {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in samples)
    return {
        "prefix_cache_hit_rate": (total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0),
        "avg_cached_tokens_per_sample": total_cached_tokens / num_samples,
    }


def _compute_rollout_metrics_from_samples(args, samples: list[Sample]) -> dict[str, float]:
    response_lengths = [sample.effective_response_length for sample in samples]
    log_dict: dict[str, float] = {}
    log_dict |= _dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([has_repetition(sample.response) for sample in samples]).item()
    log_dict["truncated_ratio"] = np.mean([sample.status == Sample.Status.TRUNCATED for sample in samples]).item()
    turns = [s.metadata.get("agentic_trace", {}).get("turn_count", 1) for s in samples]
    log_dict["num_turn/mean"] = float(np.mean(turns))
    log_dict["num_turn/max"] = float(np.max(turns))
    log_dict["num_turn/min"] = float(np.min(turns))
    return log_dict


def _compute_rollout_perf_metrics_from_samples(args, samples: list[Sample], rollout_time: float) -> dict[str, float]:
    non_generation_time = [sample.non_generation_time for sample in samples]
    log_dict: dict[str, float] = {"rollout_time": rollout_time}
    if max(non_generation_time) > 0:
        log_dict |= _dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths: list[int], non_generation_time_values: list[float], key: str = "") -> None:
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time_values) == 0:
            return

        aligned_non_generation_time = [
            value
            for value, length in zip(non_generation_time_values, response_lengths, strict=True)
            if length == max_response_length
        ]
        mean_non_generation_time = sum(aligned_non_generation_time) / len(aligned_non_generation_time)
        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")
    return log_dict


def _log_rollout_data(
    rollout_id,
    args,
    samples,
    rollout_extra_metrics,
    rollout_time,
) -> None:
    from relax.utils import tracking_utils

    save_rollout_result_jsonl(args, rollout_id, samples)

    if args.custom_rollout_log_function_path is not None:
        from relax.utils.misc import load_function

        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= _dict_add_prefix(_compute_rollout_metrics_from_samples(args, samples), "rollout/")
    log_dict |= _dict_add_prefix(_compute_rollout_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    # Agentic rollout and trainer report the same logical step from different
    # processes. Flush here as well as on the trainer so late rollout/reward
    # events, especially from the final step, cannot remain stranded in the
    # MetricsService timeline buffer with no later report trigger.
    tracking_utils.flush_metrics(args, step)


def _load_eval_dataset(args, dataset_cfg: EvalDatasetConfig):
    from relax.utils.data.data import Dataset
    from relax.utils.misc import load_function

    resources = get_agentic_runtime_resources(args).compiler
    custom_prompt_path = getattr(args, "custom_prompt_path", None)
    custom_prompt_func = load_function(custom_prompt_path) if custom_prompt_path else None
    return Dataset(
        path=dataset_cfg.path,
        tokenizer=resources.tokenizer,
        processor=resources.processor,
        max_length=args.eval_max_prompt_len,
        prompt_key=dataset_cfg.input_key,
        label_key=dataset_cfg.label_key,
        multimodal_keys=args.multimodal_keys,
        metadata_key=dataset_cfg.metadata_key,
        tool_key=dataset_cfg.tool_key,
        apply_chat_template=args.apply_chat_template,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        use_audio_in_video=args.use_audio_in_video,
        system_prompt=args.system_prompt,
        custom_prompt_func=custom_prompt_func,
    )


def _eval_sampling_params(args, dataset_cfg: EvalDatasetConfig, *, sample_slot_idx: int) -> dict[str, Any]:
    sampling_params = {
        "temperature": dataset_cfg.temperature,
        "top_p": dataset_cfg.top_p,
        "top_k": dataset_cfg.top_k,
        "max_new_tokens": dataset_cfg.max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if args.sglang_enable_deterministic_inference:
        sampling_params["sampling_seed"] = args.rollout_seed + sample_slot_idx
    return sampling_params


def _build_eval_samples(args, dataset_cfg: EvalDatasetConfig) -> list[Sample]:
    dataset = _load_eval_dataset(args, dataset_cfg)
    samples: list[Sample] = []
    sample_index = 0
    group_index = 0
    for prompt_sample in dataset.samples:
        for sample_slot_idx in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(sample.metadata)
            samples.append(sample)
        group_index += 1
    return samples


async def _advance_eval_outputs(
    *,
    runtime_domain,
    reward_domain,
    completed_samples: list[Sample],
    pbar,
    do_print: bool,
) -> tuple[bool, bool, int, int]:
    progressed = False
    runtime_materialized_group_count = 0
    runtime_dropped_group_count = 0
    discarded_group_keys = runtime_domain.drain_discarded_group_keys()
    if discarded_group_keys:
        runtime_dropped_group_count = len(discarded_group_keys)
        await reward_domain.drop_waiting_groups_by_key(discarded_group_keys)
        progressed = True
    runtime_dispatch = runtime_domain.drain_ready_execution()
    if runtime_dispatch.materialized_batches or runtime_dispatch.ready_groups:
        if runtime_dispatch.materialized_batches and not reward_domain.group_rm:
            reward_domain.accept_session_materializations(runtime_dispatch.materialized_batches)
            runtime_materialized_group_count += len(
                {record["group_key"] for batch in runtime_dispatch.materialized_batches for record in batch}
            )
        for group in runtime_dispatch.ready_groups:
            await reward_domain.ingest_groups([group])
        if reward_domain.group_rm:
            runtime_materialized_group_count += len(runtime_dispatch.ready_groups)
        progressed = True

    reward_progressed = await reward_domain.step_once()
    if reward_progressed:
        ready_reward_groups = reward_domain.drain_ready_dispatch()
        for group in ready_reward_groups:
            if do_print and group:
                sample = group[0]
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample.prompt) + sample.response]} "
                    f"reward={sample.reward}"
                )
                do_print = False
            completed_samples.extend(group)
            pbar.update(len(group))
        progressed = True
    return progressed, do_print, runtime_materialized_group_count, runtime_dropped_group_count


async def _run_eval_samples(
    args, rollout_id: int, dataset_cfg: EvalDatasetConfig, eval_samples: list[Sample]
) -> list[Sample]:
    if not eval_samples:
        return []

    completed_samples: list[Sample] = []
    do_print = True
    eval_sample_count = len(eval_samples)
    if args.group_rm:
        eval_prepare_groups: list[list[Sample]] = []
        current_group_index = object()
        for sample in eval_samples:
            if sample.group_index != current_group_index:
                eval_prepare_groups.append([])
                current_group_index = sample.group_index
            eval_prepare_groups[-1].append(sample)
    else:
        eval_prepare_groups = [[sample] for sample in eval_samples]
    total_group_count = len(eval_prepare_groups)
    eval_group_size = dataset_cfg.n_samples_per_eval_prompt if args.group_rm else 1
    train_prepare_pool_target_group_count = args.agentic_prepare_pool_size or args.over_sampling_batch_size
    train_prepare_pool_target_session_count = train_prepare_pool_target_group_count * args.n_samples_per_prompt
    eval_prepare_pool_target_group_count = args.agentic_eval_prepare_pool_size
    if eval_prepare_pool_target_group_count is None:
        eval_prepare_pool_target_group_count = (
            train_prepare_pool_target_session_count + eval_group_size - 1
        ) // eval_group_size
    eval_prepare_pool_target_group_count = min(total_group_count, eval_prepare_pool_target_group_count)
    # Cap runtime residency to training's steady-state in-flight session budget.
    # Leaving it at total_group_count lets the whole eval set queue in runtime and
    # can OOM the raylet on large eval datasets; training never hits this because
    # `over_sampling_batch_size` bounds admission per step.
    eval_runtime_admission_group_count = min(
        total_group_count,
        max(
            eval_prepare_pool_target_group_count,
            (train_prepare_pool_target_session_count + eval_group_size - 1) // eval_group_size,
        ),
    )
    pbar = tqdm(total=eval_sample_count, desc=f"Eval {dataset_cfg.name}", unit="sample")
    logger.info(
        "Agentic eval dataset=%s total_samples=%s total_group_count=%s "
        "prepare_pool_target_group_count=%s runtime_admission_group_count=%s",
        dataset_cfg.name,
        eval_sample_count,
        total_group_count,
        eval_prepare_pool_target_group_count,
        eval_runtime_admission_group_count,
    )
    eval_namespace = str(dataset_cfg.name).replace("/", "_").replace(" ", "_")
    for sample in eval_samples:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["eval_dataset"] = dataset_cfg.name
        sample.session_id = f"eval_session_{eval_namespace}_{rollout_id}_{sample.index}"

    prepare_domain = PrepareDomain(
        scope_id=_eval_scope_id(dataset_name=str(dataset_cfg.name), rollout_id=rollout_id),
        data_source=None,
        pool_target_group_count=eval_prepare_pool_target_group_count,
    )
    runtime_domain = RuntimeDomain(
        args=args,
        scope_id=prepare_domain.scope_id,
    )
    runtime_domain.rebind_step(
        args=args,
        rollout_id=rollout_id,
    )
    reward_domain = RewardDomain(
        args=args,
        group_filter=None,
        use_custom_advantage=False,
    )
    prepare_domain.configure(
        runtime_driver=runtime_domain,
        pool_target_group_count=eval_prepare_pool_target_group_count,
    )

    try:
        await runtime_domain.enter_eval()
        runtime_domain.ensure_session_runner_pool(total_requests=eval_sample_count)
        next_eval_group_idx = 0
        started_group_count = 0
        runtime_materialized_group_count = 0
        runtime_dropped_group_count = 0

        def runtime_resident_group_count() -> int:
            return started_group_count - runtime_materialized_group_count - runtime_dropped_group_count

        async def refresh_eval_ready_groups(*, wait_for_progress: bool) -> int:
            if not prepare_domain.has_warming_groups():
                return 0
            if wait_for_progress:
                await asyncio.sleep(0.05)
            return await prepare_domain.refresh_ready_groups(
                status_fetcher=runtime_domain.prepare_group_status,
                drop_completed_before_ready=True,
            )

        async def fill_prepare_pool() -> int:
            nonlocal next_eval_group_idx
            if next_eval_group_idx >= len(eval_prepare_groups):
                return 0
            snapshot = prepare_domain.accounting_snapshot()
            prepare_capacity = int(snapshot["pool_target_groups"]) - int(snapshot["pool_groups"])
            if prepare_capacity <= 0:
                return 0
            select_count = min(prepare_capacity, len(eval_prepare_groups) - next_eval_group_idx)
            selected_groups = eval_prepare_groups[next_eval_group_idx : next_eval_group_idx + select_count]
            for group in selected_groups:
                for sample_slot_idx, sample in enumerate(group):
                    eval_sample_slot_idx = (
                        sample_slot_idx if args.group_rm else sample.index % dataset_cfg.n_samples_per_eval_prompt
                    )
                    sample.sampling_params = _eval_sampling_params(
                        args,
                        dataset_cfg,
                        sample_slot_idx=eval_sample_slot_idx,
                    )
            next_eval_group_idx += len(selected_groups)
            await prepare_domain.accept_prepare(selected_groups)
            return len(selected_groups)

        await fill_prepare_pool()
        while True:
            launched_count = await prepare_domain.launch_pending()
            runtime_capacity = eval_runtime_admission_group_count - runtime_resident_group_count()
            if runtime_capacity > 0:
                if not prepare_domain.has_ready_groups():
                    if prepare_domain.has_pending_prepare():
                        await prepare_domain.launch_pending()
                    if prepare_domain.has_warming_groups() or prepare_domain.has_inflight_work():
                        await refresh_eval_ready_groups(wait_for_progress=True)
                elif prepare_domain.has_warming_groups():
                    await refresh_eval_ready_groups(wait_for_progress=False)
                batch_input = await prepare_domain.lease_ready_groups(
                    quota_group_count=runtime_capacity,
                    rollout_id=rollout_id,
                )
                if batch_input is not None:
                    started_group_count += await runtime_domain.start_batch(batch_input=batch_input)
                    await fill_prepare_pool()
                    (
                        _progressed_after_start,
                        do_print,
                        materialized_groups,
                        dropped_groups,
                    ) = await _advance_eval_outputs(
                        runtime_domain=runtime_domain,
                        reward_domain=reward_domain,
                        completed_samples=completed_samples,
                        pbar=pbar,
                        do_print=do_print,
                    )
                    runtime_materialized_group_count += materialized_groups
                    runtime_dropped_group_count += dropped_groups
                    if _progressed_after_start:
                        await fill_prepare_pool()
                    continue

            progressed, do_print, materialized_groups, dropped_groups = await _advance_eval_outputs(
                runtime_domain=runtime_domain,
                reward_domain=reward_domain,
                completed_samples=completed_samples,
                pbar=pbar,
                do_print=do_print,
            )
            runtime_materialized_group_count += materialized_groups
            runtime_dropped_group_count += dropped_groups
            if progressed:
                await fill_prepare_pool()
                continue
            if runtime_domain.has_pending_runtime_work():
                runtime_progress = await runtime_domain.wait_for_next_runtime_slot()
                if runtime_progress is not None:
                    (
                        _progressed_after_wait,
                        do_print,
                        materialized_groups,
                        dropped_groups,
                    ) = await _advance_eval_outputs(
                        runtime_domain=runtime_domain,
                        reward_domain=reward_domain,
                        completed_samples=completed_samples,
                        pbar=pbar,
                        do_print=do_print,
                    )
                    runtime_materialized_group_count += materialized_groups
                    runtime_dropped_group_count += dropped_groups
                    await fill_prepare_pool()
                    continue
            elif reward_domain.has_inflight_work():
                reward_completed = await reward_domain.wait_for_next_completion()
                if reward_completed:
                    await fill_prepare_pool()
                    continue

            has_prepare_work = (
                prepare_domain.has_pending_prepare()
                or prepare_domain.has_inflight_work()
                or prepare_domain.has_warming_groups()
                or prepare_domain.has_ready_groups()
            )
            if (
                next_eval_group_idx >= len(eval_prepare_groups)
                and not has_prepare_work
                and not runtime_domain.has_inflight_work()
                and not runtime_domain.has_ready_output()
                and not reward_domain.has_inflight_work()
                and not reward_domain.has_pending_submission_work()
                and not reward_domain.has_ready_output()
            ):
                break
            if (
                prepare_domain.has_pending_prepare()
                and launched_count == 0
                and not prepare_domain.has_warming_groups()
                and not prepare_domain.has_ready_groups()
            ):
                raise RuntimeError(
                    "Agentic eval prepare queue is stuck with pending groups but no runtime driver launch."
                )
            await asyncio.sleep(_BACKGROUND_POLL_INTERVAL_S)
    finally:
        pbar.close()
        await reward_domain.shutdown()
        await prepare_domain.shutdown()
        await runtime_domain.exit_eval()
        await runtime_domain.shutdown()

    completed_samples.sort(key=lambda sample: sample.index)
    return completed_samples


async def eval_rollout(args, rollout_id) -> RolloutFnEvalOutput:
    try:
        results = {}
        reward_key = args.eval_reward_key or args.reward_key
        for dataset_cfg in args.eval_datasets:
            data = await _run_eval_samples(args, rollout_id, dataset_cfg, _build_eval_samples(args, dataset_cfg))
            results[dataset_cfg.name] = {
                "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
                "samples": data,
            }
        return RolloutFnEvalOutput(data=results)
    finally:
        await asyncio.to_thread(_shutdown_resident_pipeline_after_deferred_eval, rollout_id)


def generate_rollout(args, rollout_id, data_source, data_system_client=None, evaluation=False):
    global _RESIDENT_PIPELINE, _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID
    if evaluation:
        return asyncio.run(eval_rollout(args=args, rollout_id=rollout_id))
    resident_pipeline = get_agentic_resident_pipeline()
    num_rollout = args.num_rollout
    terminal_step = num_rollout is not None and rollout_id + 1 >= num_rollout
    final_train_rollout_id = min(rollout_id, num_rollout - 1) if terminal_step else rollout_id
    defer_terminal_shutdown_for_eval = terminal_step and _post_train_eval_expected(
        args, final_train_rollout_id, data_source
    )
    completed = False
    final_backfill_pending = False
    try:
        output = _run_on_resident_async_loop(
            resident_pipeline.run_step(
                args=args,
                rollout_id=rollout_id,
                defer_terminal_shutdown_for_eval=defer_terminal_shutdown_for_eval,
            )
        )
        completed = True
        final_backfill_pending = resident_pipeline._final_backfill_pending
        if defer_terminal_shutdown_for_eval and not final_backfill_pending:
            with _RESIDENT_PIPELINE_LOCK:
                _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID = final_train_rollout_id
        return output
    finally:
        finalize_pipeline = terminal_step and (not completed or not final_backfill_pending)
        if finalize_pipeline and (not defer_terminal_shutdown_for_eval or not completed):
            with _RESIDENT_PIPELINE_LOCK:
                _RESIDENT_PIPELINE_DEFERRED_EVAL_ROLLOUT_ID = None
                if _RESIDENT_PIPELINE is resident_pipeline:
                    _RESIDENT_PIPELINE = None
            try:
                _run_on_resident_async_loop(resident_pipeline.shutdown())
            finally:
                _shutdown_resident_async_loop()
