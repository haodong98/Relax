# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two required local judges for complete terminal agentic trajectories."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections import Counter
from typing import Any

import httpx

from relax.agentic.profile import (
    ensure_reward_judge_trace,
    ensure_reward_trace,
    mark_agentic_event,
    mark_metadata_agentic_event,
)
from relax.agentic.session.reward_context import RewardContextError, RewardContextV1
from relax.engine.rewards.reward_projection import (
    ContextOverflowError,
    InvalidJudgeResponse,
    JudgeProjection,
    ParsedJudgeResponse,
    ProjectionError,
    aggregate_dual_reward,
    build_accuracy_projection,
    build_reasoning_projection,
    build_turn_reasoning_projection,
    parse_judge_response,
)
from relax.utils.genrm_client import get_genrm_client
from relax.utils.judge_config import JudgeServiceSpec
from relax.utils.types import Sample


class JudgeSampleRejected(RuntimeError):
    """A sample-local failure that must atomically replace its GRPO group."""

    def __init__(self, code: str, message: str, *, role: str | None = None) -> None:
        self.code = code
        self.role = role
        super().__init__(message)


class JudgeSystemicFatal(RuntimeError):
    """A configuration/client error for which replacing samples cannot help."""


def recorded_reward_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = copy.deepcopy(value)
        score = payload.get("score")
    else:
        payload = {}
        score = value
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("recorded benchmark reward must be numeric or a dict containing numeric 'score'")
    payload["score"] = float(score)
    payload.setdefault("_schema_version", "relax.benchmark_recorded_reward.v1")
    payload["_benchmark_recorded"] = True
    return payload


def recorded_reward_hash(value: Any) -> str:
    """Hash the numeric reward that the trainer consumes in recorded/shadow
    modes."""
    score = recorded_reward_payload(value)["score"]
    canonical = json.dumps(float(score), allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class DualJudgeExecutor:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.config = args.judge_services
        self._semaphores = {
            spec.role: asyncio.Semaphore(spec.max_concurrency)
            for spec in (self.config.answer_accuracy, self.config.multi_turn_reasoning)
        }
        self._metrics: Counter[str] = Counter()

    def _note_metric(self, kind: str, role: str, code: str) -> None:
        self._metrics[f"judge_{kind}/{role}/{code}"] += 1

    def drain_metrics(self) -> dict[str, int]:
        metrics = dict(self._metrics)
        self._metrics.clear()
        return metrics

    def _client(self, spec: JudgeServiceSpec):
        injected = getattr(self.args, "_judge_clients", None)
        if isinstance(injected, dict) and spec.role in injected:
            return injected[spec.role]
        return get_genrm_client(timeout=spec.timeout_s, role=spec.role)

    @staticmethod
    def _media_payload(context: RewardContextV1, projection: JudgeProjection) -> tuple[list[dict], dict[str, str]]:
        media_ids = set(projection.media_ids)
        manifest = [item for item in context.media_manifest if item["media_id"] in media_ids]
        blobs = {media_id: context.media_blobs[media_id].data_uri() for media_id in projection.media_ids}
        return manifest, blobs

    async def _call_required_judge(
        self,
        *,
        component: str,
        spec: JudgeServiceSpec,
        projection: JudgeProjection,
        context: RewardContextV1,
        trace_metadata: dict[str, Any],
    ) -> ParsedJudgeResponse:
        judge_trace = ensure_reward_judge_trace(
            trace_metadata,
            component=component,
            role=spec.role,
        )
        branch_started_ns = time.perf_counter_ns()
        queue_entered_ns = branch_started_ns
        queue_acquired_ns: int | None = None
        judge_trace.update(
            {
                "status": "queued",
                "attempt_count": 0,
                "invalid_response_count": 0,
                "payload_prep_elapsed_s": 0.0,
                "http_elapsed_s": 0.0,
                "parse_elapsed_s": 0.0,
                "backoff_elapsed_s": 0.0,
            }
        )
        mark_metadata_agentic_event(trace_metadata, f"judge_{component}_queue_enter_at")
        client = self._client(spec)
        transient_attempt = 0
        invalid_attempt = 0
        backoff = 0.5
        try:
            async with self._semaphores[spec.role]:
                queue_acquired_ns = time.perf_counter_ns()
                judge_trace["queue_elapsed_s"] = (queue_acquired_ns - queue_entered_ns) / 1e9
                judge_trace["status"] = "running"
                mark_metadata_agentic_event(trace_metadata, f"judge_{component}_queue_acquired_at")
                payload_prep_started_ns = time.perf_counter_ns()
                manifest, blobs = self._media_payload(context, projection)
                payload = {
                    "context_hash": projection.context_hash,
                    "projection_hash": projection.projection_hash,
                    "judge_role": spec.role,
                    "prompt_version": projection.prompt_version,
                    "messages": projection.messages,
                    "sampling_params": copy.deepcopy(spec.sampling_config),
                    "media_manifest": manifest,
                    "media_blobs": blobs,
                    "max_input_tokens": spec.max_input_tokens,
                }
                judge_trace["payload_prep_elapsed_s"] = (time.perf_counter_ns() - payload_prep_started_ns) / 1e9
                while True:
                    judge_trace["attempt_count"] += 1
                    try:
                        request_started_ns = time.perf_counter_ns()
                        mark_metadata_agentic_event(trace_metadata, f"judge_{component}_http_start_at")
                        try:
                            generate_once_profiled = getattr(client, "generate_once_profiled", None)
                            if callable(generate_once_profiled):
                                raw_output, server_timings = await generate_once_profiled(payload)
                                server_attempt = copy.deepcopy(server_timings)
                                judge_trace.setdefault("server_attempts", []).append(server_attempt)
                                server_cumulative = judge_trace.setdefault("server", {})
                                for name, value in server_attempt.items():
                                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                                        continue
                                    server_cumulative[name] = float(server_cumulative.get(name, 0.0)) + float(value)
                            else:
                                raw_output = await client.generate_once(payload)
                        finally:
                            request_ended_ns = time.perf_counter_ns()
                            judge_trace["http_elapsed_s"] += (request_ended_ns - request_started_ns) / 1e9
                            mark_metadata_agentic_event(trace_metadata, f"judge_{component}_http_end_at")
                        parse_started_ns = time.perf_counter_ns()
                        try:
                            result = parse_judge_response(raw_output, component=component)
                        except InvalidJudgeResponse as exc:
                            invalid_attempt += 1
                            judge_trace["invalid_response_count"] = invalid_attempt
                            if invalid_attempt <= 1:
                                self._note_metric("retry", spec.role, "invalid_response")
                                continue
                            self._note_metric("error", spec.role, "invalid_response")
                            raise JudgeSampleRejected(
                                "invalid_response",
                                f"{spec.role} returned invalid JSON twice: {exc}",
                                role=spec.role,
                            ) from exc
                        finally:
                            judge_trace["parse_elapsed_s"] += (time.perf_counter_ns() - parse_started_ns) / 1e9
                        judge_trace["status"] = "success"
                        return result
                    except asyncio.CancelledError:
                        raise
                    except JudgeSampleRejected:
                        raise
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        if status in {400, 413}:
                            code = "invalid_media" if status == 400 else "context_overflow"
                            self._note_metric("error", spec.role, code)
                            raise JudgeSampleRejected(
                                code,
                                f"{spec.role} returned HTTP {status}",
                                role=spec.role,
                            ) from exc
                        if status != 429 and status < 500:
                            self._note_metric("error", spec.role, f"http_{status}")
                            raise JudgeSystemicFatal(
                                f"{spec.role} returned non-retryable HTTP {status}; check model/template/schema config"
                            ) from exc
                        transient_attempt += 1
                        error_code = "rate_limited" if status == 429 else "server_error"
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        transient_attempt += 1
                        error_code = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection_error"
                    except (KeyError, TypeError, ValueError) as exc:
                        self._note_metric("error", spec.role, "service_contract")
                        raise JudgeSystemicFatal(f"{spec.role} service contract error: {exc}") from exc
                    if transient_attempt >= spec.max_attempts:
                        self._note_metric("error", spec.role, error_code)
                        raise JudgeSampleRejected(
                            error_code,
                            f"{spec.role} exhausted {transient_attempt} attempts",
                            role=spec.role,
                        )
                    self._note_metric("retry", spec.role, error_code)
                    backoff_started_ns = time.perf_counter_ns()
                    await asyncio.sleep(backoff)
                    judge_trace["backoff_elapsed_s"] += (time.perf_counter_ns() - backoff_started_ns) / 1e9
                    backoff *= 2
        except asyncio.CancelledError:
            judge_trace["status"] = "cancelled"
            judge_trace["error_code"] = "cancelled"
            raise
        except JudgeSampleRejected as exc:
            judge_trace["status"] = "rejected"
            judge_trace["error_code"] = exc.code
            raise
        except Exception as exc:
            judge_trace["status"] = "error"
            judge_trace["error_code"] = type(exc).__name__
            raise
        finally:
            ended_ns = time.perf_counter_ns()
            if queue_acquired_ns is None:
                judge_trace["queue_elapsed_s"] = (ended_ns - queue_entered_ns) / 1e9
            judge_trace["elapsed_s"] = (ended_ns - branch_started_ns) / 1e9
            mark_metadata_agentic_event(trace_metadata, f"judge_{component}_end_at")

    @staticmethod
    def _copy_turn_judge_events(events: dict[str, Any]) -> None:
        """Normalize the reusable request trace to the per-turn event
        schema."""
        event_pairs = {
            "judge_multi_turn_reasoning_queue_enter_at": "turn_judge_queue_enter_at",
            "judge_multi_turn_reasoning_queue_acquired_at": "turn_judge_queue_acquired_at",
            "judge_multi_turn_reasoning_http_start_at": "turn_judge_http_start_at",
            "judge_multi_turn_reasoning_http_end_at": "turn_judge_http_end_at",
            "judge_multi_turn_reasoning_end_at": "turn_judge_request_end_at",
        }
        for source, target in event_pairs.items():
            if source in events:
                events[target] = events[source]
            source_host = f"{source}__clock_host"
            if source_host in events:
                events[f"{target}__clock_host"] = events[source_host]

    async def score_turn_reasoning(
        self,
        *,
        context: RewardContextV1,
        outcome_base: dict[str, Any],
    ) -> dict[str, Any]:
        """Score one already-completed interaction and release its context.

        This is intentionally separate from :meth:`score`: session shards own
        one short-lived turn context and may run this work while the agent
        continues to the next environment interaction.  The terminal executor
        later validates and aggregates the immutable outcome records.
        """
        events = outcome_base.get("events")
        if not isinstance(events, dict):
            events = {}
            outcome_base = {**outcome_base, "events": events}
        trace_metadata: dict[str, Any] = {"agentic_trace": {"events": events}}
        projection: JudgeProjection | None = None
        try:
            projection_started_ns = time.perf_counter_ns()
            mark_agentic_event(events, "turn_judge_projection_start_at")
            loop = asyncio.get_running_loop()
            projection_future = loop.run_in_executor(
                None,
                build_turn_reasoning_projection,
                context,
                self.config.multi_turn_reasoning,
            )
            try:
                projection = await asyncio.shield(projection_future)
            except asyncio.CancelledError:
                # Projection owns a worker thread and can still be reading
                # media. Never release the context until the thread exits.
                await projection_future
                raise
            finally:
                events["turn_judge_projection_elapsed_s"] = (time.perf_counter_ns() - projection_started_ns) / 1e9
                mark_agentic_event(events, "turn_judge_projection_end_at")
            result = await self._call_required_judge(
                component="multi_turn_reasoning",
                spec=self.config.multi_turn_reasoning,
                projection=projection,
                context=context,
                trace_metadata=trace_metadata,
            )
            self._copy_turn_judge_events(events)
            reward_trace = ensure_reward_trace(trace_metadata)
            judge_trace = copy.deepcopy(reward_trace["judges"]["multi_turn_reasoning"])
            if projection.truncation:
                self._metrics["judge_projection_truncation"] += 1
            return {
                **outcome_base,
                "status": "success",
                "score": float(result.score),
                "context_hash": context.content_hash,
                "projection_hash": projection.projection_hash,
                "prompt_version": projection.prompt_version,
                "raw_output_sha256": result.raw_output_sha256,
                "truncation": copy.deepcopy(projection.truncation),
                "projection_elapsed_s": events["turn_judge_projection_elapsed_s"],
                "judge": judge_trace,
            }
        except asyncio.CancelledError:
            self._copy_turn_judge_events(events)
            raise
        except (ContextOverflowError, ProjectionError) as exc:
            self._note_metric("error", "projection", exc.code)
            return {
                **outcome_base,
                "status": "rejected",
                "error_code": exc.code,
                "error_message": str(exc),
                "role": self.config.multi_turn_reasoning.role,
            }
        except JudgeSampleRejected as exc:
            self._copy_turn_judge_events(events)
            return {
                **outcome_base,
                "status": "rejected",
                "error_code": exc.code,
                "error_message": str(exc),
                "role": exc.role or self.config.multi_turn_reasoning.role,
            }
        except Exception as exc:
            self._copy_turn_judge_events(events)
            return {
                **outcome_base,
                "status": "error",
                "error_code": type(exc).__name__,
                "error_message": str(exc)[:512],
                "role": self.config.multi_turn_reasoning.role,
                "systemic": True,
            }
        finally:
            mark_agentic_event(events, "turn_judge_end_at")
            context.release()

    def _aggregate_per_turn_reasoning(self, context: RewardContextV1) -> tuple[float, list[dict[str, Any]]]:
        outcomes = copy.deepcopy(context.per_turn_judgements)
        if not outcomes:
            raise JudgeSystemicFatal("per_turn reasoning has no completed interaction outcomes")
        scores: list[float] = []
        for outcome in outcomes:
            status = outcome.get("status")
            role = outcome.get("role") or self.config.multi_turn_reasoning.role
            if status != "success":
                code = str(outcome.get("error_code") or "turn_judge_failed")
                if outcome.get("systemic"):
                    raise JudgeSystemicFatal(f"per-turn {role} failed systemically: {code}")
                raise JudgeSampleRejected(
                    code,
                    f"per-turn {role} rejected a trajectory interaction: {code}",
                    role=str(role),
                )
            score = outcome.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                raise JudgeSystemicFatal("per-turn reasoning result has an invalid score")
            scores.append(float(score))
        return sum(scores) / len(scores), outcomes

    async def score(self, sample: Sample) -> dict[str, Any]:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        audit = sample.metadata.setdefault("reward_audit", {})
        if sample.reward is not None:
            audit.setdefault("environment_reward", copy.deepcopy(sample.reward))
        reward_trace = ensure_reward_trace(sample.metadata)
        executor_started_ns = time.perf_counter_ns()
        judge_tasks: set[asyncio.Task] = set()
        reward_trace["executor_status"] = "running"
        mark_metadata_agentic_event(sample.metadata, "reward_executor_start_at")
        context = sample.reward_context
        benchmark_mode = getattr(self.config, "benchmark_mode", "dual")
        try:
            if isinstance(context, RewardContextError):
                self._note_metric("error", "reward_context", context.code)
                raise JudgeSampleRejected(context.code, str(context)) from context
            if not isinstance(context, RewardContextV1):
                self._note_metric("error", "reward_context", "missing_context")
                raise JudgeSampleRejected("missing_context", "sample has no RewardContextV1")
            if benchmark_mode == "recorded":
                raise JudgeSystemicFatal("recorded benchmark mode must bypass DualJudgeExecutor.score")
            use_reasoning = benchmark_mode in {"dual", "dual_shadow"}
            reasoning_trigger = getattr(self.config, "reasoning_trigger", "terminal_once")
            # Keep the configured A/B dimension stable even when a K=0
            # per-turn trajectory uses the documented terminal fallback.
            reward_trace["reasoning_trigger"] = reasoning_trigger
            use_per_turn_reasoning = (
                use_reasoning and reasoning_trigger == "per_turn" and not context.per_turn_fallback_terminal_once
            )
            use_terminal_reasoning = use_reasoning and not use_per_turn_reasoning
            if use_per_turn_reasoning:
                reward_trace["reasoning_execution_trigger"] = "per_turn"
                reasoning_score, per_turn_outcomes = self._aggregate_per_turn_reasoning(context)
                reward_trace["per_turn_judges"] = copy.deepcopy(per_turn_outcomes)
                reward_trace["per_turn_judge_count"] = len(per_turn_outcomes)
            else:
                reasoning_score = None
                per_turn_outcomes: list[dict[str, Any]] = []
                if use_terminal_reasoning:
                    reward_trace["reasoning_execution_trigger"] = (
                        "terminal_once_fallback" if context.per_turn_fallback_terminal_once else "terminal_once"
                    )
                    if context.per_turn_fallback_terminal_once:
                        reward_trace["per_turn_fallback_terminal_once"] = True
                        reward_trace["per_turn_judge_count"] = 0

            async def build_projection(component: str, builder: Any, spec: JudgeServiceSpec) -> JudgeProjection:
                projection_started_ns = time.perf_counter_ns()
                try:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, builder, context, spec)
                finally:
                    reward_trace[f"{component}_projection_elapsed_s"] = (
                        time.perf_counter_ns() - projection_started_ns
                    ) / 1e9

            try:
                projection_tasks = [
                    asyncio.create_task(
                        build_projection(
                            "answer_accuracy",
                            build_accuracy_projection,
                            self.config.answer_accuracy,
                        )
                    )
                ]
                if use_terminal_reasoning:
                    projection_tasks.append(
                        asyncio.create_task(
                            build_projection(
                                "multi_turn_reasoning",
                                build_reasoning_projection,
                                self.config.multi_turn_reasoning,
                            )
                        )
                    )
                projection_future = asyncio.gather(*projection_tasks, return_exceptions=True)
                try:
                    projection_results = await asyncio.shield(projection_future)
                except asyncio.CancelledError:
                    # ``run_in_executor`` cannot stop a projection already
                    # running in a worker thread. Wait before RewardContextV1
                    # is released so the worker never observes cleared media.
                    await projection_future
                    raise
                projection_failure = next(
                    (result for result in projection_results if isinstance(result, BaseException)),
                    None,
                )
                if projection_failure is not None:
                    raise projection_failure
                accuracy_projection = projection_results[0]
                reasoning_projection = projection_results[1] if use_terminal_reasoning else None
            except ContextOverflowError as exc:
                self._note_metric("error", "projection", exc.code)
                raise JudgeSampleRejected(exc.code, str(exc)) from exc
            except ProjectionError as exc:
                self._note_metric("error", "projection", exc.code)
                raise JudgeSampleRejected(exc.code, str(exc)) from exc

            accuracy_task = asyncio.create_task(
                self._call_required_judge(
                    component="answer_accuracy",
                    spec=self.config.answer_accuracy,
                    projection=accuracy_projection,
                    context=context,
                    trace_metadata=sample.metadata,
                )
            )
            reasoning_task = None
            if reasoning_projection is not None:
                reasoning_task = asyncio.create_task(
                    self._call_required_judge(
                        component="multi_turn_reasoning",
                        spec=self.config.multi_turn_reasoning,
                        projection=reasoning_projection,
                        context=context,
                        trace_metadata=sample.metadata,
                    )
                )
            judge_tasks = {accuracy_task} if reasoning_task is None else {accuracy_task, reasoning_task}
            done, pending = await asyncio.wait(judge_tasks, return_when=asyncio.FIRST_EXCEPTION)
            failure = None
            for task in done:
                if task.cancelled():
                    failure = asyncio.CancelledError()
                    break
                task_exception = task.exception()
                if task_exception is not None:
                    failure = task_exception
                    break
            if failure is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure
            if pending:
                await asyncio.gather(*pending)
            accuracy_result = accuracy_task.result()
            reasoning_result = reasoning_task.result() if reasoning_task is not None else None
            if reasoning_result is not None:
                reasoning_score = float(reasoning_result.score)
            if benchmark_mode in {"accuracy_shadow", "dual_shadow"}:
                try:
                    reward = recorded_reward_payload(audit.get("environment_reward"))
                except ValueError as exc:
                    raise JudgeSystemicFatal(str(exc)) from exc
            elif reasoning_score is None:
                reward = {
                    "score": float(accuracy_result.score),
                    "answer_accuracy": int(accuracy_result.score),
                    "_schema_version": "relax.composite_reward.v1",
                }
            else:
                reward = aggregate_dual_reward(int(accuracy_result.score), reasoning_score)
            judges_audit = {
                "answer_accuracy": {
                    "model_path": self.config.answer_accuracy.model_path,
                    "projection_hash": accuracy_projection.projection_hash,
                    "prompt_version": accuracy_projection.prompt_version,
                    "raw_output_sha256": accuracy_result.raw_output_sha256,
                    "diagnostic_snippet": accuracy_result.diagnostic_snippet,
                }
            }
            if reasoning_result is not None and reasoning_projection is not None:
                if reasoning_projection.truncation:
                    self._metrics["judge_projection_truncation"] += 1
                judges_audit["multi_turn_reasoning"] = {
                    "model_path": self.config.multi_turn_reasoning.model_path,
                    "projection_hash": reasoning_projection.projection_hash,
                    "prompt_version": reasoning_projection.prompt_version,
                    "raw_output_sha256": reasoning_result.raw_output_sha256,
                    "diagnostic_snippet": reasoning_result.diagnostic_snippet,
                    "truncation": copy.deepcopy(reasoning_projection.truncation),
                }
            elif use_per_turn_reasoning:
                judges_audit["multi_turn_reasoning"] = {
                    "model_path": self.config.multi_turn_reasoning.model_path,
                    "trigger": "per_turn",
                    "aggregation": "mean",
                    "turn_count": len(per_turn_outcomes),
                    "score": reasoning_score,
                    "turns": [
                        {
                            "turn_index": outcome.get("turn_index"),
                            "status": outcome.get("status"),
                            "context_hash": outcome.get("context_hash"),
                            "projection_hash": outcome.get("projection_hash"),
                            "raw_output_sha256": outcome.get("raw_output_sha256"),
                            "truncation": copy.deepcopy(outcome.get("truncation", {})),
                        }
                        for outcome in per_turn_outcomes
                    ],
                }
            audit.update(
                {
                    "schema_version": "relax.reward_audit.v1",
                    "context_hash": context.content_hash,
                    "benchmark_mode": benchmark_mode,
                    "judges": judges_audit,
                }
            )
            reward_trace["benchmark_mode"] = benchmark_mode
            reward_trace["executor_status"] = "success"
            return reward
        except asyncio.CancelledError:
            for task in judge_tasks:
                if not task.done():
                    task.cancel()
            if judge_tasks:
                await asyncio.gather(*judge_tasks, return_exceptions=True)
            reward_trace["executor_status"] = "cancelled"
            reward_trace["executor_error_code"] = "cancelled"
            raise
        except JudgeSampleRejected as exc:
            if benchmark_mode in {"accuracy_shadow", "dual_shadow"}:
                try:
                    reward = recorded_reward_payload(audit.get("environment_reward"))
                except ValueError as value_error:
                    raise JudgeSystemicFatal(str(value_error)) from value_error
                reward_trace["executor_status"] = "shadow_judge_error"
                reward_trace["executor_error_code"] = exc.code
                audit.update(
                    {
                        "schema_version": "relax.reward_audit.v1",
                        "benchmark_mode": benchmark_mode,
                        "shadow_judge_error": {"code": exc.code, "role": exc.role},
                    }
                )
                self._note_metric("shadow_error", exc.role or "unknown", exc.code)
                return reward
            reward_trace["executor_status"] = "rejected"
            reward_trace["executor_error_code"] = exc.code
            raise
        except Exception as exc:
            reward_trace["executor_status"] = "error"
            reward_trace["executor_error_code"] = type(exc).__name__
            raise
        finally:
            reward_trace["executor_elapsed_s"] = (time.perf_counter_ns() - executor_started_ns) / 1e9
            mark_metadata_agentic_event(sample.metadata, "reward_executor_end_at")
            if isinstance(context, RewardContextV1):
                context.release()
            sample.reward_context = None


_EXECUTORS: dict[tuple[int, asyncio.AbstractEventLoop], DualJudgeExecutor] = {}


def _executor_key(args: Any) -> tuple[int, asyncio.AbstractEventLoop]:
    return id(args.judge_services), asyncio.get_running_loop()


def get_dual_judge_executor(args: Any) -> DualJudgeExecutor:
    key = _executor_key(args)
    executor = _EXECUTORS.get(key)
    if executor is None:
        executor = DualJudgeExecutor(args)
        _EXECUTORS[key] = executor
    return executor


def drain_dual_judge_metrics(args: Any) -> dict[str, int]:
    executor = _EXECUTORS.get(_executor_key(args))
    return executor.drain_metrics() if executor is not None else {}


async def close_dual_judge_executor_for_current_loop(args: Any) -> None:
    """Evict loop-bound semaphores and HTTP pools after a reward domain
    stops."""
    _EXECUTORS.pop(_executor_key(args), None)
    from relax.utils.genrm_client import close_genrm_clients_for_current_loop

    await close_genrm_clients_for_current_loop()


async def async_compute_dual_agentic_judge(args: Any, sample: Sample) -> dict[str, Any]:
    return await get_dual_judge_executor(args).score(sample)


async def async_compute_turn_reasoning(
    args: Any,
    *,
    context: RewardContextV1,
    outcome_base: dict[str, Any],
) -> dict[str, Any]:
    """Run the VLM branch for one completed agent-environment interaction."""
    return await get_dual_judge_executor(args).score_turn_reasoning(
        context=context,
        outcome_base=outcome_base,
    )
