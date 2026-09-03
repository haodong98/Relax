# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import math
from argparse import Namespace
from copy import deepcopy
from typing import TYPE_CHECKING

from relax.utils import tracking_utils
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.timer import Timer


if TYPE_CHECKING:
    from relax.utils.training.flops_counter import FlopsCounter


logger = get_logger(__name__)


def _merge_critical_path_rank_max(rank_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = {
        key
        for metrics in rank_metrics
        for key, value in metrics.items()
        if key.startswith("critical_path.") and isinstance(value, (int, float))
    }
    return {key: max(float(metrics.get(key, 0.0)) for metrics in rank_metrics) for key in keys}


def _critical_path_rank_max(log_dict_raw: dict[str, float], world_size: int) -> dict[str, float]:
    local_metrics = {
        key: float(value)
        for key, value in log_dict_raw.items()
        if key.startswith("critical_path.") and isinstance(value, (int, float))
    }
    if world_size <= 1:
        return local_metrics
    import torch.distributed as dist

    if not dist.is_initialized():
        return local_metrics
    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    rank_metrics: list[dict[str, float] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(rank_metrics, local_metrics, group=group)
    return _merge_critical_path_rank_max([metrics or {} for metrics in rank_metrics])


def _gather_critical_path_payload(
    local_metrics: dict[str, float],
    local_events: list[dict],
    world_size: int,
    *,
    pid_base: int = 3000,
    component: str = "trainer",
) -> tuple[dict[str, float], list[dict]]:
    """Gather rank-local phase metrics and trace events in one Gloo
    collective."""
    import torch.distributed as dist

    def decorate_events(events: list[dict], rank: int) -> list[dict]:
        decorated = []
        for event in events:
            event = deepcopy(event)
            event["pid"] = pid_base + rank
            event_args = event.setdefault("args", {})
            if isinstance(event_args, dict):
                event_args["global_rank"] = rank
                if component == "trainer":
                    event_args["trainer_rank"] = rank
                event_args["component"] = component
            decorated.append(event)
        return decorated

    if world_size <= 1 or not dist.is_initialized():
        if dist.is_initialized():
            from relax.utils.distributed_utils import get_gloo_group

            rank = dist.get_rank(group=get_gloo_group())
        else:
            rank = 0
        return local_metrics, decorate_events(local_events, rank)
    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    global_rank = dist.get_rank(group=group)
    payloads: list[dict | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(
        payloads,
        {"global_rank": global_rank, "metrics": local_metrics, "events": local_events},
        group=group,
    )
    rank_metrics = [payload.get("metrics", {}) for payload in payloads if isinstance(payload, dict)]
    events: list[dict] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        rank = int(payload.get("global_rank", 0))
        events.extend(
            decorate_events(
                [event for event in payload.get("events", []) if isinstance(event, dict)],
                rank,
            )
        )
    return _merge_critical_path_rank_max(rank_metrics), events


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    flops_counter: FlopsCounter | None = None,
    world_size: int = 1,
    *,
    pid_base: int = 3000,
    component: str = "trainer",
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    step = compute_rollout_step(args, rollout_id)
    critical_records = timer_instance.log_record_and_clear(step=step, critical_path_only=True)
    timeline_enabled = bool(getattr(args, "use_metrics_service", False) and getattr(args, "timeline_dump_dir", None))
    local_critical_metrics = {
        key: float(value)
        for key, value in log_dict_raw.items()
        if key.startswith("critical_path.") and isinstance(value, (int, float))
    }
    local_critical_events = [record.to_trace_event() for record in critical_records] if timeline_enabled else []
    if timeline_enabled:
        critical_metrics, critical_events = _gather_critical_path_payload(
            local_critical_metrics,
            local_critical_events,
            world_size,
            pid_base=pid_base,
            component=component,
        )
    else:
        critical_metrics, critical_events = local_critical_metrics, []
    log_dict_raw.update(critical_metrics)
    timer_instance.reset()

    if not is_primary_rank:
        timer_instance.log_record_and_clear(step=step, include_critical_path=False)
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if timer_instance.seq_lens:
        log_dict["perf/actor_train_tokens"] = sum(timer_instance.seq_lens)

    per_gpu_tflops: float | None = None
    if ("perf/actor_train_time" in log_dict) and (flops_counter is not None):
        seq_lens = timer_instance.seq_lens
        images_seqlens = getattr(timer_instance, "images_seqlens", None) or None
        audio_seqlens = getattr(timer_instance, "audio_seqlens", None) or None
        estimated_tflops, peak_tflops = flops_counter.estimate(
            batch_seqlens=seq_lens, delta_time=1.0, images_seqlens=images_seqlens, audio_seqlens=audio_seqlens
        )
        # estimated_tflops is total fwd+bwd TFLOPS at delta_time=1 => raw TFLOPS count
        # Normalize to per-GPU
        per_gpu_tflops = estimated_tflops / world_size

        if "perf/log_probs_time" in log_dict:
            # Forward only = fwd+bwd / 3
            log_dict["perf/log_probs_tflops"] = per_gpu_tflops / 3 / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = per_gpu_tflops / 3 / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            # Training includes fwd+bwd, use full 6N flops
            log_dict["perf/actor_train_tflops"] = per_gpu_tflops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(seq_lens) / log_dict["perf/actor_train_time"]

        # MFU = achieved_per_gpu_tflops / device_peak_tflops
        if math.isfinite(peak_tflops) and peak_tflops > 0:
            log_dict["perf/device_peak_tflops"] = peak_tflops
            if "perf/actor_train_tflops" in log_dict:
                log_dict["perf/mfu/actor_train"] = log_dict["perf/actor_train_tflops"] / peak_tflops
            if "perf/log_probs_tflops" in log_dict:
                log_dict["perf/mfu/actor_infer"] = log_dict["perf/log_probs_tflops"] / peak_tflops
            if "perf/ref_log_probs_tflops" in log_dict:
                log_dict["perf/mfu/ref_infer"] = log_dict["perf/ref_log_probs_tflops"] / peak_tflops

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time
            if timer_instance.seq_lens:
                log_dict["perf/step_token_per_s"] = sum(timer_instance.seq_lens) / total_time
            response_lens = getattr(timer_instance, "response_lens", None)
            if response_lens:
                log_dict["perf/step_resp_token_per_s"] = sum(response_lens) / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    if critical_events:
        log_dict["__timeline_events__"] = critical_events
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    # The metrics-service adapter drains records when timeline output is
    # enabled. Always clear leftovers so trace-disabled runs do not retain one
    # TimelineEvent per micro-batch for the lifetime of the actor.
    timer_instance.log_record_and_clear(step=step, include_critical_path=False)
