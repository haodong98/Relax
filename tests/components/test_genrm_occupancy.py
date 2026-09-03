# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for GenRM's event-driven request-occupancy integration and the non-
destructive ``/metrics`` endpoint (see relax.components.genrm.GenRM).

These exercise the real methods on a shell instance (``object.__new__``),
mirroring tests/components/test_rollout_weight_update_handshake.py, so the
integration arithmetic is tested without Ray Serve / a live cluster. Two of
GenRM's imports (``create_genrm_manager``, ``load_processor``/
``load_tokenizer``) pull in torch/transformers only to load real model
checkpoints, which this file never exercises; if those heavy deps are not
installed, they are stubbed so the real GenRM class can still be imported.
In a fully-provisioned environment the plain import path is used and no
stubbing happens at all.
"""

from __future__ import annotations

import asyncio
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest


def _import_genrm_class():
    try:
        from relax.components.genrm import GenRM as GenRMDeployment
    except ImportError:
        placement_group_stub = ModuleType("relax.distributed.ray.placement_group")
        placement_group_stub.create_genrm_manager = lambda *a, **k: None
        sys.modules.setdefault("relax.distributed.ray.placement_group", placement_group_stub)

        processing_utils_stub = ModuleType("relax.utils.data.processing_utils")
        processing_utils_stub.load_processor = lambda *a, **k: None
        processing_utils_stub.load_tokenizer = lambda *a, **k: None
        sys.modules.setdefault("relax.utils.data.processing_utils", processing_utils_stub)

        from relax.components.genrm import GenRM as GenRMDeployment
    # GenRM is wrapped by @serve.deployment / @serve.ingress -- reach the
    # underlying class so we can build a plain shell instance, mirroring
    # tests/components/test_rollout_weight_update_handshake.py.
    return GenRMDeployment.func_or_class


GenRM = _import_genrm_class()


def _make_instance(**overrides) -> "GenRM":
    inst = object.__new__(GenRM)
    inst._inflight_requests = 0
    inst._occupancy_anchor_ns = time.perf_counter_ns()
    inst._occupancy_epoch_wall_s = time.time()
    inst._busy_wall_s = 0.0
    inst._inflight_integral_req_s = 0.0
    inst._served_requests = 0
    inst._queued_requests = 0
    inst._server_max_concurrency = None
    inst._request_semaphore = None
    inst._stopping = False
    inst._drained = asyncio.Event()
    inst._drained.set()
    inst.role = "judge_accuracy"
    inst.config = SimpleNamespace(
        genrm_model_path="/models/fake",
        genrm_num_gpus=1,
        genrm_num_gpus_per_engine=1,
    )
    inst.genrm_manager = None
    inst._logger_instance = None  # _logger is a read-only property; this backs it (see Base._logger)
    for key, value in overrides.items():
        setattr(inst, key, value)
    return inst


def test_advance_occupancy_integrates_busy_time_and_inflight_area():
    inst = _make_instance()
    inst._inflight_requests = 2
    # Backdate the anchor instead of sleeping: deterministic and fast.
    inst._occupancy_anchor_ns -= int(0.1 * 1e9)

    inst._advance_occupancy()

    assert inst._busy_wall_s == pytest.approx(0.1, abs=0.02)
    assert inst._inflight_integral_req_s == pytest.approx(0.2, abs=0.04)


def test_advance_occupancy_does_not_accumulate_busy_time_while_idle():
    inst = _make_instance()
    inst._inflight_requests = 0
    inst._occupancy_anchor_ns -= int(0.1 * 1e9)

    inst._advance_occupancy()

    assert inst._busy_wall_s == 0.0
    assert inst._inflight_integral_req_s == 0.0


def test_advance_occupancy_integrates_correctly_across_a_transition_sequence(monkeypatch):
    """0 -> 1 -> 2 -> 1 -> 0 inflight, each level held for exactly 1s (scripted
    monotonic clock), matching the fold-before-transition call order used in
    generate(): _advance_occupancy() is always called BEFORE _inflight_requests
    changes, so it integrates the level that was active over the interval that
    just elapsed."""
    import relax.components.genrm as genrm_module

    ns_values = iter([0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000])
    monkeypatch.setattr(genrm_module.time, "perf_counter_ns", lambda: next(ns_values))

    inst = _make_instance()  # consumes ns=0 as the initial anchor
    assert inst._occupancy_anchor_ns == 0

    for target_inflight in (1, 2, 1, 0):
        inst._advance_occupancy()
        inst._inflight_requests = target_inflight

    # intervals: [0,1]@0 [1,2]@1 [2,3]@2 [3,4]@1 -> integral = 0+1+2+1 = 4, busy = 3 of 4s
    assert inst._inflight_integral_req_s == pytest.approx(4.0)
    assert inst._busy_wall_s == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_metrics_endpoint_is_non_destructive_across_repeated_reads():
    inst = _make_instance()
    inst._inflight_requests = 1
    inst._occupancy_anchor_ns -= int(0.05 * 1e9)

    first = await inst.metrics()
    second = await inst.metrics()

    assert first["cumulative_busy_wall_s"] > 0
    # A second read must never have reset a counter: only monotonic growth.
    assert second["cumulative_busy_wall_s"] >= first["cumulative_busy_wall_s"]
    assert second["cumulative_inflight_integral_req_s"] >= first["cumulative_inflight_integral_req_s"]
    assert second["cumulative_served_requests"] == first["cumulative_served_requests"] == 0
    assert second["current_inflight"] == first["current_inflight"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_reports_static_and_occupancy_fields():
    inst = _make_instance()

    payload = await inst.metrics()

    assert payload["service"] == "judge_accuracy"
    assert payload["model_path"] == "/models/fake"
    assert payload["current_inflight"] == 0
    assert payload["cumulative_served_requests"] == 0
    assert "occupancy_epoch_s" in payload
    assert "now_s" in payload


@pytest.mark.asyncio
async def test_metrics_endpoint_reports_gpu_occupancy_from_manager_only_when_explicitly_requested(monkeypatch):
    inst = _make_instance()

    class _RemoteHandle:
        def remote(self):
            return "snapshot-ref"

    class _Manager:
        get_gpu_occupancy_snapshot = _RemoteHandle()

    inst.genrm_manager = _Manager()

    import relax.components.genrm as genrm_module

    def fake_ray_get(ref):
        assert ref == "snapshot-ref"
        return [{"role": "judge_accuracy", "tick_count": 3}]

    monkeypatch.setattr(genrm_module.ray, "get", fake_ray_get)

    payload = await inst.metrics(include_gpu_occupancy=True)

    assert payload["gpu_occupancy"] == [{"role": "judge_accuracy", "tick_count": 3}]


@pytest.mark.asyncio
async def test_metrics_endpoint_default_does_not_call_gpu_occupancy_manager(monkeypatch):
    inst = _make_instance()

    class _RemoteHandle:
        def remote(self):
            return "snapshot-ref"

    class _Manager:
        get_gpu_occupancy_snapshot = _RemoteHandle()

    inst.genrm_manager = _Manager()

    import relax.components.genrm as genrm_module

    def unexpected_ray_get(_ref):
        raise AssertionError("default /metrics must not request the GPU occupancy manager snapshot")

    monkeypatch.setattr(genrm_module.ray, "get", unexpected_ray_get)

    payload = await inst.metrics()

    assert payload["gpu_occupancy"] is None


@pytest.mark.asyncio
async def test_metrics_endpoint_gpu_occupancy_is_none_when_manager_lacks_sampler():
    inst = _make_instance()
    inst.genrm_manager = object()  # no get_gpu_occupancy_snapshot attribute at all

    payload = await inst.metrics()

    assert payload["gpu_occupancy"] is None


@pytest.mark.asyncio
async def test_metrics_endpoint_tolerates_gpu_snapshot_failure(monkeypatch):
    """A dead/unreachable GenRMManager must not turn /metrics itself into a
    failure -- occupancy reporting is best-effort observability, not a
    dependency of the judge service's core request path."""
    inst = _make_instance()

    class _RemoteHandle:
        def remote(self):
            return "snapshot-ref"

    class _Manager:
        get_gpu_occupancy_snapshot = _RemoteHandle()

    inst.genrm_manager = _Manager()

    import relax.components.genrm as genrm_module

    def fake_ray_get(_ref):
        raise RuntimeError("actor unavailable")

    monkeypatch.setattr(genrm_module.ray, "get", fake_ray_get)

    payload = await inst.metrics(include_gpu_occupancy=True)

    assert payload["gpu_occupancy"] is None


@pytest.mark.asyncio
async def test_generate_server_admission_serializes_role_requests_and_reports_queue_time():
    inst = _make_instance()
    inst._server_max_concurrency = 1
    inst._request_semaphore = asyncio.Semaphore(1)
    engine_started = asyncio.Event()
    release_engine = asyncio.Event()
    call_count = 0

    inst._restore_media = lambda _request: ([], [])

    async def fake_call_engine(_messages, _sampling_params, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            engine_started.set()
            await release_engine.wait()
        return {"text": "ok", "meta_info": {"completion_tokens": 1}}

    inst._call_engine = fake_call_engine
    request = SimpleNamespace(sampling_params=None, max_input_tokens=None)

    first_task = asyncio.create_task(inst.generate(request))
    await engine_started.wait()
    second_task = asyncio.create_task(inst.generate(request))
    await asyncio.sleep(0.01)
    assert inst._queued_requests == 1

    release_engine.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first.response == second.response == "ok"
    assert first.timings["server_queue_elapsed_s"] >= 0.0
    assert second.timings["server_queue_elapsed_s"] > 0.0
    assert inst._inflight_requests == 0
    assert inst._queued_requests == 0
