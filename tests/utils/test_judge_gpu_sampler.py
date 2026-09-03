# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for JudgeGpuSampler (relax.utils.metrics.judge_gpu_sampler).

Exercises the NVML + SGLang-Prometheus tick logic with a fake pynvml module and
a fake ``requests.get`` so these run without a real GPU or a live judge engine.
``_tick``/``snapshot``/``_write_manifest`` are called directly rather than
through the background thread for the aggregation-math tests, so they are
deterministic; a separate smoke test exercises the real thread with a short
interval to confirm ``start``/``stop`` lifecycle wiring.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_judge_gpu_sampler_module() -> ModuleType:
    """Import judge_gpu_sampler.py directly by file path, bypassing
    ``relax.utils.metrics.__init__`` (which eagerly imports
    ``TimelineTraceAdapter`` -> ``relax.utils.timer`` -> ``relax.utils.misc``

    -> torch, a dependency this leaf module itself does not need and that
    may be unavailable in a lightweight test environment). The module's own
    absolute imports (``relax.utils.autoscaler.metrics_collector``,
    ``relax.utils.logging_utils``) are untouched and resolve normally; in a
    fully-provisioned environment this is equivalent to a plain import.
    """
    cached = sys.modules.get("relax.utils.metrics.judge_gpu_sampler")
    if cached is not None:
        return cached
    module_path = Path(__file__).resolve().parents[2] / "relax" / "utils" / "metrics" / "judge_gpu_sampler.py"
    if "relax.utils.metrics" not in sys.modules:
        package_stub = ModuleType("relax.utils.metrics")
        package_stub.__path__ = [str(module_path.parent)]
        sys.modules["relax.utils.metrics"] = package_stub
    spec = importlib.util.spec_from_file_location("relax.utils.metrics.judge_gpu_sampler", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = _load_judge_gpu_sampler_module()


class _FakeNvml:
    """Stand-in for the pynvml module: handles are the GPU index itself."""

    def __init__(self, *, util_by_index, mem_by_index, uuid_by_index, fail_handle_indices=()):
        self.util_by_index = util_by_index
        self.mem_by_index = mem_by_index
        self.uuid_by_index = uuid_by_index
        self.fail_handle_indices = set(fail_handle_indices)
        self.init_called = False
        self.shutdown_called = False

    def nvmlInit(self):
        self.init_called = True

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetHandleByIndex(self, index):
        if index in self.fail_handle_indices:
            raise RuntimeError(f"no such device {index}")
        return index

    def nvmlDeviceGetUUID(self, handle):
        return self.uuid_by_index[handle]

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=self.util_by_index[handle], memory=0)

    def nvmlDeviceGetMemoryInfo(self, handle):
        used, total = self.mem_by_index[handle]
        return SimpleNamespace(used=used, total=total)


class _FailingInitNvml(_FakeNvml):
    def nvmlInit(self):
        raise RuntimeError("driver mismatch")


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def _make_sampler(tmp_path, *, scrape_url=None, base_gpu_id=4, num_gpus_per_engine=2, interval_s=0.2):
    return m.JudgeGpuSampler(
        role="judge_multiturn_vlm",
        engine_rank=0,
        base_gpu_id=base_gpu_id,
        num_gpus_per_engine=num_gpus_per_engine,
        model_path="/models/fake",
        scrape_url=scrape_url,
        sample_dir=str(tmp_path),
        server_host="10.0.0.1",
        dist_init_addr="10.0.0.1:15002",
        interval_s=interval_s,
    )


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_try_init_nvml_returns_none_when_module_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert m._try_init_nvml() is None


def test_try_init_nvml_returns_none_when_nvml_init_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _FailingInitNvml(util_by_index={}, mem_by_index={}, uuid_by_index={}))
    assert m._try_init_nvml() is None


def test_try_init_nvml_returns_module_on_success(monkeypatch):
    fake = _FakeNvml(util_by_index={}, mem_by_index={}, uuid_by_index={})
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert m._try_init_nvml() is fake
    assert fake.init_called


def test_resolve_gpu_handles_populates_indices_and_uuids(tmp_path):
    sampler = _make_sampler(tmp_path, base_gpu_id=4, num_gpus_per_engine=2)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 10, 5: 20},
        mem_by_index={4: (1, 2), 5: (3, 4)},
        uuid_by_index={4: "GPU-a", 5: "GPU-b"},
    )
    sampler._resolve_gpu_handles()
    assert [index for index, _handle in sampler._gpu_handles] == [4, 5]
    assert sampler._uuid_by_index == {4: "GPU-a", 5: "GPU-b"}


def test_resolve_gpu_handles_disables_nvml_on_handle_failure(tmp_path):
    sampler = _make_sampler(tmp_path, base_gpu_id=4, num_gpus_per_engine=2)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 10},
        mem_by_index={4: (1, 2)},
        uuid_by_index={4: "GPU-a"},
        fail_handle_indices={5},
    )
    sampler._resolve_gpu_handles()
    assert sampler._nvml is None
    assert sampler._gpu_handles == []
    assert sampler._uuid_by_index == {}


def test_tick_aggregates_nvml_and_sglang_readings_across_multiple_calls(tmp_path, monkeypatch):
    sampler = _make_sampler(tmp_path, scrape_url="http://127.0.0.1:9/metrics", base_gpu_id=4, num_gpus_per_engine=1)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 50},
        mem_by_index={4: (1000, 2000)},
        uuid_by_index={4: "GPU-a"},
    )
    sampler._resolve_gpu_handles()

    prom_text = "sglang:num_running_reqs{} 1.0\nsglang:num_queue_reqs{} 0.0\nsglang:token_usage{} 0.25\n"
    monkeypatch.setattr(m.requests, "get", lambda *a, **k: _FakeResponse(prom_text))

    sampler._tick()
    sampler._tick()

    snapshot = sampler.snapshot()
    assert snapshot["tick_count"] == 2
    assert snapshot["nvml_enabled"] is True
    [gpu_entry] = snapshot["gpu"]
    assert gpu_entry["index"] == 4
    assert gpu_entry["uuid"] == "GPU-a"
    assert gpu_entry["sample_count"] == 2
    assert gpu_entry["util_percent_sum"] == pytest.approx(100.0)
    assert gpu_entry["util_nonzero_count"] == 2
    assert gpu_entry["mem_used_bytes_sum"] == pytest.approx(2000.0)
    assert gpu_entry["mem_total_bytes"] == 2000
    assert snapshot["sglang_tick_count"] == 2
    assert snapshot["sglang_metrics_enabled"] is True
    assert snapshot["sglang_sum"]["sglang:num_running_reqs"] == pytest.approx(2.0)
    assert snapshot["sglang_sum"]["sglang:token_usage"] == pytest.approx(0.5)

    records = _read_jsonl(sampler._file_path())
    assert len(records) == 2
    assert records[0]["record_type"] == "sample"
    assert records[0]["role"] == "judge_multiturn_vlm"
    assert records[0]["gpu"][0]["uuid"] == "GPU-a"
    assert records[0]["sglang"]["sglang:num_running_reqs"] == pytest.approx(1.0)


def test_tick_zero_utilization_does_not_count_as_nonidle(tmp_path):
    sampler = _make_sampler(tmp_path, base_gpu_id=4, num_gpus_per_engine=1)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 0},
        mem_by_index={4: (0, 2000)},
        uuid_by_index={4: "GPU-a"},
    )
    sampler._resolve_gpu_handles()

    sampler._tick()

    [gpu_entry] = sampler.snapshot()["gpu"]
    assert gpu_entry["sample_count"] == 1
    assert gpu_entry["util_percent_sum"] == pytest.approx(0.0)
    assert gpu_entry["util_nonzero_count"] == 0


def test_sglang_scrape_failure_does_not_block_nvml_collection(tmp_path, monkeypatch):
    sampler = _make_sampler(tmp_path, scrape_url="http://127.0.0.1:9/metrics", base_gpu_id=4, num_gpus_per_engine=1)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 77},
        mem_by_index={4: (1, 2)},
        uuid_by_index={4: "GPU-a"},
    )
    sampler._resolve_gpu_handles()

    def _raise(*_args, **_kwargs):
        raise m.requests.exceptions.ConnectionError("engine not enabled for metrics")

    monkeypatch.setattr(m.requests, "get", _raise)

    sampler._tick()

    snapshot = sampler.snapshot()
    [gpu_entry] = snapshot["gpu"]
    assert gpu_entry["util_percent_sum"] == pytest.approx(77.0)
    assert snapshot["sglang_tick_count"] == 0
    assert snapshot["sglang_metrics_enabled"] is False
    assert sampler._sglang_scrape_warned is True

    records = _read_jsonl(sampler._file_path())
    assert records[0]["sglang"] is None
    assert len(records[0]["gpu"]) == 1


def test_tick_without_nvml_still_writes_sample_with_empty_gpu_list(tmp_path):
    sampler = _make_sampler(tmp_path, base_gpu_id=4, num_gpus_per_engine=1)
    assert sampler._nvml is None  # never resolved

    sampler._tick()

    snapshot = sampler.snapshot()
    assert snapshot["nvml_enabled"] is False
    [gpu_entry] = snapshot["gpu"]
    # No NVML query completed, so this must be an unavailable denominator,
    # not a zero-utilization observation.
    assert gpu_entry["sample_count"] == 0
    assert gpu_entry["util_percent_sum"] == 0.0
    assert gpu_entry["util_nonzero_count"] == 0
    records = _read_jsonl(sampler._file_path())
    assert records[0]["gpu"] == []


def test_write_manifest_records_static_engine_info(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/tmp/relax-flashinfer/job-42")
    sampler = _make_sampler(tmp_path, scrape_url="http://x/metrics", base_gpu_id=4, num_gpus_per_engine=2)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 1, 5: 2},
        mem_by_index={4: (1, 2), 5: (1, 2)},
        uuid_by_index={4: "GPU-a", 5: "GPU-b"},
    )
    sampler._resolve_gpu_handles()

    sampler._write_manifest()

    [manifest] = _read_jsonl(sampler._file_path())
    assert manifest["record_type"] == "manifest"
    assert manifest["role"] == "judge_multiturn_vlm"
    assert manifest["base_gpu_id"] == 4
    assert manifest["num_gpus_per_engine"] == 2
    assert manifest["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert manifest["model_path"] == "/models/fake"
    assert manifest["flashinfer_workspace_base"] == "/tmp/relax-flashinfer/job-42"
    assert manifest["nvml_enabled"] is True
    assert manifest["scrape_configured"] is True


def test_snapshot_is_non_destructive_across_repeated_reads(tmp_path):
    sampler = _make_sampler(tmp_path, base_gpu_id=4, num_gpus_per_engine=1)
    sampler._nvml = _FakeNvml(
        util_by_index={4: 10},
        mem_by_index={4: (1, 2)},
        uuid_by_index={4: "GPU-a"},
    )
    sampler._resolve_gpu_handles()

    sampler._tick()
    first = sampler.snapshot()
    second = sampler.snapshot()

    assert first["tick_count"] == second["tick_count"] == 1
    assert first["gpu"][0]["sample_count"] == second["gpu"][0]["sample_count"] == 1
    assert first["gpu"][0]["util_percent_sum"] == second["gpu"][0]["util_percent_sum"]


def test_start_stop_lifecycle_writes_manifest_and_at_least_one_sample(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # keep it fast/deterministic: NVML disabled
    sampler = _make_sampler(tmp_path, interval_s=0.01)

    sampler.start()
    try:
        sampler._stop_event.wait(0.2)
    finally:
        sampler.stop(timeout_s=2.0)

    records = _read_jsonl(sampler._file_path())
    assert records[0]["record_type"] == "manifest"
    assert records[0]["server_host"] == "10.0.0.1"
    assert records[0]["dist_init_addr"] == "10.0.0.1:15002"
    assert any(record["record_type"] == "sample" for record in records)
    assert records[-1]["record_type"] == "sample"
    assert records[-1]["final_sample"] is True


def test_disabled_sampler_without_sample_dir_is_a_noop():
    sampler = m.JudgeGpuSampler(
        role="judge_accuracy",
        engine_rank=0,
        base_gpu_id=0,
        num_gpus_per_engine=1,
        model_path=None,
        scrape_url=None,
        sample_dir=None,
    )
    assert sampler.enabled is False
    sampler.start()
    assert sampler._thread is None
    sampler.stop()  # must not raise
