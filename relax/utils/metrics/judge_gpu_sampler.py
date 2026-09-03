# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Background GPU-occupancy sampler for dedicated reward-model (judge) and
rollout SGLang engines.

Runs a daemon thread inside an ``SGLangEngine``/``GenRMEngine`` Ray actor that
periodically reads NVML utilization for the physical GPUs the engine owns and
scrapes the engine's own SGLang Prometheus endpoint, so latency-benchmark
tooling can answer "were the judge GPUs actually busy" rather than only
"how long did the reward request take". See ``examples/agentic_dual_judge/``.

All accumulators are cumulative sums/counts since the sampler started, never
drain-and-reset, so ``snapshot()`` can be read repeatedly (e.g. once per
rollout publication round) and the caller derives a windowed mean by
differencing two reads. This mirrors the request-occupancy integrator in
``relax.components.genrm.GenRM``.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any

import requests

from relax.utils.autoscaler.metrics_collector import parse_prometheus_metrics
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_SGLANG_SCRAPE_TIMEOUT_S = 2.0
_SGLANG_METRIC_KEYS = (
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:gen_throughput",
    "sglang:cache_hit_rate",
)


def _try_init_nvml() -> Any:
    """Best-effort NVML init.

    Mirrors the soft-dependency pattern in
    ``relax.utils.device.set_numa_affinity``: pynvml ships only transitively
    (via the ``gpustat`` requirement), so a missing or broken install must
    degrade to a silent no-op rather than fail engine startup.
    """
    try:
        import pynvml
    except ImportError:
        logger.info("pynvml not available, judge GPU sampling disabled")
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.info(f"Failed to initialize NVML, judge GPU sampling disabled: {exc}")
        return None
    return pynvml


class JudgeGpuSampler:
    """Periodically samples NVML + SGLang Prometheus metrics for one engine
    shard and appends them to a JSONL sidecar file.

    One instance per ``SGLangEngine``/``GenRMEngine`` actor. Disabled (no
    thread started) unless ``sample_dir`` is truthy.
    """

    def __init__(
        self,
        *,
        role: str,
        engine_rank: int,
        base_gpu_id: int,
        num_gpus_per_engine: int,
        model_path: str | None,
        scrape_url: str | None,
        sample_dir: str | None,
        server_host: str | None = None,
        dist_init_addr: str | None = None,
        interval_s: float = 0.2,
    ) -> None:
        self._role = role
        self._engine_rank = engine_rank
        self._base_gpu_id = base_gpu_id
        self._num_gpus_per_engine = max(1, num_gpus_per_engine)
        self._model_path = model_path
        self._scrape_url = scrape_url
        self._sample_dir = sample_dir
        self._server_host = server_host
        self._dist_init_addr = dist_init_addr
        self._interval_s = max(0.01, interval_s)
        self._clock_host = socket.gethostname()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._nvml: Any = None
        self._gpu_handles: list[tuple[int, Any]] = []
        self._uuid_by_index: dict[int, str] = {}
        self._sglang_scrape_warned = False

        # Cumulative accumulators (protected by ``_lock``); never reset.
        self._tick_count = 0
        # A sampler heartbeat is not necessarily a GPU observation: NVML can
        # be unavailable or an individual device query can fail. Keep an
        # explicit per-GPU denominator so consumers never interpret a missing
        # NVML observation as a zero-utilization sample.
        self._gpu_sample_count: dict[int, int] = {}
        self._gpu_util_percent_sum: dict[int, float] = {}
        self._gpu_util_nonzero_count: dict[int, int] = {}
        self._gpu_mem_used_bytes_sum: dict[int, float] = {}
        self._gpu_mem_total_bytes_last: dict[int, int] = {}
        self._sglang_tick_count = 0
        self._sglang_metrics_ever_enabled = False
        self._sglang_sum: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._sample_dir)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        try:
            os.makedirs(self._sample_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(f"{self._role} judge GPU sampler cannot create {self._sample_dir}: {exc}; disabling")
            self._sample_dir = None
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"judge-gpu-sampler-{self._role}-{self._engine_rank}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None

    def _file_path(self) -> str:
        return os.path.join(self._sample_dir, f"{self._role}_{self._clock_host}_rank{self._engine_rank}.jsonl")

    def _run(self) -> None:
        self._nvml = _try_init_nvml()
        if self._nvml is not None:
            self._resolve_gpu_handles()
        self._write_manifest()
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(self._interval_s)
        # Capture the engine's last observable request state before shutdown.
        # This is the fail-closed drain witness: non-zero WIP here means the run
        # terminated with right-censored SGLang work.
        self._tick(final_sample=True)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _resolve_gpu_handles(self) -> None:
        handles: list[tuple[int, Any]] = []
        try:
            for index in range(self._base_gpu_id, self._base_gpu_id + self._num_gpus_per_engine):
                handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
                uuid = self._nvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8", "replace")
                handles.append((index, handle))
                self._uuid_by_index[index] = uuid
        except Exception as exc:
            logger.info(
                f"{self._role} failed to resolve NVML handles for GPUs "
                f"{self._base_gpu_id}..{self._base_gpu_id + self._num_gpus_per_engine - 1}: {exc}; "
                "disabling NVML sampling for this engine"
            )
            self._nvml = None
            handles = []
            self._uuid_by_index = {}
        self._gpu_handles = handles

    def _scrape_sglang(self) -> dict[str, float] | None:
        if self._scrape_url is None:
            return None
        try:
            response = requests.get(self._scrape_url, timeout=_SGLANG_SCRAPE_TIMEOUT_S)
            response.raise_for_status()
            parsed = parse_prometheus_metrics(response.text)
        except Exception as exc:
            if not self._sglang_scrape_warned:
                logger.warning(
                    f"{self._role} judge GPU sampler could not scrape SGLang metrics at "
                    f"{self._scrape_url} (NVML sampling continues regardless); this engine's "
                    "server_args needs enable_metrics=true for this signal. First error: "
                    f"{exc}"
                )
                self._sglang_scrape_warned = True
            return None
        record = {key: parsed[key] for key in _SGLANG_METRIC_KEYS if key in parsed}
        return record or None

    def _tick(self, *, final_sample: bool = False) -> None:
        now_wall_s = time.time()
        gpu_records = []
        if self._nvml is not None:
            for index, handle in self._gpu_handles:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_records.append(
                        {
                            "index": index,
                            "uuid": self._uuid_by_index.get(index),
                            "util_percent": float(util.gpu),
                            "mem_util_percent": float(util.memory),
                            "mem_used_bytes": int(mem.used),
                            "mem_total_bytes": int(mem.total),
                        }
                    )
                except Exception as exc:
                    logger.info(f"{self._role} NVML sample failed for GPU {index}: {exc}")
        sglang_record = self._scrape_sglang()

        with self._lock:
            self._tick_count += 1
            for record in gpu_records:
                idx = record["index"]
                self._gpu_sample_count[idx] = self._gpu_sample_count.get(idx, 0) + 1
                self._gpu_util_percent_sum[idx] = self._gpu_util_percent_sum.get(idx, 0.0) + record["util_percent"]
                self._gpu_util_nonzero_count[idx] = self._gpu_util_nonzero_count.get(idx, 0) + (
                    1 if record["util_percent"] > 0 else 0
                )
                self._gpu_mem_used_bytes_sum[idx] = (
                    self._gpu_mem_used_bytes_sum.get(idx, 0.0) + record["mem_used_bytes"]
                )
                self._gpu_mem_total_bytes_last[idx] = record["mem_total_bytes"]
            if sglang_record is not None:
                self._sglang_tick_count += 1
                self._sglang_metrics_ever_enabled = True
                for key, value in sglang_record.items():
                    self._sglang_sum[key] = self._sglang_sum.get(key, 0.0) + value

        self._append_jsonl(
            {
                "schema_version": 1,
                "record_type": "sample",
                "ts": now_wall_s,
                "clock_host": self._clock_host,
                "role": self._role,
                "engine_rank": self._engine_rank,
                "final_sample": final_sample,
                "gpu": gpu_records,
                "sglang": sglang_record,
            }
        )

    def _write_manifest(self) -> None:
        self._append_jsonl(
            {
                "schema_version": 1,
                "record_type": "manifest",
                "ts": time.time(),
                "clock_host": self._clock_host,
                "role": self._role,
                "engine_rank": self._engine_rank,
                "base_gpu_id": self._base_gpu_id,
                "num_gpus_per_engine": self._num_gpus_per_engine,
                "gpu_uuids": [self._uuid_by_index.get(i) for i in self._gpu_handle_indices()],
                "model_path": self._model_path,
                "server_host": self._server_host,
                "dist_init_addr": self._dist_init_addr,
                "flashinfer_workspace_base": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
                "nvml_enabled": self._nvml is not None,
                "scrape_configured": self._scrape_url is not None,
                "interval_s": self._interval_s,
            }
        )

    def _gpu_handle_indices(self) -> list[int]:
        return [index for index, _ in self._gpu_handles]

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        try:
            with open(self._file_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning(f"{self._role} judge GPU sampler failed to write {self._file_path()}: {exc}")

    def snapshot(self) -> dict[str, Any]:
        """Return the current cumulative accumulators.

        Every field is a running sum/count since the sampler started, not a
        mean and not a windowed value — callers that want a per-interval mean
        (e.g. "utilization during this publication round") must cache a
        previous snapshot and difference the sums/counts themselves. In
        particular, use each GPU's ``sample_count`` rather than the sampler
        heartbeat ``tick_count`` as the utilization denominator.
        """
        with self._lock:
            gpu = [
                {
                    "index": index,
                    "uuid": self._uuid_by_index.get(index),
                    "sample_count": self._gpu_sample_count.get(index, 0),
                    "util_percent_sum": self._gpu_util_percent_sum.get(index, 0.0),
                    "util_nonzero_count": self._gpu_util_nonzero_count.get(index, 0),
                    "mem_used_bytes_sum": self._gpu_mem_used_bytes_sum.get(index, 0.0),
                    "mem_total_bytes": self._gpu_mem_total_bytes_last.get(index),
                }
                for index in range(self._base_gpu_id, self._base_gpu_id + self._num_gpus_per_engine)
            ]
            return {
                "role": self._role,
                "engine_rank": self._engine_rank,
                "now_s": time.time(),
                "tick_count": self._tick_count,
                "nvml_enabled": self._nvml is not None,
                "gpu": gpu,
                "sglang_tick_count": self._sglang_tick_count,
                "sglang_metrics_enabled": self._sglang_metrics_ever_enabled,
                "sglang_sum": dict(self._sglang_sum),
            }
