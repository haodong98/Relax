#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dedicated 4-GPU Judge topology smoke for MobileGym dual-Judge runs.

This driver deliberately bypasses Controller: Controller always creates core
training roles, while G2-A must exercise only the production ORM TP2 and VLM
TP2 Judge services. It still uses the normal argument parser, JudgeServiceSpec,
Service placement groups, GenRM, and SGLang implementation used by training.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import ray
import requests
import yaml
from ray import serve

from relax.components.genrm import GenRM
from relax.core.service import Service
from relax.utils.arguments import parse_args
from relax.utils.autoscaler.metrics_collector import parse_prometheus_metrics
from relax.utils.genrm_client import GenRMClient
from relax.utils.health_system import HealthStatus
from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_serve_url, post_process_env


logger = get_logger(__name__)

_ROLES = ("judge_accuracy", "judge_multiturn_vlm")
_SWEEP_CONCURRENCIES = (1, 2, 4, 8)
_FAULT_DRAIN_TIMEOUT_S = 90.0


def _prepend_unique_library_path(path: str) -> None:
    """Put an overlay library directory first, even if sitecustomize added one.

    The EDF image's Python startup can prepend OpenCV's bundled libraries to
    ``LD_LIBRARY_PATH``. Judge engines are launched only after this driver
    starts, so restore the venv cuDNN overlay before deriving their runtime
    environment.
    """
    existing = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry and entry != path]
    os.environ["LD_LIBRARY_PATH"] = ":".join([path, *existing])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _host_gpu_uuid_by_index() -> dict[int, str]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    result: dict[int, str] = {}
    for line in output.splitlines():
        index, uuid = (part.strip() for part in line.split(",", maxsplit=1))
        result[int(index)] = uuid
    return result


def _service_gpu_ids(service: Service) -> list[int]:
    if not isinstance(service.pgs, tuple) or len(service.pgs) != 3:
        raise RuntimeError(f"{service.role} has no dedicated placement-group GPU mapping: {service.pgs!r}")
    return [int(gpu_id) for gpu_id in service.pgs[2]]


def _require_contiguous_disjoint_pairs(services: dict[str, Service]) -> dict[str, list[int]]:
    mapping = {role: _service_gpu_ids(service) for role, service in services.items()}
    for role, gpu_ids in mapping.items():
        if len(gpu_ids) != 2 or gpu_ids != list(range(gpu_ids[0], gpu_ids[0] + 2)):
            raise RuntimeError(f"{role} requires a contiguous TP2 GPU pair, got {gpu_ids}")
    if set(mapping[_ROLES[0]]) & set(mapping[_ROLES[1]]):
        raise RuntimeError(f"Judge TP2 GPU pairs overlap: {mapping}")
    return mapping


def _request_payload() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "Return a JSON object with a single key named ok."}],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
        "max_input_tokens": 256,
    }


def _run_sweep(service: Service) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for concurrency in _SWEEP_CONCURRENCIES:

        def request_once() -> dict[str, Any]:
            started = time.perf_counter()
            # Service._http_call deliberately retains its legacy query-parameter
            # POST behaviour. The FastAPI Judge endpoint expects a JSON body,
            # so use requests directly for this production request sweep.
            response = requests.post(
                f"{get_serve_url(route_prefix=f'/{service.role}')}/generate",
                json=_request_payload(),
                timeout=300,
            )
            response.raise_for_status()
            response = response.json()
            return {
                "elapsed_s": time.perf_counter() - started,
                "response_chars": len(str(response.get("response", ""))),
                "timings": response.get("timings", {}),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            request_results = list(executor.map(lambda _: request_once(), range(concurrency)))
        metrics = service._http_call(service.role, "/metrics", params={"include_gpu_occupancy": "true"}, timeout=30)
        results.append({"concurrency": concurrency, "requests": request_results, "metrics": metrics})
    return results


def _service_url(service: Service) -> str:
    return get_serve_url(route_prefix=f"/{service.role}")


def _service_metrics(service: Service) -> dict[str, Any]:
    response = requests.get(f"{_service_url(service)}/metrics", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{service.role} returned non-object metrics: {payload!r}")
    return payload


def _engine_base_urls(service: Service) -> list[str]:
    manager = service.handle.get_genrm_manager.remote().result()
    hosts_ports = ray.get(manager.get_engine_hosts_ports.remote())
    if not hosts_ports:
        raise RuntimeError(f"{service.role} has no live SGLang engine address")
    return [f"http://{host}:{port}" for host, port in hosts_ports]


def _engine_metrics(engine_base_url: str) -> dict[str, float]:
    response = requests.get(f"{engine_base_url}/metrics", timeout=30)
    response.raise_for_status()
    return parse_prometheus_metrics(response.text)


class _AttemptCountingGenRMClient(GenRMClient):
    """Observe the production GenRM retry loop without changing its policy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attempts: list[dict[str, Any]] = []

    async def generate_once(self, payload: dict[str, Any]) -> str:
        attempt: dict[str, Any] = {"started_at_s": time.time()}
        self.attempts.append(attempt)
        try:
            return await super().generate_once(payload)
        except Exception as exc:
            attempt["exception_type"] = type(exc).__name__
            raise


def _wait_for_drain(service: Service, engine_base_urls: list[str]) -> dict[str, Any]:
    deadline = time.monotonic() + _FAULT_DRAIN_TIMEOUT_S
    last_metrics: dict[str, Any] = {}
    engine_polls: list[dict[str, Any]] = []
    consecutive_idle_polls = 0
    while time.monotonic() < deadline:
        last_metrics = _service_metrics(service)
        engine_polls = []
        for engine_base_url in engine_base_urls:
            metrics = _engine_metrics(engine_base_url)
            engine_polls.append(
                {
                    "engine_base_url": engine_base_url,
                    "num_running_reqs": metrics.get("sglang:num_running_reqs"),
                    "num_queue_reqs": metrics.get("sglang:num_queue_reqs"),
                }
            )
        engine_idle = all(poll["num_running_reqs"] == 0 and poll["num_queue_reqs"] == 0 for poll in engine_polls)
        drained = last_metrics.get("current_inflight") == 0 and last_metrics.get("current_queued") == 0 and engine_idle
        if drained:
            consecutive_idle_polls += 1
            if consecutive_idle_polls >= 2:
                flush_statuses = []
                for engine_base_url in engine_base_urls:
                    flush_response = requests.get(f"{engine_base_url}/flush_cache", timeout=30)
                    flush_statuses.append(flush_response.status_code)
                if all(status == 200 for status in flush_statuses):
                    return {
                        "service_metrics": last_metrics,
                        "engine_metrics": engine_polls,
                        "flush_cache_statuses": flush_statuses,
                    }
        else:
            consecutive_idle_polls = 0
        time.sleep(0.5)
    raise TimeoutError(f"{service.role} did not drain after fault injection: {last_metrics}")


async def _inject_client_timeout_and_retry(service: Service) -> dict[str, Any]:
    """Force the real GenRM client through its transient timeout retry
    ladder."""
    client = _AttemptCountingGenRMClient(_service_url(service), timeout=0.005, role=service.role)
    try:
        await client.generate(
            [{"role": "user", "content": "Return a JSON object with a single key named ok."}],
            sampling_params={"temperature": 0.0, "max_new_tokens": 64},
        )
    except httpx.TimeoutException as exc:
        if len(client.attempts) != 3:
            raise RuntimeError(
                f"{service.role} timeout injection made {len(client.attempts)} attempts instead of 3"
            ) from exc
        return {"exception": type(exc).__name__, "attempts": client.attempts}
    finally:
        await client.aclose()
    raise RuntimeError(f"{service.role} timeout injection unexpectedly completed without a timeout")


def _long_request_payload() -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": "Summarize this text only after reading all words: " + ("critical-path " * 8000),
            }
        ],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 512},
        "max_input_tokens": 30000,
    }


def _wait_for_inflight_request(service: Service, engine_base_urls: list[str]) -> dict[str, Any]:
    deadline = time.monotonic() + 30.0
    last_service_metrics: dict[str, Any] = {}
    last_engine_metrics: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last_service_metrics = _service_metrics(service)
        last_engine_metrics = []
        for engine_base_url in engine_base_urls:
            metrics = _engine_metrics(engine_base_url)
            last_engine_metrics.append(
                {
                    "engine_base_url": engine_base_url,
                    "num_running_reqs": metrics.get("sglang:num_running_reqs"),
                    "num_queue_reqs": metrics.get("sglang:num_queue_reqs"),
                }
            )
        engine_has_work = any(
            (item["num_running_reqs"] or 0) + (item["num_queue_reqs"] or 0) > 0 for item in last_engine_metrics
        )
        if last_service_metrics.get("current_inflight", 0) > 0 and engine_has_work:
            return {"service_metrics": last_service_metrics, "engine_metrics": last_engine_metrics}
        time.sleep(0.05)
    raise TimeoutError(
        f"{service.role} long request never reached SGLang: "
        f"service={last_service_metrics}, engines={last_engine_metrics}"
    )


def _run_fault_and_drain_gate(service: Service) -> dict[str, Any]:
    """Exercise malformed-input, timeout/retry, and active-request abort
    paths."""
    service_url = _service_url(service)
    engine_base_urls = _engine_base_urls(service)
    served_before_malformed = int(_service_metrics(service)["cumulative_served_requests"])
    malformed = requests.post(
        f"{service_url}/generate",
        data=b'{"messages":',
        headers={"content-type": "application/json"},
        timeout=30,
    )
    if malformed.status_code not in {400, 422}:
        raise RuntimeError(f"{service.role} malformed JSON returned {malformed.status_code}: {malformed.text}")
    after_malformed = _wait_for_drain(service, engine_base_urls)
    if int(after_malformed["service_metrics"]["cumulative_served_requests"]) != served_before_malformed:
        raise RuntimeError(f"{service.role} malformed JSON unexpectedly reached the Judge engine")

    served_before_timeout = int(_service_metrics(service)["cumulative_served_requests"])
    timeout_result = asyncio.run(_inject_client_timeout_and_retry(service))
    after_timeout = _wait_for_drain(service, engine_base_urls)
    reissue = requests.post(f"{service_url}/generate", json=_request_payload(), timeout=300)
    reissue.raise_for_status()
    after_reissue = _wait_for_drain(service, engine_base_urls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        request_future = executor.submit(
            requests.post,
            f"{service_url}/generate",
            json=_long_request_payload(),
            timeout=_FAULT_DRAIN_TIMEOUT_S,
        )
        admitted = _wait_for_inflight_request(service, engine_base_urls)
        abort_statuses = []
        for engine_base_url in engine_base_urls:
            abort_response = requests.post(f"{engine_base_url}/abort_request", json={"abort_all": True}, timeout=30)
            abort_statuses.append(abort_response.status_code)
        if not all(status == 200 for status in abort_statuses):
            raise RuntimeError(f"{service.role} SGLang abort_request failed: {abort_statuses}")
        try:
            cancelled_response = request_future.result(timeout=_FAULT_DRAIN_TIMEOUT_S)
            try:
                cancelled_payload = cancelled_response.json()
            except ValueError:
                cancelled_payload = None
            timings = cancelled_payload.get("timings") if isinstance(cancelled_payload, dict) else None
            cancelled_outcome: dict[str, Any] = {
                "kind": "http_response",
                "status_code": cancelled_response.status_code,
                "engine_attempt_count": timings.get("engine_attempt_count") if isinstance(timings, dict) else None,
            }
        except requests.RequestException as exc:
            cancelled_outcome = {"kind": "request_exception", "type": type(exc).__name__}

    after_cancel = _wait_for_drain(service, engine_base_urls)
    post_abort_probe = requests.post(f"{service_url}/generate", json=_request_payload(), timeout=300)
    post_abort_probe.raise_for_status()
    after_probe = _wait_for_drain(service, engine_base_urls)
    recovery = service.wait_ready(timeout=300)
    return {
        "malformed_input": {
            "status_code": malformed.status_code,
            "service_metrics_after": after_malformed["service_metrics"],
        },
        "timeout_reissue": {
            "timeout_exception": timeout_result["exception"],
            "client_attempt_count": len(timeout_result["attempts"]),
            "client_attempts": timeout_result["attempts"],
            "successful_responses_delta_after_timeout": int(
                after_timeout["service_metrics"]["cumulative_served_requests"]
            )
            - served_before_timeout,
            "reissue_status_code": reissue.status_code,
            "drain": after_reissue,
        },
        "abort": {
            "inflight_observed": admitted,
            "abort_http_statuses": abort_statuses,
            "long_request_outcome": cancelled_outcome,
        },
        "drain": {
            "after_abort": after_cancel,
            "after_post_abort_probe": after_probe,
        },
        "recovery": recovery,
    }


def _engine_runtime_identities(service: Service) -> list[dict[str, Any]]:
    # Ray Serve 2.x returns a DeploymentResponse from a deployment handle;
    # unlike an ObjectRef it must be resolved through its own API.
    manager = service.handle.get_genrm_manager.remote().result()
    engines, _, _ = ray.get(manager.get_genrm_engines_and_lock.remote())
    return ray.get([engine.get_runtime_identity.remote() for engine in engines])


def _read_sampler_records(sample_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(sample_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    return records


def _validate_sampler_records(
    records: list[dict[str, Any]], gpu_pairs: dict[str, list[int]], uuid_by_index: dict[int, str]
) -> None:
    manifests = {record["role"]: record for record in records if record.get("record_type") == "manifest"}
    for role, gpu_ids in gpu_pairs.items():
        manifest = manifests.get(role)
        if manifest is None or not manifest.get("nvml_enabled"):
            raise RuntimeError(f"{role} did not emit an NVML sampler manifest")
        expected_uuids = [uuid_by_index[gpu_id] for gpu_id in gpu_ids]
        if manifest.get("gpu_uuids") != expected_uuids:
            raise RuntimeError(f"{role} sampler UUIDs do not match its placement group: {manifest}")
        samples = [
            record for record in records if record.get("record_type") == "sample" and record.get("role") == role
        ]
        if not any(item.get("util_percent", 0.0) > 0.0 for sample in samples for item in sample.get("gpu", [])):
            raise RuntimeError(f"{role} has no non-idle NVML sample after the request sweep")
        sglang_samples = [sample["sglang"] for sample in samples if isinstance(sample.get("sglang"), dict)]
        required_sglang_keys = {"sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage"}
        if not any(required_sglang_keys <= set(sample) for sample in sglang_samples):
            raise RuntimeError(f"{role} has no complete SGLang metrics sample")
        if not any(sample.get("sglang:num_running_reqs", 0.0) > 0.0 for sample in sglang_samples):
            raise RuntimeError(f"{role} sampler never observed an active SGLang request")
        if not any(sample.get("sglang:token_usage", 0.0) > 0.0 for sample in sglang_samples):
            raise RuntimeError(f"{role} sampler never observed positive SGLang token usage")


def main() -> None:
    args = parse_args()
    if args.judge_services is None:
        raise ValueError("G2 Judge smoke requires --rm-type dual-agentic-judge and --judge-services-config")
    if set(args.resource) != set(_ROLES):
        raise ValueError(f"G2 Judge smoke accepts only dedicated Judge resources, got {args.resource}")

    output_dir = Path(os.environ["G2_JUDGE_SMOKE_DIR"]).resolve()
    sample_dir = Path(os.environ["RELAX_JUDGE_GPU_SAMPLE_DIR"]).resolve()
    expected_cudnn_dir = os.environ["CUDNN_LIB_DIR"]
    _prepend_unique_library_path(expected_cudnn_dir)
    if not sample_dir:
        raise ValueError("RELAX_JUDGE_GPU_SAMPLE_DIR must be set")
    if "RELAX_JUDGE_GPU_SAMPLE_DIR" not in os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""):
        raise ValueError("RELAX_JUDGE_GPU_SAMPLE_DIR must be listed in RELAX_PROPAGATE_ENV_VARS")
    if "LD_LIBRARY_PATH" not in os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""):
        raise ValueError("LD_LIBRARY_PATH must be listed in RELAX_PROPAGATE_ENV_VARS")
    if "CUDNN_LIB_DIR" not in os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""):
        raise ValueError("CUDNN_LIB_DIR must be listed in RELAX_PROPAGATE_ENV_VARS")

    with (Path(__file__).resolve().parents[2] / "configs" / "env.yaml").open(encoding="utf-8") as file:
        runtime_env = post_process_env(args, yaml.safe_load(file))
    if runtime_env["env_vars"].get("LD_LIBRARY_PATH", "").split(":")[0] != expected_cudnn_dir:
        raise RuntimeError("The Ray runtime environment does not lead with the venv cuDNN library")

    ray.init(address="auto", runtime_env=runtime_env)
    serve.start(http_options={"host": "0.0.0.0", "port": 8000}, detached=True)
    healthy = HealthStatus.remote()
    services: dict[str, Service] = {}
    report: dict[str, Any] = {"schema_version": 1, "roles": {}, "status": "failed"}
    try:
        for role in _ROLES:
            service = Service(GenRM, role, healthy, args, num_gpus=2, runtime_env=runtime_env)
            services[role] = service
            readiness = service.wait_ready(timeout=600)
            spec = args.judge_services.by_role(role)
            if readiness.get("service") != role or readiness.get("model_path") != spec.model_path:
                raise RuntimeError(f"{role} readiness identity mismatch: {readiness}")
            if role == "judge_multiturn_vlm" and readiness.get("processor") is not True:
                raise RuntimeError(f"{role} readiness did not validate the multimodal processor")

        gpu_pairs = _require_contiguous_disjoint_pairs(services)
        uuid_by_index = _host_gpu_uuid_by_index()
        for role, service in services.items():
            identities = _engine_runtime_identities(service)
            for identity in identities:
                if (identity.get("ld_library_path") or "").split(":", maxsplit=1)[0] != expected_cudnn_dir:
                    raise RuntimeError(f"{role} GenRMEngine does not lead with the venv cuDNN overlay: {identity}")
            report["roles"][role] = {
                "readiness": service.wait_ready(timeout=300),
                "placement_gpu_ids": gpu_pairs[role],
                "engine_runtime_identities": identities,
                "sweep": _run_sweep(service),
                "fault_and_drain": _run_fault_and_drain_gate(service),
            }

        time.sleep(1.0)
        records = _read_sampler_records(sample_dir)
        _validate_sampler_records(records, gpu_pairs, uuid_by_index)
        report["host_gpu_uuid_by_index"] = uuid_by_index
        report["sampler_records"] = records
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        for service in reversed(list(services.values())):
            service.shutdown_owned()
        _write_json(output_dir / "g2_judge_tp2_smoke_report.json", report)
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
