# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Service Implementation.

This module provides a Ray Serve deployment for Generative Reward Model
(genRM), which evaluates responses using LLM-based preference prediction.
"""

import asyncio
import base64
import hashlib
import time
from argparse import Namespace
from itertools import cycle
from typing import Any, List, Optional, Union

import httpx
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.ray.placement_group import create_genrm_manager
from relax.utils.data.processing_utils import load_processor, load_tokenizer
from relax.utils.env import Envs


app = FastAPI()

# Max concurrent in-flight requests per GenRM Serve replica. Ray Serve's default
# of 5 throttles judge dispatch and leaves the SGLang engines idle. The replica
# is a pure-async CPU proxy (tokenize + forward), so a high cap lets one replica
# saturate the engines. Override via env for tuning.
GENRM_SERVE_MAX_ONGOING_REQUESTS = Envs.GENRM_SERVE_MAX_ONGOING_REQUESTS

# NOTE: GENRM_SERVE_MAX_ONGOING_REQUESTS above must stay module-level — it feeds
# the @serve.deployment decorator, which runs at import. The retry count has no
# such constraint, so it is read at the call site to stay lazy.


class Message(BaseModel):
    """Single chat message."""

    role: str
    content: Any


class GenerateRequest(BaseModel):
    """Request model for genRM generation (OpenAI chat format).

    Accepts a list of messages in OpenAI format with optional sampling params.
    """

    messages: Union[List[Message], List[dict]]
    sampling_params: Optional[dict] = None
    context_hash: Optional[str] = None
    projection_hash: Optional[str] = None
    judge_role: Optional[str] = None
    prompt_version: Optional[str] = None
    media_manifest: List[dict] = Field(default_factory=list)
    media_blobs: dict[str, str] = Field(default_factory=dict)
    max_input_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    """Response model for genRM generation.

    Returns the raw model response text.
    """

    response: str
    timings: dict[str, float | int] = Field(default_factory=dict)


@serve.deployment(
    max_ongoing_requests=GENRM_SERVE_MAX_ONGOING_REQUESTS,
    logging_config=LoggingConfig(
        log_level="WARNING",
        enable_access_log=False,  # 关闭 HTTP 访问日志
    ),
)
@serve.ingress(app)
class GenRM(Base):
    """GenRM Service for generative reward model evaluation.

    This service uses SGLang engines to perform preference evaluation by
    comparing model responses against ground truth or standards.
    """

    def __init__(
        self,
        healthy: Any,
        pg: Optional[Any],
        num_gpus: int,
        config: Namespace,
        role: str,
        runtime_env: Optional[dict] = None,
    ) -> None:
        """Initialize GenRM service.

        Args:
            healthy: Remote health manager actor handle.
            pg: Placement group for resource allocation.
            num_gpus: Number of GPUs allocated (used by Service framework).
            config: Runtime configuration namespace.
            role: Role name (should be "genrm").
            runtime_env: Optional Ray runtime environment dict.
        """
        super().__init__()
        if role in {"judge_accuracy", "judge_multiturn_vlm"}:
            config = config.judge_services.by_role(role).as_genrm_namespace(config)
        self.config = config
        self.healthy = healthy
        self.role = role

        # Initialize GenRM Manager
        if role == "genrm":
            self.genrm_manager = create_genrm_manager(config, pg, runtime_env=runtime_env)
        else:
            self.genrm_manager = create_genrm_manager(config, pg, runtime_env=runtime_env, role=role)

        # Store engine addresses for HTTP-based generation
        self._engine_hosts_ports = None
        self._engine_index = 0
        self._logger.info("GenRM service initialized successfully")
        # Shared HTTP client for engine calls (avoids per-request connection overhead).
        # Raise pool limits well above httpx's default 100 so one replica can fan out
        # many concurrent engine requests; keepalive_expiry >> the default 5s so
        # idle-then-reused connections aren't reaped mid-burst (avoids ReadError/500).
        self._http_client = httpx.AsyncClient(
            timeout=1800,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048, keepalive_expiry=600),
        )

        # Load tokenizer for prompt encoding
        self.tokenizer = load_tokenizer(config.genrm_model_path, trust_remote_code=True)
        self.processor = (
            load_processor(config.genrm_model_path, trust_remote_code=True) if role == "judge_multiturn_vlm" else None
        )
        if role == "judge_multiturn_vlm" and self.processor is None:
            raise RuntimeError(f"Failed to load multimodal processor for {config.genrm_model_path}")
        self._stopping = False
        self._inflight_requests = 0
        self._queued_requests = 0
        configured_max_concurrency = getattr(config, "genrm_max_concurrency", None)
        if configured_max_concurrency is not None:
            if isinstance(configured_max_concurrency, bool) or not isinstance(configured_max_concurrency, int):
                raise TypeError(
                    f"{role} genrm_max_concurrency must be a positive integer when configured, "
                    f"got {configured_max_concurrency!r}"
                )
            if configured_max_concurrency <= 0:
                raise ValueError(
                    f"{role} genrm_max_concurrency must be a positive integer when configured, "
                    f"got {configured_max_concurrency}"
                )
        # Client-side semaphores are local to their session-shard event loop,
        # so they cannot cap aggregate pressure from all shards. Dedicated
        # judge roles carry this role-level admission limit in their scoped
        # config; the Serve replica is the one shared chokepoint.
        self._server_max_concurrency = configured_max_concurrency
        self._request_semaphore = (
            asyncio.Semaphore(configured_max_concurrency) if configured_max_concurrency is not None else None
        )
        self._drained = asyncio.Event()
        self._drained.set()

        # Event-driven request-occupancy integration (exact, no sampling error).
        # `_advance_occupancy` folds the in-flight level held since the last call
        # into the running integral before any transition, so the accumulators
        # are always consistent with the wall clock at the moment they're read.
        # Duration bookkeeping uses the monotonic clock; `_occupancy_epoch_wall_s`
        # is wall-clock only so /metrics readers can align it against other
        # processes without assuming perf_counter shares an epoch across them.
        self._occupancy_epoch_wall_s = time.time()
        self._occupancy_anchor_ns = time.perf_counter_ns()
        self._busy_wall_s = 0.0
        self._inflight_integral_req_s = 0.0
        self._served_requests = 0

    def run(self):
        """GenRM is a passive HTTP service, no background loop needed.

        Unlike Actor or Rollout, GenRM only responds to incoming requests and
        does not actively produce work. Return None so the Controller training
        loop does not block on it.
        """
        return None

    def _advance_occupancy(self) -> None:
        """Fold the in-flight level held since the last call into the running
        integral, then re-anchor.

        Called immediately before every transition of ``_inflight_requests`` so
        ``_busy_wall_s`` / ``_inflight_integral_req_s`` are exact integrals of
        an event-driven step function, not a sampled estimate.
        """
        now_ns = time.perf_counter_ns()
        delta_s = (now_ns - self._occupancy_anchor_ns) / 1e9
        self._occupancy_anchor_ns = now_ns
        self._inflight_integral_req_s += self._inflight_requests * delta_s
        if self._inflight_requests > 0:
            self._busy_wall_s += delta_s

    @app.post("/generate")
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate response for given chat messages.

        Takes OpenAI-style messages as input, sends to SGLang engine,
        and returns the raw model response. The caller is responsible
        for formatting the prompt and parsing the response.

        Args:
            request: GenerateRequest containing messages list and optional sampling_params

        Returns:
            GenerateResponse containing raw model response text
        """
        request_started_ns = time.perf_counter_ns()
        timings: dict[str, float | int] = {}
        request_semaphore = getattr(self, "_request_semaphore", None)
        semaphore_acquired = False
        if request_semaphore is not None:
            self._queued_requests += 1
            try:
                await request_semaphore.acquire()
                semaphore_acquired = True
            finally:
                self._queued_requests -= 1
        timings["server_queue_elapsed_s"] = (time.perf_counter_ns() - request_started_ns) / 1e9
        admitted = False
        try:
            if self._stopping:
                raise HTTPException(status_code=503, detail={"code": "stopping"})
            self._advance_occupancy()
            self._inflight_requests += 1
            admitted = True
            self._drained.clear()
            try:
                media_started_ns = time.perf_counter_ns()
                messages, image_data = self._restore_media(request)
                timings["media_restore_elapsed_s"] = (time.perf_counter_ns() - media_started_ns) / 1e9
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "invalid_media", "message": str(exc)},
                ) from exc
            # Call SGLang engine via GenRMManager
            output = await self._call_engine(
                messages,
                request.sampling_params,
                image_data=image_data,
                max_input_tokens=request.max_input_tokens,
                timings=timings,
            )

            # Return raw response text
            response = output.get("text", "").strip()
            meta_info = output.get("meta_info")
            if isinstance(meta_info, dict):
                completion_tokens = meta_info.get("completion_tokens")
                if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
                    timings["output_tokens"] = completion_tokens
            timings["request_total_elapsed_s"] = (time.perf_counter_ns() - request_started_ns) / 1e9
            self._served_requests += 1
            return GenerateResponse(response=response, timings=timings)

        except Exception as e:
            self._logger.error(f"GenRM generation failed: {e}")
            raise
        finally:
            if admitted:
                self._advance_occupancy()
                self._inflight_requests -= 1
                if self._inflight_requests == 0:
                    self._drained.set()
            if semaphore_acquired:
                request_semaphore.release()

    @staticmethod
    def _message_dict(message: Message | dict) -> dict:
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return message.dict()

    def _restore_media(self, request: GenerateRequest) -> tuple[list[dict], list[str]]:
        manifest = {item.get("media_id"): item for item in request.media_manifest}
        image_data: list[str] = []
        messages = [self._message_dict(message) for message in request.messages]
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            restored = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    restored.append(part)
                    continue
                media_id = part.get("media_id")
                data_uri = request.media_blobs.get(media_id)
                item = manifest.get(media_id)
                if not isinstance(media_id, str) or not isinstance(data_uri, str) or item is None:
                    raise ValueError(f"missing media payload for {media_id!r}")
                try:
                    header, encoded = data_uri.split(",", 1)
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"invalid media payload for {media_id}") from exc
                digest = hashlib.sha256(raw).hexdigest()
                if media_id != f"sha256:{digest}" or item.get("size_bytes") != len(raw):
                    raise ValueError(f"media digest/size mismatch for {media_id}")
                if not header.startswith("data:image/"):
                    raise ValueError(f"unsupported media for {media_id}")
                restored.append({"type": "image_url", "image_url": {"url": data_uri}})
                image_data.append(encoded)
            message["content"] = restored
        if len(image_data) != len(request.media_blobs):
            raise ValueError("media placeholder and blob counts do not match")
        return messages, image_data

    async def _call_engine(
        self,
        messages: list,
        sampling_params: Optional[dict] = None,
        *,
        image_data: Optional[list[str]] = None,
        max_input_tokens: Optional[int] = None,
        timings: Optional[dict[str, float | int]] = None,
    ) -> dict:
        """Call an SGLang engine for text generation.

        Uses the engine addresses obtained from GenRMManager to send HTTP
        requests to the underlying SGLang server.

        Args:
            messages: List of chat messages in OpenAI format.
            sampling_params: Optional per-request sampling params that override defaults.

        Returns:
            Dict containing at least {"text": str} from the SGLang server.
        """
        retry_attempts = Envs.GENRM_ENGINE_RETRY_ATTEMPTS
        if retry_attempts < 1:
            raise RuntimeError(f"GENRM_ENGINE_RETRY_ATTEMPTS must be at least 1, got {retry_attempts}.")

        # Lazily fetch engine host/port information and build cycle iterator
        call_started_ns = time.perf_counter_ns()
        address_started_ns = call_started_ns
        if self._engine_hosts_ports is None:
            self._engine_hosts_ports = ray.get(self.genrm_manager.get_engine_hosts_ports.remote())
            self._engine_cycle = cycle(range(len(self._engine_hosts_ports)))
        if timings is not None:
            timings["engine_address_elapsed_s"] = (time.perf_counter_ns() - address_started_ns) / 1e9

        if not self._engine_hosts_ports:
            raise RuntimeError("No genRM engines available")

        # Thread-safe round-robin via itertools.cycle (next() is atomic in CPython)
        idx = next(self._engine_cycle)
        host, port = self._engine_hosts_ports[idx]

        url = f"http://{host}:{port}/generate"
        # ensure plain list — some tokenizers return BatchEncoding which is not JSON-serializable
        # Tokenization (chat-template render + encode) is synchronous CPU work; run it in a
        # worker thread so it does not block this replica's event loop. Fast (Rust) tokenizers
        # release the GIL during encode, so concurrent requests tokenize in parallel instead of
        # serializing — without this a single replica throttles dispatch and starves the engines.
        # Forward chat_template_kwargs from --genrm-sampling-config through to
        # the jinja template — e.g. `{"enable_thinking": false}` for Qwen3+ to
        # suppress the default <think> block. Keys unused by the template are
        # silently dropped by transformers, so this is safe across model families.
        tokenize_started_ns = time.perf_counter_ns()
        chat_template_kwargs = self.config.genrm_sampling_config.get("chat_template_kwargs", {}) or {}
        if self.processor is None:
            input_ids = await asyncio.to_thread(
                self.tokenizer.apply_chat_template,
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        else:
            # Multimodal processors own the model-specific conversion from
            # structured image parts to visual placeholder tokens. Render only
            # the chat template here; SGLang receives the corresponding pixels
            # separately through the ordered ``image_data`` field.
            rendered_prompt = await asyncio.to_thread(
                self.processor.apply_chat_template,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            input_ids = await asyncio.to_thread(
                self.tokenizer.encode,
                rendered_prompt,
                add_special_tokens=False,
            )

        if not isinstance(input_ids, list):
            input_ids = (
                input_ids["input_ids"]
                if hasattr(input_ids, "__getitem__") and "input_ids" in input_ids
                else list(input_ids)
            )
        if timings is not None:
            timings["tokenize_elapsed_s"] = (time.perf_counter_ns() - tokenize_started_ns) / 1e9
            timings["input_tokens"] = len(input_ids)
        if max_input_tokens is not None and len(input_ids) > max_input_tokens:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "context_overflow",
                    "message": f"projection token count {len(input_ids)} exceeds max_input_tokens={max_input_tokens}",
                },
            )

        # Merge per-request sampling params with default config
        default_sampling = {
            "temperature": self.config.genrm_sampling_config.get("temperature", 0.2),
            "top_p": self.config.genrm_sampling_config.get("top_p", 1.0),
            "top_k": self.config.genrm_sampling_config.get("top_k", -1),
            "max_new_tokens": self.config.genrm_sampling_config.get("max_response_len", 1024),
        }
        # Override defaults with per-request params
        if sampling_params:
            request_sampling = dict(sampling_params)
            if "max_response_len" in request_sampling:
                request_sampling["max_new_tokens"] = request_sampling.pop("max_response_len")
            request_sampling.pop("chat_template_kwargs", None)
            allowed_sampling_keys = {"temperature", "top_p", "top_k", "max_new_tokens"}
            default_sampling.update(
                {key: value for key, value in request_sampling.items() if key in allowed_sampling_keys}
            )

        payload = {
            "input_ids": input_ids,
            "sampling_params": default_sampling,
        }
        if image_data:
            payload["image_data"] = image_data

        # Retry transient resets (transport-level or 5xx) with short backoff so
        # bursty colocate contention doesn't surface as a 500; 4xx is a client bug
        # (terminal) and a cancellation (caller timeout) is never retried.
        # Serve→engine retry attempts for transient resets (ReadError / 5xx under
        # bursty colocate contention), absorbing them before they surface as a 500
        # to the client. Set 1 to disable.
        engine_http_elapsed_s = 0.0
        engine_backoff_elapsed_s = 0.0
        for _attempt in range(1, retry_attempts + 1):
            try:
                engine_request_started_ns = time.perf_counter_ns()
                try:
                    resp = await self._http_client.post(url, json=payload)
                finally:
                    engine_http_elapsed_s += (time.perf_counter_ns() - engine_request_started_ns) / 1e9
                resp.raise_for_status()
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                status = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                if (status == 0 or status >= 500) and _attempt < retry_attempts:
                    backoff_started_ns = time.perf_counter_ns()
                    await asyncio.sleep(0.3 * _attempt)
                    engine_backoff_elapsed_s += (time.perf_counter_ns() - backoff_started_ns) / 1e9
                    continue
                raise
        if timings is not None:
            timings["engine_attempt_count"] = _attempt
            timings["engine_http_elapsed_s"] = engine_http_elapsed_s
            timings["engine_backoff_elapsed_s"] = engine_backoff_elapsed_s
            timings["engine_call_elapsed_s"] = (time.perf_counter_ns() - call_started_ns) / 1e9
        return resp.json()

    @app.get("/health")
    async def health(self) -> dict:
        """Health check endpoint."""
        try:
            # Check if genRM engines are healthy
            is_healthy = ray.get(self.genrm_manager.health_check.remote())
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "service": self.role,
            }
        except Exception as e:
            self._logger.error(f"GenRM health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": self.role,
                "error": str(e),
            }

    @app.get("/metrics")
    async def metrics(self, include_gpu_occupancy: bool = False) -> dict:
        """Metrics endpoint.

        Returns monotonically-increasing cumulative counters, never a drain-
        and-reset snapshot: several independent ad-hoc pollers can read this
        endpoint at any cadence and each computes its own interval by diffing
        two reads, without stealing state from one another. The expensive
        cross-actor GPU sampler snapshot is deliberately opt-in via
        ``?include_gpu_occupancy=true``; normal metrics reads never block a
        rollout/judge critical path on a manager RPC.
        """
        self._advance_occupancy()
        payload = {
            "service": self.role,
            "model_path": self.config.genrm_model_path,
            "num_gpus": self.config.genrm_num_gpus,
            "num_engines": self.config.genrm_num_gpus // self.config.genrm_num_gpus_per_engine,
            "occupancy_epoch_s": self._occupancy_epoch_wall_s,
            "now_s": time.time(),
            "cumulative_busy_wall_s": self._busy_wall_s,
            "cumulative_inflight_integral_req_s": self._inflight_integral_req_s,
            "cumulative_served_requests": self._served_requests,
            "current_inflight": self._inflight_requests,
            "current_queued": getattr(self, "_queued_requests", 0),
            "server_max_concurrency": getattr(self, "_server_max_concurrency", None),
            "gpu_occupancy": None,
        }
        if include_gpu_occupancy:
            try:
                snapshot_fn = getattr(self.genrm_manager, "get_gpu_occupancy_snapshot", None)
                if snapshot_fn is not None:
                    payload["gpu_occupancy"] = await asyncio.to_thread(ray.get, snapshot_fn.remote())
            except Exception as exc:
                self._logger.warning(f"{self.role} GPU occupancy snapshot unavailable: {exc}")
        return payload

    def get_genrm_manager(self) -> Any:
        """Get the underlying GenRM manager."""
        return self.genrm_manager

    @app.get("/live")
    async def live(self) -> dict:
        return {"status": "stopping" if self._stopping else "live", "service": self.role}

    @app.get("/ready")
    async def ready(self) -> dict:
        if self._stopping:
            return {"status": "unready", "service": self.role}
        healthy = await asyncio.to_thread(ray.get, self.genrm_manager.health_check.remote())
        if not healthy:
            return {"status": "unready", "service": self.role}
        await self._call_engine(
            [{"role": "user", "content": "Return JSON."}],
            {"temperature": 0.0, "max_new_tokens": 1},
        )
        return {
            "status": "ready",
            "service": self.role,
            "model_path": self.config.genrm_model_path,
            "processor": self.processor is not None,
        }

    @app.post("/stop_service")
    async def stop_service(self) -> dict:
        if self._stopping:
            return {"status": "stopped", "service": self.role}
        self._stopping = True
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=30)
        except asyncio.TimeoutError:
            self._logger.warning(f"{self.role} timed out draining {self._inflight_requests} requests")
        await self._http_client.aclose()
        await asyncio.to_thread(ray.get, self.genrm_manager.dispose.remote())
        return {"status": "stopped", "service": self.role}

    def onload(self) -> None:
        """Load genRM model weights to GPU."""
        self._logger.info("GenRM onload requested")
        ray.get(self.genrm_manager.onload.remote())

    def offload(self) -> None:
        """Offload genRM model weights from GPU."""
        self._logger.info("GenRM offload requested")
        ray.get(self.genrm_manager.offload.remote())


# ── Compatibility wrapper for old imports ─────────────────────────────────
GENRM_ROLE = "genrm"


def register_genrm(config, algo: dict) -> list[str]:
    """Compatibility wrapper; optional-role wiring lives in ``relax.core``."""
    from relax.core.optional_roles import register_genrm as _register_genrm

    return _register_genrm(config, algo)
