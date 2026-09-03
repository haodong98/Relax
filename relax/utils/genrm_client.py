# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Client for accessing GenRM service.

This module provides an async client to communicate with the GenRM service for
generative reward model evaluations.
"""

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_serve_url


# Retry policy for transient network failures against the GenRM service.
# Judge calls are cheap (short generations) so a bounded retry with modest
# backoff is safe. 5xx and transport-level errors are retried; 4xx is a
# client bug and re-raised immediately.
_GENRM_MAX_ATTEMPTS = 3
_GENRM_INITIAL_BACKOFF_SEC = 0.5


logger = get_logger(__name__)

# Clients are isolated by role, normalized URL and timeout. A single global
# instance silently routed the second judge to the first judge's URL.
_genrm_clients: dict[tuple[asyncio.AbstractEventLoop | None, str, str, float], "GenRMClient"] = {}


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class GenRMClient:
    """Async client for GenRM service.

    Provides async methods to interact with the GenRM service for
    reward/preference evaluations. Uses httpx.AsyncClient to avoid blocking the
    event loop in async rollout contexts.
    """

    def __init__(self, service_url: Optional[str] = None, timeout: float = 1800.0, *, role: str = "genrm"):
        """Initialize GenRM client.

        Args:
            service_url: URL of the GenRM service. If None, will use get_serve_url("genrm")
            timeout: Request timeout in seconds
        """
        if service_url is None:
            service_url = get_serve_url(role)

        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.role = role
        # Raise pool limits above httpx's default 100 — a reward step can fire
        # thousands of concurrent judge calls and a small pool serializes them;
        # keepalive_expiry >> the default 5s so reused connections aren't reaped
        # mid-burst.
        self._async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048, keepalive_expiry=600),
        )
        # Keep a sync client for health checks during init and non-async contexts
        self._sync_client = httpx.Client(timeout=timeout)

        logger.info(f"GenRMClient initialized with service URL: {self.service_url}")

    async def generate(
        self,
        messages: List[dict],
        sampling_params: Optional[Dict] = None,
    ) -> str:
        """Async generate response for given chat messages.

        Takes OpenAI-style messages as input, sends to GenRM service,
        and returns the raw model response. The caller is responsible
        for formatting the messages and parsing the response.

        Args:
            messages: List of messages with role and content
                (e.g., [{"role": "user", "content": "..."}])
            sampling_params: Optional sampling parameters to override defaults.
                Supported keys: temperature, top_p, top_k, max_new_tokens.
                Example: {"temperature": 0.3, "top_p": 0.9}

        Returns:
            Raw response string from the GenRM model
        """
        payload: Dict = {
            "messages": messages,
        }
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params

        backoff = _GENRM_INITIAL_BACKOFF_SEC
        for attempt in range(1, _GENRM_MAX_ATTEMPTS + 1):
            try:
                return await self.generate_once(payload)
            except httpx.HTTPStatusError as e:
                # 4xx is a client bug — don't retry.
                if e.response.status_code < 500:
                    logger.error(f"GenRM generate 4xx (not retrying): {e}")
                    raise
                if attempt >= _GENRM_MAX_ATTEMPTS:
                    logger.error(f"GenRM generate 5xx after {attempt} attempts: {e}")
                    raise
                logger.warning(
                    f"GenRM generate 5xx attempt {attempt}/{_GENRM_MAX_ATTEMPTS}, retrying in {backoff:.2f}s: {e}"
                )
            except httpx.HTTPError as e:
                # Transport-level errors (ReadError, ConnectError, timeouts, ...).
                if attempt >= _GENRM_MAX_ATTEMPTS:
                    logger.error(f"GenRM generate transport error after {attempt} attempts ({type(e).__name__}): {e}")
                    raise
                logger.warning(
                    f"GenRM generate transport error attempt {attempt}/{_GENRM_MAX_ATTEMPTS} "
                    f"({type(e).__name__}), retrying in {backoff:.2f}s: {e}"
                )
            except Exception as e:
                logger.error(f"Unexpected error in GenRM generate: {e}")
                raise

            await asyncio.sleep(backoff)
            backoff *= 2

    async def generate_once(self, payload: dict[str, Any]) -> str:
        """Make one request without retries (dual-judge executor owns
        policy)."""
        response, _timings = await self.generate_once_profiled(payload)
        return response

    async def generate_once_profiled(self, payload: dict[str, Any]) -> tuple[str, dict[str, float | int]]:
        """Make one request and return server-side timing decomposition."""
        resp = await self._async_client.post(f"{self.service_url}/generate", json=payload)
        resp.raise_for_status()
        result = resp.json()
        if not isinstance(result, dict) or not isinstance(result.get("response"), str):
            raise ValueError("GenRM service response must contain a string 'response'")
        raw_timings = result.get("timings", {})
        if not isinstance(raw_timings, dict):
            raise ValueError("GenRM service response 'timings' must be a mapping")
        timings: dict[str, float | int] = {}
        for key, value in raw_timings.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("GenRM service response 'timings' must contain numeric values")
            timings[key] = value
        return result["response"], timings

    def health_check(self) -> Dict:
        """Check health status of GenRM service (sync, safe to call at init).

        Returns:
            Dictionary with status information
        """
        url = f"{self.service_url}/health"
        try:
            resp = self._sync_client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GenRM health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_metrics(self, *, include_gpu_occupancy: bool = False) -> Dict:
        """Get metrics from GenRM service.

        ``include_gpu_occupancy`` is intentionally opt-in because it causes
        the Serve replica to make a manager RPC for raw sampler snapshots.
        Normal health/occupancy reads stay local to the replica.

        Returns:
            Dictionary with service metrics
        """
        url = f"{self.service_url}/metrics"
        try:
            resp = self._sync_client.get(url, params={"include_gpu_occupancy": include_gpu_occupancy})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GenRM metrics request failed: {e}")
            return {}

    async def aget_metrics(self, *, include_gpu_occupancy: bool = False) -> Dict:
        """Async counterpart of ``get_metrics`` for callers already running in
        an event loop.

        Returns:
            Dictionary with service metrics, or {} on any failure. Never
            raises: this is an observability side-channel and must not be
            able to interrupt the reward/rollout path it is reporting on.
        """
        url = f"{self.service_url}/metrics"
        try:
            resp = await self._async_client.get(url, params={"include_gpu_occupancy": include_gpu_occupancy})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"GenRM async metrics request failed: {e}")
            return {}

    async def aclose(self) -> None:
        """Close both HTTP clients without leaking the async connection
        pool."""
        self._sync_client.close()
        await self._async_client.aclose()

    def close(self):
        """Compatibility close for synchronous callers."""
        self._sync_client.close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._async_client.aclose())
        else:
            loop.create_task(self._async_client.aclose())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_genrm_client(
    service_url: Optional[str] = None,
    timeout: float = 1800.0,
    *,
    role: str = "genrm",
) -> GenRMClient:
    """Get or create a role/URL/timeout-scoped GenRM client.

    Uses module-level keyed caching to avoid creating a new HTTP client per
    call without allowing two dedicated judges to share the wrong URL.

    Args:
        service_url: URL of the GenRM service
        timeout: Request timeout in seconds

    Returns:
        Cached GenRMClient instance for the exact key
    """
    normalized_url = (service_url or get_serve_url(role)).rstrip("/")
    key = (_running_loop(), role, normalized_url, float(timeout))
    client = _genrm_clients.get(key)
    if client is None:
        client = GenRMClient(service_url=normalized_url, timeout=timeout, role=role)
        _genrm_clients[key] = client
    return client


async def close_all_genrm_clients() -> None:
    """Atomically detach and close every cached client."""
    clients = list(_genrm_clients.values())
    _genrm_clients.clear()
    if clients:
        await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)


async def close_genrm_clients_for_current_loop() -> None:
    """Close clients owned by the caller's asyncio loop without touching live
    peers."""
    loop = _running_loop()
    keys = [key for key in _genrm_clients if key[0] is loop]
    clients = [_genrm_clients.pop(key) for key in keys]
    if clients:
        await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)


async def close_all() -> None:
    """Public short alias used by controller/test teardown."""
    await close_all_genrm_clients()
