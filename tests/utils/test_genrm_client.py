# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import httpx
import pytest

from relax.utils import genrm_client


class _Client:
    def __init__(self, service_url, timeout, *, role):
        self.service_url = service_url
        self.timeout = timeout
        self.role = role
        self.closed = False

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_genrm_clients_are_cached_by_role_url_and_timeout(monkeypatch):
    await genrm_client.close_all()
    monkeypatch.setattr(genrm_client, "GenRMClient", _Client)
    first = genrm_client.get_genrm_client("http://host/a/", 3, role="judge_accuracy")
    same = genrm_client.get_genrm_client("http://host/a", 3.0, role="judge_accuracy")
    other_role = genrm_client.get_genrm_client("http://host/a", 3, role="judge_multiturn_vlm")
    other_timeout = genrm_client.get_genrm_client("http://host/a", 4, role="judge_accuracy")
    assert first is same
    assert other_role is not first
    assert other_timeout is not first
    await genrm_client.close_all()
    assert first.closed and other_role.closed and other_timeout.closed


def test_genrm_clients_are_not_reused_across_event_loops(monkeypatch):
    asyncio.run(genrm_client.close_all())
    monkeypatch.setattr(genrm_client, "GenRMClient", _Client)

    async def get_client():
        return genrm_client.get_genrm_client("http://host/a", 3, role="judge_accuracy")

    first = asyncio.run(get_client())
    second = asyncio.run(get_client())
    asyncio.run(genrm_client.close_all())

    assert first is not second


@pytest.mark.asyncio
async def test_generate_once_profiled_returns_validated_server_timings():
    client = genrm_client.GenRMClient("http://judge", role="judge_accuracy")

    async def handler(request):
        return httpx.Response(
            200,
            request=request,
            json={"response": "ok", "timings": {"engine_http_elapsed_s": 0.25, "input_tokens": 8}},
        )

    await client._async_client.aclose()
    client._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response, timings = await client.generate_once_profiled({"messages": []})
    finally:
        await client.aclose()

    assert response == "ok"
    assert timings == {"engine_http_elapsed_s": 0.25, "input_tokens": 8}


@pytest.mark.asyncio
async def test_async_metrics_passes_gpu_snapshot_opt_in_as_a_query_parameter():
    client = genrm_client.GenRMClient("http://judge", role="judge_accuracy")

    async def handler(request):
        assert request.url.params["include_gpu_occupancy"] == "true"
        return httpx.Response(200, request=request, json={"gpu_occupancy": ["snapshot"]})

    await client._async_client.aclose()
    client._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await client.aget_metrics(include_gpu_occupancy=True)
    finally:
        await client.aclose()

    assert result == {"gpu_occupancy": ["snapshot"]}
