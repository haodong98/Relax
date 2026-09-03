# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU admission tests for managed agent-session runners."""

from types import SimpleNamespace

from relax.agentic.pipeline import runtime


def test_session_runner_reserves_measured_cpu_per_launch_slot(monkeypatch) -> None:
    reservations: list[int] = []

    class _RemoteBuilder:
        def remote(self, **kwargs):
            return SimpleNamespace(**kwargs)

    class _ManagedSessionRunner:
        @staticmethod
        def options(**kwargs):
            reservations.append(kwargs["num_cpus"])
            return _RemoteBuilder()

    class _Resources:
        @staticmethod
        def build_session_runner_pool(*, total_requests, factory):
            assert total_requests == 8
            return factory([3, 5])

    monkeypatch.setattr(runtime, "ManagedSessionRunner", _ManagedSessionRunner)
    monkeypatch.setattr(runtime, "get_agentic_runtime_resources", lambda _args: _Resources())

    pool = runtime.create_managed_session_runner_pool(SimpleNamespace(), total_requests=8)

    assert reservations == [6, 10]
    assert pool is not None
    assert pool.available_launch_slots() == 8
