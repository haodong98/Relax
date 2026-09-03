# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for SGLang engine telemetry sampler wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


pytest.importorskip("sglang.srt.server_args")

try:
    import relax.backends.sglang.sglang_engine as m
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.sglang.sglang_engine unavailable: {_exc}", allow_module_level=True)


def test_sampler_scrapes_the_bound_sglang_host(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeSampler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(m, "JudgeGpuSampler", _FakeSampler)
    monkeypatch.setenv("RELAX_JUDGE_GPU_SAMPLE_DIR", str(tmp_path))

    engine = m.SGLangEngine.__new__(m.SGLangEngine)
    engine.args = SimpleNamespace(hf_checkpoint="/models/fake")
    engine.node_rank = 0
    engine.rank = 0
    engine.base_gpu_id = 2
    engine.num_gpus_per_engine = 2
    engine.server_host = "10.100.48.23"
    engine.server_port = 17000

    engine._maybe_start_judge_gpu_sampler()

    assert captured["scrape_url"] == "http://10.100.48.23:17000/metrics"
    assert captured["started"] is True
