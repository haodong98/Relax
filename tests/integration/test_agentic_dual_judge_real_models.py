# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in real-model smoke for the two dedicated local judges."""

import asyncio
import base64
import io
import os
import time
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pytest


_REQUIRED = (
    "RELAX_TEST_QWEN3_32B",
    "RELAX_TEST_QWEN25_VL",
    "RELAX_TEST_ACCURACY_GPUS",
    "RELAX_TEST_VLM_GPUS",
)
_MISSING = [name for name in _REQUIRED if not os.environ.get(name)]


@dataclass
class _Node:
    kind: str
    state_hash: str
    rollout_id: int
    messages_delta: list[dict]
    backend_image_data_delta: list[str] = field(default_factory=list)
    backend_audio_data_delta: list[str] = field(default_factory=list)
    backend_video_data_delta: list[str] = field(default_factory=list)
    status: str | None = None


def _data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _png(color: str) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(output, format="PNG")
    return output.getvalue()


def _judge_config(accuracy_model: str, vlm_model: str, accuracy_gpus: int, vlm_gpus: int):
    from relax.utils.judge_config import parse_judge_services_config

    def service(model: str, gpus: int, port: int, *, media: bool = False) -> dict:
        value = {
            "model_path": model,
            "num_gpus_per_engine": gpus,
            "engine_config": {"max_context_len": 32768},
            "sampling_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_response_len": 256,
            },
            "max_input_tokens": 32512,
            "max_output_tokens": 256,
            "timeout_s": 300,
            "max_attempts": 3,
            "max_concurrency": 1,
            "port_base": port,
        }
        if media:
            value.update(
                max_media_items=16,
                max_media_total_bytes=64 * 1024 * 1024,
                max_pixels_per_item=4 * 1024 * 1024,
            )
        else:
            value["sampling_config"]["chat_template_kwargs"] = {"enable_thinking": False}
        return value

    return parse_judge_services_config(
        {
            "schema_version": 1,
            "max_group_replacements_per_step": 8,
            "answer_accuracy": service(accuracy_model, accuracy_gpus, 16000),
            "multi_turn_reasoning": service(vlm_model, vlm_gpus, 17000, media=True),
        }
    )


def _runtime_args(judge_services, visible_gpus: int) -> Namespace:
    return Namespace(
        judge_services=judge_services,
        enable_affinity=False,
        num_gpus_per_node=visible_gpus,
        debug_train_only=False,
        fully_async=False,
        rollout_num_gpus=0,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=1,
        use_distributed_post=False,
        sglang_dp_size=1,
        seed=1,
        rollout_external=False,
        use_rollout_routing_replay=False,
        fp16=False,
        warm_hf_checkpoint_page_cache=False,
    )


def _context(final_answer: str):
    from relax.agentic.session.reward_context import build_reward_context

    lineage = [
        _Node(
            "obs",
            "root",
            0,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is 2 + 2? Inspect the red image before answering."},
                        {"type": "image_url"},
                    ],
                }
            ],
            [_data_uri(_png("red"))],
        ),
        _Node(
            "resp",
            "tool-call",
            0,
            [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should inspect the crop and then answer the arithmetic question.",
                    "tool_calls": [{"id": "crop-1", "function": {"name": "crop", "arguments": "{}"}}],
                }
            ],
        ),
        _Node(
            "obs",
            "tool-result",
            0,
            [
                {
                    "role": "tool",
                    "tool_call_id": "crop-1",
                    "content": [
                        {"type": "text", "text": "The crop remains a solid-color image."},
                        {"type": "image_url"},
                    ],
                }
            ],
            [_data_uri(_png("blue"))],
        ),
        _Node(
            "resp",
            "final",
            0,
            [{"role": "assistant", "content": final_answer}],
            status="completed",
        ),
    ]
    return build_reward_context(
        session_id=f"real-smoke-{final_answer}",
        group_index=0,
        sample_index=0,
        leaf_state_hash="final",
        lineage=lineage,
        reference_answer="4",
        static_metadata={"data_source": "dual-judge-real-smoke"},
        tools=[{"type": "function", "function": {"name": "crop", "parameters": {"type": "object"}}}],
        terminal_status="completed",
        remove_sample=False,
    )


def _wait_until(predicate, timeout_s: float = 30) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.25)
    return False


@pytest.mark.gpu
@pytest.mark.skipif(bool(_MISSING), reason=f"dual-judge real-model smoke missing: {', '.join(_MISSING)}")
def test_agentic_dual_judge_real_models():
    import ray
    import torch
    from ray import serve
    from ray.util.placement_group import placement_group_table

    from relax.components.genrm import GenRM
    from relax.core.service import Service
    from relax.engine.rewards.dual_agentic_judge import DualJudgeExecutor
    from relax.utils.genrm_client import close_all_genrm_clients
    from relax.utils.health_system import HealthStatus
    from relax.utils.types import Sample

    accuracy_model = os.environ["RELAX_TEST_QWEN3_32B"]
    vlm_model = os.environ["RELAX_TEST_QWEN25_VL"]
    missing_paths = [path for path in (accuracy_model, vlm_model) if not Path(path).exists()]
    if missing_paths:
        pytest.skip(f"dual-judge model path does not exist: {', '.join(missing_paths)}")
    accuracy_gpus = int(os.environ["RELAX_TEST_ACCURACY_GPUS"])
    vlm_gpus = int(os.environ["RELAX_TEST_VLM_GPUS"])
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < accuracy_gpus + vlm_gpus:
        pytest.skip(f"dual-judge smoke needs {accuracy_gpus + vlm_gpus} GPUs, only {visible_gpus} are visible")

    judge_services = _judge_config(accuracy_model, vlm_model, accuracy_gpus, vlm_gpus)
    args = _runtime_args(judge_services, visible_gpus)
    services: list[Service] = []
    placement_groups = []
    health = None
    ray.init(ignore_reinit_error=True)
    try:
        health = HealthStatus.remote()
        for role, num_gpus in (("judge_accuracy", accuracy_gpus), ("judge_multiturn_vlm", vlm_gpus)):
            service = Service(GenRM, role, health, args, num_gpus=num_gpus)
            services.append(service)
            placement_groups.append(service.pgs[0])
        readiness = [service.wait_ready(timeout=1800) for service in services]
        assert [item["service"] for item in readiness] == ["judge_accuracy", "judge_multiturn_vlm"]
        assert readiness[1]["processor"] is True

        async def score_both():
            executor = DualJudgeExecutor(args)
            try:
                good_result = await executor.score(Sample(reward_context=_context("4")))
                bad_result = await executor.score(Sample(reward_context=_context("5")))
                return good_result, bad_result
            finally:
                await close_all_genrm_clients()

        good, bad = asyncio.run(score_both())
        assert good["answer_accuracy"] == 1
        assert bad["answer_accuracy"] == 0
        assert 0.0 <= good["multi_turn_reasoning"] <= 1.0
        assert 0.0 <= bad["multi_turn_reasoning"] <= 1.0
        assert good["_schema_version"] == "relax.composite_reward.v1"
    finally:
        for service in reversed(services):
            service.shutdown_owned()
        if health is not None:
            ray.kill(health)

        assert _wait_until(lambda: not serve.status().applications)
        for role in ("judge_accuracy", "judge_multiturn_vlm"):
            manager_name = f"relax_genrm_manager_{role}"

            def manager_is_gone(name=manager_name):
                try:
                    ray.get_actor(name)
                except ValueError:
                    return True
                return False

            assert _wait_until(manager_is_gone)
        for pg in placement_groups:
            assert _wait_until(lambda group=pg: placement_group_table(group).get("state") == "REMOVED")
        serve.shutdown()
        ray.shutdown()
