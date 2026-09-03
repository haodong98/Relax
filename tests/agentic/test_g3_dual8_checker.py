# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
from pathlib import Path


def _load_checker():
    module_path = Path("examples/mobilegym_agentic/scripts/check_g3_dual8.py")
    spec = importlib.util.spec_from_file_location("g3_dual8_checker", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g3_dual8_checker_accepts_clean_per_turn_trace_and_disjoint_gpu_topology(tmp_path: Path) -> None:
    request = {
        "terminal_orm": {"raw_count": 4, "clean_count": 4},
        "terminal_vlm": {"raw_count": 0, "clean_count": 0},
        "per_turn_vlm": {"raw_count": 4, "clean_count": 4},
    }
    direct = {
        "analysis_mode": "standalone_direct",
        "variants": {
            "per_turn": {
                "observed_benchmark_modes": ["dual"],
                "observed_reasoning_triggers": ["per_turn"],
                "request": request,
                "trajectory": {"raw_count": 4, "clean_count": 4, "fallback_count": 0},
                "group": {
                    "group_finalize_count": 2,
                    "complete_group_count": 2,
                    "missing_terminal_admission_group_count": 0,
                },
            }
        },
    }
    (tmp_path / "direct_report.json").write_text(json.dumps(direct), encoding="utf-8")
    result_dir = tmp_path / "rollout_result" / "train"
    result_dir.mkdir(parents=True)
    reward = {
        "pipeline_status": "success",
        "executor_status": "success",
        "reasoning_execution_trigger": "per_turn",
        "judges": {"answer_accuracy": {"status": "success", "attempt_count": 1}},
        "per_turn_judge_count": 1,
        "per_turn_judges": [
            {
                "status": "success",
                "response_state_hash": "response",
                "observation_state_hash": "observation",
                "judge": {"attempt_count": 1, "invalid_response_count": 0},
            }
        ],
    }
    rows = [{"status": "completed", "latency_trace": {"reward": reward}} for _ in range(4)]
    (result_dir / "0.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    debug = tmp_path / "debug_rollout"
    debug.mkdir()
    (debug / "0.pt").write_bytes(b"capture")
    timeline = tmp_path / "timeline"
    timeline.mkdir()
    (timeline / "timeline_step_0.json").write_text("[]", encoding="utf-8")

    samples = tmp_path / "gpu_samples"
    samples.mkdir()
    specs = [("rollout", rank, 1) for rank in range(4)] + [
        ("judge_accuracy", 0, 2),
        ("judge_multiturn_vlm", 0, 2),
    ]
    uuid_index = 0
    for file_index, (role, rank, gpu_count) in enumerate(specs):
        uuids = [f"GPU-{index}" for index in range(uuid_index, uuid_index + gpu_count)]
        uuid_index += gpu_count
        records = [
            {
                "record_type": "manifest",
                "role": role,
                "engine_rank": rank,
                "num_gpus_per_engine": gpu_count,
                "gpu_uuids": uuids,
                "nvml_enabled": True,
            },
            {
                "record_type": "sample",
                "gpu": [{"util_percent": 1.0}],
                "sglang": {"sglang:gen_throughput": 1.0},
            },
        ]
        (samples / f"{role}_{file_index}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )

    report = _load_checker().validate(tmp_path, "per_turn")

    assert report["status"] == "passed"
    assert report["allocated_gpu_uuids"] == 8
    assert report["per_turn_judge_count"] == 4
