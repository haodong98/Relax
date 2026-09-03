# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
from pathlib import Path


def _load_checker():
    module_path = Path("examples/mobilegym_agentic/scripts/check_g3_rollout8.py")
    spec = importlib.util.spec_from_file_location("g3_rollout8_checker", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g3_rollout8_checker_recovers_two_node_engine_provenance(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    log_path = exp_dir / "driver.log"
    log_rows = []
    for rank in range(8):
        host = "10.0.0.1" if rank < 4 else "10.0.0.2"
        gpu = rank % 4
        log_rows.append(f"bundle {rank:4}, actual_bundle_index: {rank:4}, node: {host}, gpu: {gpu}")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("\n".join(log_rows), encoding="utf-8")

    sample_dir = exp_dir / "gpu_samples"
    sample_dir.mkdir()
    for rank in range(8):
        host = "node-a" if rank < 4 else "node-b"
        records = [
            {
                "record_type": "manifest",
                "role": "rollout",
                "engine_rank": rank,
                "clock_host": host,
                "server_host": "10.0.0.1" if rank < 4 else "10.0.0.2",
                "dist_init_addr": f"{'10.0.0.1' if rank < 4 else '10.0.0.2'}:{15002 + rank * 40}",
                "nvml_enabled": True,
            },
            {
                "record_type": "sample",
                "engine_rank": rank,
                "sglang": {"sglang:gen_throughput": 1.0},
            },
        ]
        (sample_dir / f"rollout_{host}_rank{rank}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )

    result_dir = exp_dir / "rollout_result" / "train"
    result_dir.mkdir(parents=True)
    (result_dir / "0.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "status": "completed" if rank < 4 else "truncated",
                    "sample_index": rank,
                    "response": "generated",
                    "response_token_count": 1,
                }
            )
            for rank in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    debug_dir = exp_dir / "debug_rollout"
    debug_dir.mkdir()
    (debug_dir / "0.pt").write_bytes(b"capture")
    timeline_dir = exp_dir / "timeline"
    timeline_dir.mkdir()
    (timeline_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")

    report = _load_checker().validate(exp_dir, log_path, expected_samples=8)

    assert report["status"] == "passed"
    assert report["active_sampler_ranks"] == list(range(8))
    assert report["committed_samples"] == 8
    assert report["rollout_status_counts"] == {"completed": 4, "truncated": 4}
