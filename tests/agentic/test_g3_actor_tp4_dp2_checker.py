# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
from pathlib import Path


def _load_checker():
    module_path = Path("examples/mobilegym_agentic/scripts/check_g3_actor_tp4_dp2.py")
    spec = importlib.util.spec_from_file_location("g3_actor_checker", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(root: Path) -> None:
    checkpoint = root / "checkpoint"
    iteration = checkpoint / "iter_0000001"
    iteration.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    (iteration / "rank.distcp").write_bytes(b"checkpoint")


def test_g3_actor_checker_requires_two_steps_on_eight_ranks_and_two_hosts(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    timeline = tmp_path / "train_timeline"
    timeline.mkdir()
    for step in range(2):
        rows = []
        for rank in range(8):
            for name in ("critical_path.training_schedule", "critical_path.optimizer_step"):
                rows.append(
                    {
                        "name": name,
                        "args": {
                            "step": step,
                            "global_rank": rank,
                            "clock_host": "node-a" if rank < 4 else "node-b",
                        },
                    }
                )
        (timeline / f"timeline_step_{step}.json").write_text(json.dumps(rows), encoding="utf-8")
    log = tmp_path / "train.log"
    log.write_text("Ray job succeeded\n", encoding="utf-8")

    report = _load_checker().validate(tmp_path, "train", log)

    assert report["training_steps"] == [0, 1]
    assert report["actor_ranks"] == list(range(8))
    assert report["actor_hosts"] == {"node-a": 4, "node-b": 4}


def test_g3_actor_checker_requires_reload_iteration_and_starting_step(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    log = tmp_path / "reload.log"
    log.write_text(
        "loading distributed checkpoint from /checkpoint at iteration 1\n"
        "successfully loaded checkpoint from /checkpoint at iteration 1\n"
        "Actor initialized with starting step 2\n",
        encoding="utf-8",
    )

    report = _load_checker().validate(tmp_path, "reload", log)

    assert report["loaded_iteration"] == 1
    assert report["starting_step"] == 2
