# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build leak-free AndroidLab task manifests for Relax managed rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "androidlab.relax.v1"
OPERATION_TYPES = frozenset({"operation", "operations"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("PyYAML is required by AndroidLab's task configuration") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AndroidLab config is not a mapping: {path}")
    return payload


def _metric_module(value: Any, path: Path) -> str:
    if not isinstance(value, str) or not value.startswith("evaluation.tasks."):
        raise ValueError(f"invalid metric_func in {path}: {value!r}")
    return value


def load_inventory(androidlab_repo: Path) -> list[dict[str, Any]]:
    config_dir = androidlab_repo / "evaluation" / "config"
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"no AndroidLab configs under {config_dir}")
    inventory: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for config_path in configs:
        config = _load_yaml(config_path)
        app = config.get("APP")
        package = config.get("package")
        if not isinstance(app, str) or not app or not isinstance(package, str) or not package:
            raise ValueError(f"AndroidLab app/package missing from {config_path}")
        tasks = config.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError(f"AndroidLab tasks missing from {config_path}")
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError(f"AndroidLab task must be a mapping in {config_path}")
            task_id = task.get("task_id")
            instruction = task.get("task")
            metric_type = task.get("metric_type")
            if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                raise ValueError(f"duplicate or invalid AndroidLab task ID: {task_id!r}")
            if not isinstance(instruction, str) or not instruction or not isinstance(metric_type, str):
                raise ValueError(f"incomplete AndroidLab task {task_id}")
            task_ids.add(task_id)
            source = config_path.relative_to(androidlab_repo).as_posix()
            normalized_metric_type = "operation" if metric_type in OPERATION_TYPES else metric_type
            inventory.append(
                {
                    "adb_query": task.get("adb_query"),
                    "app": app,
                    "category": task.get("category"),
                    "instruction": instruction,
                    "metric_module": _metric_module(task.get("metric_func"), config_path),
                    "metric_type": normalized_metric_type,
                    "package": package,
                    "source_metric_type": metric_type,
                    "source_path": source,
                    "task_id": task_id,
                    "task_manifest_digest": sha256_bytes(canonical_json(task).encode("utf-8")),
                }
            )
    inventory.sort(key=lambda item: item["task_id"])
    return inventory


def _policy_row(item: dict[str, Any], *, subset: str) -> dict[str, Any]:
    return {
        "input": [{"role": "user", "content": item["instruction"]}],
        "metadata": {
            "app": item["app"],
            "environment": "androidlab",
            "environment_seed": 0,
            "subset": subset,
            "task_id": item["task_id"],
            "task_manifest_digest": item["task_manifest_digest"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def build_dataset(androidlab_repo: Path, output_dir: Path) -> dict[str, Any]:
    inventory = load_inventory(androidlab_repo)
    metric_counts = Counter(item["metric_type"] for item in inventory)
    if len(inventory) != 138 or metric_counts != {"operation": 93, "query_detect": 45}:
        raise ValueError(
            "unexpected AndroidLab inventory; expected 138 tasks with 93 operation-like and 45 query_detect, "
            f"got count={len(inventory)} metrics={dict(metric_counts)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    operation_items = [item for item in inventory if item["metric_type"] == "operation"]
    query_items = [item for item in inventory if item["metric_type"] == "query_detect"]
    _write_jsonl(output_dir / "benchmark_all.jsonl", [_policy_row(item, subset="benchmark_all") for item in inventory])
    _write_jsonl(output_dir / "operation.jsonl", [_policy_row(item, subset="operation") for item in operation_items])
    _write_jsonl(output_dir / "query.jsonl", [_policy_row(item, subset="query") for item in query_items])

    registry = {
        "schema_version": "androidlab.trusted_registry.v1",
        "tasks": {item["task_id"]: dict(item) for item in inventory},
    }
    (output_dir / "trusted_registry.json").write_text(canonical_json(registry) + "\n", encoding="utf-8")
    manifest = {
        "app_counts": dict(sorted(Counter(item["app"] for item in inventory).items())),
        "assignment_digest": sha256_bytes(
            "".join(
                canonical_json({"task_id": item["task_id"], "digest": item["task_manifest_digest"]}) + "\n"
                for item in inventory
            ).encode("utf-8")
        ),
        "counts": {"benchmark_all": len(inventory), "operation": len(operation_items), "query": len(query_items)},
        "schema_version": SCHEMA_VERSION,
        "source_metric_type_anomalies": [
            {"source_metric_type": item["source_metric_type"], "task_id": item["task_id"]}
            for item in inventory
            if item["source_metric_type"] != item["metric_type"]
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--androidlab-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(build_dataset(args.androidlab_repo.resolve(), args.output_dir.resolve())))


if __name__ == "__main__":
    main()
