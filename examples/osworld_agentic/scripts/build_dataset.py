# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a leakage-resistant Relax manifest from an OSWorld checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_VERSION = "osworld-relax-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_file(root: Path, domain: str, task_id: str) -> Path:
    candidates = (
        root / "evaluation_examples" / "examples" / domain / f"{task_id}.json",
        root / "evaluation_examples" / "examples_windows" / domain / f"{task_id}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"OSWorld task file not found: {domain}/{task_id}")


def load_tasks(repo: Path) -> list[dict[str, Any]]:
    index_path = repo / "evaluation_examples" / "test_all.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tasks = []
    for domain, task_ids in sorted(index.items()):
        for task_id in task_ids:
            path = _task_file(repo, domain, task_id)
            config = json.loads(path.read_text(encoding="utf-8"))
            config["id"] = task_id
            instruction = config.get("instruction") or config.get("task")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"OSWorld task has no instruction: {domain}/{task_id}")
            tasks.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "task_config": config,
                    "task_path": path.relative_to(repo).as_posix(),
                    "task_manifest_digest": digest(path.read_bytes()),
                }
            )
    if not tasks:
        raise ValueError("OSWorld inventory is empty")
    return tasks


def assign_splits(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_domain[task["domain"]].append(task)
    result = []
    for domain, domain_tasks in sorted(by_domain.items()):
        ordered = sorted(domain_tasks, key=lambda item: item["task_id"])
        count = len(ordered)
        train_count = max(1, round(count * 0.7))
        dev_count = max(0, round(count * 0.15))
        for index, task in enumerate(ordered):
            split = "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
            result.append({**task, "split": split, "split_version": SPLIT_VERSION})
    return result


def build_dataset(repo: Path, output_dir: Path) -> None:
    assignments = assign_splits(load_tasks(repo))
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = []
        for item in assignments:
            if item["split"] != split:
                continue
            rows.append(
                {
                    "input": [
                        {
                            "role": "user",
                            "content": item["task_config"].get("instruction", item["task_config"].get("task")),
                        }
                    ],
                    "metadata": {
                        "domain": item["domain"],
                        "environment": "osworld",
                        "environment_seed": 0,
                        "split": split,
                        "split_version": SPLIT_VERSION,
                        "task_id": item["task_id"],
                        "task_manifest_digest": item["task_manifest_digest"],
                    },
                }
            )
        (output_dir / f"{split}.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
        )
    smoke = next(item for item in assignments if item["domain"] == "os" and item["split"] == "train")
    smoke_row = {
        "input": [
            {"role": "user", "content": smoke["task_config"].get("instruction", smoke["task_config"].get("task"))}
        ],
        "metadata": {
            "domain": smoke["domain"],
            "environment": "osworld",
            "environment_seed": 0,
            "split": "train",
            "split_version": SPLIT_VERSION,
            "task_id": smoke["task_id"],
            "task_manifest_digest": smoke["task_manifest_digest"],
        },
    }
    (output_dir / "smoke_train.jsonl").write_text(canonical_json(smoke_row) + "\n", encoding="utf-8")
    registry = {
        "schema_version": "osworld.trusted_registry.v1",
        "split_version": SPLIT_VERSION,
        "tasks": {
            item["task_id"]: {
                key: item[key]
                for key in ("domain", "split", "split_version", "task_config", "task_path", "task_manifest_digest")
            }
            for item in assignments
        },
    }
    (output_dir / "trusted_registry.json").write_text(canonical_json(registry) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "osworld.dataset_manifest.v1",
        "assignment_digest": digest(
            "".join(f"{item['domain']}\0{item['task_id']}\0{item['split']}\n" for item in assignments).encode()
        ),
        "counts": {split: sum(item["split"] == split for item in assignments) for split in ("train", "dev", "test")},
        "task_count": len(assignments),
    }
    (output_dir / "dataset_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_dataset(args.osworld_repo.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
