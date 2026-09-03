# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a deterministic, leakage-resistant Relax dataset from WAA public
tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_VERSION = "waa-relax-154-v1"
SMOKE_DOMAIN = "notepad"
SMOKE_TASK_ID = "366de66e-cbae-4d72-b042-26390db2b145-WOS"
TARGET_COUNTS = {"train": 108, "dev": 23, "test": 23}
DOMAIN_FORWARD_PORTS = {
    "chrome": [5000, 9222],
    "msedge": [5000, 9222],
    "vlc": [5000, 8080],
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hamilton(capacities: dict[str, int], target: int) -> dict[str, int]:
    total = sum(capacities.values())
    quotas = {name: target * count // total for name, count in capacities.items()}
    remainder = target - sum(quotas.values())
    order = sorted(capacities, key=lambda name: (-(target * capacities[name] % total), name))
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _load_inventory(waa_repo: Path) -> list[dict[str, Any]]:
    client_root = waa_repo / "src/win-arena-container/client"
    examples_root = client_root / "evaluation_examples_windows/examples"
    index = json.loads((client_root / "evaluation_examples_windows/test_all.json").read_text(encoding="utf-8"))
    inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for domain, task_ids in index.items():
        for task_id in task_ids:
            key = (domain, task_id)
            if key in seen:
                raise ValueError(f"duplicate task in WAA inventory: {domain}/{task_id}")
            seen.add(key)
            task_path = examples_root / domain / f"{task_id}.json"
            config = json.loads(task_path.read_text(encoding="utf-8"))
            source_task_id = config.get("id")
            # Six files in the pinned public corpus have stale/case-mismatched
            # internal ids. The canonical identity is test_all.json + path;
            # normalize only the trusted copy and retain the anomaly for audit.
            config["id"] = task_id
            inventory.append(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "task_path": task_path.relative_to(waa_repo).as_posix(),
                    "task_config": config,
                    "task_manifest_digest": sha256_bytes(task_path.read_bytes()),
                    "source_task_id": source_task_id,
                }
            )
    if len(inventory) != 154:
        raise ValueError(f"expected the pinned 154-task WAA inventory, found {len(inventory)}")
    return inventory


def assign_splits(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in inventory:
        by_domain[item["domain"]].append(item)
    train_quota = _hamilton({name: len(items) for name, items in by_domain.items()}, TARGET_COUNTS["train"])
    remaining = {name: len(items) - train_quota[name] for name, items in by_domain.items()}
    dev_quota = _hamilton(remaining, TARGET_COUNTS["dev"])

    assignments: list[dict[str, Any]] = []
    for domain, items in sorted(by_domain.items()):
        ordered = sorted(
            items,
            key=lambda item: (
                hashlib.sha256(f"{SPLIT_VERSION}\0{domain}\0{item['task_id']}".encode()).hexdigest(),
                item["task_id"],
            ),
        )
        if domain == SMOKE_DOMAIN:
            smoke = next(item for item in ordered if item["task_id"] == SMOKE_TASK_ID)
            ordered = [smoke] + [item for item in ordered if item is not smoke]
        for index, item in enumerate(ordered):
            split = (
                "train"
                if index < train_quota[domain]
                else "dev"
                if index < train_quota[domain] + dev_quota[domain]
                else "test"
            )
            assignments.append({**item, "split": split, "split_version": SPLIT_VERSION})
    return sorted(assignments, key=lambda item: (item["domain"], item["task_id"]))


def assignment_digest(assignments: list[dict[str, Any]]) -> str:
    payload = "".join(
        canonical_json(
            {
                "domain": item["domain"],
                "split": item["split"],
                "split_version": item["split_version"],
                "task_id": item["task_id"],
            }
        )
        + "\n"
        for item in assignments
    )
    return sha256_bytes(payload.encode())


def _policy_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": [{"role": "user", "content": item["task_config"]["instruction"]}],
        "metadata": {
            "domain": item["domain"],
            "environment_seed": 0,
            "split": item["split"],
            "split_version": item["split_version"],
            "task_id": item["task_id"],
            "task_manifest_digest": item["task_manifest_digest"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _normalized_instruction(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _capability_manifest(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    """Record static readiness facts without claiming guest execution
    readiness."""

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_instruction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in assignments:
        by_domain[item["domain"]].append(item)
        by_instruction[_normalized_instruction(item["task_config"]["instruction"])].append(item)
    return {
        "schema_version": "waa.capability_manifest.v1",
        "domains": {
            domain: {
                "required_guest_forward_ports": DOMAIN_FORWARD_PORTS.get(domain, [5000]),
                "status": "hardware_gate_required",
                "task_count": len(items),
            }
            for domain, items in sorted(by_domain.items())
        },
        "instruction_clusters": [
            {
                "instruction": instruction,
                "task_ids": [item["task_id"] for item in sorted(items, key=lambda item: item["task_id"])],
            }
            for instruction, items in sorted(by_instruction.items())
            if len(items) > 1
        ],
        "notes": [
            "hardware_gate_required means ingestion is complete but setup/getter/evaluator guest execution is unverified",
            "instruction_clusters are exact normalized duplicates only; curate semantic task families before reporting generalization",
        ],
        "task_status": {item["task_id"]: "ingested" for item in assignments},
    }


def build_dataset(waa_repo: Path, output_dir: Path) -> dict[str, Any]:
    assignments = assign_splits(_load_inventory(waa_repo))
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in TARGET_COUNTS:
        _write_jsonl(
            output_dir / f"{split}.jsonl", [_policy_row(item) for item in assignments if item["split"] == split]
        )
    smoke = next(item for item in assignments if item["task_id"] == SMOKE_TASK_ID and item["domain"] == SMOKE_DOMAIN)
    _write_jsonl(output_dir / "smoke_train.jsonl", [_policy_row(smoke)])

    registry = {
        "schema_version": "waa.trusted_registry.v1",
        "split_version": SPLIT_VERSION,
        "tasks": {
            item["task_id"]: {
                "domain": item["domain"],
                "split": item["split"],
                "task_manifest_digest": item["task_manifest_digest"],
                "task_path": item["task_path"],
                "task_config": item["task_config"],
                "source_task_id": item["source_task_id"],
            }
            for item in assignments
        },
    }
    (output_dir / "trusted_registry.json").write_text(canonical_json(registry) + "\n", encoding="utf-8")

    counts = {split: sum(item["split"] == split for item in assignments) for split in TARGET_COUNTS}
    manifest = {
        "assignment_digest": assignment_digest(assignments),
        "counts": counts,
        "domain_counts": {
            domain: {
                split: sum(item["domain"] == domain and item["split"] == split for item in assignments)
                for split in TARGET_COUNTS
            }
            for domain in sorted({item["domain"] for item in assignments})
        },
        "schema_version": "waa.split_manifest.v1",
        "smoke_task": {"domain": SMOKE_DOMAIN, "task_id": SMOKE_TASK_ID},
        "split_version": SPLIT_VERSION,
        "source_id_anomalies": [
            {
                "domain": item["domain"],
                "source_task_id": item["source_task_id"],
                "task_id": item["task_id"],
            }
            for item in assignments
            if item["source_task_id"] != item["task_id"]
        ],
    }
    (output_dir / "split_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    capability_manifest = _capability_manifest(assignments)
    (output_dir / "capability_manifest.json").write_text(canonical_json(capability_manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waa-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_dataset(args.waa_repo.resolve(), args.output_dir.resolve())
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
