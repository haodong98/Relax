# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cache trusted WAA cloud-file evaluator assets before a GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _cloud_assets(value: Any) -> list[tuple[str, str]]:
    assets: list[tuple[str, str]] = []
    if isinstance(value, dict):
        if value.get("type") == "cloud_file":
            paths = value.get("path") if isinstance(value.get("path"), list) else [value.get("path")]
            destinations = value.get("dest") if isinstance(value.get("dest"), list) else [value.get("dest")]
            if len(paths) != len(destinations):
                raise ValueError("cloud_file path/dest arity mismatch")
            for url, destination in zip(paths, destinations):
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise ValueError("trusted cloud_file URL must use HTTPS")
                if not isinstance(destination, str) or Path(destination).name != destination:
                    raise ValueError("cloud_file destination must be a basename")
                assets.append((url, destination))
        else:
            for nested in value.values():
                assets.extend(_cloud_assets(nested))
    elif isinstance(value, list):
        for nested in value:
            assets.extend(_cloud_assets(nested))
    return assets


def cache_assets(registry_path: Path, output_dir: Path, task_ids: set[str] | None = None) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for task_id, task in sorted(registry["tasks"].items()):
        if task_ids is not None and task_id not in task_ids:
            continue
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        evaluator = task["task_config"].get("evaluator", {})
        for url, destination in _cloud_assets(evaluator.get("expected")):
            output = task_dir / destination
            if not output.exists():
                temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
                with urllib.request.urlopen(url, timeout=120) as response:
                    if response.status != 200:
                        raise RuntimeError(f"asset download returned HTTP {response.status}")
                    temporary.write_bytes(response.read())
                temporary.replace(output)
            content = output.read_bytes()
            if not content:
                raise RuntimeError(f"cached asset is empty: {output}")
            records.append(
                {
                    "destination": destination,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "task_id": task_id,
                    "url": url,
                }
            )
    manifest = {"assets": records, "schema_version": "waa.asset_cache.v1"}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=None)
    args = parser.parse_args()
    result = cache_assets(args.registry, args.output_dir, None if args.task_id is None else set(args.task_id))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
