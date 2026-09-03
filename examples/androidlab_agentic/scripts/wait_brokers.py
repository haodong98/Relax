# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wait for all AndroidLab broker manifests and health endpoints."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def ready_urls(manifest_dir: Path) -> set[str]:
    urls: set[str] = set()
    for path in manifest_dir.glob("broker-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        url = payload.get("broker_url") if isinstance(payload, dict) else None
        if payload.get("schema_version") == "androidlab.broker_manifest.v1" and isinstance(url, str):
            try:
                with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as response:
                    health = json.loads(response.read())
                if response.status == 200 and health.get("ok") is True:
                    urls.add(url)
            except Exception:
                continue
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        urls = ready_urls(args.manifest_dir)
        if len(urls) == args.expected:
            print(json.dumps({"brokers": sorted(urls), "count": len(urls)}))
            return
        time.sleep(1)
    raise TimeoutError(
        f"expected {args.expected} ready AndroidLab brokers, found {len(ready_urls(args.manifest_dir))}"
    )


if __name__ == "__main__":
    main()
