# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Wait for an exact set of WAA broker manifests and probe every endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def probe(manifest_dir: Path, expected: int) -> list[dict[str, str]]:
    manifests = []
    for path in sorted(manifest_dir.glob("broker-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "waa.broker_manifest.v1":
            raise RuntimeError(f"invalid broker manifest schema: {path}")
        manifests.append(payload)
    if len(manifests) != expected:
        raise RuntimeError(f"expected {expected} broker manifests, found {len(manifests)}")
    if len({item["hostname"] for item in manifests}) != expected:
        raise RuntimeError("broker manifests do not cover distinct hosts")
    if len({item["broker_url"] for item in manifests}) != expected:
        raise RuntimeError("broker URLs are not unique")
    for item in manifests:
        with urllib.request.urlopen(item["broker_url"].rstrip("/") + "/health", timeout=5) as response:
            health = json.loads(response.read())
            if response.status != 200 or health.get("status") != "ok":
                raise RuntimeError(f"broker is not healthy: {item['hostname']}")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            manifests = probe(args.manifest_dir, args.expected)
            print(json.dumps({"brokers": manifests}, sort_keys=True))
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"broker readiness timeout: {last_error}")


if __name__ == "__main__":
    main()
