# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Write one machine-readable post-broker cleanup record per Slurm node."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node-root", type=Path, required=True)
    parser.add_argument("--container-prefix", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()

    containers = subprocess.run(
        ["podman", "ps", "-a", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    matching_containers = sorted(name for name in containers if name.startswith(args.container_prefix))
    manifests = sorted(path.name for path in args.manifest_dir.glob("broker-*.json"))
    result = {
        "broker_manifests": manifests,
        "containers": matching_containers,
        "hostname": socket.gethostname(),
        "node_root_absent": not args.node_root.exists(),
        "schema_version": "waa.cleanup_audit.v1",
        "token_absent": not args.token_file.exists(),
    }
    if matching_containers or manifests or not result["node_root_absent"] or not result["token_absent"]:
        raise RuntimeError(f"incomplete WAA cleanup: {result}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{socket.gethostname()}.json"
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
