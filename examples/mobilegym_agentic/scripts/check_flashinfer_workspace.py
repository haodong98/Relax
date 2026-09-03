# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-fast probe for the node-local FlashInfer JIT workspace."""

import argparse
import fcntl
import json
import os
import shutil
import socket
from pathlib import Path


_MIN_FREE_BYTES = 4 * 1024**3
_REQUIRED_ROOT = Path("/tmp/relax-flashinfer")


def check_workspace(path: str) -> dict[str, int | str]:
    workspace = Path(path).resolve()
    try:
        workspace.relative_to(_REQUIRED_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"FLASHINFER_WORKSPACE_BASE must be under {_REQUIRED_ROOT}, got {workspace}") from exc

    workspace.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(workspace).free
    if free_bytes < _MIN_FREE_BYTES:
        raise RuntimeError(
            f"FlashInfer workspace has only {free_bytes} free bytes; require at least {_MIN_FREE_BYTES}: {workspace}"
        )

    probe = workspace / ".flock_probe"
    with probe.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return {
        "free_bytes": free_bytes,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "workspace": str(workspace),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not workspace:
        raise RuntimeError("FLASHINFER_WORKSPACE_BASE is required")
    serialized = json.dumps(check_workspace(workspace), sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(f"{args.output.suffix}.{os.getpid()}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
