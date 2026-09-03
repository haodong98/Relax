#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import socket
import subprocess


def main() -> None:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ips = subprocess.run(["hostname", "-I"], check=True, capture_output=True, text=True).stdout.split()
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid = (value.strip() for value in line.split(",", maxsplit=1))
        gpus.append({"index": int(index), "uuid": uuid})
    print(
        json.dumps(
            {
                "schema_version": 1,
                "hostname": socket.gethostname(),
                "ip": socket.gethostbyname(socket.gethostname()),
                "ips": ips,
                "gpus": gpus,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
