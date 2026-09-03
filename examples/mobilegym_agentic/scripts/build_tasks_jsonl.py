# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a Relax --prompt-data jsonl from MobileGym's own task split files.

Agentic mode does not need the dataset to carry the task instruction text or
any tool schema (see examples/mobilegym_agentic/README.md) -- MobileGym's own
``bench_env.run`` renders each task's instruction from its templates and owns
the whole multi-turn loop. Each row only needs a placeholder ``input`` (mini
_swe_agent's dummy-data pattern) plus a ``task_id`` for app/agent.py to select.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read_split(split_path: Path) -> list[str]:
    task_ids = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            task_ids.append(line)
    return task_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobilegym-repo", type=Path, required=True, help="Path to the mobilegym checkout.")
    parser.add_argument("--split", default="train", choices=["train", "test", "payment", "high_risk"])
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each task_id N times (e.g. for GRPO group diversity across rollout_batch_size).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_path = args.mobilegym_repo / "bench_env" / "splits" / f"{args.split}.txt"
    task_ids = _read_split(split_path)
    if not task_ids:
        raise ValueError(f"No task ids found in {split_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for task_id in task_ids:
            for repeat_index in range(args.repeat):
                # Keep the task instance stable across the GRPO samples derived
                # from this row while making repeated source rows reproducible.
                # The wrapper passes this value to MobileGym verbatim.
                sample_seed = int.from_bytes(
                    hashlib.sha256(f"{task_id}:{repeat_index}".encode("utf-8")).digest()[:4],
                    byteorder="big",
                )
                row = {
                    "input": [{"role": "user", "content": ""}],
                    "metadata": {"task_id": task_id, "sample_seed": sample_seed},
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(task_ids) * args.repeat} rows ({len(task_ids)} unique task_id x {args.repeat}) to {args.output}"
    )


if __name__ == "__main__":
    main()
