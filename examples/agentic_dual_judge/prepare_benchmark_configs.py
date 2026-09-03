# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create the two real dual-training Judge configs for the direct
experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRIGGERS = ("terminal_once", "per_turn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name("judge_services.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for trigger in TRIGGERS:
        config = dict(base)
        config["benchmark_mode"] = "dual"
        config["reasoning_trigger"] = trigger
        output = args.output_dir / f"judge_services_dual_{trigger}.json"
        output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
