# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-fast Qwen3-VL Bridge and runtime dependency gate inside the training
EDF."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import json
import os
import socket
from pathlib import Path


def _cudnn_runtime_version() -> int | None:
    """Return the version from the libcudnn selected by the process loader."""
    library_name = ctypes.util.find_library("cudnn") or "libcudnn.so.9"
    try:
        library = ctypes.CDLL(library_name)
        version = library.cudnnGetVersion
        version.restype = ctypes.c_size_t
        return int(version())
    except (OSError, AttributeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import ray
    import sglang
    import torch
    import transfer_queue
    import transformer_engine
    from megatron.bridge import AutoBridge

    torch_cudnn_version = int(torch.backends.cudnn.version() or 0)
    loaded_cudnn_version = _cudnn_runtime_version()
    if loaded_cudnn_version is not None and torch_cudnn_version and loaded_cudnn_version != torch_cudnn_version:
        raise RuntimeError(
            "cuDNN runtime mismatch: "
            f"torch={torch_cudnn_version}, loaded={loaded_cudnn_version}, "
            f"CUDNN_LIB_DIR={os.environ.get('CUDNN_LIB_DIR', '')}"
        )

    bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    contract = {
        "mrope_section": list(provider.mrope_section),
        "position_embedding_type": str(provider.position_embedding_type),
        "rotary_base": int(provider.rotary_base),
        "share_embeddings_and_output_weights": bool(provider.share_embeddings_and_output_weights),
    }
    expected = {
        "mrope_section": [24, 20, 20],
        "position_embedding_type": "mrope",
        "rotary_base": 5_000_000,
        "share_embeddings_and_output_weights": True,
    }
    if contract != expected:
        raise RuntimeError(f"Qwen3-VL Bridge provider contract mismatch: {contract}")

    transfer_queue_file = getattr(transfer_queue, "__file__", None)
    result = {
        "contract": contract,
        "cudnn_loaded_version": loaded_cudnn_version,
        "cudnn_torch_version": torch_cudnn_version,
        "hostname": socket.gethostname(),
        "model": str(Path(args.model).resolve()),
        "ray_version": ray.__version__,
        "schema_version": "waa.training_env_preflight.v1",
        "sglang_version": getattr(sglang, "__version__", None) or importlib.metadata.version("sglang"),
        "transfer_queue_module": str(Path(transfer_queue_file).resolve()) if transfer_queue_file else "namespace",
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{socket.gethostname()}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
