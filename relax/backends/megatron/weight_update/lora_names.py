# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bridge-independent canonical names for LoRA base and adapter tensors.

These string-only helpers are shared by the optional Megatron Bridge export
path and the checkpoint service.  Keeping them here prevents importing the
optional ``megatron.bridge`` package merely to normalize parameter names.
"""

import re


def base_param_prefix(name: str) -> str:
    """Return the shared prefix for a base LoRA target parameter."""
    return re.sub(r"\.to_wrap\.weight\d*$", "", name)


def adapter_base_prefix(name: str) -> str:
    """Return the shared prefix for either LoRA adapter tensor."""
    return name.replace(".adapter.linear_in.weight", "").replace(".adapter.linear_out.weight", "")
