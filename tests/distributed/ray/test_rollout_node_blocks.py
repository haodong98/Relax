# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.distributed.ray.placement_group import _validate_rollout_engine_node_blocks
from relax.distributed.ray.utils import propagate_allowlisted_env_vars


def test_rollout_node_blocks_accept_two_contiguous_four_gpu_nodes() -> None:
    mapping = {rank: "10.0.0.1" if rank < 4 else "10.0.0.2" for rank in range(8)}

    _validate_rollout_engine_node_blocks(mapping, rank_offset=0, num_engines_per_node=4)


def test_rollout_node_blocks_reject_fragmented_placement_before_engine_init() -> None:
    mapping = {
        0: "10.0.0.1",
        1: "10.0.0.1",
        2: "10.0.0.2",
        3: "10.0.0.2",
        4: "10.0.0.2",
        5: "10.0.0.2",
        6: "10.0.0.3",
        7: "10.0.0.3",
    }

    with pytest.raises(RuntimeError, match="ambiguous dist-init addresses"):
        _validate_rollout_engine_node_blocks(mapping, rank_offset=0, num_engines_per_node=4)


def test_rollout_engine_runtime_env_copies_explicit_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_PROPAGATE_ENV_VARS", "FLASHINFER_WORKSPACE_BASE, MISSING_VALUE")
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/tmp/relax-flashinfer/job-42")
    env_vars = {"EXISTING": "value"}

    propagate_allowlisted_env_vars(env_vars)

    assert env_vars == {
        "EXISTING": "value",
        "FLASHINFER_WORKSPACE_BASE": "/tmp/relax-flashinfer/job-42",
    }
