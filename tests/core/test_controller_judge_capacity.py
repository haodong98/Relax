# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static topology checks for dedicated Judge STRICT_PACK placement."""

import pytest


def _strict_pack_blocks_fit():
    try:
        from relax.core.controller import _strict_pack_blocks_fit as checker
    except (ImportError, AssertionError) as exc:
        pytest.skip(f"relax.core.controller unavailable: {exc}")
    return checker


def test_dedicated_judge_blocks_require_single_node_capacity():
    checker = _strict_pack_blocks_fit()

    assert not checker([4, 2], [4, 1, 1])
    assert checker([4, 2], [4, 2])
    assert checker([4, 2], [8])


def test_dedicated_judge_blocks_consume_capacity_when_co_located():
    checker = _strict_pack_blocks_fit()

    assert not checker([4, 4], [6, 2])
    assert checker([4, 4], [8])
