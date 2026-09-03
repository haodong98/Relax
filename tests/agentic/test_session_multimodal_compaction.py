# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.agentic.session.service import _copy_training_multimodal_inputs


def test_copy_training_multimodal_inputs_compacts_only_floating_pixels_for_bf16():
    pixel_values = torch.randn(4, 8, dtype=torch.float32)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    metadata = {"source": ["screen-1"]}

    copied = _copy_training_multimodal_inputs(
        {"pixel_values": pixel_values, "image_grid_thw": grid, "metadata": metadata},
        use_bf16=True,
    )

    assert copied is not None
    assert copied["pixel_values"].dtype == torch.bfloat16
    torch.testing.assert_close(copied["pixel_values"].float(), pixel_values.to(torch.bfloat16).float())
    assert copied["image_grid_thw"].dtype == torch.int64
    assert torch.equal(copied["image_grid_thw"], grid)
    assert copied["image_grid_thw"] is not grid
    assert copied["metadata"] == metadata
    assert copied["metadata"] is not metadata


def test_copy_training_multimodal_inputs_preserves_float32_when_bf16_is_disabled():
    pixel_values = torch.randn(2, 4, dtype=torch.float32)

    copied = _copy_training_multimodal_inputs({"pixel_values": pixel_values}, use_bf16=False)

    assert copied is not None
    assert copied["pixel_values"].dtype == torch.float32
    assert copied["pixel_values"] is not pixel_values
