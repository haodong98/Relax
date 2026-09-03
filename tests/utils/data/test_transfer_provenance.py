# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.data.transfer_provenance import (
    TRANSFER_PROVENANCE_KEY,
    build_transfer_provenance,
    extract_data_wait_provenance,
)


class _BatchMeta:
    def __init__(self, custom_meta, partition_ids):
        self._custom_meta = custom_meta
        self.partition_ids = partition_ids

    def get_all_custom_meta(self):
        return self._custom_meta


def _row(sample_index, group_key):
    return {
        TRANSFER_PROVENANCE_KEY: build_transfer_provenance(
            sample_key=["session", sample_index, None],
            group_key=group_key,
            partition_id="train_7",
            weight_versions=[f"policy-{sample_index}"],
        )
    }


def test_extract_data_wait_provenance_preserves_selected_row_order_and_deduplicates_groups():
    group_a = [["session", 0, None], ["session", 1, None]]
    group_b = [["session", 2, None], ["session", 3, None]]
    meta = _BatchMeta(
        [_row(3, group_b), _row(1, group_a), _row(2, group_b)],
        ["train_7", "train_7", "train_7"],
    )

    attributes = extract_data_wait_provenance(
        meta,
        partition_id="train_7",
        task_name="actor_train",
        dp_rank=1,
        batch_index=4,
        required=True,
    )

    assert attributes["returned_sample_keys"] == [
        ["session", 3, None],
        ["session", 1, None],
        ["session", 2, None],
    ]
    assert attributes["returned_group_keys"] == [group_b, group_a]
    assert attributes["returned_weight_versions"] == [["policy-3"], ["policy-1"], ["policy-2"]]
    assert attributes["trainer_batch_id"] == "train_7:actor_train:4"


def test_extract_data_wait_provenance_fails_closed_for_dual_batch_without_keys():
    meta = SimpleNamespace(get_all_custom_meta=lambda: [{}], partition_ids=["train_7"])
    with pytest.raises(ValueError, match="missing 'relax_provenance'"):
        extract_data_wait_provenance(
            meta,
            partition_id="train_7",
            task_name="actor_train",
            dp_rank=0,
            batch_index=0,
            required=True,
        )
