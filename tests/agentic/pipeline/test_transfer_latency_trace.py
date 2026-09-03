# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.agentic.pipeline.transfer import TransferDomain, _transfer_batch_to_data_system
from relax.utils.data.transfer_provenance import TRANSFER_PROVENANCE_KEY, TRANSFER_PROVENANCE_SCHEMA_VERSION
from relax.utils.types import Sample


@pytest.mark.asyncio
async def test_transfer_trace_uses_actual_backfill_partition(monkeypatch):
    async def fake_transfer(**_kwargs):
        return []

    monkeypatch.setattr("relax.agentic.pipeline.transfer._transfer_batch_to_data_system", fake_transfer)
    args = Namespace(
        rollout_batch_size=2,
        over_sampling_batch_size=0,
        n_samples_per_prompt=1,
        colocate=True,
        global_batch_size=2,
        num_iters_per_train_update=1,
        wandb_always_use_train_step=False,
    )
    domain = TransferDomain(args=args, data_system_client=object())
    domain.rollout_id = 3
    previous = Sample(metadata={"agentic_trace": {"events": {}, "reward": {}}})
    current = Sample(metadata={"agentic_trace": {"events": {}, "reward": {}}})

    await domain._dispatch_transfer_batch(groups=[[previous]], partition_rollout_id=2)
    await domain._dispatch_transfer_batch(groups=[[current]], partition_rollout_id=3)

    previous_trace = previous.metadata["agentic_trace"]["reward"]
    current_trace = current.metadata["agentic_trace"]["reward"]
    assert previous_trace["collection_rollout_id"] == 3
    assert previous_trace["train_partition_rollout_id"] == 2
    assert previous_trace["train_partition_step"] == 2
    assert current_trace["train_partition_rollout_id"] == 3
    assert current_trace["train_partition_step"] == 3


@pytest.mark.asyncio
async def test_transfer_custom_meta_carries_ordered_dual_judge_provenance(monkeypatch):
    class _RolloutBatch(dict):
        def numel(self):
            return len(self["total_lengths"])

        def __repr__(self):
            raise AssertionError("the transfer hot path must not format tensor batches")

    class _Client:
        async def async_put(self, **kwargs):
            self.kwargs = kwargs

    samples = []
    group_key = [["session-1", 0, None], ["session-1", 1, None]]
    for index in range(2):
        samples.append(
            Sample(
                session_id="session-1",
                index=index,
                reward=1.0,
                weight_versions=["policy-v3"],
                metadata={
                    "agentic_trace": {
                        "reward": {
                            "sample_key": group_key[index],
                            "group_key": group_key,
                            "collection_rollout_id": 4,
                            "train_partition_rollout_id": 3,
                            "train_partition_step": 3,
                        }
                    }
                },
            )
        )

    monkeypatch.setattr(
        "relax.utils.utils.convert_samples_to_train_data",
        lambda _args, _samples: _RolloutBatch(total_lengths=[11, 22]),
    )
    client = _Client()
    await _transfer_batch_to_data_system(
        args=Namespace(rm_type="dual-agentic-judge", reward_key="score"),
        batch_samples=[samples],
        batch_count=1,
        rollout_id=3,
        data_system_client=client,
    )

    assert client.kwargs["partition_id"] == "train_3"
    assert [row["total_lengths"] for row in client.kwargs["custom_meta"]] == [11, 22]
    provenance = [row[TRANSFER_PROVENANCE_KEY] for row in client.kwargs["custom_meta"]]
    assert [row["sample_key"] for row in provenance] == group_key
    assert all(row["group_key"] == group_key for row in provenance)
    assert all(row["partition_id"] == "train_3" for row in provenance)
    assert all(row["schema_version"] == TRANSFER_PROVENANCE_SCHEMA_VERSION for row in provenance)
    assert all(row["weight_versions"] == ["policy-v3"] for row in provenance)
