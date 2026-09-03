# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from typing import Any


TRANSFER_PROVENANCE_KEY = "relax_provenance"
TRANSFER_PROVENANCE_SCHEMA_VERSION = "relax.transfer_provenance.v1"


def _sample_key(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a three-element sample key, got {value!r}")
    return list(value)


def build_transfer_provenance(
    *,
    sample_key: Any,
    group_key: Any,
    partition_id: str,
    collection_rollout_id: Any = None,
    train_partition_rollout_id: Any = None,
    train_partition_step: Any = None,
    weight_versions: Any = None,
) -> dict[str, Any]:
    """Build the JSON-safe per-row provenance attached atomically to TQ
    data."""
    if not isinstance(partition_id, str) or not partition_id:
        raise ValueError(f"partition_id must be a non-empty string, got {partition_id!r}")
    normalized_sample_key = _sample_key(sample_key, field="sample_key")
    if not isinstance(group_key, (list, tuple)) or not group_key:
        raise ValueError(f"group_key must be a non-empty sequence, got {group_key!r}")
    normalized_group_key = [_sample_key(item, field="group_key item") for item in group_key]
    if normalized_sample_key not in normalized_group_key:
        raise ValueError(
            "sample_key must be a member of group_key: "
            f"sample_key={normalized_sample_key!r}, group_key={normalized_group_key!r}"
        )
    return {
        "schema_version": TRANSFER_PROVENANCE_SCHEMA_VERSION,
        "sample_key": normalized_sample_key,
        "group_key": normalized_group_key,
        "partition_id": partition_id,
        "collection_rollout_id": collection_rollout_id,
        "train_partition_rollout_id": train_partition_rollout_id,
        "train_partition_step": train_partition_step,
        "weight_versions": list(weight_versions) if isinstance(weight_versions, (list, tuple)) else [],
    }


def extract_data_wait_provenance(
    batch_meta: Any,
    *,
    partition_id: str,
    task_name: str,
    dp_rank: int,
    batch_index: int,
    required: bool,
) -> dict[str, Any]:
    """Extract the keys of the exact TQ rows returned to one trainer fetch."""
    attributes: dict[str, Any] = {
        "returned_batch": True,
        "partition_id": partition_id,
        "task_name": task_name,
        "dp_rank": int(dp_rank),
        "batch_index": int(batch_index),
        "trainer_batch_id": f"{partition_id}:{task_name}:{batch_index}",
    }
    get_all_custom_meta = getattr(batch_meta, "get_all_custom_meta", None)
    if not callable(get_all_custom_meta):
        if required:
            raise ValueError("returned TransferQueue BatchMeta has no get_all_custom_meta() method")
        return attributes
    rows = get_all_custom_meta()
    if not isinstance(rows, list):
        raise ValueError(f"BatchMeta custom_meta must be a list, got {type(rows).__name__}")

    sample_keys: list[list[Any]] = []
    weight_versions: list[list[Any]] = []
    group_keys: list[list[list[Any]]] = []
    seen_groups: set[tuple[tuple[Any, ...], ...]] = set()
    for row_index, row in enumerate(rows):
        provenance = row.get(TRANSFER_PROVENANCE_KEY) if isinstance(row, dict) else None
        if not isinstance(provenance, dict):
            if required:
                raise ValueError(f"custom_meta row {row_index} is missing {TRANSFER_PROVENANCE_KEY!r}")
            continue
        if provenance.get("schema_version") != TRANSFER_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"custom_meta row {row_index} has unsupported provenance schema {provenance.get('schema_version')!r}"
            )
        if provenance.get("partition_id") != partition_id:
            raise ValueError(
                f"custom_meta row {row_index} partition mismatch: "
                f"expected={partition_id!r}, observed={provenance.get('partition_id')!r}"
            )
        sample_key = _sample_key(provenance.get("sample_key"), field=f"custom_meta row {row_index} sample_key")
        raw_group_key = provenance.get("group_key")
        if not isinstance(raw_group_key, (list, tuple)) or not raw_group_key:
            raise ValueError(f"custom_meta row {row_index} group_key must be a non-empty sequence")
        group_key = [_sample_key(item, field=f"custom_meta row {row_index} group_key item") for item in raw_group_key]
        if sample_key not in group_key:
            raise ValueError(f"custom_meta row {row_index} sample_key is not present in group_key")
        sample_keys.append(sample_key)
        raw_weight_versions = provenance.get("weight_versions", [])
        if not isinstance(raw_weight_versions, list):
            raise ValueError(f"custom_meta row {row_index} weight_versions must be a list")
        weight_versions.append(list(raw_weight_versions))
        group_identity = tuple(tuple(item) for item in group_key)
        if group_identity not in seen_groups:
            seen_groups.add(group_identity)
            group_keys.append(group_key)

    if required and len(sample_keys) != len(rows):
        raise ValueError(
            "returned TransferQueue batch is missing required row provenance: "
            f"rows={len(rows)}, provenance_rows={len(sample_keys)}"
        )
    partition_ids = getattr(batch_meta, "partition_ids", None)
    if isinstance(partition_ids, list) and any(value != partition_id for value in partition_ids):
        raise ValueError(
            f"BatchMeta partition_ids do not match requested partition {partition_id!r}: {partition_ids!r}"
        )
    attributes.update(
        {
            "provenance_schema_version": TRANSFER_PROVENANCE_SCHEMA_VERSION,
            "returned_sample_keys": sample_keys,
            "returned_group_keys": group_keys,
            "returned_weight_versions": weight_versions,
        }
    )
    return attributes
