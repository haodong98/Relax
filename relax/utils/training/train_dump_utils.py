# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import hashlib
import json
import os
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.stats import get_sample_multimodal_stats


logger = get_logger(__name__)
_DIGEST_MODULUS = 1 << 256


def _dual_judge_marker_dir(args: Any) -> Path | None:
    marker_dir = Envs.RELAX_DUAL_JUDGE_MARKER_DIR
    if (
        marker_dir is None
        or getattr(args, "rm_type", None) != "dual-agentic-judge"
        or getattr(args, "judge_services", None) is None
    ):
        return None
    return Path(marker_dir)


def _append_marker_record(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("Failed to append critical-path marker to %s: %s", path, exc)


def append_weight_serving_ready_marker(
    args: Any,
    *,
    step: int,
    wall_time_s: float,
    monotonic_ns: int,
) -> None:
    """Append one low-volume ready boundary usable with timeline tracing
    disabled."""
    marker_dir = _dual_judge_marker_dir(args)
    if marker_dir is None:
        return
    judge_services = getattr(args, "judge_services", None)
    benchmark_mode = getattr(judge_services, "benchmark_mode", None)
    from relax.utils.judge_config import dual_judge_benchmark_invariant_hash

    invariant_hash = dual_judge_benchmark_invariant_hash(args)
    record = {
        "schema_version": 1,
        "event": "weight_serving_ready",
        "step": step,
        "wall_time_s": wall_time_s,
        "monotonic_ns": monotonic_ns,
        "clock_host": socket.gethostname(),
        "pid": os.getpid(),
        "benchmark_mode": benchmark_mode,
        "benchmark_invariant_hash": invariant_hash,
        "timeline_enabled": bool(
            getattr(args, "use_metrics_service", False) and getattr(args, "timeline_dump_dir", None)
        ),
    }
    _append_marker_record(marker_dir / "weight_serving_ready.jsonl", record)


def append_agentic_accounting_marker(args: Any, *, step: int, snapshot: dict[str, Any]) -> None:
    """Persist the terminal dataflow snapshot used to reject right-censored
    runs."""
    marker_dir = _dual_judge_marker_dir(args)
    if marker_dir is None:
        return
    record = {
        "schema_version": 1,
        "event": "agentic_accounting_end",
        "step": step,
        "wall_time_s": time.time(),
        "clock_host": socket.gethostname(),
        "pid": os.getpid(),
        "snapshot": copy.deepcopy(snapshot),
    }
    _append_marker_record(marker_dir / "agentic_accounting_end.jsonl", record)


def _reward_marker_metadata(
    samples: list[Any],
    reward_trace_snapshots: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], bool]]:
    records = [
        (sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}, False) for sample in samples
    ]
    committed_keys = {
        repr(reward_trace.get("sample_key"))
        for metadata, _is_snapshot in records
        if isinstance((trace := metadata.get("agentic_trace")), dict)
        and isinstance((reward_trace := trace.get("reward")), dict)
        and reward_trace.get("sample_key") is not None
    }
    for snapshot in reward_trace_snapshots:
        if not isinstance(snapshot, dict):
            continue
        trace = snapshot.get("agentic_trace")
        reward_trace = trace.get("reward") if isinstance(trace, dict) else None
        snapshot_key = repr(reward_trace.get("sample_key")) if isinstance(reward_trace, dict) else None
        if snapshot_key is not None and snapshot_key in committed_keys:
            continue
        records.append((snapshot, True))
    return records


def _counter_to_json(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def append_reward_workload_markers(
    args: Any,
    *,
    collection_rollout_id: int,
    default_step: int,
    committed_samples: list[Any],
    reward_trace_snapshots: list[dict[str, Any]],
) -> None:
    """Append mergeable per-partition workload evidence without serializing
    trajectories."""
    marker_dir = _dual_judge_marker_dir(args)
    if marker_dir is None:
        return
    from relax.utils.judge_config import dual_judge_benchmark_invariant_hash

    invariant_hash = dual_judge_benchmark_invariant_hash(args)
    fragments: defaultdict[int, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    for metadata, is_snapshot in _reward_marker_metadata(committed_samples, reward_trace_snapshots):
        trace = metadata.get("agentic_trace")
        reward_trace = trace.get("reward") if isinstance(trace, dict) else None
        partition_step = reward_trace.get("train_partition_step") if isinstance(reward_trace, dict) else None
        if isinstance(partition_step, bool) or not isinstance(partition_step, int):
            partition_step = default_step
        fragments[partition_step].append((metadata, is_snapshot))

    for partition_step, metadata_records in fragments.items():
        identity_sum = 0
        identity_xor = 0
        invalid_identity_count = 0
        invalid_recorded_reward_hash_count = 0
        group_sample_counts: Counter[int] = Counter()
        outcome_counts: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()
        branch_signature_counts: Counter[str] = Counter()
        terminal_snapshot_count = 0
        for metadata, is_snapshot in metadata_records:
            terminal_snapshot_count += int(is_snapshot)
            trace = metadata.get("agentic_trace")
            reward_trace = trace.get("reward") if isinstance(trace, dict) else None
            reward_trace = reward_trace if isinstance(reward_trace, dict) else {}
            group_index = reward_trace.get("group_index")
            sample_index = reward_trace.get("sample_index")
            context_hash = reward_trace.get("context_hash")
            recorded_reward_hash = reward_trace.get("recorded_reward_hash")
            identity_is_valid = (
                isinstance(group_index, int)
                and not isinstance(group_index, bool)
                and isinstance(sample_index, int)
                and not isinstance(sample_index, bool)
                and isinstance(context_hash, str)
                and bool(context_hash)
            )
            if identity_is_valid:
                group_sample_counts[group_index] += 1
            else:
                invalid_identity_count += 1
            if reward_trace.get("benchmark_mode") in {"recorded", "accuracy_shadow", "dual_shadow"} and not (
                isinstance(recorded_reward_hash, str) and recorded_reward_hash
            ):
                invalid_recorded_reward_hash_count += 1
            canonical_identity = json.dumps(
                [group_index, sample_index, context_hash, recorded_reward_hash],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            identity_value = int.from_bytes(hashlib.sha256(canonical_identity).digest(), "big")
            identity_sum = (identity_sum + identity_value) % _DIGEST_MODULUS
            identity_xor ^= identity_value
            outcome_key = json.dumps(
                [
                    reward_trace.get("pipeline_status"),
                    reward_trace.get("executor_status"),
                    reward_trace.get("terminal_outcome"),
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            outcome_counts[outcome_key] += 1
            mode_counts[str(reward_trace.get("benchmark_mode"))] += 1
            judges = reward_trace.get("judges")
            branch_signature = []
            if isinstance(judges, dict):
                branch_signature = sorted(
                    [component, value.get("status") if isinstance(value, dict) else None]
                    for component, value in judges.items()
                )
            branch_signature_counts[json.dumps(branch_signature, ensure_ascii=False, separators=(",", ":"))] += 1
        record = {
            "schema_version": 1,
            "event": "reward_workload_fragment",
            "step": partition_step,
            "collection_rollout_id": collection_rollout_id,
            "identity_count": len(metadata_records),
            "identity_digest_sum": f"{identity_sum:064x}",
            "identity_digest_xor": f"{identity_xor:064x}",
            "invalid_identity_count": invalid_identity_count,
            "invalid_recorded_reward_hash_count": invalid_recorded_reward_hash_count,
            "group_sample_counts": [[group, group_sample_counts[group]] for group in sorted(group_sample_counts)],
            "reward_outcome_counts": _counter_to_json(outcome_counts),
            "benchmark_mode_counts": _counter_to_json(mode_counts),
            "judge_branch_signature_counts": _counter_to_json(branch_signature_counts),
            "terminal_snapshot_count": terminal_snapshot_count,
            "benchmark_invariant_hash": invariant_hash,
            "clock_host": socket.gethostname(),
            "pid": os.getpid(),
        }
        _append_marker_record(marker_dir / "reward_workload.jsonl", record)


def _decode_tokens_to_chars(tokens: List[int], tokenizer) -> List[str]:
    """Decode each token to its corresponding string representation."""
    if tokenizer is None or not tokens:
        return []
    try:
        decoded_chars = []
        for token_id in tokens:
            try:
                char = tokenizer.decode([token_id], skip_special_tokens=False)
                decoded_chars.append(char)
            except Exception:
                decoded_chars.append(f"<{token_id}>")
        return decoded_chars
    except Exception as e:
        logger.warning(f"Failed to decode tokens: {e}")
        return []


def _tensor_to_list(tensor_or_list):
    """Convert tensor to list if needed."""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    return tensor_or_list


def _reorganize_train_data_by_sample(rollout_data: Dict[str, Any], tokenizer=None) -> List[Dict[str, Any]]:
    """Reorganize train data from field-oriented to sample-oriented format.

    Args:
        rollout_data: Dict with keys like 'tokens', 'rewards', etc., each containing lists of values
        tokenizer: Optional tokenizer for decoding tokens

    Returns:
        List of sample dicts, each containing all fields for that sample
    """
    # Determine the number of samples from any available field
    num_samples = 0
    for key, value in rollout_data.items():
        if isinstance(value, (list, tuple)) and len(value) > 0:
            num_samples = len(value)
            break

    if num_samples == 0:
        return []

    # Use rollout_data keys for per-sample fields to keep extensibility
    per_sample_fields = list(rollout_data.keys())
    logger.info(
        "Reorganize train data with fields: %s",
        ", ".join(per_sample_fields) if per_sample_fields else "<empty>",
    )

    samples = []
    for i in range(num_samples):
        sample = {"sample_index": i}

        for field in per_sample_fields:
            if field in rollout_data:
                value = rollout_data[field]
                if isinstance(value, (list, tuple)) and i < len(value):
                    sample[field] = _tensor_to_list(value[i])
                elif isinstance(value, (list, tuple)):
                    sample[field] = None

        # Add decoded_tokens if tokens are available
        if "tokens" in sample and sample["tokens"] is not None:
            tokens = sample["tokens"]
            if tokenizer:
                sample["decoded_tokens"] = _decode_tokens_to_chars(tokens, tokenizer)
            else:
                sample["decoded_tokens"] = []

        samples.append(sample)

    return samples


def save_debug_train_data(args, *, rollout_id, rollout_data, tokenizer=None):
    """Save debug train data reorganized by sample with tokenizer decoding.

    Args:
        args: Arguments containing save_debug_train_data path template
        rollout_id: The rollout ID
        rollout_data: Dict with training data organized by field
        tokenizer: Optional tokenizer for decoding tokens. If provided, will be used
                  to decode tokens to readable strings. Pass from caller context
                  (e.g., Actor.tokenizer) for efficiency.
    """
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reorganize data by sample
        samples = _reorganize_train_data_by_sample(rollout_data, tokenizer)

        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                samples=samples,
            ),
            path,
        )


def save_debug_rollout_data(args, data, rollout_id: int, evaluation: bool, tokenizer=None) -> None:
    """Save debug rollout data with tokenizer decoding support.

    Args:
        args: Arguments containing save_debug_rollout_data path template
        data: Either a list of Sample objects, or a dict (for evaluation data)
        rollout_id: The rollout ID
        evaluation: Whether this is evaluation data
        tokenizer: Optional tokenizer for decoding tokens. If provided, will be used
                  to decode tokens to readable strings. Pass from caller context
                  (e.g., GenerateState.tokenizer) for efficiency.
    """
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        if evaluation:
            # Evaluation data is a dict with dataset_name -> info structure
            samples_list = [sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
        else:
            # Regular rollout data is a list of Sample objects
            samples_list = [sample.to_dict() for sample in data]

        # Add decoded_tokens field for each sample if tokenizer is provided
        for sample_dict in samples_list:
            tokens = sample_dict.get("tokens", [])
            if tokens and tokenizer:
                sample_dict["decoded_tokens"] = _decode_tokens_to_chars(tokens, tokenizer)
            else:
                sample_dict["decoded_tokens"] = []

        dump_data = dict(samples=samples_list)
        torch.save(dict(rollout_id=rollout_id, **dump_data), path)


def _summarize_multimodal_inputs(mm_inputs: dict) -> dict:
    """Summarize raw multimodal inputs (images, videos, audio) for logging."""
    summary = {}
    for key, value in mm_inputs.items():
        if isinstance(value, list):
            items = []
            for item in value:
                try:
                    if isinstance(item, Image.Image):
                        items.append(str(item))
                        continue
                except ImportError:
                    pass
                items.append(str(type(item).__name__))
            summary[key] = items if items else len(value)
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _summarize_multimodal_train_inputs(mm_train_inputs: dict) -> dict:
    """Summarize processed multimodal train inputs (tensors) for logging."""
    import torch

    summary = {}
    for key, value in mm_train_inputs.items():
        if isinstance(value, torch.Tensor):
            summary[key] = f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            summary[key] = [f"Tensor(shape={list(t.shape)}, dtype={t.dtype})" for t in value]
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _compact_latency_trace(metadata: dict) -> dict | None:
    agentic_trace = metadata.get("agentic_trace")
    reward_trace = agentic_trace.get("reward") if isinstance(agentic_trace, dict) else None
    if not isinstance(reward_trace, dict):
        return None
    turns = agentic_trace.get("turns")
    compact_trace = {
        "events": copy.deepcopy(agentic_trace.get("events", {})),
        "reward": copy.deepcopy(reward_trace),
        "turns": [
            {
                key: copy.deepcopy(turn[key])
                for key in ("generation_elapsed_s", "wall_elapsed_s", "events")
                if key in turn
            }
            for turn in (turns if isinstance(turns, list) else [])
            if isinstance(turn, dict)
        ],
    }
    assistant_turn_count = agentic_trace.get("per_turn_assistant_turn_count")
    if isinstance(assistant_turn_count, int) and not isinstance(assistant_turn_count, bool):
        compact_trace["per_turn_assistant_turn_count"] = assistant_turn_count
    if agentic_trace.get("reasoning_trigger") == "per_turn":
        off_lineage_count = agentic_trace.get("per_turn_off_lineage_judge_count", 0)
        if not isinstance(off_lineage_count, int) or isinstance(off_lineage_count, bool) or off_lineage_count < 0:
            raise ValueError("agentic_trace.per_turn_off_lineage_judge_count must be a non-negative integer")
        compact_trace["per_turn_off_lineage_judge_count"] = off_lineage_count
    return compact_trace


def _sample_to_summary_record(sample, rollout_id: int, idx: int, dataset_name: str | None = None) -> dict:
    total_length = len(sample.tokens) if sample.tokens else 0
    response_length = sample.response_length
    prompt_length = max(total_length - response_length, 0)
    multimodal_stats = get_sample_multimodal_stats(sample)
    metadata = sample.metadata or {}
    record = {
        "rollout_id": rollout_id,
        "sample_index": idx,
        "prompt": sample.prompt,
        "response": sample.response,
        "reward": sample.reward,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "total_length": total_length,
        "prompt_token_count": prompt_length,
        "response_token_count": response_length,
        "total_token_count": total_length,
        "image_count": multimodal_stats["image_count"],
        "image_token_count": multimodal_stats["image_token_count"],
        "multimodal_token_count": multimodal_stats["multimodal_token_count"],
        "agent_turns": metadata.get("rollout_turns", 1),
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
        "group_index": sample.group_index,
        "weight_versions": list(getattr(sample, "weight_versions", []) or []),
    }
    if sample.label is not None:
        record["label"] = sample.label
    if sample.multimodal_inputs is not None:
        record["multimodal_inputs"] = _summarize_multimodal_inputs(sample.multimodal_inputs)
    if sample.multimodal_train_inputs is not None:
        record["multimodal_train_inputs"] = _summarize_multimodal_train_inputs(sample.multimodal_train_inputs)
    if dataset_name is not None:
        record["dataset"] = dataset_name
    latency_trace = _compact_latency_trace(metadata)
    if latency_trace is not None:
        record["latency_trace"] = latency_trace
    return record


def _reward_snapshot_to_summary_record(metadata: dict, rollout_id: int, idx: int) -> dict | None:
    latency_trace = _compact_latency_trace(metadata)
    if latency_trace is None:
        return None
    reward_trace = latency_trace["reward"]
    return {
        "record_type": "reward_terminal_trace",
        "rollout_id": rollout_id,
        "collection_rollout_id": rollout_id,
        "trace_index": idx,
        "sample_index": reward_trace.get("sample_index", idx),
        "group_index": reward_trace.get("group_index"),
        "sample_key": copy.deepcopy(reward_trace.get("sample_key")),
        "group_key": copy.deepcopy(reward_trace.get("group_key")),
        "status": reward_trace.get("terminal_outcome", reward_trace.get("pipeline_status", "unknown")),
        "latency_trace": latency_trace,
    }


def _write_summary_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_rollout_result_jsonl(args, rollout_id: int, samples: list) -> None:
    """Save a lightweight per-step rollout result as a JSONL file.

    Always-on: writes one JSONL per rollout step with prompt, response,
    reward and sequence length for every sample.

    Args:
        args: Arguments containing ``rollout_result_dir``.
        rollout_id: The rollout ID (used as step number in the filename).
        samples: A list of :class:`Sample` objects from the rollout batch.
    """
    result_dir = getattr(args, "rollout_result_dir", None)
    if result_dir is None:
        return

    path = Path(result_dir) / "train" / f"{rollout_id}.jsonl"
    records = [_sample_to_summary_record(s, rollout_id, i) for i, s in enumerate(samples)]

    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved rollout result ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save rollout result to {path}: {e}")


def save_reward_trace_snapshots_jsonl(
    args,
    rollout_id: int,
    snapshots: list[dict],
    *,
    committed_samples: list | None = None,
) -> None:
    """Persist terminal reward attempts separately from real rollout
    samples."""
    result_dir = getattr(args, "rollout_result_dir", None)
    if result_dir is None or not snapshots:
        return
    committed_keys = {
        repr(trace.get("reward", {}).get("sample_key"))
        for sample in committed_samples or []
        if isinstance((trace := getattr(sample, "metadata", {}).get("agentic_trace")), dict)
    }
    records = []
    for index, metadata in enumerate(snapshots):
        if not isinstance(metadata, dict):
            continue
        latency_trace = _compact_latency_trace(metadata)
        if latency_trace is None or repr(latency_trace["reward"].get("sample_key")) in committed_keys:
            continue
        record = _reward_snapshot_to_summary_record(metadata, rollout_id, index)
        if record is not None:
            records.append(record)
    if not records:
        return
    path = Path(result_dir) / "reward_terminal_traces" / f"{rollout_id}.jsonl"
    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved terminal reward traces ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save terminal reward traces to {path}: {e}")


def save_eval_summary_jsonl(args, rollout_id: int, data: dict) -> None:
    """Save a lightweight per-step eval summary as a JSONL file.

    Similar to :func:`save_rollout_result_jsonl` but for evaluation rollouts.
    Each record includes an extra ``dataset`` field indicating which eval
    dataset the sample belongs to.

    Args:
        args: Arguments containing ``rollout_result_dir``.
        rollout_id: The rollout ID.
        data: Eval data dict — ``{dataset_name: {"samples": list[Sample], ...}}``.
    """
    result_dir = getattr(args, "rollout_result_dir", None)
    if result_dir is None:
        return

    eval_dir = Path(result_dir) / "eval"
    path = eval_dir / f"{rollout_id}.jsonl"

    records = []
    for dataset_name, info in data.items():
        samples = info.get("samples")
        if not samples:
            continue
        for i, s in enumerate(samples):
            records.append(_sample_to_summary_record(s, rollout_id, i, dataset_name=dataset_name))

    if not records:
        return

    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved eval summary ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save eval summary to {path}: {e}")
