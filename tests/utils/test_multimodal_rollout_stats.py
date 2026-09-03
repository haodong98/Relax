# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.utils.multimodal.stats import get_multimodal_token_counts, get_sample_multimodal_stats
from relax.utils.training.train_dump_utils import _reward_snapshot_to_summary_record, _sample_to_summary_record
from relax.utils.types import Sample


def test_multimodal_stats_counts_images_and_tokens():
    sample = Sample(
        tokens=list(range(100)),
        response_length=20,
        multimodal_inputs={"images": ["a.png", "b.png"]},
        multimodal_train_inputs={
            "image_grid_thw": torch.tensor([[1, 14, 14], [2, 10, 10]]),
            "video_grid_thw": torch.tensor([[3, 8, 8]]),
        },
        metadata={"rollout_turns": 3},
    )

    token_counts = get_multimodal_token_counts(sample.multimodal_train_inputs)
    assert token_counts == {
        "image": 99,
        "video": 48,
        "audio": 0,
        "total": 147,
    }
    assert get_sample_multimodal_stats(sample) == {
        "image_count": 2,
        "image_token_count": 99,
        "video_token_count": 48,
        "audio_token_count": 0,
        "multimodal_token_count": 147,
    }


def test_rollout_summary_record_includes_token_and_agent_stats():
    sample = Sample(
        prompt="hello",
        response="world",
        tokens=list(range(12)),
        response_length=5,
        reward=1.0,
        multimodal_inputs={"images": ["image.png"]},
        multimodal_train_inputs={"image_grid_thw": [[1, 8, 8]]},
        metadata={"rollout_turns": 2},
    )

    record = _sample_to_summary_record(sample, rollout_id=7, idx=0)

    assert record["prompt_token_count"] == 7
    assert record["response_token_count"] == 5
    assert record["total_token_count"] == 12
    assert record["prompt_length"] == 7
    assert record["image_count"] == 1
    assert record["image_token_count"] == 16
    assert record["multimodal_token_count"] == 16
    assert record["agent_turns"] == 2


def test_rollout_summary_record_preserves_compact_reward_latency_trace():
    sample = Sample(
        tokens=[1, 2],
        response_length=1,
        metadata={
            "agentic_trace": {
                "events": {"reward_arrive_at": 1.0, "reward_end_at": 2.0},
                "reward": {"schema_version": 1, "pipeline_elapsed_s": 1.0},
                "reasoning_trigger": "per_turn",
                "per_turn_assistant_turn_count": 2,
                "turns": [
                    {
                        "generation_elapsed_s": 0.5,
                        "events": {"generation_start_at": 0.0, "generation_end_at": 0.5},
                        "prompt_text": "must not be copied",
                    }
                ],
            }
        },
    )

    record = _sample_to_summary_record(sample, rollout_id=1, idx=0)

    assert record["latency_trace"]["reward"]["pipeline_elapsed_s"] == 1.0
    assert record["latency_trace"]["turns"][0]["generation_elapsed_s"] == 0.5
    assert record["latency_trace"]["per_turn_assistant_turn_count"] == 2
    assert record["latency_trace"]["per_turn_off_lineage_judge_count"] == 0
    assert "prompt_text" not in record["latency_trace"]["turns"][0]


def test_reward_trace_without_turns_is_persisted_for_failed_sample():
    metadata = {
        "agentic_trace": {
            "events": {"reward_arrive_at": 1.0, "reward_end_at": 2.0},
            "reward": {
                "pipeline_elapsed_s": 1.0,
                "pipeline_status": "cancelled",
                "terminal_outcome": "group_rejected",
                "sample_index": 4,
            },
        }
    }

    record = _reward_snapshot_to_summary_record(metadata, rollout_id=2, idx=0)

    assert record["record_type"] == "reward_terminal_trace"
    assert record["sample_index"] == 4
    assert record["status"] == "group_rejected"
    assert record["latency_trace"]["turns"] == []
