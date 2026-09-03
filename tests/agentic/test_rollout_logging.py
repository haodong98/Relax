# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.agentic.rollout import _sample_log_summary
from relax.utils.types import Sample


class _TensorLike:
    shape = (81600, 1536)

    def __repr__(self):
        raise AssertionError("sample logging must not format multimodal tensor contents")


class _UnformattableText:
    def __str__(self):
        raise AssertionError("sample logging must not format trajectory contents")


def test_sample_log_summary_is_bounded_and_does_not_format_payloads():
    sample = Sample(
        session_id="session-1",
        index=7,
        prompt="p" * 1000,
        response="r" * 2000,
        tokens=[1, 2, 3, 4, 5],
        response_length=3,
        multimodal_train_inputs={"pixel_values": _TensorLike()},
        reward={"score": 0.5, "rationale": _UnformattableText()},
        weight_versions=["policy-v3"],
    )

    summary = _sample_log_summary(sample)

    assert summary["prompt_chars"] == 1000
    assert summary["response_chars"] == 2000
    assert summary["prompt_tokens"] == 2
    assert summary["response_tokens"] == 3
    assert summary["multimodal_shapes"] == {"pixel_values": (81600, 1536)}
    assert summary["reward"] == {"score": 0.5}
    assert summary["weight_versions"] == ["policy-v3"]
    assert "prompt" not in summary
    assert "response" not in summary
