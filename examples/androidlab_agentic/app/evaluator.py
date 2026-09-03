# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict AndroidLab terminal evaluator and local-query Judge boundary."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class EvaluationInfraError(RuntimeError):
    """An evaluator dependency failed; this is not a valid task reward."""

    def __init__(self, code: str, phase: str) -> None:
        super().__init__(f"{phase}:{code}")
        self.code = code
        self.phase = phase


@dataclass(frozen=True)
class EvaluationOutcome:
    score: float
    reason: str
    partial_subgoals: dict[str, bool]


class LocalAnswerJudge:
    """Call a local, OpenAI-compatible answer Judge held outside policy
    scope."""

    def __init__(self, url: str, *, timeout_s: float = 120.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("local Judge URL must be an HTTP(S) URL")
        self._url = url.rstrip("/")
        self._timeout_s = timeout_s

    def score(self, *, question: str, reference_answer: str, answer: str) -> float:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Judge whether the answer matches the reference for the Android task. "
                        'Return exactly JSON: {"score": 0 or 1}.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "answer": answer,
                            "question": question,
                            "reference_answer": reference_answer,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                if response.status != 200:
                    raise EvaluationInfraError("judge_http", "query")
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EvaluationInfraError("judge_unavailable", "query") from exc
        score = _extract_judge_score(result)
        if not math.isfinite(score) or score not in (0.0, 1.0):
            raise EvaluationInfraError("judge_invalid_score", "query")
        return score


def _extract_judge_score(payload: Any) -> float:
    if isinstance(payload, dict) and isinstance(payload.get("score"), (int, float)):
        return float(payload["score"])
    if not isinstance(payload, dict):
        raise EvaluationInfraError("judge_invalid_payload", "query")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise EvaluationInfraError("judge_invalid_payload", "query")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise EvaluationInfraError("judge_invalid_payload", "query")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EvaluationInfraError("judge_invalid_json", "query") from exc
    if not isinstance(decoded, dict) or isinstance(decoded.get("score"), bool):
        raise EvaluationInfraError("judge_invalid_payload", "query")
    score = decoded.get("score")
    if not isinstance(score, (int, float)):
        raise EvaluationInfraError("judge_invalid_payload", "query")
    return float(score)


def outcome_from_result(result: dict[str, Any]) -> EvaluationOutcome:
    if not isinstance(result, dict):
        raise EvaluationInfraError("metric_invalid_result", "metric")
    complete = result.get("complete")
    if not isinstance(complete, bool):
        raise EvaluationInfraError("metric_missing_complete", "metric")
    partial = {key: value for key, value in result.items() if key != "complete" and isinstance(value, bool)}
    return EvaluationOutcome(score=float(complete), reason="native_complete", partial_subgoals=partial)
