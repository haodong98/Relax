# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict wrapper around the native WAA terminal evaluator contract."""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class EvaluationInfraError(RuntimeError):
    """Evaluator infrastructure failed; callers must drop the rollout group."""

    def __init__(self, code: str, phase: str):
        super().__init__(f"{phase}:{code}")
        self.code = code
        self.phase = phase


@dataclass(frozen=True)
class EvaluationOutcome:
    score: float
    reason: str


Getter = Callable[[Any, dict[str, Any]], Any]
Metric = Callable[..., Any]


class StrictTerminalEvaluator:
    """Preserve native WAA scoring while separating valid zero from infra
    failure."""

    def __init__(
        self,
        evaluator: dict[str, Any],
        *,
        getter_resolver: Callable[[str], Getter],
        metric_resolver: Callable[[str], Metric],
        postconfig: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        self._config = copy.deepcopy(evaluator)
        self._getter_resolver = getter_resolver
        self._metric_resolver = metric_resolver
        self._postconfig = postconfig
        self._expected_cache: list[Any] | None = None

    @staticmethod
    def _as_list(value: Any, length: int, default: Any) -> list[Any]:
        if isinstance(value, list):
            if len(value) != length:
                raise EvaluationInfraError("arity_mismatch", "preflight")
            return value
        return [value if value is not None else default for _ in range(length)]

    def _components(self) -> tuple[list[str], list[Any], list[Any], list[dict[str, Any]]]:
        functions = self._config.get("func")
        names = functions if isinstance(functions, list) else [functions]
        if not names or any(not isinstance(name, str) for name in names):
            raise EvaluationInfraError("invalid_metric", "preflight")
        length = len(names)
        results = self._as_list(self._config.get("result"), length, None)
        expected = self._as_list(self._config.get("expected"), length, None)
        options = self._as_list(self._config.get("options"), length, {})
        if any(option is not None and not isinstance(option, dict) for option in options):
            raise EvaluationInfraError("invalid_options", "preflight")
        return names, results, expected, [option or {} for option in options]

    def preflight_expected(self, env: Any) -> None:
        """Resolve all gold state before the first policy call and cache it."""

        names, _, expected_configs, _ = self._components()
        if names == ["infeasible"]:
            self._expected_cache = [None]
            return
        cache: list[Any] = []
        for config in expected_configs:
            if config is None:
                cache.append(None)
                continue
            try:
                getter = self._getter_resolver(config["type"])
                expected = getter(env, copy.deepcopy(config))
                if expected is None:
                    raise EvaluationInfraError("expected_unavailable", "preflight")
                cache.append(expected)
            except Exception as exc:
                raise EvaluationInfraError("expected_unavailable", "preflight") from exc
        self._expected_cache = cache

    @staticmethod
    def _score(value: Any) -> float:
        if isinstance(value, bool):
            return float(value)
        if not isinstance(value, numbers.Real):
            raise EvaluationInfraError("metric_non_numeric", "metric")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise EvaluationInfraError("metric_out_of_range", "metric")
        return score

    def evaluate(self, env: Any, *, last_action: str | None) -> EvaluationOutcome:
        if self._expected_cache is None:
            raise EvaluationInfraError("expected_not_preflighted", "evaluate")
        try:
            self._postconfig(copy.deepcopy(self._config.get("postconfig", [])))
        except FileNotFoundError:
            # A task-created target may legitimately not exist. Result getters
            # below still decide the policy outcome.
            pass
        except Exception as exc:
            raise EvaluationInfraError("postconfig_failed", "postconfig") from exc

        functions, result_configs, _, options = self._components()
        if functions == ["infeasible"]:
            score = 1.0 if last_action == "FAIL" else 0.0
            return EvaluationOutcome(score, "infeasible_declared" if score else "infeasible_not_declared")
        if last_action == "FAIL":
            return EvaluationOutcome(0.0, "feasible_declared_failed")

        scores: list[float] = []
        conjunction = self._config.get("conj", "and")
        if conjunction not in ("and", "or"):
            raise EvaluationInfraError("invalid_conjunction", "preflight")
        for index, name in enumerate(functions):
            result_config = result_configs[index]
            try:
                result = (
                    None if result_config is None else self._getter_resolver(result_config["type"])(env, result_config)
                )
            except FileNotFoundError:
                score = 0.0
            except Exception as exc:
                raise EvaluationInfraError("result_unavailable", "result") from exc
            else:
                try:
                    metric = self._metric_resolver(name)
                    expected = self._expected_cache[index]
                    value = (
                        metric(result, expected, **options[index])
                        if expected is not None
                        else metric(result, **options[index])
                    )
                    score = self._score(value)
                except EvaluationInfraError:
                    raise
                except Exception as exc:
                    raise EvaluationInfraError("metric_failed", "metric") from exc
            if conjunction == "and" and score == 0.0:
                return EvaluationOutcome(0.0, "and_short_circuit")
            if conjunction == "or" and score == 1.0:
                return EvaluationOutcome(1.0, "or_short_circuit")
            scores.append(score)
        if not scores:
            raise EvaluationInfraError("empty_metric", "metric")
        score = sum(scores) / len(scores) if conjunction == "and" else max(scores)
        return EvaluationOutcome(score, conjunction)
