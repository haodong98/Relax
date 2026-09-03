# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from argparse import Namespace

import pytest

from examples.agentic_dual_judge.analyze_latency import (
    _events_from_latency_trace,
    _expected_trainer_components,
    _optimizer_steps_per_publication_round,
    _paired_makespan_delta,
    _reject_duplicate_names,
    _sample_identity_coverage,
    _stage_for_event,
    analyze_direct_events,
    analyze_events,
    load_variant_events,
)


def _event(name: str, stage: str, start: float, end: float, step: int = 0) -> dict:
    return {"name": name, "stage": stage, "start_s": start, "end_s": end, "step": step}


def _reward_events(
    start: float,
    end: float,
    *,
    step: int,
    context_hash: str,
    group_index: int = 0,
    sample_index: int = 0,
    mode: str = "recorded",
    pipeline_status: str | None = None,
    executor_status: str | None = None,
    invariant_hash: str = "invariant",
    recorded_reward_hash: str = "reward",
    expected_trainer_components: list[str] | None = None,
    clock_host: str | None = None,
) -> list[dict]:
    expected_status = "bypassed" if mode == "recorded" else "success"
    attributes = {
        "benchmark_mode": mode,
        "pipeline_status": pipeline_status if pipeline_status is not None else expected_status,
        "executor_status": executor_status if executor_status is not None else expected_status,
        "group_index": group_index,
        "sample_index": sample_index,
        "context_hash": context_hash,
        "benchmark_invariant_hash": invariant_hash,
    }
    if expected_trainer_components is not None:
        attributes["expected_trainer_components"] = expected_trainer_components
    if clock_host is not None:
        attributes["clock_host"] = clock_host
    if mode in {"recorded", "accuracy_shadow", "dual_shadow"}:
        attributes["recorded_reward_hash"] = recorded_reward_hash
    events = [{**_event("critical_path.reward", "reward", start, end, step), "attributes": attributes}]
    if mode != "recorded":
        events.append(
            {
                **_event("critical_path.reward.answer_accuracy", "reward", start, end, step),
                "attributes": attributes,
            }
        )
    if mode in {"dual", "dual_shadow"}:
        events.append(
            {
                **_event("critical_path.reward.multi_turn_reasoning", "reward", start, end, step),
                "attributes": attributes,
            }
        )
    return events


def test_latency_analyzer_does_not_classify_evaluation_as_generation():
    assert _stage_for_event("critical_path.rollout_generation") == "generation"
    assert _stage_for_event("critical_path.rollout_evaluation") == "evaluation"
    assert _stage_for_event("critical_path.rollout_queue") == "rollout_queue"
    assert _stage_for_event("critical_path.optimizer_step") == "training"
    assert _stage_for_event("critical_path.weight_gate_wait") == "weight_gate_wait"
    assert _stage_for_event("critical_path.turn_judge") == "turn_judge"
    assert _stage_for_event("critical_path.judge_request") == "request"
    assert _stage_for_event("critical_path.session_terminal_admission") == "rollout_orchestration"


def test_direct_analyzer_recovers_request_to_trainer_dependency_chain():
    attributes = {
        "sample_key": ["session-0", 0, None],
        "group_key": [["session-0", 0, None]],
        "group_index": 0,
        "sample_index": 0,
        "terminal_outcome": "success",
        "executor_status": "success",
        "reasoning_trigger": "terminal_once",
    }
    request_attributes = {
        **attributes,
        "request_status": "success",
        "attempt_count": 1,
        "invalid_response_count": 0,
        "per_turn_off_lineage_judge_count": None,
        "queue_elapsed_s": 0.2,
        "http_elapsed_s": 1.5,
        "server": {"server_queue_elapsed_s": 0.1, "engine_elapsed_s": 1.0},
    }
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, 0),
        _event("critical_path.weight_serving_ready", "weight_update", 10.0, 11.0, 1),
        {
            **_event("critical_path.session_terminal_admission", "rollout_orchestration", 2.0, 2.0, 1),
            "attributes": attributes,
        },
        {**_event("critical_path.reward_context_build", "reward", 2.2, 2.5, 1), "attributes": attributes},
        {**_event("critical_path.reward", "reward", 3.0, 5.0, 1), "attributes": attributes},
        {**_event("critical_path.reward_group_finalize", "reward", 5.0, 6.0, 1), "attributes": attributes},
        {
            **_event("critical_path.judge_request", "request", 3.2, 4.2, 1),
            "attributes": {**request_attributes, "request_kind": "terminal_orm"},
        },
        {
            **_event("critical_path.judge_request", "request", 3.1, 4.9, 1),
            "attributes": {**request_attributes, "request_kind": "terminal_vlm"},
        },
        {
            **_event("critical_path.data_wait", "data_wait", 4.0, 7.0, 1),
            "attributes": {
                "returned_batch": True,
                "trainer_batch_id": "train_1:actor_train:0",
                "returned_group_keys": [[["session-0", 0, None]]],
                "observed_other_blocker_intervals": [{"start_s": 4.5, "end_s": 5.5}],
            },
        },
    ]

    report = analyze_direct_events(events, warmup_steps=0, measure_updates=1)

    assert report["request"]["terminal_orm"]["clean"]["client_branch_sojourn_s"]["p50_s"] == pytest.approx(1.0)
    assert report["trajectory"]["clean"]["reward_gate_s"]["p50_s"] == pytest.approx(2.3)
    assert report["trajectory"]["clean"]["terminal_orm_s"]["p50_s"] == pytest.approx(1.0)
    assert report["group"]["group_reward_closure_s"]["p50_s"] == pytest.approx(4.0)
    assert report["trainer"]["inclusive_reward_ancestor_wait_s"]["p50_s"] == pytest.approx(2.0)
    assert report["trainer"]["exclusive_reward_wait_s"]["p50_s"] == pytest.approx(1.0)
    assert report["trainer"]["reward_plus_other_blocker_wait_s"]["p50_s"] == pytest.approx(1.0)
    assert report["window_s"] == pytest.approx([2.0, 11.0])
    assert report["publication"]["per_step"] == [
        {
            "step": 1,
            "previous_ready_step": 0,
            "ready_interval_s": 10.0,
            "reward_union_s": 2.3,
            "sidecar_union_s": 0.0,
            "barrier_union_s": 0.0,
            "reward_sidecar_barrier_union_s": 2.3,
            "request_count": 2,
            "trajectory_count": 1,
            "group_count": 1,
        }
    ]


def test_direct_trainer_derives_other_blockers_from_returned_group_ancestors():
    group_key = [["session-0", 0, None]]
    attributes = {"group_key": group_key, "group_index": 0, "sample_key": group_key[0]}
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, 0),
        _event("critical_path.weight_serving_ready", "weight_update", 10.0, 11.0, 1),
        {
            **_event("critical_path.session_terminal_admission", "rollout_orchestration", 2.0, 2.0, 1),
            "attributes": attributes,
        },
        {**_event("critical_path.reward", "reward", 3.0, 5.0, 1), "attributes": attributes},
        {**_event("critical_path.reward_group_finalize", "reward", 5.0, 6.0, 1), "attributes": attributes},
        {**_event("critical_path.rollout_generation", "generation", 4.5, 5.5, 1), "attributes": attributes},
        {
            **_event("critical_path.data_wait", "data_wait", 4.0, 7.0, 1),
            "attributes": {
                "returned_batch": True,
                "trainer_batch_id": "train_1:actor_train:0",
                "returned_group_keys": [group_key],
            },
        },
    ]

    report = analyze_direct_events(events, warmup_steps=0, measure_updates=1)

    assert report["trainer"]["exclusive_and_other_blocker_available"] is True
    assert report["trainer"]["exclusive_reward_wait_s"]["p50_s"] == pytest.approx(1.0)
    assert report["trainer"]["reward_plus_other_blocker_wait_s"]["p50_s"] == pytest.approx(1.0)


def test_direct_analyzer_rejects_inconsistent_duplicate_group_finalize_spans():
    attributes = {"group_key": [["session-0", 0, None]], "group_index": 0}
    events = [
        {
            **_event("critical_path.session_terminal_admission", "rollout_orchestration", 1.0, 1.0, 0),
            "attributes": attributes,
        },
        {**_event("critical_path.reward_group_finalize", "reward", 2.0, 3.0, 0), "attributes": attributes},
        {**_event("critical_path.reward_group_finalize", "reward", 2.0, 4.0, 0), "attributes": attributes},
    ]

    with pytest.raises(ValueError, match="group-finalize timestamps differ"):
        analyze_direct_events(events, warmup_steps=0)


def test_direct_analyzer_rejects_reward_ready_invariant_mismatch():
    events = [
        {
            **_event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, 0),
            "attributes": {"benchmark_invariant_hash": "ready"},
        },
        {
            **_event("critical_path.weight_serving_ready", "weight_update", 2.0, 3.0, 1),
            "attributes": {"benchmark_invariant_hash": "ready"},
        },
        {
            **_event("critical_path.reward", "reward", 1.1, 1.5, 1),
            "attributes": {
                "benchmark_mode": "dual",
                "reasoning_trigger": "terminal_once",
                "benchmark_invariant_hash": "reward",
            },
        },
    ]

    with pytest.raises(ValueError, match="invariant mismatch"):
        analyze_direct_events(events, warmup_steps=0, measure_updates=1)


def test_direct_analyzer_validates_trigger_and_includes_gpu_samples():
    attributes = {
        "sample_key": ["session-0", 0, None],
        "group_key": [["session-0", 0, None]],
        "benchmark_mode": "dual",
        "reasoning_trigger": "terminal_once",
        "executor_status": "success",
    }
    events = [
        {**_event("critical_path.reward", "reward", 1.0, 2.0, 0), "attributes": attributes},
    ]
    manifests = [
        {
            "record_type": "manifest",
            "role": "judge_accuracy",
            "clock_host": "host-0",
            "engine_rank": 0,
            "num_gpus_per_engine": 1,
            "interval_s": 0.5,
            "nvml_enabled": True,
        }
    ]
    samples = [
        {
            "record_type": "sample",
            "role": "judge_accuracy",
            "clock_host": "host-0",
            "engine_rank": 0,
            "ts": 1.5,
            "gpu": [{"index": 0, "uuid": "GPU-0", "util_percent": 75.0}],
            "sglang": {"sglang:num_running_reqs": 1.0, "sglang:token_usage": 0.25},
        }
    ]

    report = analyze_direct_events(
        events,
        warmup_steps=0,
        expected_reasoning_trigger="terminal_once",
        gpu_sample_manifests=manifests,
        gpu_sample_records=samples,
    )

    assert report["judge_gpu_efficiency"]["judge_accuracy"]["util_percent"]["p50_percent"] == 75.0
    with pytest.raises(ValueError, match="reasoning trigger mismatch"):
        analyze_direct_events(events, warmup_steps=0, expected_reasoning_trigger="per_turn")


def test_latency_analyzer_reports_overlap_and_additive_wall_time_share():
    events = [
        _event("critical_path.rollout_generation", "generation", 0.0, 4.0),
        _event("critical_path.reward", "reward", 2.0, 6.0),
        _event("critical_path.training", "training", 6.0, 9.0),
        _event("critical_path.weight_serving_ready", "weight_update", 9.0, 10.0),
    ]

    report = analyze_events(events)

    assert report["inclusive_occupancy_percent"]["generation"] == pytest.approx(40.0)
    assert report["inclusive_occupancy_percent"]["reward"] == pytest.approx(40.0)
    assert report["active_set_percent"]["generation+reward"] == pytest.approx(20.0)
    assert sum(report["overlap_split_wall_time_percent"].values()) == pytest.approx(100.0)
    assert report["step_makespan"]["p50_s"] == pytest.approx(10.0)
    assert report["measurement_makespan_s"] == pytest.approx(10.0)


def test_optimizer_steps_are_counted_per_publication_round_without_rank_duplication():
    events = []
    for rank in (0, 1):
        for optimizer_step_id in (0, 1, 2):
            events.append(
                {
                    **_event("critical_path.optimizer_step", "training", 1.0, 1.1, step=4),
                    "pid": 3000 + rank,
                    "attributes": {
                        "component": "actor",
                        "global_rank": rank,
                        "optimizer_step_id": optimizer_step_id,
                    },
                }
            )

    report = _optimizer_steps_per_publication_round(events, [4], require_step_ids=True)

    assert report["actor"]["per_round"] == [{"publication_round": 4, "optimizer_step_count": 3}]
    assert report["actor"]["distribution"]["mean_steps"] == pytest.approx(3.0)


def test_optimizer_trace_requires_actor_and_observed_critic_every_round():
    actor = {
        **_event("critical_path.optimizer_step", "training", 1.0, 1.1, step=1),
        "attributes": {"component": "actor", "global_rank": 0, "optimizer_step_id": 0},
    }
    critic = {
        **_event("critical_path.optimizer_step", "training", 1.0, 1.1, step=1),
        "attributes": {"component": "critic", "global_rank": 1, "optimizer_step_id": 0},
    }
    actor_step_two = {
        **actor,
        "step": 2,
    }

    with pytest.raises(ValueError, match="critic.*missing publication round 2"):
        _optimizer_steps_per_publication_round(
            [actor, critic, actor_step_two],
            [1, 2],
            require_step_ids=True,
        )
    with pytest.raises(ValueError, match="actor.*missing publication round"):
        _optimizer_steps_per_publication_round([critic], [1], require_step_ids=True)


def test_expected_trainer_components_infer_critic_and_merge_assertions():
    inferred_events = _reward_events(
        1.0,
        1.1,
        step=1,
        context_hash="sample",
        expected_trainer_components=["actor", "critic"],
    )

    required, summary = _expected_trainer_components(
        inferred_events,
        [1],
        asserted_components={"critic"},
    )

    assert required == {"actor", "critic"}
    assert summary["inferred_from_reward"] == ["actor", "critic"]
    assert summary["asserted_by_cli"] == ["critic"]


def test_latency_analyzer_auto_requires_inferred_critic_optimizer_trace():
    actor_optimizer = {
        **_event("critical_path.optimizer_step", "training", 1.5, 1.6, step=1),
        "attributes": {"component": "actor", "global_rank": 0, "optimizer_step_id": 0},
    }
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.9, 2.0, step=1),
        *_reward_events(
            1.1,
            1.4,
            step=1,
            context_hash="sample",
            expected_trainer_components=["actor", "critic"],
        ),
        actor_optimizer,
    ]

    with pytest.raises(ValueError, match="critic.*missing publication round"):
        analyze_events(
            events,
            warmup_steps=1,
            measure_updates=1,
            require_optimizer_step_ids=True,
        )


def test_latency_analyzer_uses_fixed_k_ready_to_ready_window():
    events = [
        _event("critical_path.rollout_generation", "generation", -100.0, -90.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 9.0, 10.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 19.0, 20.0, step=1),
        _event("critical_path.weight_serving_ready", "weight_update", 34.0, 35.0, step=2),
        *_reward_events(10.0, 10.0, step=1, context_hash="one"),
        *_reward_events(20.0, 20.0, step=2, context_hash="two"),
    ]

    report = analyze_events(events, warmup_steps=1, measure_updates=2)

    assert report["ready_boundary_step"] == 0
    assert report["steps"] == [1, 2]
    assert report["fixed_k_ready_to_ready_makespan_s"] == pytest.approx(25.0)
    assert report["per_update_ready_interval"]["p50_s"] == pytest.approx(12.5)
    assert report["per_update_inclusive_occupancy_percent"]["generation"]["mean_percent"] == pytest.approx(0.0)
    assert report["per_update_inclusive_occupancy_percent"]["weight_update"]["mean_percent"] == pytest.approx(
        (10.0 + 100.0 / 15.0) / 2.0
    )


def test_candidate_analysis_uses_baseline_selected_ready_steps():
    candidate = [
        _event("critical_path.weight_serving_ready", "weight_update", -2.0, -1.0, step=-1),
        _event("critical_path.weight_serving_ready", "weight_update", -1.0, 0.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 9.0, 10.0, step=1),
        _event("critical_path.weight_serving_ready", "weight_update", 19.0, 20.0, step=2),
        *_reward_events(0.0, 0.0, step=1, context_hash="one"),
        *_reward_events(10.0, 10.0, step=2, context_hash="two"),
    ]

    report = analyze_events(
        candidate,
        warmup_steps=1,
        measure_updates=2,
        required_ready_steps=[0, 1, 2],
    )

    assert report["ready_boundary_step"] == 0
    assert report["steps"] == [1, 2]
    assert report["fixed_k_ready_to_ready_makespan_s"] == pytest.approx(20.0)


def test_fixed_k_strict_coverage_rejects_missing_critical_stages():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(1.1, 1.5, step=1, context_hash="sample"),
    ]

    with pytest.raises(ValueError, match="incomplete critical-stage trace coverage"):
        analyze_events(
            events,
            warmup_steps=1,
            measure_updates=1,
            require_complete_coverage=True,
        )


def test_sample_identity_coverage_requires_exact_sample_local_spans():
    reward_parent = _reward_events(1.0, 1.5, step=1, context_hash="sample")[0]
    attributes = dict(reward_parent["attributes"])
    events = [
        reward_parent,
        {**_event("critical_path.reward_context_build", "reward", 0.9, 1.0, step=1), "attributes": attributes},
        {**_event("critical_path.transfer", "transfer", 1.5, 1.6, step=1), "attributes": attributes},
        {**_event("critical_path.rollout_generation", "generation", 0.2, 0.8, step=1), "attributes": attributes},
        {**_event("critical_path.rollout_generation", "generation", 0.8, 0.9, step=1), "attributes": attributes},
    ]

    coverage = _sample_identity_coverage(events, [1])

    assert coverage["complete"] is True
    assert coverage["expected_identity_count"] == 1
    assert coverage["expected_identity_count_by_step"] == {"1": 1}

    duplicate_transfer = {**events[2], "start_s": 1.6, "end_s": 1.7}
    incomplete = _sample_identity_coverage([*events, duplicate_transfer], [1])
    assert incomplete["complete"] is False
    assert incomplete["issues"][0]["violations"]["critical_path.transfer"]["observed"] == 2


def test_strict_analyzer_rejects_global_stage_coverage_without_sample_identity_coverage():
    reward_parent = _reward_events(1.1, 1.3, step=1, context_hash="sample")[0]
    attributes = dict(reward_parent["attributes"])
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.9, 1.0, step=0),
        _event("critical_path.rollout_generation", "generation", 1.0, 1.1, step=1),
        reward_parent,
        {**_event("critical_path.reward_context_build", "reward", 1.0, 1.1, step=1), "attributes": attributes},
        {**_event("critical_path.transfer", "transfer", 1.3, 1.4, step=1), "attributes": attributes},
        {**_event("critical_path.training_schedule", "training", 1.4, 1.6, step=1), "pid": 1},
        _event("critical_path.optimizer_step", "training", 1.5, 1.6, step=1),
        _event("critical_path.weight_update", "weight_update", 1.6, 1.9, step=1),
        _event("critical_path.weight_serving_ready", "weight_update", 1.9, 2.0, step=1),
    ]

    with pytest.raises(ValueError, match="sample-identity coverage"):
        analyze_events(
            events,
            warmup_steps=1,
            measure_updates=1,
            require_complete_coverage=True,
        )


def test_clock_gate_checks_nonoverlapping_reward_wip_before_wall_filter():
    boundary_zero = {
        **_event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        "attributes": {"clock_host": "host-a"},
    }
    boundary_one = {
        **_event("critical_path.weight_serving_ready", "weight_update", 1.9, 2.0, step=1),
        "attributes": {"clock_host": "host-a"},
    }
    measured = _reward_events(1.1, 1.5, step=1, context_hash="measured", clock_host="host-a")
    untrusted_future = _reward_events(100.0, 101.0, step=2, context_hash="future")

    with pytest.raises(ValueError, match="missing clock_host provenance"):
        analyze_events(
            [boundary_zero, boundary_one, *measured, *untrusted_future],
            warmup_steps=1,
            measure_updates=1,
            require_clock_host=True,
        )


def test_paired_fixed_k_comparison_rejects_missing_ready_step():
    baseline = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        _event("critical_path.weight_serving_ready", "weight_update", 2.0, 3.0, step=2),
    ]
    candidate = [baseline[0], baseline[2]]

    with pytest.raises(ValueError, match="missing serving-ready steps"):
        _paired_makespan_delta(baseline, candidate, warmup_steps=1, measure_updates=2)


def test_paired_fixed_k_comparison_requires_reward_workload_identity_each_step():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(1.1, 1.5, step=2, context_hash="future"),
    ]

    with pytest.raises(ValueError, match="missing reward workload identities"):
        _paired_makespan_delta(events, events, warmup_steps=1, measure_updates=1)


def test_paired_fixed_k_comparison_rejects_changed_trajectory_content():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    baseline = boundaries + _reward_events(1.1, 1.5, step=1, group_index=2, sample_index=3, context_hash="baseline")
    candidate = boundaries + _reward_events(1.1, 1.6, step=1, group_index=2, sample_index=3, context_hash="candidate")

    with pytest.raises(ValueError, match="identities, or contents differ"):
        _paired_makespan_delta(baseline, candidate, warmup_steps=1, measure_updates=1)


def test_paired_fixed_k_uses_trajectory_hash_when_judge_context_scopes_differ():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    baseline_reward = _reward_events(1.1, 1.5, step=1, group_index=2, sample_index=3, context_hash="full")
    candidate_reward = _reward_events(1.1, 1.6, step=1, group_index=2, sample_index=3, context_hash="accuracy")
    for event in [*baseline_reward, *candidate_reward]:
        event["attributes"]["trajectory_hash"] = "same-whole-trajectory"

    report = _paired_makespan_delta(
        boundaries + baseline_reward,
        boundaries + candidate_reward,
        warmup_steps=1,
        measure_updates=1,
    )

    assert report["overlapping_reward_workload"]["equal"] is True


def test_paired_fixed_k_comparison_rejects_changed_recorded_training_reward():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    baseline = boundaries + _reward_events(
        1.1,
        1.5,
        step=1,
        context_hash="same",
        recorded_reward_hash="reward-a",
    )
    candidate = boundaries + _reward_events(
        1.1,
        1.5,
        step=1,
        context_hash="same",
        recorded_reward_hash="reward-b",
    )

    with pytest.raises(ValueError, match="identities, or contents differ"):
        _paired_makespan_delta(baseline, candidate, warmup_steps=1, measure_updates=1)


def test_paired_fixed_k_comparison_rejects_reward_changing_modes_by_default():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    accuracy = boundaries + _reward_events(1.1, 1.5, step=1, context_hash="same", mode="accuracy")
    dual = boundaries + _reward_events(1.1, 1.5, step=1, context_hash="same", mode="dual")

    with pytest.raises(ValueError, match="requires recorded/shadow modes"):
        _paired_makespan_delta(accuracy, dual, warmup_steps=1, measure_updates=1)

    exploratory = _paired_makespan_delta(
        accuracy,
        dual,
        warmup_steps=1,
        measure_updates=1,
        allow_reward_changing_pair=True,
    )
    assert exploratory["training_reward_paired"] is False
    assert exploratory["valid_for_causal_latency_inference"] is False
    assert exploratory["validity_issues"] == ["training_reward_not_paired"]


def test_paired_fixed_k_comparison_rejects_overlapping_future_work_by_default():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    measured = _reward_events(1.1, 1.5, step=1, context_hash="same")
    baseline = boundaries + measured + _reward_events(1.6, 1.8, step=2, context_hash="future")
    candidate = boundaries + measured

    with pytest.raises(ValueError, match="reward work overlapping.*differs"):
        _paired_makespan_delta(baseline, candidate, warmup_steps=1, measure_updates=1)


def test_paired_fixed_k_can_label_exploratory_overlapping_workload_mismatch():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    measured = _reward_events(1.1, 1.5, step=1, context_hash="same")
    baseline = boundaries + measured + _reward_events(1.6, 1.8, step=2, context_hash="future")
    candidate = boundaries + measured

    report = _paired_makespan_delta(
        baseline,
        candidate,
        warmup_steps=1,
        measure_updates=1,
        allow_overlapping_workload_mismatch=True,
    )

    assert report["overlapping_reward_workload"]["equal"] is False
    assert report["valid_for_causal_latency_inference"] is False
    assert report["validity_issues"] == ["overlapping_reward_workload_mismatch"]


def test_fixed_k_rejects_baseline_step_gap():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        _event("critical_path.weight_serving_ready", "weight_update", 2.0, 3.0, step=3),
    ]

    with pytest.raises(ValueError, match="not contiguous"):
        analyze_events(events, warmup_steps=1, measure_updates=2)


def test_fixed_k_rejects_shadow_judge_failure():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(
            1.1,
            1.5,
            step=1,
            context_hash="failed",
            mode="dual_shadow",
            executor_status="shadow_judge_error",
        ),
    ]

    with pytest.raises(ValueError, match="invalid reward outcomes"):
        analyze_events(events, warmup_steps=1, measure_updates=1)


def test_terminal_reward_snapshot_survives_jsonl_round_trip(tmp_path):
    from relax.utils.training.train_dump_utils import save_reward_trace_snapshots_jsonl

    snapshot = {
        "agentic_trace": {
            "events": {"reward_arrive_at": 1.0, "reward_end_at": 2.0},
            "reward": {
                "sample_key": ["failed", 0],
                "sample_index": None,
                "pipeline_status": "cancelled",
                "terminal_outcome": "group_rejected",
            },
        }
    }
    save_reward_trace_snapshots_jsonl(
        Namespace(rollout_result_dir=str(tmp_path)),
        rollout_id=4,
        snapshots=[snapshot],
    )

    events, source = load_variant_events(tmp_path)
    report = analyze_events(events, source=source)

    assert source == "rollout_jsonl"
    assert report["event_count"] == 1
    assert events[0]["step"] == 4


def test_rollout_jsonl_loader_expands_reward_per_turn_judges(tmp_path):
    record = {
        "rollout_id": 0,
        "sample_index": 0,
        "latency_trace": {
            "events": {"reward_arrive_at": 3.0, "reward_end_at": 4.0},
            "reward": {
                "reasoning_trigger": "per_turn",
                "reasoning_execution_trigger": "per_turn",
                "per_turn_judge_count": 1,
                "per_turn_judges": [
                    {
                        "turn_index": 0,
                        "role": "judge_multiturn_vlm",
                        "status": "success",
                        "events": {
                            "turn_judge_trigger_at": 1.0,
                            "turn_judge_queue_enter_at": 1.1,
                            "turn_judge_request_end_at": 2.0,
                            "turn_judge_end_at": 2.1,
                        },
                        "judge": {
                            "attempt_count": 1,
                            "invalid_response_count": 0,
                            "queue_elapsed_s": 0.1,
                            "http_elapsed_s": 0.9,
                        },
                    }
                ],
            },
            "turns": [{"events": {}}],
        },
    }
    path = tmp_path / "0.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    events, source = load_variant_events(path)
    requests = [event for event in events if event["name"] == "critical_path.judge_request"]

    assert source == "rollout_jsonl"
    assert len(requests) == 1
    assert requests[0]["attributes"]["request_kind"] == "per_turn_vlm"
    assert requests[0]["attributes"]["attempt_count"] == 1


def test_fixed_k_rejects_missing_reward_statuses():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        {
            **_event("critical_path.reward", "reward", 1.1, 1.5, step=1),
            "attributes": {
                "benchmark_mode": "recorded",
                "group_index": 0,
                "sample_index": 0,
                "context_hash": "missing-status",
            },
        },
    ]

    with pytest.raises(ValueError, match="invalid reward outcomes"):
        analyze_events(events, warmup_steps=1, measure_updates=1)


def test_fixed_k_rejects_wrong_judge_branches_for_mode():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(1.1, 1.5, step=1, context_hash="missing-vlm", mode="accuracy"),
    ]
    events[2]["attributes"]["benchmark_mode"] = "dual"
    events[3]["attributes"]["benchmark_mode"] = "dual"

    with pytest.raises(ValueError, match="invalid reward outcomes"):
        analyze_events(events, warmup_steps=1, measure_updates=1)


def _rejected_group_reward_event(context_hash: str) -> dict:
    return {
        **_event("critical_path.reward", "reward", 1.1, 1.5, step=1),
        "attributes": {
            "benchmark_mode": "dual",
            "group_index": 0,
            "sample_index": 0,
            "context_hash": context_hash,
            "executor_status": "rejected",
            "pipeline_status": "error",
            "terminal_outcome": "group_rejected",
        },
    }


def test_fixed_k_rejects_group_rejection_by_default():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        _rejected_group_reward_event("rejected-group"),
    ]

    with pytest.raises(ValueError, match=r"invalid reward outcomes.*rejected=1, tolerance=0"):
        analyze_events(events, warmup_steps=1, measure_updates=1)


def test_fixed_k_allow_rejected_samples_tolerates_bounded_group_rejections():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        _rejected_group_reward_event("rejected-group"),
    ]

    report = analyze_events(events, warmup_steps=1, measure_updates=1, max_rejected_samples=1)
    assert report["observed_benchmark_mode"] == "dual"


def test_fixed_k_allow_rejected_samples_still_rejects_unrelated_violations():
    other_violation_events = _reward_events(
        1.2, 1.6, step=1, context_hash="missing-vlm", mode="accuracy", group_index=1
    )
    other_violation_events[0]["attributes"]["benchmark_mode"] = "dual"
    other_violation_events[1]["attributes"]["benchmark_mode"] = "dual"
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        _rejected_group_reward_event("rejected-group"),
        *other_violation_events,
    ]

    # A tolerated group rejection must not mask an unrelated real violation (here: a
    # "dual" sample missing its multi_turn_reasoning judge branch) --
    # max_rejected_samples only ever widens the group-rejection allowance, never the
    # other checks.
    with pytest.raises(ValueError, match=r"invalid reward outcomes.*rejected=1, tolerance=1, other=1"):
        analyze_events(events, warmup_steps=1, measure_updates=1, max_rejected_samples=1)


def test_fixed_k_accepts_per_turn_judges_as_a_separate_branch() -> None:
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    reward_events = _reward_events(1.1, 1.5, step=1, context_hash="per-turn", mode="dual_shadow")
    # The terminal answer judge remains a reward branch; the reasoning VLM
    # moves to its own interaction span rather than pretending to be terminal
    # reward work.
    reward_events = reward_events[:2]
    attributes = reward_events[0]["attributes"]
    attributes.update(
        {
            "reasoning_trigger": "per_turn",
            "reasoning_execution_trigger": "per_turn",
            "per_turn_judge_count": 1,
            "trajectory_hash": "trajectory-per-turn",
        }
    )
    reward_events[1]["attributes"] = attributes
    events.extend(reward_events)
    events.append(
        {
            **_event("critical_path.turn_judge", "turn_judge", 1.15, 1.35, step=1),
            "attributes": attributes,
        }
    )

    report = analyze_events(
        events,
        warmup_steps=1,
        measure_updates=1,
        expected_benchmark_mode="dual_shadow",
        expected_reasoning_trigger="per_turn",
    )

    assert report["observed_reasoning_trigger"] == "per_turn"
    assert report["reasoning_execution_trigger_counts"] == {"per_turn": 1}
    assert report["inclusive_occupancy_percent"]["turn_judge"] > 0.0


def test_fixed_k_allows_terminal_fallback_for_zero_interaction_per_turn_sample() -> None:
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(1.1, 1.5, step=1, context_hash="fallback", mode="dual_shadow"),
    ]
    for event in events[2:]:
        event["attributes"].update(
            {
                "reasoning_trigger": "per_turn",
                "reasoning_execution_trigger": "terminal_once_fallback",
                "per_turn_judge_count": 0,
            }
        )

    report = analyze_events(
        events,
        warmup_steps=1,
        measure_updates=1,
        expected_benchmark_mode="dual_shadow",
        expected_reasoning_trigger="per_turn",
    )

    assert report["observed_reasoning_trigger"] == "per_turn"
    assert report["per_turn_fallback_sample_count"] == 1.0


def test_fixed_k_rejects_declared_mode_mismatch():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(1.1, 1.5, step=1, context_hash="mode", mode="recorded"),
    ]

    with pytest.raises(ValueError, match="benchmark mode mismatch"):
        analyze_events(
            events,
            warmup_steps=1,
            measure_updates=1,
            expected_benchmark_mode="dual_shadow",
        )


def test_fixed_k_rejects_overlapping_future_step_failure():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 2.0, 3.0, step=1),
        *_reward_events(
            1.5,
            2.5,
            step=2,
            context_hash="future-failure",
            mode="dual_shadow",
            executor_status="shadow_judge_error",
        ),
    ]

    with pytest.raises(ValueError, match="invalid reward outcomes"):
        analyze_events(events, warmup_steps=1, measure_updates=1)


def test_paired_fixed_k_rejects_incomplete_expected_group_cardinality():
    events = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
        *_reward_events(1.1, 1.5, step=1, context_hash="only-one"),
    ]

    with pytest.raises(ValueError, match="workload cardinality mismatch"):
        _paired_makespan_delta(
            events,
            events,
            warmup_steps=1,
            measure_updates=1,
            expected_groups_per_round=1,
            expected_samples_per_group=2,
        )


def test_paired_fixed_k_rejects_changed_benchmark_invariants():
    boundaries = [
        _event("critical_path.weight_serving_ready", "weight_update", 0.0, 1.0, step=0),
        _event("critical_path.weight_serving_ready", "weight_update", 1.0, 2.0, step=1),
    ]
    baseline = boundaries + _reward_events(1.1, 1.5, step=1, context_hash="same", invariant_hash="a")
    candidate = boundaries + _reward_events(1.1, 1.5, step=1, context_hash="same", invariant_hash="b")

    with pytest.raises(ValueError, match="changed benchmark invariants"):
        _paired_makespan_delta(
            baseline,
            candidate,
            warmup_steps=1,
            measure_updates=1,
            require_benchmark_invariant_hash=True,
        )


def test_duplicate_variant_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate variant"):
        _reject_duplicate_names([("dual", "a"), ("dual", "b")], label="variant")


def test_timeline_dedup_preserves_distinct_contexts_at_same_timestamp(tmp_path):
    path = tmp_path / "timeline_step_1.json"
    base = {
        "name": "critical_path.reward",
        "ph": "X",
        "ts": 1_000_000,
        "dur": 0,
        "pid": 1,
        "tid": 1,
    }
    path.write_text(
        json.dumps(
            [
                {**base, "args": {"step": 1, "context_hash": "a"}},
                {**base, "args": {"step": 1, "context_hash": "b"}},
            ]
        ),
        encoding="utf-8",
    )

    events, source = load_variant_events(path)

    assert source == "timeline"
    assert len(events) == 2


def test_timeline_loader_preserves_optimizer_spans(tmp_path):
    path = tmp_path / "timeline_step_1.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "critical_path.optimizer_step",
                    "ph": "X",
                    "ts": 1_000_000,
                    "dur": 100_000,
                    "pid": 1,
                    "tid": 1,
                    "args": {
                        "step": 1,
                        "component": "actor",
                        "global_rank": 0,
                        "optimizer_step_id": 0,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    events, source = load_variant_events(path)

    assert source == "timeline"
    assert events[0]["stage"] == "training"


def test_rollout_loader_adds_explicit_weight_serving_ready_markers(tmp_path):
    rollout_path = tmp_path / "rollout.jsonl"
    rollout_path.write_text("", encoding="utf-8")
    marker_path = tmp_path / "weight_serving_ready.jsonl"
    marker_path.write_text(
        json.dumps(
            {
                "event": "weight_serving_ready",
                "step": 3,
                "wall_time_s": 12.5,
                "clock_host": "node-a",
                "pid": 42,
                "benchmark_mode": "dual",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events, source = load_variant_events(rollout_path, ready_marker_path=marker_path)

    assert source == "rollout_jsonl+ready_markers"
    assert events == [
        {
            "name": "critical_path.weight_serving_ready",
            "stage": "weight_update",
            "start_s": 12.5,
            "end_s": 12.5,
            "step": 3,
            "pid": 42,
            "tid": 0,
            "attributes": {
                "event": "weight_serving_ready",
                "step": 3,
                "wall_time_s": 12.5,
                "clock_host": "node-a",
                "pid": 42,
                "benchmark_mode": "dual",
            },
        }
    ]


def test_rollout_trace_loader_preserves_per_turn_lineage_counts():
    raw_events = _events_from_latency_trace(
        {
            "rollout_id": 0,
            "status": "completed",
            "agent_turns": 8,
            "prompt_token_count": 100,
            "response_token_count": 20,
            "total_token_count": 120,
            "image_count": 8,
            "image_token_count": 4096,
            "weight_versions": ["2"],
            "latency_trace": {
                "events": {"reward_arrive_at": 1.0, "reward_end_at": 2.0},
                "reward": {
                    "reasoning_trigger": "per_turn",
                    "per_turn_judge_count": 7,
                },
                "turns": [],
                "per_turn_assistant_turn_count": 8,
                "per_turn_off_lineage_judge_count": 0,
            },
        }
    )

    reward_event = next(event for event in raw_events if event["name"] == "critical_path.reward")
    assert reward_event["args"]["per_turn_assistant_turn_count"] == 8
    assert reward_event["args"]["per_turn_off_lineage_judge_count"] == 0
    workload_event = next(event for event in raw_events if event["name"] == "critical_path.sample_workload")
    assert workload_event["args"]["rollout_status"] == "completed"
    assert workload_event["args"]["input_tokens"] == 100
    assert workload_event["args"]["weight_versions"] == ["2"]


def test_rollout_loader_preserves_agentic_accounting_markers(tmp_path):
    path = tmp_path / "agentic_accounting_end.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "agentic_accounting_end",
                "step": 2,
                "wall_time_s": 12.5,
                "clock_host": "node-a",
                "pid": 42,
                "snapshot": {"judge_group_replacements": 0, "interrupted_groups": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events, source = load_variant_events(path)

    assert source == "rollout_jsonl"
    assert events[0]["name"] == "critical_path.agentic_accounting_end"
    assert events[0]["attributes"]["judge_group_replacements"] == 0
