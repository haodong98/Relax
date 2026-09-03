# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.judge_config import (
    dual_judge_benchmark_invariant_hash,
    parse_judge_services_config,
    validate_dual_judge_args,
)


def _service(model: str, gpus: int, port: int, *, media: bool = False) -> dict:
    value = {
        "model_path": model,
        "num_gpus_per_engine": gpus,
        "engine_config": {"max_context_len": 1024},
        "sampling_config": {"temperature": 0.0, "max_response_len": 24},
        "max_input_tokens": 1000,
        "max_output_tokens": 24,
        "timeout_s": 30,
        "max_attempts": 3,
        "max_concurrency": 2,
        "port_base": port,
    }
    if media:
        value.update(max_media_items=4, max_media_total_bytes=1024, max_pixels_per_item=4096)
    return value


def _config() -> dict:
    return {
        "schema_version": 1,
        "max_group_replacements_per_step": 8,
        "answer_accuracy": _service("/accuracy", 1, 16000),
        "multi_turn_reasoning": _service("/vlm", 2, 17000, media=True),
    }


def test_judge_config_builds_role_scoped_legacy_namespace():
    config = parse_judge_services_config(_config())
    scoped = config.multi_turn_reasoning.as_genrm_namespace(Namespace(num_gpus_per_node=4, offload_rollout=True))
    assert scoped.genrm_model_path == "/vlm"
    assert scoped.genrm_num_gpus == scoped.genrm_num_gpus_per_engine == 2
    assert scoped.genrm_port_base == 17000
    assert scoped.genrm_max_concurrency == 2
    assert scoped.offload_rollout is False


def test_judge_config_rejects_overlap_and_unknown_keys():
    value = _config()
    value["multi_turn_reasoning"]["port_base"] = 16500
    with pytest.raises(ValueError, match="overlap"):
        parse_judge_services_config(value)
    value = _config()
    value["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        parse_judge_services_config(value)


def test_judge_config_accepts_benchmark_modes_and_defaults_to_dual():
    assert parse_judge_services_config(_config()).benchmark_mode == "dual"
    for mode in ("recorded", "accuracy", "accuracy_shadow", "dual", "dual_shadow"):
        value = _config()
        value["benchmark_mode"] = mode
        assert parse_judge_services_config(value).benchmark_mode == mode
    value = _config()
    value["benchmark_mode"] = "reasoning_only"
    with pytest.raises(ValueError, match="benchmark_mode"):
        parse_judge_services_config(value)


def test_judge_config_accepts_explicit_reasoning_triggers():
    assert parse_judge_services_config(_config()).reasoning_trigger == "terminal_once"
    value = _config()
    value["reasoning_trigger"] = "per_turn"
    assert parse_judge_services_config(value).reasoning_trigger == "per_turn"
    value["reasoning_trigger"] = "invalid"
    with pytest.raises(ValueError, match="reasoning_trigger"):
        parse_judge_services_config(value)


def test_judge_config_accepts_optional_turn_judge_barrier_timeout():
    assert parse_judge_services_config(_config()).turn_judge_barrier_timeout_s is None
    value = _config()
    value["turn_judge_barrier_timeout_s"] = 120
    assert parse_judge_services_config(value).turn_judge_barrier_timeout_s == 120.0
    value["turn_judge_barrier_timeout_s"] = 0
    with pytest.raises(ValueError, match="turn_judge_barrier_timeout_s"):
        parse_judge_services_config(value)


def test_turn_judge_barrier_timeout_is_part_of_the_benchmark_invariant():
    """It bounds exposed wait, so a paired comparison must hold it fixed."""
    baseline = _config()
    changed = _config()
    changed["turn_judge_barrier_timeout_s"] = 120
    baseline_args = Namespace(judge_services=parse_judge_services_config(baseline))
    changed_args = Namespace(judge_services=parse_judge_services_config(changed))

    assert dual_judge_benchmark_invariant_hash(baseline_args) != dual_judge_benchmark_invariant_hash(changed_args)


def test_dual_judge_args_enforce_agentic_resources_and_conflicts():
    args = Namespace(
        judge_services_config=_config(),
        rm_type="dual-agentic-judge",
        use_agentic_rollout=True,
        group_rm=False,
        custom_rm_path=None,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        genrm_model_path=None,
        genrm_engine_config=None,
        genrm_sampling_config=None,
        reward_key="score",
        resource={"judge_accuracy": [1, 1], "judge_multiturn_vlm": [1, 2]},
        num_gpus_per_node=4,
    )
    validate_dual_judge_args(args)
    assert args.judge_services.max_group_replacements_per_step == 8
    assert not hasattr(args, "dual_judge_benchmark_invariant_hash")
    args.group_rm = True
    with pytest.raises(ValueError, match="group-rm"):
        validate_dual_judge_args(args)


def test_benchmark_invariant_hash_excludes_ab_dimensions_and_tracks_runtime_settings():
    recorded = _config()
    recorded["benchmark_mode"] = "recorded"
    dual = _config()
    dual["benchmark_mode"] = "dual_shadow"
    recorded_args = Namespace(
        judge_services=parse_judge_services_config(recorded),
        reward_max_concurrency=64,
        resource={"judge_accuracy": [1, 1], "judge_multiturn_vlm": [1, 2]},
    )
    dual_args = Namespace(
        judge_services=parse_judge_services_config(dual),
        reward_max_concurrency=64,
        resource={"judge_accuracy": [1, 1], "judge_multiturn_vlm": [1, 2]},
    )

    assert dual_judge_benchmark_invariant_hash(recorded_args) == dual_judge_benchmark_invariant_hash(dual_args)

    dual_args.reward_max_concurrency = 32
    assert dual_judge_benchmark_invariant_hash(recorded_args) != dual_judge_benchmark_invariant_hash(dual_args)

    dual_args.reward_max_concurrency = 64
    recorded_args.micro_batch_size = dual_args.micro_batch_size = 1
    recorded_args.use_critic = dual_args.use_critic = False
    assert dual_judge_benchmark_invariant_hash(recorded_args) == dual_judge_benchmark_invariant_hash(dual_args)

    dual_args.micro_batch_size = 8
    assert dual_judge_benchmark_invariant_hash(recorded_args) != dual_judge_benchmark_invariant_hash(dual_args)
    dual_args.micro_batch_size = 1
    dual_args.use_critic = True
    assert dual_judge_benchmark_invariant_hash(recorded_args) != dual_judge_benchmark_invariant_hash(dual_args)


def test_benchmark_invariant_hash_honors_controller_frozen_value():
    args = Namespace(judge_services=parse_judge_services_config(_config()), reward_max_concurrency=64)
    expected = dual_judge_benchmark_invariant_hash(args)
    args._dual_judge_frozen_invariant_hash = expected
    args.reward_max_concurrency = 1

    assert dual_judge_benchmark_invariant_hash(args) == expected


def test_benchmark_invariant_hash_excludes_declared_trigger_dimension():
    terminal = _config()
    terminal["reasoning_trigger"] = "terminal_once"
    per_turn = _config()
    per_turn["reasoning_trigger"] = "per_turn"
    terminal_args = Namespace(judge_services=parse_judge_services_config(terminal))
    per_turn_args = Namespace(judge_services=parse_judge_services_config(per_turn))

    assert dual_judge_benchmark_invariant_hash(terminal_args) == dual_judge_benchmark_invariant_hash(per_turn_args)


@pytest.mark.parametrize(
    ("field_name", "baseline", "changed"),
    [
        ("balance_data", False, True),
        ("hybrid", False, True),
        ("true_on_policy_mode", False, True),
        ("use_dynamic_batch_size", False, True),
        ("use_dynamic_global_batch_size", False, True),
    ],
)
def test_benchmark_invariant_hash_tracks_resolved_pipeline_settings(field_name, baseline, changed):
    config = parse_judge_services_config(_config())
    baseline_args = Namespace(judge_services=config, **{field_name: baseline})
    changed_args = Namespace(judge_services=config, **{field_name: changed})

    assert dual_judge_benchmark_invariant_hash(baseline_args) != dual_judge_benchmark_invariant_hash(changed_args)
