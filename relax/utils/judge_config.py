# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validation and role-scoped configuration for agentic local judges."""

from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Any


DUAL_JUDGE_RM_TYPE = "dual-agentic-judge"
JUDGE_ROLE_BY_COMPONENT = {
    "answer_accuracy": "judge_accuracy",
    "multi_turn_reasoning": "judge_multiturn_vlm",
}
JUDGE_COMPONENT_BY_ROLE = {value: key for key, value in JUDGE_ROLE_BY_COMPONENT.items()}
JUDGE_BENCHMARK_MODES = {
    "recorded",
    "accuracy",
    "accuracy_shadow",
    "dual",
    "dual_shadow",
}
JUDGE_REASONING_TRIGGERS = {"terminal_once", "per_turn"}
_TOP_LEVEL_REQUIRED_KEYS = {"schema_version", "max_group_replacements_per_step", *JUDGE_ROLE_BY_COMPONENT}
_TOP_LEVEL_OPTIONAL_KEYS = {"benchmark_mode", "reasoning_trigger", "turn_judge_barrier_timeout_s"}
_SPEC_REQUIRED_KEYS = {
    "model_path",
    "num_gpus_per_engine",
    "engine_config",
    "sampling_config",
    "max_input_tokens",
    "max_output_tokens",
    "timeout_s",
    "max_attempts",
    "max_concurrency",
    "port_base",
}
_SPEC_OPTIONAL_KEYS = {"max_media_items", "max_media_total_bytes", "max_pixels_per_item"}
JUDGE_PORT_BAND_SIZE = 1000
_BENCHMARK_INVARIANT_ARG_NAMES = (
    "actor_num_gpus_per_node",
    "actor_num_nodes",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "advantage_estimator",
    "agentic_prepare_pool_size",
    "agentic_reasoning_parser",
    "agentic_tool_call_parser",
    "apply_chat_template",
    "apply_chat_template_kwargs",
    "attention_backend",
    "balance_data",
    "bf16",
    "clip_grad",
    "colocate",
    "compute_advantages_and_returns",
    "context_parallel_size",
    "critic_load",
    "critic_lr",
    "critic_num_gpus_per_node",
    "critic_num_nodes",
    "custom_generate_function_path",
    "custom_prompt_path",
    "data_source_path",
    "data_parallel_sharding_strategy",
    "debug_rollout_only",
    "debug_train_only",
    "disable_cuda_graph",
    "disable_rollout_trim_samples",
    "dynamic_context_parallel",
    "dynamic_sampling_filter_path",
    "entropy_coef",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "fp16",
    "fp8",
    "ffn_hidden_size",
    "fully_async",
    "gamma",
    "genrm_engine_config",
    "genrm_model_path",
    "genrm_num_gpus_per_engine",
    "genrm_sampling_config",
    "global_batch_size",
    "grpo_std_normalization",
    "hidden_size",
    "hybrid",
    "input_key",
    "keep_old_actor",
    "label_key",
    "kl_coef",
    "kl_loss_coef",
    "lambd",
    "load",
    "log_probs_max_tokens_per_gpu",
    "loss_type",
    "lr",
    "max_staleness",
    "max_tokens_per_gpu",
    "micro_batch_size",
    "metadata_key",
    "multimodal_keys",
    "n_samples_per_prompt",
    "normalize_advantages",
    "num_critic_only_steps",
    "num_data_storage_units",
    "num_gpus_per_node",
    "num_iters_per_train_update",
    "num_microbatches",
    "num_attention_heads",
    "num_experts",
    "num_layers",
    "num_layers_per_virtual_pipeline_stage",
    "num_nodes",
    "num_query_groups",
    "num_rollout",
    "num_steps_per_rollout",
    "offload_rollout",
    "offload_train",
    "optimizer",
    "optimizer_cpu_offload",
    "optimizer_offload_fraction",
    "over_sampling_batch_size",
    "partial_rollout",
    "partial_rollout_max_aborted_count",
    "per_rank_fetch",
    "polling_mode",
    "prefetch_chunk_size",
    "prefetch_max_cached",
    "prefetch_num_workers",
    "pipeline_model_parallel_size",
    "prompt_data",
    "qkv_format",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "ref_load",
    "resource",
    "reward_max_concurrency",
    "reward_key",
    "rm_type",
    "rollout_batch_size",
    "rollout_external",
    "rollout_http_timeout",
    "rollout_max_context_len",
    "rollout_max_prompt_len",
    "rollout_max_response_len",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "rollout_seed",
    "rollout_shuffle",
    "rollout_stop_token_ids",
    "rollout_temperature",
    "rollout_top_k",
    "rollout_top_p",
    "sequence_parallel",
    "selective_offload",
    "seed",
    "sglang_attention_backend",
    "sglang_chunked_prefill_size",
    "sglang_cuda_graph_bs",
    "sglang_cuda_graph_max_bs",
    "sglang_disable_cuda_graph",
    "sglang_disable_radix_cache",
    "sglang_dp_size",
    "sglang_enable_dp_attention",
    "sglang_ep_size",
    "sglang_expert_parallel_size",
    "sglang_hf_checkpoint",
    "sglang_max_prefill_tokens",
    "sglang_max_running_requests",
    "sglang_mem_fraction_static",
    "sglang_page_size",
    "sglang_server_concurrency",
    "sglang_speculative_algorithm",
    "sglang_speculative_num_draft_tokens",
    "sglang_speculative_num_steps",
    "streaming_buffer_size",
    "system_prompt",
    "tensor_model_parallel_size",
    "train_backend",
    "train_iters",
    "train_memory_margin_bytes",
    "tool_key",
    "true_on_policy_mode",
    "update_weight_buffer_size",
    "update_weights_interval",
    "use_critic",
    "use_dynamic_batch_size",
    "use_dynamic_global_batch_size",
    "use_kl_loss",
    "use_streaming_dataset",
    "use_metrics_service",
    "use_rollout_logprobs",
    "use_tis",
)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class JudgeServiceSpec:
    component: str
    role: str
    model_path: str
    num_gpus_per_engine: int
    engine_config: dict[str, Any]
    sampling_config: dict[str, Any]
    max_input_tokens: int
    max_output_tokens: int
    timeout_s: float
    max_attempts: int
    max_concurrency: int
    port_base: int
    max_media_items: int | None = None
    max_media_total_bytes: int | None = None
    max_pixels_per_item: int | None = None

    @property
    def max_context_len(self) -> int:
        return int(self.engine_config["max_context_len"])

    @classmethod
    def from_dict(cls, component: str, value: Any) -> "JudgeServiceSpec":
        if not isinstance(value, dict):
            raise TypeError(f"judge_services_config.{component} must be a JSON object")
        keys = set(value)
        missing = _SPEC_REQUIRED_KEYS - keys
        unknown = keys - _SPEC_REQUIRED_KEYS - _SPEC_OPTIONAL_KEYS
        if missing or unknown:
            raise ValueError(
                f"judge_services_config.{component} has invalid keys: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        model_path = value["model_path"]
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError(f"judge_services_config.{component}.model_path must be a non-empty string")
        engine_config = copy.deepcopy(value["engine_config"])
        sampling_config = copy.deepcopy(value["sampling_config"])
        if not isinstance(engine_config, dict) or not isinstance(sampling_config, dict):
            raise TypeError(f"judge_services_config.{component} engine_config and sampling_config must be objects")
        max_context_len = _positive_int(engine_config.get("max_context_len"), f"{component}.max_context_len")
        max_input_tokens = _positive_int(value["max_input_tokens"], f"{component}.max_input_tokens")
        max_output_tokens = _positive_int(value["max_output_tokens"], f"{component}.max_output_tokens")
        sampling_max_output = sampling_config.get("max_response_len")
        if sampling_max_output != max_output_tokens:
            raise ValueError(
                f"{component}.sampling_config.max_response_len must equal max_output_tokens "
                f"({sampling_max_output!r} != {max_output_tokens})"
            )
        if max_input_tokens + max_output_tokens > max_context_len:
            raise ValueError(
                f"{component}: max_input_tokens + max_output_tokens must be <= max_context_len "
                f"({max_input_tokens} + {max_output_tokens} > {max_context_len})"
            )
        optional_limits = {}
        for key in _SPEC_OPTIONAL_KEYS:
            optional_limits[key] = _positive_int(value[key], f"{component}.{key}") if key in value else None
        return cls(
            component=component,
            role=JUDGE_ROLE_BY_COMPONENT[component],
            model_path=model_path,
            num_gpus_per_engine=_positive_int(value["num_gpus_per_engine"], f"{component}.num_gpus_per_engine"),
            engine_config=engine_config,
            sampling_config=sampling_config,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            timeout_s=_positive_number(value["timeout_s"], f"{component}.timeout_s"),
            max_attempts=_positive_int(value["max_attempts"], f"{component}.max_attempts"),
            max_concurrency=_positive_int(value["max_concurrency"], f"{component}.max_concurrency"),
            port_base=_positive_int(value["port_base"], f"{component}.port_base"),
            **optional_limits,
        )

    def as_genrm_namespace(self, args: Namespace) -> Namespace:
        """Copy the global args and overlay the legacy GenRM engine fields."""
        scoped = Namespace(**vars(args))
        scoped.genrm_model_path = self.model_path
        scoped.genrm_num_gpus = self.num_gpus_per_engine
        scoped.genrm_num_gpus_per_engine = self.num_gpus_per_engine
        scoped.genrm_engine_config = copy.deepcopy(self.engine_config)
        scoped.genrm_sampling_config = copy.deepcopy(self.sampling_config)
        # Dedicated Judge requests may originate from every agentic session
        # shard. Keep the authoritative admission limit at the single GenRM
        # Serve replica instead of treating this client-side value as global.
        scoped.genrm_max_concurrency = self.max_concurrency
        scoped.genrm_port_base = self.port_base
        scoped.genrm_port_band_size = JUDGE_PORT_BAND_SIZE
        scoped.genrm_role = self.role
        scoped.offload_rollout = False
        scoped._genrm_colocate_with_rollout = False
        return scoped


@dataclass(frozen=True)
class JudgeServicesConfig:
    answer_accuracy: JudgeServiceSpec
    multi_turn_reasoning: JudgeServiceSpec
    max_group_replacements_per_step: int = 8
    schema_version: int = 1
    benchmark_mode: str = "dual"
    reasoning_trigger: str = "terminal_once"
    # Hard ceiling on the terminal join of ``per_turn`` sidecar work. Each
    # individual call is already bounded by ``timeout_s * max_attempts`` plus
    # backoff, but ``per_turn`` has ~T times more calls in flight than
    # ``terminal_once``, so a single wedged Judge would otherwise hold the
    # session lock for that full per-call bound. ``None`` keeps the per-call
    # bounds as the only limit.
    turn_judge_barrier_timeout_s: float | None = None

    def by_role(self, role: str) -> JudgeServiceSpec:
        return getattr(self, JUDGE_COMPONENT_BY_ROLE[role])


def dual_judge_benchmark_invariant_hash(args: Namespace) -> str:
    """Hash settings fixed across the declared Judge latency variants.

    ``benchmark_mode`` and ``reasoning_trigger`` are the two explicit A/B
    dimensions. Their observed values remain in every reward trace, while the
    invariant hash proves all other resolved settings were held constant.
    """
    frozen = getattr(args, "_dual_judge_frozen_invariant_hash", None)
    if isinstance(frozen, str) and frozen:
        return frozen
    config = asdict(args.judge_services)
    config.pop("benchmark_mode", None)
    config.pop("reasoning_trigger", None)
    payload = {
        "schema_version": 3,
        "judge_services_excluding_benchmark_mode": config,
        "runtime": {name: getattr(args, name, None) for name in _BENCHMARK_INVARIANT_ARG_NAMES},
    }
    # Some invariant args hold library types rather than JSON scalars -- SGLang
    # >=0.5.10 makes ``attention_backend`` an ``AttnBackend`` enum, which raises
    # "Object of type AttnBackend is not JSON serializable" and kills the reward
    # pipeline at startup. ``default=str`` keeps the hash well-defined for any
    # such value; its string form is just as stable for proving a setting was
    # held constant. schema_version is bumped to 3 because this changes the
    # canonical encoding, so hashes are not comparable with earlier runs.
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_judge_services_config(value: Any) -> JudgeServicesConfig:
    if not isinstance(value, dict):
        raise TypeError("--judge-services-config must be a JSON object")
    keys = set(value)
    missing = _TOP_LEVEL_REQUIRED_KEYS - keys
    unknown = keys - _TOP_LEVEL_REQUIRED_KEYS - _TOP_LEVEL_OPTIONAL_KEYS
    if missing or unknown:
        raise ValueError(
            f"--judge-services-config has invalid keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if value["schema_version"] != 1:
        raise ValueError("--judge-services-config schema_version must be 1")
    benchmark_mode = value.get("benchmark_mode", "dual")
    if not isinstance(benchmark_mode, str) or benchmark_mode not in JUDGE_BENCHMARK_MODES:
        raise ValueError(
            f"judge_services_config.benchmark_mode must be one of {sorted(JUDGE_BENCHMARK_MODES)}, "
            f"got {benchmark_mode!r}"
        )
    reasoning_trigger = value.get("reasoning_trigger", "terminal_once")
    if not isinstance(reasoning_trigger, str) or reasoning_trigger not in JUDGE_REASONING_TRIGGERS:
        raise ValueError(
            f"judge_services_config.reasoning_trigger must be one of {sorted(JUDGE_REASONING_TRIGGERS)}, "
            f"got {reasoning_trigger!r}"
        )
    barrier_timeout_s = value.get("turn_judge_barrier_timeout_s")
    if barrier_timeout_s is not None:
        barrier_timeout_s = _positive_number(barrier_timeout_s, "turn_judge_barrier_timeout_s")
    config = JudgeServicesConfig(
        answer_accuracy=JudgeServiceSpec.from_dict("answer_accuracy", value["answer_accuracy"]),
        multi_turn_reasoning=JudgeServiceSpec.from_dict("multi_turn_reasoning", value["multi_turn_reasoning"]),
        max_group_replacements_per_step=_positive_int(
            value["max_group_replacements_per_step"], "max_group_replacements_per_step"
        ),
        benchmark_mode=benchmark_mode,
        reasoning_trigger=reasoning_trigger,
        turn_judge_barrier_timeout_s=barrier_timeout_s,
    )
    accuracy_band = range(
        config.answer_accuracy.port_base,
        config.answer_accuracy.port_base + JUDGE_PORT_BAND_SIZE,
    )
    reasoning_band = range(
        config.multi_turn_reasoning.port_base,
        config.multi_turn_reasoning.port_base + JUDGE_PORT_BAND_SIZE,
    )
    if accuracy_band.stop > 65536 or reasoning_band.stop > 65536:
        raise ValueError("judge port band exceeds TCP port 65535")
    if max(accuracy_band.start, reasoning_band.start) < min(accuracy_band.stop, reasoning_band.stop):
        raise ValueError("judge port bands overlap; each role reserves 1000 ports from port_base")
    return config


def validate_dual_judge_args(args: Namespace) -> None:
    raw_config = getattr(args, "judge_services_config", None)
    enabled = raw_config is not None or getattr(args, "rm_type", None) == DUAL_JUDGE_RM_TYPE
    if not enabled:
        args.judge_services = None
        return
    if raw_config is None:
        raise ValueError(f"--rm-type {DUAL_JUDGE_RM_TYPE} requires --judge-services-config")
    if getattr(args, "rm_type", None) != DUAL_JUDGE_RM_TYPE:
        raise ValueError(f"--judge-services-config requires --rm-type {DUAL_JUDGE_RM_TYPE}")
    if not getattr(args, "use_agentic_rollout", False):
        raise ValueError("dual-agentic-judge v1 requires --use-agentic-rollout")
    if getattr(args, "group_rm", False):
        raise ValueError("dual-agentic-judge requires --group-rm to remain disabled")
    conflicts = {
        "--custom-rm-path": getattr(args, "custom_rm_path", None),
        "--custom-reward-post-process-path": getattr(args, "custom_reward_post_process_path", None),
        "--agentic-custom-advantage-path": getattr(args, "agentic_custom_advantage_path", None),
        "--genrm-model-path": getattr(args, "genrm_model_path", None),
        "--genrm-engine-config": getattr(args, "genrm_engine_config", None),
        "--genrm-sampling-config": getattr(args, "genrm_sampling_config", None),
        "--rm-type-fallback": getattr(args, "rm_type_fallback", None),
        "--rm-type-infer": True if getattr(args, "rm_type_infer", False) else None,
        "--rm-url": getattr(args, "rm_url", None),
        "--defer-reward-to-post-process": (True if getattr(args, "defer_reward_to_post_process", False) else None),
    }
    active_conflicts = [name for name, value in conflicts.items() if value is not None]
    if active_conflicts:
        raise ValueError(f"dual-agentic-judge is mutually exclusive with: {', '.join(active_conflicts)}")
    if getattr(args, "reward_key", None) != "score":
        raise ValueError("dual-agentic-judge requires --reward-key score")

    config = parse_judge_services_config(raw_config)
    resource = getattr(args, "resource", None) or {}
    for spec in (config.answer_accuracy, config.multi_turn_reasoning):
        entry = resource.get(spec.role)
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"--resource must contain {spec.role!r}: [1, {spec.num_gpus_per_engine}]")
        service_count, gpu_count = entry
        if (
            isinstance(service_count, bool)
            or not isinstance(service_count, int)
            or isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or service_count != 1
            or gpu_count != spec.num_gpus_per_engine
        ):
            raise ValueError(f"--resource {spec.role!r} must be [1, {spec.num_gpus_per_engine}], got {entry!r}")
        num_gpus_per_node = getattr(args, "num_gpus_per_node", None)
        if num_gpus_per_node is not None and spec.num_gpus_per_engine > num_gpus_per_node:
            raise ValueError(
                f"{spec.role}.num_gpus_per_engine={spec.num_gpus_per_engine} exceeds "
                f"num_gpus_per_node={num_gpus_per_node}; dedicated judges are single-node in v1"
            )
    args.judge_services = config
