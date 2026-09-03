# Agentic dual local judges

This is a two-reward ablation: answer accuracy (Qwen3-32B) and complete multi-turn visual reasoning
(Qwen2.5-VL). It does **not** implement the Tool-Calling Reward from the Qwen3-VL reward design.
That design creates the expert target tool-call count offline with Qwen2.5-VL-72B and compares the runtime call count
against it. The offline target-generation model is therefore outside training's critical path; the runtime count
comparison would still have a small local cost, but it is intentionally outside this online-model latency study.

Add the following options to an existing agentic training command:

```bash
--use-agentic-rollout \
--rm-type dual-agentic-judge \
--reward-key score \
--judge-services-config "$(tr -d '\n' < examples/agentic_dual_judge/judge_services.json)" \
--resource '{"actor":[1,4],"rollout":[1,4],"judge_accuracy":[1,1],"judge_multiturn_vlm":[1,2]}'
```

Keep `--group-rm` disabled. Each terminal trajectory is scored concurrently by both required judges and
the only GRPO normalization value is `score = 0.8 * answer_accuracy + 0.2 * multi_turn_reasoning`. A required
judge failure rejects and replaces the entire GRPO group in `accuracy`/`dual`; it is never converted to zero or
reweighted. In the two shadow modes, the failure is traced but the recorded reward and sample set are retained so
the systems A/B workload stays paired.

`--reward-max-concurrency` limits concurrent sample-level reward tasks (default 64) in each rollout process. Within
one sample, the accuracy and VLM projections run off the resident event loop and both terminal Judge requests are
concurrent. `max_concurrency` is enforced authoritatively at the single role-specific GenRM Serve replica, so all
session shards share one admission queue; the per-process executor semaphore only reduces local bursts. With the
example value `1`, samples are serial **within each Judge role**, while the two role queues operate independently;
there is no rollout-wide lock that waits for one sample before admitting every later sample. The exposed delay
appears when these queues and the reward/transfer pipeline create backpressure for the next trainable partition.

A four-GH200 host can be used for a judge-only smoke depending on measured model memory use. Concurrent actor
training needs additional GPU capacity. Always measure the selected models and tensor-parallel sizes in the target
container; do not infer fit from parameter count.

## Latency benchmark

### Terminal-once versus per-turn VLM

`reasoning_trigger` is the explicit implementation comparison axis:

- `terminal_once` (default): the VLM reads the complete materialized trajectory once at terminal reward time.
- `per_turn`: after a new environment/tool observation is committed, the VLM reads only the task prefix plus that
  response-to-observation interaction. Calls run as sidecars while later agent turns continue; terminal materialization
  waits only for unfinished calls. Their arithmetic mean is the `multi_turn_reasoning` component.

The terminal answer-accuracy Judge still runs once in both modes. A trajectory with zero completed interactions is
explicitly recorded as `per_turn_fallback_terminal_once` and uses the full-trajectory VLM once rather than assigning an
invented score. Per-turn results do not yet create turn/token-local advantages; all rollout tokens still receive the
same final scalar reward.

The two triggers do not read the same amount of a trajectory, and a reward comparison must state this rather than
infer it. A per-turn Judge scores one completed `response -> observation` pair, so the trajectory's **final response
has no following observation and is covered by the answer Judge alone**, while `terminal_once` does read it. Every
per-turn trace therefore exports both `per_turn_judge_count` (scored interactions, `K`) and
`per_turn_assistant_turn_count` (assistant turns on the exported lineage); `turn_judge/turn_coverage_ratio` reports
`K / assistant_turns` per publication round. A trajectory of `T` assistant turns normally yields `K = T - 1`.

Judgements are scheduled per completed interaction, but the session forest is prefix-matched: an agent that trims or
replays its message history branches off an interior node, and the abandoned branch's interactions are **not** part of
the exported trajectory. Only judgements whose response node lies on the exported lineage are averaged into
`multi_turn_reasoning` or counted in `per_turn_judge_count`; anything dropped is reported as
`per_turn_off_lineage_judge_count` and `turn_judge/off_lineage_judge_count`. A trajectory whose judgements are all
off-lineage has no scored interaction of its own and takes the `per_turn_fallback_terminal_once` path. Context-trimming
agents (common for long GUI trajectories) hit this, so watch that metric — a persistently non-zero value means Judge
GPU time is being spent on interactions that never reach training.

`turn_judge_barrier_timeout_s` (optional, top level) bounds the terminal join of sidecar work. Each call is already
bounded by `timeout_s * max_attempts` plus backoff, but per-turn keeps roughly `T` times more calls in flight, so one
wedged Judge would otherwise hold a session for that full per-call bound. On expiry the stragglers are cancelled and
recorded as `barrier_timeout` rejections, which flow through the ordinary sample-rejection path and therefore consume
`max_group_replacements_per_step`. Size it below `timeout_s * max_attempts` so it actually binds, but well above the
observed p99 of `turn_judge/sidecar_elapsed_time` so merely-slow Judges are not rejected. It is part of the benchmark
invariant hash and must be identical across paired variants. Omit it (or set `null`) to keep the per-call bounds as the
only limit.

The VLM sees assistant tool calls and the following observation/tool messages, including image parts. An environment
"meta result" must therefore be serialized into that observation message's `content` (or a supported image part);
arbitrary session/output `metadata` is intentionally not forwarded into a Judge prompt.

The config generator keeps the historical five terminal-once files and additionally writes
`judge_services_dual_per_turn.json` and `judge_services_dual_shadow_per_turn.json`. For the latency comparison, pair
`dual_shadow` with `dual_shadow_per_turn`: both train on the recorded environment reward, while the latter exposes the
per-round VLM request pattern. The timeline emits separate `critical_path.turn_judge` and terminal
`critical_path.turn_judge_barrier` spans, rather than folding those calls into terminal `critical_path.reward.*` spans.
The workload pairing identity is the full terminal state hash, not the scoped Judge-context hash, so these two valid
implementations remain comparable despite reading different context scopes.

The reward config supports five paired variants. `recorded` calls no Judge and consumes the pre-existing environment
reward. `accuracy` calls only the answer Judge. `dual` calls both Judges concurrently. The two `*_shadow` modes run
the named Judge workload but still train on the recorded environment reward, which isolates systems latency from
policy drift caused by changing the reward. The paired analyzer accepts only `recorded`/shadow modes by default;
`--allow-reward-changing-pair` is an explicit opt-in to a confounded operational comparison.

Generate configs from the same model and concurrency settings:

```bash
python examples/agentic_dual_judge/prepare_benchmark_configs.py \
  --output-dir /tmp/dual-judge-latency-configs
```

For each variant, keep the model checkpoint, data order, random seed, resource allocation, and number of publication
rounds fixed. Replace only `--judge-services-config`. Keep both Judge services allocated even for `recorded` if the
question is the request-path overhead at fixed resources; remove those resources only for a separate capacity-cost
experiment. Use fresh timeline and rollout-result directories for every run. Randomize or balance variant execution
order to reduce cluster-temperature and background-load bias.

`recorded` uses the lightweight terminal context needed to retain a stable trajectory identity; it does not decode
the full trajectory/media envelope used by terminal-once VLM evaluation. Therefore `dual_shadow - recorded` measures
the combined online Judge and terminal-trajectory-context overhead, not a request-only delta. For the direct
implementation comparison, pair `dual_shadow` with `dual_shadow_per_turn`; use
`critical_path.reward_context_build`, `critical_path.turn_judge`, and `critical_path.turn_judge_barrier` to separate
their context construction, sidecar work, and exposed final wait. A full legacy comparison additionally needs a
native/no-dual baseline with an external replay manifest; do not silently call the current `recorded` run that native
baseline.

Enable the existing centralized timeline collector:

```bash
--use-metrics-service \
--timeline-dump-dir "${EXP_DIR}/timeline" \
--rollout-result-dir "${EXP_DIR}/rollout_result"
```

The trace contains these intervals:

- `critical_path.rollout_pre_generation`, `rollout_queue`, `rollout_generation`,
  `rollout_post_generation`, `rollout_managed_session`, and `rollout_finalize`
- `critical_path.rollout_evaluation` (reported separately from training-data generation)
- `critical_path.reward_context_build`, `critical_path.reward`, group finalization, and one nested interval per Judge
  role
- `critical_path.transfer_buffer_wait`, `critical_path.transfer`, and `critical_path.data_wait`
- `critical_path.training_forward`, `critical_path.training_schedule`, and `critical_path.optimizer_step`
- `critical_path.weight_gate_wait`, `critical_path.weight_update`, and
  `critical_path.weight_serving_ready`

Reward metrics additionally report p50/p90/p95/p99/max for global queueing, projection, per-role semaphore queueing,
bounded payload/media preparation, client HTTP RTT, parsing, retry backoff, branch latency, server media restoration,
tokenization, engine HTTP, and token counts. Rejected and cancelled groups are included through terminal trace
snapshots instead of being silently omitted.

After all paired runs finish, analyze their Chrome traces together:

```bash
python examples/agentic_dual_judge/analyze_latency.py \
  --variant recorded=/path/to/recorded/timeline \
  --variant accuracy_shadow=/path/to/accuracy-shadow/timeline \
  --variant dual_shadow=/path/to/dual-shadow/timeline \
  --warmup-publication-rounds 2 \
  --measure-publication-rounds 100 \
  --expected-groups-per-round 64 \
  --expected-samples-per-group 8 \
  --output /path/to/latency-report.json
```

Use the real `rollout_batch_size` (groups) and `n_samples_per_prompt` for the two expected-cardinality values. The
analyzer selects the fixed-K publication sequence once from the baseline and requires every candidate's detailed
report to use that exact sequence. It expects consecutive step IDs by default. If the run intentionally logs rollout
updates with a larger fixed stride, pass that value through `--expected-step-stride`; an extra or missing boundary is
a hard error. For paired timeline runs every measured update must contain unique reward workload identities, and the
analyzer compares each sample's full terminal trajectory hash (falling back to a legacy `RewardContext` content hash);
matching slot indices with different trajectory contents are rejected as an unpaired workload. Recorded and shadow runs also hash the numeric reward
consumed by the trainer, so equal trajectories with different training rewards are rejected. It enforces the
declared benchmark mode, exact Judge branches, successful terminal statuses, expected group cardinality, and a
manifest hash computed from the final resolved service config and covering Judge settings plus rollout,
model-parallel, optimizer, batching, async, and weight-publication settings. It also requires every measured sample
to have exactly one context-build/reward/transfer span and at least one generation span. A failure from another
logical step whose wall span overlaps the measured window also invalidates the run.

Trainer events include `optimizer_step_id`, so `optimizer_steps_per_publication_round` reports optimizer attempts per
actor/critic round without multiplying by distributed rank count; an overflow-skipped attempt is not currently
distinguished from a successful parameter update. Actor/critic expectations are carried in every reward trace, so
PPO automatically requires critic telemetry. `--expected-optimizer-component` is an additional operator assertion.
Required components must appear in every measured round and rank sets must remain fixed. The CLI name is deliberately
“publication rounds”: one ready-to-ready interval can contain multiple optimizer attempts.

Interpret the report as follows:

- `paired_makespan_delta_vs_baseline` is the primary detailed-trace result. It compares the same fixed K publication
  IDs using
  ready-to-ready wall time: the end of the last measured `weight_serving_ready` minus the ready boundary immediately
  before the measured window. At least one warmup update is therefore required.
- `inclusive_occupancy_percent` measures each stage's interval union; values can sum above 100% because work overlaps.
- `per_update_ready_interval` and `per_update_inclusive_occupancy_percent` report p50/p90/p95/p99/max across the
  individual ready-to-ready publication intervals, rather than only one aggregate window. Use substantially more
  than 20 rounds before quoting p99; 20 is only a smoke test.
- `active_set_percent` shows exact observed overlap states and sums to 100% with `no_observed_span`; that label does
  not prove the hardware was idle -- see `judge_gpu_efficiency` below for a direct GPU-occupancy measurement of the
  judge (and rollout) engines, and `exposure_vs_baseline` for whether reward growth actually stalled the trainer.
- `equal_split_observed_wall_percent` is an additive visualization that splits overlap evenly. It is not a causal
  critical-path proof, so do not present it as one.
- `event_latency_by_name` contains full durations for events touching the window;
  `event_clipped_duration_by_name` clips them to the measurement boundaries.
- `overlapping_reward_workload` reports whether all completed reward work touching each run's wall window matches,
  including successful future-step work. A mismatch is a hard error by default. Only exploratory reports may pass
  `--allow-overlapping-workload-mismatch`, and those reports are marked invalid for causal latency inference.
- `exposure_vs_baseline` (per candidate variant, alongside `paired_makespan_delta_vs_baseline`) reports
  `delta_data_wait_s`, `delta_reward_occupancy_s`, and `exposure_ratio = delta_data_wait_s / delta_reward_occupancy_s`,
  derived from each variant's own `inclusive_occupancy_percent`. A ratio near 0 means the added reward work was
  hidden by async overlap (the trainer never waited longer for data); a ratio near 1 means it was fully exposed as
  added trainer stall. This is the direct answer to "is the reward workload on the critical path", complementing the
  aggregate `paired_makespan_delta_vs_baseline`. `None` fields mean one side's report lacked the underlying stage
  (e.g. a rollout-JSONL-only variant).
- `judge_gpu_efficiency` (per variant, one entry per judge/rollout `role`) reports GPU occupancy from the sidecar
  sampler described below: `nonidle_fraction` and `util_percent` (NVML `utilization.gpu`, sampled -- see the
  low-overhead channel section for why this is called "non-idle fraction" and not "utilization"),
  `zero_request_time_fraction` and `running_reqs`/`token_usage` (from the judge engine's own SGLang Prometheus
  endpoint), `gpu_hours_in_window`, `allocated_gpu_seconds_per_training_sample`, and `sample_coverage_fraction` (observed
  sampler ticks versus the window duration divided by the configured sampling interval -- low coverage means the
  other fields in this section are unreliable). `judge_gpu_efficiency_issues` carries soft warnings
  (`no_gpu_samples_in_window`, `low_gpu_sample_coverage:<role>`, `gpu_sample_clock_host_mismatch`) rather than hard
  errors, since this sidecar channel is a best-effort secondary signal, not a validated benchmark input like the
  timeline trace. Absent (`null`) when `--gpu-samples` was not passed for that variant.

Training schedule spans are collected from every trainer rank. For streaming fully-async training, the analyzer
subtracts rank-local `data_wait` intervals and calls the remainder `training_compute_and_orchestration`; it is not a
pure GPU-kernel measurement. A rollout-result JSONL contains no trainer or weight spans and is accepted only for
rollout/reward diagnostics, never for an E2E comparison.

The detailed timeline contains only completed spans. End a production measurement with an admission stop/drain and
at least one later timeline flush; otherwise Judge work still in flight at the final ready boundary is right-censored
and cannot appear in `overlapping_reward_workload`. The current analyzer reports completed observed work and does not
manufacture a causal attribution for an undrained tail.

The weight serving-ready boundary waits for every rollout engine's `continue_generation` response. Every detailed
event records `clock_host`; the analyzer checks all measured-step events and all reward-WIP candidates before using
wall-time overlap. A multi-host report is rejected unless `--allow-synchronized-multi-host-clock` explicitly asserts
that measured NTP/PTP skew is acceptable. The flag is an operator assertion, not an automatic skew measurement, so
retain the cluster's offset/uncertainty evidence with the report.

## Low-overhead E2E channel and trace calibration

Set a fresh dedicated directory for every run. In standard Ray Job mode, shell exports made outside the submitted
driver are not enough: after the entrypoint has initialized `RUNTIME_ENV_JSON` and before `ray job submit`, inject
both the directory and the generic passthrough declaration into the runtime environment:

```bash
export RELAX_DUAL_JUDGE_MARKER_DIR="${EXP_DIR}/latency_markers"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_DUAL_JUDGE_MARKER_DIR"
export RUNTIME_ENV_JSON="$(
  jq -c \
    --arg marker_dir "${RELAX_DUAL_JUDGE_MARKER_DIR}" \
    --arg propagate "${RELAX_PROPAGATE_ENV_VARS}" \
    '.env_vars.RELAX_DUAL_JUDGE_MARKER_DIR = $marker_dir
     | .env_vars.RELAX_PROPAGATE_ENV_VARS = $propagate' \
    <<<"${RUNTIME_ENV_JSON}"
)"
```

This channel is independent of `--rollout-result-dir`: rank 0 appends one monotonic serving-ready boundary per
publication round, while the rollout process writes one small mergeable workload fragment per closed partition.
It does not serialize prompts, responses, media, or full reward traces. The directory must be on shared persistent
storage visible to actor and rollout processes; a missing marker is a benchmark failure even though training only
logs the write failure. Use it for the primary low-volume makespan and to quantify timeline trace
export/aggregation/dump overhead, not the cost of all instrumentation:

```bash
python examples/agentic_dual_judge/analyze_ready_markers.py \
  --variant trace_off=/path/to/trace-off/latency_markers \
  --expected-mode trace_off=dual_shadow \
  --variant trace_on=/path/to/trace-on/latency_markers \
  --expected-mode trace_on=dual_shadow \
  --trace-timeline trace_on=/path/to/trace-on/timeline \
  --warmup-publication-rounds 2 \
  --measure-publication-rounds 100 \
  --expected-groups-per-round 64 \
  --expected-samples-per-group 8 \
  --require-trace-calibration \
  --output /path/to/trace-overhead.json
```

The marker analyzer verifies that the trace-on timeline contains real serving-ready events for the selected
boundary and every measured round. It rejects actor restarts anywhere from the first warmup marker through the
measurement endpoint, missing/duplicate steps, mode changes, invariant-config changes, incomplete group
cardinality, failed Judge outcomes, changed trajectory/context digests, terminal failures anywhere in the fresh
workload marker files, and changed numeric training rewards. Mergeable SHA-256 accumulators allow a train partition
to be completed by multiple resident-rollout fragments without writing per-sample content. In fully async runs,
successful look-ahead work that has not yet closed into a partition can still overlap the window; fixed seed alone
is insufficient if arrival/admission order is nondeterministic, so retain the detailed trace as the attribution
channel.

For a tracing-overhead pair, keep `--use-metrics-service` enabled in both runs and vary only
`--timeline-dump-dir`; changing the metrics service itself is a different ablation and changes the invariant hash.

## Judge/rollout GPU occupancy sampling

The timeline trace measures request latency, not whether the judge GPUs were doing anything with that time.
`no_observed_span` in `active_set_percent` is not proof of hardware idleness (see above), so a separate sidecar
channel samples NVML utilization and each judge engine's own SGLang Prometheus endpoint directly inside the
`GenRMEngine`/rollout `SGLangEngine` actor process. It is gated entirely on an env var and costs nothing when unset.
Set the same directory-plus-passthrough pattern as the marker channel:

```bash
export RELAX_JUDGE_GPU_SAMPLE_DIR="${EXP_DIR}/gpu_samples"
export RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S=0.2  # optional, defaults to 0.2s
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_JUDGE_GPU_SAMPLE_DIR,RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S"
export RUNTIME_ENV_JSON="$(
  jq -c \
    --arg sample_dir "${RELAX_JUDGE_GPU_SAMPLE_DIR}" \
    --arg interval "${RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S}" \
    --arg propagate "${RELAX_PROPAGATE_ENV_VARS}" \
    '.env_vars.RELAX_JUDGE_GPU_SAMPLE_DIR = $sample_dir
     | .env_vars.RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S = $interval
     | .env_vars.RELAX_PROPAGATE_ENV_VARS = $propagate' \
    <<<"${RUNTIME_ENV_JSON}"
)"
```

Also set `"enable_metrics": true` in each role's `engine_config` in `judge_services_config` (see
`judge_services.json` in this directory) so the sampler's Prometheus scrape has something to read; SGLang metrics
are off by default for judge/GenRM engines (unlike the rollout engine, where Relax enables them unconditionally).
This is part of the benchmark invariant hash, so keep it identical across every paired variant.

Each engine process writes its own append-only `{role}_{clock_host}_rank{N}.jsonl` file: one manifest line, then one
sample line roughly every `RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S` seconds with per-GPU NVML `util_percent`/
`mem_used_bytes` and the scraped `sglang:num_running_reqs`/`num_queue_reqs`/`token_usage`/`gen_throughput`. It does
not serialize prompts, responses, or reward content -- only occupancy numbers. NVML's `util_percent` means "at least
one kernel was running in the sampling window", not SM occupancy; a `max_concurrency: 1` judge role can show
`util_percent` near 100 with a `running_reqs` of 1 the whole time, which is a saturated *request queue*, not a
saturated *GPU* -- always read `util_percent` next to `running_reqs`/`token_usage`, never alone. Since this hooks the
same engine-init path used by the plain rollout engine, rollout GPUs are sampled too (`role: "rollout"`) whenever the
env var is set; filter by `role` downstream if you only want the judges.

Feed the directory to the latency analyzer alongside the matching `--variant`, once per paired run:

```bash
python examples/agentic_dual_judge/analyze_latency.py \
  --variant recorded=/path/to/recorded/timeline \
  --variant dual_shadow=/path/to/dual-shadow/timeline \
  --gpu-samples recorded=/path/to/recorded/gpu_samples \
  --gpu-samples dual_shadow=/path/to/dual-shadow/gpu_samples \
  --expected-mode recorded=recorded --expected-mode dual_shadow=dual_shadow \
  --warmup-publication-rounds 2 \
  --measure-publication-rounds 100 \
  --expected-groups-per-round 64 \
  --expected-samples-per-group 8 \
  --output /path/to/latency-report.json
```

`--gpu-samples` is optional and per-variant; omitting it for a variant leaves `judge_gpu_efficiency` `null` for that
variant only. See the report-field documentation above for `judge_gpu_efficiency` and `exposure_vs_baseline`.

## Independent replicates

Run each complete paired experiment at least three times with independent pair IDs. A pair must use the same seed
and invariant manifest internally; different pairs may use different seeds and therefore different invariant hashes.
Aggregate one global delta per pair, never K autocorrelated publication intervals as K independent samples:

```bash
python examples/agentic_dual_judge/aggregate_latency_replicates.py \
  --report pair_01=/path/to/pair_01/latency-report.json \
  --report pair_02=/path/to/pair_02/latency-report.json \
  --report pair_03=/path/to/pair_03/latency-report.json \
  --candidate dual_shadow \
  --output /path/to/latency-replicates.json
```

The output includes raw run-level deltas, mean/median, sample standard deviation, and bootstrap 95% intervals over
independent paired runs. These intervals are descriptive with very few pairs. Stage occupancy shows where work was
active; a causal “rollout xx%, reward yy%” decomposition requires explicit dependency edges through generation,
reward, transfer, batch consumption, optimizer steps, weight publication, and first policy use, or controlled
component ablations. Ready-to-ready makespan is a throughput/availability metric, not per-policy causal latency.
