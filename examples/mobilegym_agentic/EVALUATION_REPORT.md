# Agentic MLLM RL Training — Latency Evaluation Report

Synthesizes the full latency-bottleneck investigation for MobileGym agentic training on
Relax. This is a structured summary of the working notes in `LATENCY_FINDINGS.md`
(same directory) — that file remains the primary, chronologically-ordered lab record with
the raw commands, log excerpts, and code diffs; this report organizes the same evidence by
question, for reference and hand-off.

**Status at time of writing:** diagnostic questions (§4.1–4.4) answered from historical
data. Experimental-validation questions (§4.5) blocked — the E2 cluster experiment hit two
reproducible infrastructure bugs across 4 attempts and is paused pending a decision on how
to fix them.

______________________________________________________________________

## 1. Purpose and Motivation

**Purpose.** Evaluate the end-to-end latency distribution of an **agentic multimodal-LLM
RL training system** — Relax, running the **MobileGym** environment (a browser-hosted
mobile-GUI task suite) as the agentic rollout — and identify where the makespan actually
goes: rollout generation, reward computation, training, or weight update, including the
overlap between them under Relax's fully-async execution.

This system's reward stage is itself a variable under evaluation, not a fixed black box:
it runs a **dual-judge design**, combining an **ORM** (outcome reward model —
`answer_accuracy`, scores only the final answer) with a **PRM** (process reward model —
`multi_turn_reasoning`, scores the reasoning across turns). The PRM pipeline in turn
supports **two distinct judgement modes**: **`terminal_once`** (final-level — the PRM reads
the complete trajectory once, after the episode ends) and **`per_turn`** (turn-level — the
PRM scores each turn as a sidecar while the trajectory is still being generated). Comparing
these two PRM modes against each other, and against the ORM-only cost, is one of the two
central axes of this evaluation (the other being the reward-service concurrency
configuration, §4.3/§4.6).

The starting ask was a Figure-1(b)-style time breakdown for end-to-end agentic MLLM RL
training — split the makespan into rollout generation / reward computation / training /
weight update, plus the overlap between them — modeled on Sheng et al., *"Laminar: A
Scalable Asynchronous RL Post-Training Framework"* (arXiv:2510.12633), whose Figure 1(b)
reports "the generation stage... accounting for up to 83.1% of total execution time in
reasoning tasks."

That framing needed correcting before it could be answered honestly: Laminar's number is a
**synchronous-pipeline occupancy share**, where occupancy and critical-path share coincide
because the stages are serial. Relax's MobileGym setup runs **fully-async**, where a stage
can be 100% occupied (rollout engines never idle) while contributing 0% to why the trainer
is stalled. The evaluation therefore had to build three distinct, non-interchangeable
measurement views (§3.3) before any "X% is the bottleneck" claim could be trusted.

## 2. System Under Test

| axis                                 | value                                                                                                                                                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| framework                            | Relax (Ray Serve + Megatron-LM + SGLang)                                                                                                                                                                                        |
| environment                          | MobileGym (browser-hosted mobile GUI simulator), `examples/mobilegym_agentic/`                                                                                                                                                  |
| policy model                         | Qwen3-VL-4B-Instruct                                                                                                                                                                                                            |
| reward system                        | dual local-judge, evaluated as a variable, not a fixed component: **ORM** (`answer_accuracy`, Qwen3-4B, text-only, final-answer only) + **PRM** (`multi_turn_reasoning`, Qwen2.5-VL-7B, multimodal, scores the reasoning trace) |
| PRM judgement modes (both evaluated) | **final-level** — `terminal_once`: PRM reads the complete trajectory once, after the episode ends · **turn-level** — `per_turn`: PRM scores each turn as a sidecar while generation continues                                   |
| training mode                        | fully-async (`--fully-async`), not colocate                                                                                                                                                                                     |
| algorithm                            | GRPO, `--kl-loss-coef 0.00`                                                                                                                                                                                                     |
| resource layout                      | `G5_FULL24_ONLY`: **6 nodes × 4 GH200 GPUs = 24 GPUs** — `actor` 4 (TP=4), `rollout` 12×1, `judge_accuracy` 4 (TP=4), `judge_multiturn_vlm` 4 (TP=4)                                                                            |
| batch shape                          | `rollout_batch_size=8 × n_samples_per_prompt=8 = 64` samples/round = 8 groups; `global_batch_size=64`                                                                                                                           |

Correction logged mid-investigation: earlier notes mislabeled this as a 16-GPU/4-node
setup, reading `run_mobilegym_e2e.sh`'s header comment literally. `submit_mobilegym_e2e.sh`
actually asserts `NUM_NODES -eq 6` for `G5_FULL24_ONLY=1`. All measured numbers in this
report are unaffected — only the resource-count framing was wrong.

### 2.1 Analyzed runs (all read-only, pre-existing before this evaluation)

| job                  | workload                        | role                                                                                                    |
| -------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `3124850`            | `terminal_once` (ORM+PRM-final) | primary reference, 9 rounds, used for most numbers below                                                |
| `3118393`, `3117772` | `terminal_once`                 | secondary cross-checks                                                                                  |
| `3124159`            | `per_turn` (ORM+PRM-turn)       | primary per_turn comparison run                                                                         |
| `3118665`, `3118224` | `per_turn`                      | secondary cross-checks; `3118665` also the reference case for pre-existing engine-init flakiness (§4.6) |

## 3. Measurement Methodology

### 3.1 Existing tooling reused, not rebuilt

- `examples/agentic_dual_judge/analyze_latency.py` (3,211 lines) — event loading from
  Chrome-trace timeline dumps or rollout-result JSONL, fixed-K window selection,
  `_union_duration`, `_active_set_durations`, clock-skew gating, paired-workload
  validation (benchmark invariant hash).
- `examples/mobilegym_agentic/scripts/stage_breakdown.py` — marginalizes
  `analyze_latency.py`'s `active_set_percent` into a stall-decomposition table and an
  ORM/PRM wall-clock split.
- `relax/utils/metrics/judge_gpu_sampler.py` — NVML + SGLang-Prometheus sidecar for
  judge/rollout GPU occupancy.

### 3.2 New tooling built during this evaluation

- **`examples/mobilegym_agentic/scripts/chain_decomposition.py`** — per-sample dependency
  chain decomposition, reading `rollout_result/*.jsonl` directly. For each sample, walks
  the ordered timestamps in `latency_trace.events` from "last turn's generation finished"
  to "released into the training queue" and attributes every second to exactly one named
  segment (A–H, §4.1). Because every segment boundary is `events.get(key)` for consecutive
  keys, segments sum to the tail by construction — verified to reproduce a
  **0.000000 s residual** on 576/576 samples across both `3124850` and `3124159`.

### 3.3 Three non-interchangeable measurement views

| view                                                                                | question answered                                                           | sums to                     | validity                                                                                                          |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **V1 — inclusive occupancy** (`analyze_latency.py`'s `inclusive_occupancy_percent`) | "how much wall-clock was each stage's span union active"                    | ≥100% (stages overlap)      | descriptive only; **not** a bottleneck claim                                                                      |
| **V2 — active-set partition** (`active_set_percent`)                                | "at each instant, which stage(s) were solo-active vs. overlapping vs. idle" | exactly 100%                | correct partition, but "solo-active" coverage can be small under high concurrency, limiting what it can attribute |
| **V3 — causal chain decomposition** (`chain_decomposition.py`)                      | "for each sample, where did the post-generation tail actually go"           | exactly 100% (residual ≈ 0) | this is the one load-bearing for bottleneck ranking                                                               |

Early in the investigation, V2 alone was tried on `3124850`: only **11.7%** of the window
had a single solo-active stage (67.0% was `rollout+reward` jointly active), which is too
thin a base to rank stages from directly — this is what motivated building V3.

______________________________________________________________________

## 4. Sub-Evaluation Questions, Analysis, and Results

### 4.1 What is the end-to-end time breakdown, and which stage is the bottleneck?

**Method:** `chain_decomposition.py` against `3124850` (`terminal_once`) and `3124159`
(`per_turn`), ≥2 warmup rounds excluded, all measured rounds included.

**Segments** (mean seconds, `terminal_once` / `per_turn`, % is share of tail **G**):

| segment | meaning                                                            | terminal_once | % of G |  per_turn | % of G |
| ------- | ------------------------------------------------------------------ | ------------: | -----: | --------: | -----: |
| A       | trajectory generation (first turn arrive → last turn's `chat_end`) |         340.0 |      — |     441.3 |      — |
| B       | finalize (`chat_end` → `finalize_end`)                             |          14.0 |   2.5% |     141.3 |  40.6% |
| C       | wait to enter reward pipeline                                      |          73.6 |  13.2% |      19.4 |   5.6% |
| D       | reward compute                                                     |         168.1 |  30.2% |       4.5 |   1.3% |
| E       | round barrier (straggler wait)                                     |         195.5 |  35.2% |      78.6 |  22.6% |
| F       | transfer release                                                   |         105.0 |  18.9% |     104.2 |  30.0% |
| **G**   | **post-generation tail** (B+C+D+E+F)                               |     **556.2** |   100% | **347.9** |   100% |
| **H**   | **full chain** (A+G)                                               |     **896.2** |      — | **789.2** |      — |

**Result:** generation (A) is not on the critical path in the way one might assume —
**the tail after generation (G) is 1.6× longer than generation itself** in `terminal_once`.
Root-cause attribution of the tail: **reward-related segments (C+D+E) = 78.6%**,
transfer (F) = 18.9%, **rollout generation = 0%** (it already finished, in segment A).

**Cross-check** (V2, coincidence-based, not causal): `stage_breakdown.py`'s stall
decomposition on the same window — while the trainer was blocked on `data_wait`, reward was
involved 59.18% of that time, transfer 53.80%, rollout generation 6.71% (`terminal_once`);
30.95% / 40.07% / 6.20% (`per_turn`). Same ranking, independent method — reward first,
generation last.

**Conclusion: reward computation is the primary bottleneck, not rollout generation.**

*References: `relax/agentic/rollout.py` (event marks: `finalize_start_at`,
`reward_arrive_at`, `transfer_release_start_at`, etc.), `relax/agentic/pipeline/transfer.py`
(transfer release gating, see §4.5), `examples/mobilegym_agentic/scripts/stage_breakdown.py`.*

### 4.2 Why does "trainer stalled" happen at all under fully-async, and how much idle time does it represent?

`stall = critical_path.data_wait` active (`relax/utils/data/stream_dataloader.py:1342`) —
the training loop is polling `TransferQueue` for its partition and not getting data yet.

**Conceptual point, confirmed against the code and data:** fully-async removes *lockstep
barriers* between stages, not the *data dependency* between them. Training for round N
still needs round N's scored batch; if the production chain (rollout → reward → transfer)
produces batches slower than the trainer consumes them, the trainer waits regardless of how
asynchronous the scheduling is. Throughput of a pipeline is bounded by its slowest stage.

**Measured:** `trainer_not_stalled = 18.15%` (`terminal_once`) / `34.36%` (`per_turn`) of
the window — i.e., the trainer is idle roughly 82% / 66% of the time.

**Is the GPU itself idle during this, or just the Python-level polling loop?** Checked
directly: `data_wait` never overlaps `optimizer_step` or `weight_update`
(`0.000%` intersection, both metrics), and 83.0% of total wait time sits in episodes >5 s —
too long to be CUDA kernel drain. This is a strong inference, not an NVML measurement (the
judge/rollout GPU sampler wasn't originally wired to the actor process). **Fixed during this
evaluation** — see §5, item M4 — but not yet exercised on a completed run, since E2 (the
runs that would produce this data) is blocked (§4.6).

### 4.3 Why is judge/reward-side GPU utilization low (~12%) while latency is still high — is it upstream starvation?

**Method:** decomposed the `multi_turn_reasoning` (PRM) reward branch into its constituent
timings from the rollout-result JSONL's `latency_trace.reward.judges.*` fields, and
cross-checked against the judge GPU sampler's NVML + SGLang-Prometheus sidecar.

**Branch decomposition** (`terminal_once`, 576 samples, `multi_turn_reasoning`):

| layer                                                          |                p50 | share of 149.4 s branch |
| -------------------------------------------------------------- | -----------------: | ----------------------: |
| **branch total**                                               |            149.4 s |                    100% |
| `queue_elapsed_s` (client-side `max_concurrency: 8` semaphore) |         **97.6 s** |               **65.1%** |
| `http_elapsed_s`                                               |             47.7 s |                   34.9% |
| — Ray Serve admission gap                                      | 0.3 s (p90 62.7 s) |                   10.4% |
| — server `request_total` ≈ `engine_http_elapsed_s`             |         **38.1 s** |                   24.5% |
| — — `media_restore` / `tokenize` / `server_queue`              |  ~0.015 s combined |                     ~0% |

**GPU occupancy** (NVML + SGLang Prometheus, `judge_multiturn_vlm`): NVML idle rate
**88.41%**, `num_running_reqs` mean 1.33/8, `num_queue_reqs` mean 0.31. GPU time per
request ≈ 0.93 s (5.15 GPU-hours in window ÷ 4 GPUs × 11.59% busy ÷ 576 requests) against
38.1 s of engine time ⇒ **only 2.4% of engine time is actual GPU compute.**

**Is this upstream starvation (nothing to do) or self-inflicted queueing?** Checked via
measured arrival rate vs. capacity: peak observed arrival **0.339 req/s** (round 1 of
`3124850`: 64 arrivals over 189.0 s) vs. capacity **8 slots ÷ 47.7 s = 0.168 req/s** — demand
exceeds capacity by **1.3–2.0× every round**. This is not upstream starvation; the judge is
provably overloaded against its own concurrency cap.

**Result: 88% GPU idle decomposes as ~65% blocked at the client's own semaphore, ~24%
CPU-side work inside the engine, \<1% genuinely no work available.** This directly motivated
the `max_concurrency: 8 → 24` fix design (§4.5).

**Verification:** the C2 instrumentation (§5) reproduces these exact figures (queue p50
97.64 s, http p50 47.74 s) by running the actual production timeline-emission function
against 576 real samples, not just replaying the JSONL by hand.

### 4.4 What causes the PRM-vs-ORM latency gap, and the `per_turn`-vs-`terminal_once` difference?

**PRM vs ORM — measured, not yet attributed:**

|                          | ORM (`answer_accuracy`, Qwen3-4B, text) | PRM-final (`multi_turn_reasoning`, Qwen2.5-VL-7B, multimodal) |
| ------------------------ | --------------------------------------: | ------------------------------------------------------------: |
| engine time              |                                  0.32 s |                                             38.1 s (**119×**) |
| engine output throughput |                             253.7 tok/s |                                                    2.33 tok/s |

Three variables move at once (model size 4B→7B, modality text→multimodal, input size) —
**not attributable from this data alone.** Leading hypothesis: CPU-side vision
preprocessing inside SGLang (PIL decode/resize/patchify), since GPU compute is only 2.4% of
engine time. Counter-evidence against a base64/serialization explanation: GenRM-side
`media_restore_elapsed_s` (base64 decode + SHA-256) is only **0.008 s**. **This remains an
open question — see §4.7 (E0), not blocked by the E2 pause.**

**`per_turn` vs. `terminal_once` — mechanism, corrected mid-investigation.** An earlier
hypothesis ("per-turn judge calls overlap the next turn's generation") was checked directly
and found wrong: only **22.1%** of `turn_judge` execution time overlaps its *own*
trajectory's subsequent generation. The actual mechanism, established from the chain
decomposition (§4.1):

1. **Work moves, it doesn't disappear.** D (reward) drops 168.1→4.5 s, but B (finalize,
   which now absorbs the `turn_judge_barrier`) rises 14.0→141.3 s — net saving only ~27 s
   from this alone.
2. **Burst smoothing (the main win).** `terminal_once` slams all 64 samples into the
   judge's 8-slot queue at 1.3–2.0× overload (§4.3); `per_turn` spreads judge calls across
   the trajectory's lifetime, collapsing C (wait to enter reward) from 73.6→19.4 s.
3. **Variance reduction shrinks the round barrier.** E drops 195.5→78.6 s — **the single
   largest gain (−117 s)** — because more evenly-timed completions make the round's `max`
   over 64 samples cheaper.

**E2E comparison** (directional only — not replicated, see §4.5/§4.6): ready-to-ready
round interval medians — `terminal_once` 587 s (6.6 samples/min) vs. `per_turn` 423 s / 352 s
across two runs (7.5–10.9 samples/min). **`per_turn` is faster, by a wide enough margin that
the direction is not in doubt, but the exact magnitude has no error bars.**

### 4.5 Data volume — is rollout→reward / env→reward network transfer a hidden cost?

**Method:** measured raw screenshot files on disk, cross-referenced against the transport
code path.

| item                      | value                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| screenshot size           | p50 105.6 KB, mean 123.8 KB, 8 images/trajectory (~990 KB raw)                                          |
| policy-side vision tokens | `image_token_count = 20,400` (8 × 2,550); `pixel_values` tensor `[71400,1536]` bf16 ≈ 219 MB/trajectory |
| transport                 | base64 data-URI inside JSON, **two hops** (rollout → GenRM Serve → SGLang engine), 4/3 base64 inflation |

**Per-round transfer estimate** (64 trajectories): ORM ~0.6 MB, PRM-final ~173 MB, PRM-turn
~152 MB. Over a ~580 s round, that is **~0.29 MB/s ≈ 2.3 Mbps — bandwidth is not a
bottleneck**, nowhere close even with a large safety margin.

**Side finding, flagged as a risk:** `input_tokens` reported in the reward trace
under-counts multimodal input by **~7×** (2,886 reported vs. ~20,400 real vision tokens for
the same 8 images) — `relax/engine/rewards/reward_projection.py:283-291` counts tokens of
the *serialized JSON text*, where each image is a short placeholder marker, not the actual
vision-token expansion. Any capacity planning based on `input_tokens` alone is wrong.

**Also flagged:** `MOBILEGYM_ENV_URL` points at the login node's nginx gateway — an
architectural risk for scaling, though not currently a bottleneck (measured env-gap union
is only 7.3% of the window, derived offline from `chat_end_at` → next
`chat_request_arrive_at`, since environment stepping has no dedicated span).

### 4.6 Can the proposed fixes be validated experimentally? (E2 / M5 / D1)

**Design.** Four-arm paired experiment: `A1` (`terminal_once`, `max_concurrency=8`,
baseline) / `A2` (`terminal_once`, `max_concurrency=24`) / `B1` (`per_turn`, `mc=8`) / `B2`
(`per_turn`, `mc=24`). `max_concurrency=24` derived from §4.3's measured arrival rate: hold
time 47.7 s, peak arrival 0.339 req/s, break-even `c = 0.339 × 47.7 = 16.2`, chose 24 for
1.48× headroom. Protocol called for ≥2 warmup + ≥20 measured rounds, ≥3 replicates per arm
with seed blocking across arms.

**Outcome: blocked.** The mandatory smoke test (one `A1` replicate, required before
committing cluster time to the other 11 runs) was attempted **4 times** on a 24-GPU
allocation; **zero attempts reached round 1 of 9.** Two distinct, independently-reproducible
bugs accounted for all four failures:

| bug                               |                   occurrences | mechanism                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | code location                                                                                                |
| --------------------------------- | ----------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Session double-activation**     | 2 of 4 (`3138562`, `3139160`) | `RuntimeDomain.start_batch` leases 64 requests but activates 128 — always immediately after a pre-existing, already-documented `SGLangEngine.init()` `BlockingIOError` crash + Ray Serve replica auto-restart. `_resident_dataflow_loop` (`relax/agentic/rollout.py:537`) catches the error and **returns permanently with no internal retry**. In one occurrence the `HealthChecker` never restarted the service (job stalled silently for 9+ min, confirmed alive via `py-spy`); in the other, the error correctly propagated and the driver exited cleanly (exit code 1)           | `relax/agentic/pipeline/runtime.py:2770`, `relax/agentic/rollout.py:537`, `relax/distributed/ray/rollout.py` |
| **Premature post-run validation** | 2 of 4 (`3138724`, `3139260`) | Round 0 completes *cleanly* (rollout+reward+transfer+advantages all succeed), then `analyze_latency.py`'s `--ready-markers weight_serving_ready.jsonl` check argparse-fails (marker doesn't exist yet — round 0's weight sync hasn't happened) with exit code 2; `set -e` in `run_mobilegym_e2e.sh` propagates it, killing the whole 6-node job. Root mechanism (why `ray job submit`, which should block until all 9 rounds finish, returns after only 1) not fully identified — ruled out `RAY_NO_WAIT` leakage, the training loop believing it's done, and a CUDA/NCCL-class crash | `examples/mobilegym_agentic/run_mobilegym_e2e.sh` (example script, not core orchestration)                   |

Neither bug is caused by this evaluation's own instrumentation changes (§5) — both trigger
in code paths (`RuntimeDomain`/`RolloutManager` session-leasing; the post-run validation
shell block) that C1/C2/M4/M6 never touch, and the engine-init crash that seeds the first
bug is itself pre-existing, documented flakiness (also seen 12× in `3118665`, which still
completed successfully).

**Decision (2026-08-21, user-directed):** pause E2/M5/D1 rather than keep retrying or
attempt a fix mid-session. The first bug is core orchestration code
(`RuntimeDomain`/`RolloutManager`) under this repo's Ask-First policy for Controller/Service
logic; the second is a tractable, in-scope script bug but its root mechanism wasn't nailed
down in the time available. Full forensic detail — `py-spy` stack dumps, log excerpts, every
ruled-out hypothesis — is in `LATENCY_FINDINGS.md` §8.

**What this means for §4.1–4.5:** those results stand independently — they come from
already-completed historical runs, not from the four failed E2 attempts. What's missing is
the **causal confirmation** ("does raising `max_concurrency` actually shrink the tail by the
predicted amount, and does `per_turn`'s advantage hold up with error bars") — that requires
E2 to actually run.

### 4.7 What is the unattributed 37 s of PRM engine time? (E0 — open, independent of the E2 pause)

Still unresolved (§4.4). A dedicated offline microbenchmark was designed but never run:
stand up `judge_multiturn_vlm` alone (4 GPUs, no training job needed), replay real
trajectories, sweep `n_images ∈ {0,1,2,4,8}` (0 is the key control), `max_pixels_per_item`,
and client concurrency, plus a Qwen3-4B-on-text-only control to separate model size from
modality. **This does not require the 24-GPU cluster allocation and is not blocked by the
E2 pause** — it can be run independently if this specific question becomes a priority.

______________________________________________________________________

## 5. Instrumentation Contributions Made During This Evaluation

All items below are additive (new spans/fields), independently verified against real data,
and do not change any existing metric's meaning.

| id     | change                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | verification                                                                                                                                                                               |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **C1** | Split the misleading `critical_path.rollout_pre_generation` span (`relax/agentic/rollout.py`, mirrored in `analyze_latency.py` and `stage_breakdown.py`) into `admission_setup` / `admission_wait` / `dispatch`, after tracing the actual mark sites in `relax/agentic/session/service.py` (an earlier guess that the second half was "preprocessing" was checked and found wrong)                                                                                           | Ran the real production function (`_build_agentic_critical_path_timeline_events`) against 576 real samples: `admission_wait` = 99.7% of the old span (sum 166,273.1 s vs. 543.6 s + 1.2 s) |
| **C2** | Added `.queue` / `.http` child spans plus full judge sub-timing attributes on `critical_path.reward.{component}` (`relax/agentic/rollout.py:1521`), mirrored in `analyze_latency.py`'s direct-JSONL path as `critical_path.judge_request.queue`/`.http` (required widening `_stage_for_event`'s exact-match check to `startswith`, audited against every other exact-name check in that file to rule out double-counting)                                                    | Same live-function test: exact match to §4.3's numbers (queue p50 97.64 s, http p50 47.74 s), zero missing pairs across 576 samples                                                        |
| **M4** | Actor-side NVML sampling (`relax/distributed/ray/train_actor.py`, `TrainRayActor.init()`), reusing `ray_get_device_ids()` (already used in the same file for `get_local_gpu_id()`) rather than parsing `CUDA_VISIBLE_DEVICES` directly, so it's correct under Ray's device remapping                                                                                                                                                                                         | Compiles/lints clean; **not yet exercised on a completed run** — blocked on E2                                                                                                             |
| **M1** | Root-caused why one judge's SGLang engine logs never appear in the driver log: that judge is colocated with the Ray head node, and Ray's log deduplication (stated explicitly in the driver log) collapses its structurally-identical log lines. Fix (`RAY_DEDUP_LOGS=0`, exported before `ray start`) landed in `scripts/entrypoint/spmd-multinode.sh` with explicit user confirmation (touches launcher infra, Ask-First scope)                                            | Fix landed; not yet re-verified on a completed run                                                                                                                                         |
| **M6** | Understood `transfer`'s round-barrier gating (`relax/agentic/pipeline/transfer.py:113,200`): deliberately tied to `num_iters_per_train_update`, confirmed as the same formula used independently in the non-agentic rollout path (`relax/engine/rollout/sglang_rollout.py:929`), and already tracked in the benchmark-invariant-hash config list. Not a safe drop-in fix (changes advantage-normalization semantics per `relax/backends/megatron/actor.py:1373`'s docstring) | No code change needed; confirms the existing plan's separate "T1" arm (varying `num_iters_per_train_update`) is the correct way to test streaming the transfer, not a bug to patch         |
| —      | New tool `examples/mobilegym_agentic/scripts/chain_decomposition.py` (§3.2)                                                                                                                                                                                                                                                                                                                                                                                                  | Residual 0.000000 s on 576/576 samples, both workloads                                                                                                                                     |

______________________________________________________________________

## 6. Summary: Answered vs. Blocked

| #   | question                                                  | status                                                       | confidence                                                                                                         |
| --- | --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| 1   | E2E time breakdown (Fig-1(b)-style)                       | **Answered** (§4.1)                                          | high — accounting identity, residual ≈ 0                                                                           |
| 2   | Which stage is the bottleneck                             | **Answered: reward** (§4.1)                                  | high — corroborated by two independent methods (V3 chain decomposition, V2 stall coincidence)                      |
| 3   | Why "stalled" happens under fully-async                   | **Answered conceptually** (§4.2)                             | high — definitional + directly measured `data_wait`                                                                |
| 4   | GPU idle but latency high — why                           | **Answered: self-inflicted queueing, not starvation** (§4.3) | high — measured arrival rate exceeds measured capacity; C2-verified                                                |
| 5a  | PRM vs. ORM 119× gap — root cause                         | **Not answered** (§4.4, §4.7)                                | needs E0 (independent of E2 pause)                                                                                 |
| 5b  | `per_turn` vs. `terminal_once` mechanism                  | **Answered** (§4.4)                                          | high, but corrected once already — trust the chain-decomposition evidence, not the first-pass "overlap" hypothesis |
| 6   | Data volume / network cost                                | **Answered: not a bottleneck** (§4.5)                        | medium — some figures are estimated from static analysis, not directly measured wire bytes                         |
| 7a  | Does `max_concurrency → 24` fix the queueing              | **Not answered — blocked on E2** (§4.6)                      | prediction only, strong analytical basis, unconfirmed                                                              |
| 7b  | `per_turn` advantage, quantified with error bars          | **Not answered — blocked on E2/M5** (§4.6)                   | direction confirmed, magnitude not                                                                                 |
| 7c  | Does fixing reward just move the bottleneck to `transfer` | **Not answerable without E2** (§4.6)                         | inherently a causal question                                                                                       |

______________________________________________________________________

## 7. Recommended Next Steps

In priority order (see `LATENCY_FINDINGS.md` §8 "Status" for the full resume checklist):

1. **E0** (§4.7) — can proceed immediately, independent of the E2 pause, no 24-GPU
   allocation needed. Resolves the last open diagnostic question.
2. **Investigate the `run_mobilegym_e2e.sh` premature-validation bug** (§4.6, second row) —
   no cluster time needed (four complete failure logs already exist to work from), and it's
   in-scope to fix directly (example script, not Ask-First orchestration code). Fixing it
   alone would have unblocked 2 of the 4 failed attempts.
3. **The `RuntimeDomain` double-activation bug** (§4.6, first row) — needs sign-off before
   any fix attempt, since it's core Controller/Service-adjacent code.
4. **Resume E2** once either bug is fixed — rerun the `A1` smoke test before committing to
   the full 4-arm × 3-replicate matrix.

______________________________________________________________________

## 8. References

**Paper:** Sheng, Tong, Wan, et al., *"Laminar: A Scalable Asynchronous RL Post-Training
Framework,"* arXiv:2510.12633 — motivating comparison for the Figure 1(b)-style breakdown;
note its occupancy-share framing does not directly transfer to a fully-async system (§1).

**Primary working document:** `examples/mobilegym_agentic/LATENCY_FINDINGS.md` — the
chronological lab notebook this report summarizes; contains every raw command, log
excerpt, and intermediate correction.

**Analyzed job IDs** (all read-only historical data unless noted):
`3124850`, `3124159`, `3118665`, `3118393`, `3117772`, `3118224` (diagnostic runs);
`3138562`, `3138724`, `3139160`, `3139260` (four blocked E2 smoke-test attempts, §4.6).

**Key code references** (by section):

- §4.1: `relax/agentic/rollout.py` (agentic event marks), `relax/agentic/pipeline/transfer.py`
- §4.2: `relax/utils/data/stream_dataloader.py:1342`
- §4.3: `relax/engine/rewards/dual_agentic_judge.py` (judge request path),
  `relax/utils/metrics/judge_gpu_sampler.py`
- §4.4: `relax/engine/rewards/reward_projection.py:283-291`,
  `relax/backends/megatron/actor.py:1373`
- §4.5: `relax/engine/rewards/reward_projection.py` (media limits),
  `relax/components/genrm.py` (media transport)
- §4.6: `relax/agentic/pipeline/runtime.py:2770` (`RuntimeDomain.start_batch`),
  `relax/agentic/rollout.py:537` (`_resident_dataflow_loop`),
  `relax/distributed/ray/rollout.py`, `examples/mobilegym_agentic/run_mobilegym_e2e.sh`
- §5: `relax/agentic/session/service.py` (admission-gate marks),
  `relax/distributed/ray/train_actor.py`, `scripts/entrypoint/spmd-multinode.sh`,
  `relax/agentic/pipeline/transfer.py:113,200`, `relax/engine/rollout/sglang_rollout.py:929`

**Analysis tooling:**
`examples/agentic_dual_judge/analyze_latency.py`,
`examples/mobilegym_agentic/scripts/stage_breakdown.py`,
`examples/mobilegym_agentic/scripts/chain_decomposition.py` (new, this evaluation),
`examples/agentic_dual_judge/README.md` (methodology and field-semantics reference).
