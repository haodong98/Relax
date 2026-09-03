# MobileGym agentic MLLM RL — latency bottleneck findings & next experiment

Status: **preliminary**. All numbers below come from single, un-replicated smoke runs
(8–9 publication rounds each). Directionally strong, not yet quotable.

Analysed runs (`/iopsstor/scratch/cscs/$USER/mobilegym_e2e/exp/<job>`):

| workload | naming here | runs |
|---|---|---|
| `reasoning_trigger=terminal_once` | ORM + PRM-**final**_judge | `3124850`, `3118393`, `3117772` |
| `reasoning_trigger=per_turn` | ORM + PRM-**turn**_judge | `3124159`, `3118665`, `3118224` |

`3124850` vs `3124159` are **config-identical except `reasoning_trigger`** (verified by
diffing the full entrypoint argv: same `--resource`, `--seed 42`, `--max-staleness 1`,
`--rollout-batch-size 8`, `--n-samples-per-prompt 8`, `--num-rollout 9`). They are a valid
controlled pair; the only weakness is round count and lack of replicates.

Setup: 24 GPUs (6 nodes, `G5_FULL24_ONLY=1` — see section 7's correction), fully-async.
actor 4 (TP=4, DP=1), rollout 12×1, judge_accuracy 4
(Qwen3-4B, text), judge_multiturn_vlm 4 (Qwen2.5-VL-7B). 64 samples/round.
Clock skew audited: `max_pairwise_offset_ms = 5.92` — cross-host span alignment is sound.

---

## 1. Conclusion: the bottleneck is **reward**, not rollout generation

Trainer is blocked on `critical_path.data_wait` **81.85%** of the window
(`trainer_not_stalled = 18.15%`).

Per-sample dependency-chain decomposition, from "trajectory finished generating" to
"admitted into the training queue". Means; residual is exactly 0, so it sums to 100%:

| segment | terminal_once | per_turn |
|---|---|---|
| A. trajectory generation (p50, *before* the tail) | 362.8 s | 428.9 s |
| B. finalize (per_turn: incl. `turn_judge_barrier`) | 14.0 s (2.5%) | 141.3 s (40.6%) |
| C. wait to enter reward pipeline | 73.6 s (13.2%) | 19.4 s (5.6%) |
| D. **reward compute** | 168.1 s (30.2%) | 4.5 s (1.3%) |
| E. **round barrier (straggler)** | 195.5 s (35.2%) | 78.6 s (22.6%) |
| F. transfer release | 105.0 s (18.9%) | 104.2 s (30.0%) |
| **post-generation tail** | **556.2 s** | **347.9 s** |
| full chain (first request → transfer end) | 967.9 s | 803.1 s |

**Generation takes 363 s; the tail after it takes 556 s.** The trainer never waits on
generation — it waits on scoring, group completion and transfer.

Root-cause attribution of the tail (terminal_once):
`reward-related (C+D+E) = 78.6%`, `transfer = 18.9%`, `rollout generation = 0%`.

Cross-check — coincidence-based stall decomposition (`scripts/stage_breakdown.py`,
interval intersection with `data_wait`, *not* causal proof): reward involved **59.18%**,
transfer 53.80%, rollout_gen **6.71%**. Same direction.

### Trainer GPU really is idle (strong inference, not NVML-measured)

- `data_wait` ∩ `optimizer_step` = **0.000%**; `data_wait` ∩ `weight_update` = **0.000%**
- `data_wait`: p50 2.18 s, p90 14.04 s, p99 185.7 s, max 719.9 s
- **83.0%** of total wait time sits in episodes > 5 s (far longer than CUDA drain)

Gap: `judge_gpu_sampler.py` only hooks inference-engine init, so **actor has no NVML
sampling**. See TODO M4.

---

## 2. Why reward is slow: the GPU is 88% idle and it is *not* waiting for upstream

Reward branch decomposition (`multi_turn_reasoning`, terminal_once, 576 samples):

| layer | p50 | share of branch |
|---|---|---|
| **branch total** | **149.4 s** | 100% |
| ├ `queue_elapsed_s` — client-side `max_concurrency: 8` semaphore | **97.6 s** | **65.1%** |
| └ `http_elapsed_s` | 47.7 s | 34.9% |
| &nbsp;&nbsp;├ Ray Serve admission gap | 0.3 s (p90 62.7 s) | 10.4% |
| &nbsp;&nbsp;└ server `request_total` ≈ `engine_http_elapsed_s` | **38.1 s** | 24.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;└ `media_restore` 0.008 s / `tokenize` 0.007 s / `server_queue` ≈0 | | ~0% |

Judge GPU occupancy (NVML + SGLang Prometheus sidecar):

| | judge_multiturn_vlm | judge_accuracy |
|---|---|---|
| NVML idle rate | **88.41%** | 99.11% |
| `num_running_reqs` mean (cap 8) | 1.33 | 0.34 |
| `num_queue_reqs` mean | 0.31 | — |
| KV-cache `token_usage` mean | 0.007 | 0.0002 |
| `gen_throughput` when non-zero | 1006 tok/s (p90 1878) | — |

**GPU time per request = 0.93 s** (5.15 GPU-h in window ÷ 4 GPUs × 11.59% busy ÷ 576 req)
against **38.1 s** of engine time ⇒ **only 2.4% of engine time is GPU**.

Composition of the 88% judge-GPU idle: **65% blocked at its own concurrency semaphore,
~24% CPU-side work inside the engine, <1% genuinely no work available.**

### The semaphore is provably overloaded

| round | arrivals | spread | arrival rate |
|---|---|---|---|
| 0 | 64 | 285.6 s | 0.224 req/s |
| 1 | 64 | 189.0 s | **0.339 req/s** |
| 2 | 64 | 259.2 s | 0.247 req/s |
| 3 | 64 | 251.0 s | 0.255 req/s |
| 4 | 64 | 222.8 s | 0.287 req/s |

Capacity = 8 slots ÷ 47.7 s = **0.168 req/s**. Demand exceeds capacity by **1.3–2.0×
every round** ⇒ backlog ≈ 22 requests ⇒ drain ≈ 131 s, matching the measured 97.6 s p50 /
211 s p90 queue.

### Straggler amplification

**Correction:** the barrier is coarser than a GRPO group. `transfer_batch_group_count =
global_batch_size // num_iters_per_train_update // n_samples_per_prompt = 64 // 1 // 8 = 8`
groups (`relax/agentic/pipeline/transfer.py:113`), and `_spawn_transfer` only dispatches
once the buffer holds ≥ 8 groups (`transfer.py:207`) — i.e. it waits for **all 64 samples
of the round**, not one group of 8. The within-group `transfer`-start spread of 0.00 s is a
symptom of this: every group in the round is released in the same dispatch, so of course
they start together. Segment E (195.5 s) is a round barrier, and within-round
reward-completion spread is **p50 223.5 s / p90 413.5 s / max 565 s**.

`_dispatch_transfer_batch` (`transfer.py:167-201`) then writes the whole round into
TransferQueue: ~64 samples × `pixel_values` `Tensor[71400,1536]` bf16 ≈ **14 GB in
~105 s ≈ 133 MB/s**, which is why segment F (`critical_path.transfer`) is near-uniform
across both workloads (p50 107.2 s, max 118.8 s) — it is bulk tensor I/O, not a
reward-dependent gate.

### PRM vs ORM: 119× — not yet attributed

| | ORM (`answer_accuracy`) | PRM-final (`multi_turn_reasoning`) |
|---|---|---|
| model | Qwen3-**4B**, text | Qwen2.5-**VL-7B**, multimodal |
| TP / `max_concurrency` | 4 / 8 | 4 / 8 (identical) |
| reported `input_tokens` | 1,079 | 2,886 |
| output tokens | 87 | 86 |
| **engine time** | **0.32 s** | **38.1 s** |
| engine output throughput | 253.7 tok/s | **2.33 tok/s** |

Three variables moved at once (model size, modality, input size). **Not attributable
without the offline sweep (E0).**

---

## 3. per_turn vs terminal_once

**per_turn is faster end-to-end.** Ready-to-ready interval, 64 samples/round:

| run | workload | intervals (s) | median | throughput |
|---|---|---|---|---|
| `3124850` | terminal_once | 465, 721, 518, 510, 726, 655, 691, 352 | **587 s** | 6.6 samples/min |
| `3124159` | per_turn | 321, 435, *1286*, 455, 411, 156 | **423 s** | 7.5 samples/min |
| `3118665` | per_turn | 375, 352, 345, 412, 274 | **352 s** | 10.9 samples/min |

**⚠️ Corrected mechanism.** An earlier hypothesis — "per-turn judge calls overlap the same
trajectory's later turns" — is **wrong as stated**. Measured: `turn_judge` total execution
= 366,033 s, of which only **22.1%** overlaps its *own* trajectory's subsequent
`rollout_generation`. The real mechanism is:

1. **Work moves, it does not disappear.** D 168.1→4.5 s but B 14.0→141.3 s; net only ~27 s.
2. **Burst smoothing (main win).** terminal_once slams all 64 samples into an 8-slot queue
   at 1.3–2.0× overload; per_turn spreads calls across the trajectory lifetime, so
   queueing collapses (C −54 s, and the 65% queue share inside D disappears).
3. **Variance reduction ⇒ shorter round barrier.** E 195.5→78.6 s (**−117 s, the single
   largest gain**), because completion times are more clustered so `max` of 8 is cheaper.

Overlapping with *unrelated* work is not the differentiator — measured overlap with *any*
`rollout_generation` is 1263% (heavy concurrency) in both modes.

**Total vs exposed** — these must not be conflated:

| metric | kind | terminal_once | per_turn |
|---|---|---|---|
| PRM branch total execution | total work | 87,476 s | **366,033 s (4.2×)** |
| reward stage union occupancy | wall occupancy | 74.26% | 57.47% |
| `stall:reward*` | **exposed** | 59.18% | 30.95% |
| chain segment D | **critical-path contribution** | 168.1 s | 4.5 s |

per_turn does **4.2× more total judge work** yet halves its exposed cost.

---

## 4. Data volume: bandwidth is not a problem; per-request CPU cost might be

Measured from the raw screenshots on disk (`mobilegym_runs/*/trajectory/*/step_*.jpg`):

| item | value |
|---|---|
| screenshots | 4,096 (576 trajectories × ~7.1 turns) |
| **JPEG size** | **p50 105.6 KB / p90 197.4 KB / max 455.6 KB / mean 123.8 KB** |
| images per trajectory | 8 |
| raw image bytes per trajectory | ~990 KB |
| policy-side `image_token_count` | **20,400** (8 × 2,550) |
| policy-side `pixel_values` | `Tensor[71400, 1536]` bf16 |

Transport (`dual_agentic_judge.py:104,156` → `genrm.py:290-322`): images travel as
**base64 data-URIs inside a JSON body, over two hops** (rollout → GenRM Serve → SGLang).

| call | images | raw | base64 | **two hops** |
|---|---|---|---|---|
| ORM | 0 | ~4 KB | ~5 KB | ~10 KB |
| PRM-final | ≤8 (cap 12) | ~990 KB | ~1.32 MB | **~2.7 MB** |
| PRM-turn | 1 | ~124 KB | ~165 KB | **~340 KB** |

Per round (64 samples): ORM ~0.6 MB, PRM-final **~173 MB**, PRM-turn ~152 MB.
Over a ~580 s round that is **~0.3 MB/s ≈ 2.3 Mbps — bandwidth is nowhere near a limit.**

**⚠️ `input_tokens` under-reports multimodal input by ~7×.** `reward_projection.py:283-291`
counts tokens of the *serialized JSON text*, where each image is a short marker; images are
only split back into image parts afterwards. Reported 2,886 vs ~20,400 real vision tokens
for the same 8 images. Any capacity planning based on `input_tokens` is wrong.

---

## 5. Open questions

1. **What are the 37 non-GPU seconds inside `engine_http_elapsed_s`?** Leading suspect:
   CPU-side vision preprocessing (PIL decode + resize + patchify) inside SGLang.
   **Counter-evidence against base64 being the cause:** GenRM-side `media_restore_elapsed_s`
   (base64 decode + SHA-256) is only **0.008 s**. Unresolved — this is E0's job.
2. **VLM judge SGLang logs are absent from the driver log** (accuracy judge has 1,420
   `Prefill batch` lines; the VLM judge has none), so real prompt length and prefill/decode
   split are unmeasured. **Root cause found:** `judge_multiturn_vlm` (`10.100.120.66`) is
   placed on the same node as the Ray head — `RAY_ADDRESS`/GCS in the driver log is also
   `10.100.120.66:6379` — while `judge_accuracy` runs on a separate node (`10.100.120.79`).
   The driver log itself states *"Ray deduplicates logs by default... set
   `RAY_DEDUP_LOGS=0` to disable"*; both judges emit structurally-identical
   `Prefill batch, #new-seq: ..., #new-token: ...` lines, so this is the leading
   explanation for why one engine's lines survive and the other's collapse into
   `[repeated Nx across cluster]` markers attributed elsewhere. **Fix:** export
   `RAY_DEDUP_LOGS=0` before `ray start` — that call lives in
   `scripts/entrypoint/spmd-multinode.sh` (Launcher logic, CLAUDE.md "Ask First"), not in
   `run_mobilegym_e2e.sh`, since `spmd-multinode.sh` runs `ray start --head`/`ray start` on
   every node *before* invoking `run_mobilegym_e2e.sh` as the driver — env vars exported
   inside `run_mobilegym_e2e.sh` are too late to reach the raylet's log monitor. Not
   applied without confirming the launcher-script edit.
3. ✅ **Landed:** actor NVML sampling. `TrainRayActor.init()` in
   `relax/distributed/ray/train_actor.py` now starts a `JudgeGpuSampler` (`role="actor"`)
   when `RELAX_JUDGE_GPU_SAMPLE_DIR` is set, using `ray_get_device_ids()` (already imported
   in that file for `get_local_gpu_id()`) for the physical GPU id rather than
   CUDA_VISIBLE_DEVICES parsing, so it is correct regardless of Ray's visible-device
   remapping. No launcher change needed — `run_mobilegym_e2e.sh` already propagates
   `RELAX_JUDGE_GPU_SAMPLE_DIR`/`RELAX_JUDGE_GPU_SAMPLE_INTERVAL_S` into every actor's
   runtime env via `RELAX_PROPAGATE_ENV_VARS` (`relax/distributed/ray/actor_group.py:106`
   merges `runtime_env["env_vars"]` into every `TrainRayActor`, including train actors, not
   just inference engines). Needs a fresh run to produce data — trainer GPU idleness is
   still an inference on any *existing* dump.
4. ✅ **Understood (M6).** `transfer` is flat ~105 s in both modes (p50 107.2 s, max
   118.8 s, groups release simultaneously) because `TransferDomain._spawn_transfer`
   (`relax/agentic/pipeline/transfer.py:200`) refuses to flush the buffer until it holds
   `transfer_batch_group_count` groups, and `drain_ready_group_payloads` already calls
   `_spawn_ready_transfers()` after every batch of newly-scored groups — so streaming per
   group as they finish, rather than waiting for the round, is not a missing code path, it's
   already gated purely by that one threshold. `transfer_batch_group_count = global_batch_size
   // num_iters_per_train_update // n_samples_per_prompt` (`transfer.py:113`); with
   `num_iters_per_train_update=1` in every reference run this equals the full round
   (64 // 1 // 8 = 8), which is why F never overlaps D/E. The identical formula appears
   independently in `relax/engine/rollout/sglang_rollout.py:929` for the non-agentic path,
   so this is a deliberate, load-bearing convention, not agentic-specific plumbing, and
   `num_iters_per_train_update` is already tracked in the benchmark-invariant-hash config
   list (`relax/utils/judge_config.py:114`) — the paired-run validator already treats it as
   a first-class axis. **Not a safe drop-in fix**, though: per `actor.py:1373`'s docstring,
   raising it changes how advantages get normalized (sub-batches are collected, *then*
   normalized across the full batch and DP group) — a training-numerics change, not just an
   I/O-timing one. This confirms the plan's existing **T1 arm** (varying
   `num_iters_per_train_update` as a separate, explicitly-labeled systems-only comparison,
   not folded into A1/A2/B1/B2) was already the right way to test streaming the transfer —
   no new code needed, and no change made outside that arm.

---

## 6. TODO

| id | item | cost | unblocks |
|---|---|---|---|
| **D1** | `max_concurrency: 8 → 16/32` in both judge specs | 1 line JSON | 65% of the reward branch |
| **E0** | offline judge microbenchmark (images × resolution × concurrency) | 4 GPUs, no training | Q: the 37 s; PRM-vs-ORM attribution |
| **M1** | 🔍 **root cause found**, fix not applied | see Open Questions §2 | needs launcher-script confirmation (Ask First) |
| **M4** | ✅ **done** — actor NVML sampler | landed | see above |
| **M6** | ✅ **done** — mechanism understood, no code change needed | landed | see Open Questions §4 |
| **C1** | ✅ **done** — split `rollout_pre_generation` 3-way | landed | see below |
| **C2** | ✅ **done** — judge `queue`/`http` sub-spans + attributes | landed | see below |
| **M5** | ⏸ **paused** — blocked on E2 (needs a clean run to build on), see §8 | user decision 2026-08-21 | makes §3 quotable |

**New tool:** `examples/mobilegym_agentic/scripts/chain_decomposition.py` computes the
per-sample B/C/D/E/F/G/H table in section 1 directly from `rollout_result/*.jsonl` (no
timeline dump needed). Run against `3124850`/`3124159`: 576/576 samples had a complete
chain on both, **max residual 0.000000 s**, and every mean/percentage reproduces this
document's section 1 exactly.

### C1 result: it is admission-gate wait, not preprocessing

The original plan guessed the second half of `rollout_pre_generation` was preprocessing
(`admission_wait` + `preproc`). Tracing the actual mark sites in
`relax/agentic/session/service.py` shows the old span (`chat_request_arrive_at` →
`generation_queue_enter_at`) crosses three ordered points, not two:
`ir_created_at` (`_create_inflight_request`, after chat-template/tokenization/vision-prep
already computed by the caller) → `ir_activated_at` (`_maybe_start_next_ir_locked`, when
the per-session admission gate pops this IR) → `generation_queue_enter_at` (`_run_ir`,
lock/requeue bookkeeping only). So it was split three ways instead —
`rollout_admission_setup` / `rollout_admission_wait` / `rollout_dispatch` — landed in
`relax/agentic/rollout.py:1574` and mirrored in
`examples/agentic_dual_judge/analyze_latency.py` (`_stage_for_event`, `STAGE_ORDER`, and
the direct-JSONL ingestion path), plus `ROLLOUT_GEN_STAGES` in
`examples/mobilegym_agentic/scripts/stage_breakdown.py` now includes
`rollout_admission_wait` so it counts toward `stall:rollout_gen` instead of disappearing
into `stall:nothing_observed`.

Measured on `3124850` (4,039 turns): **`admission_wait` is 99.7% of the old span**
(sum 166,273 s vs 543.6 s setup + 1.2 s dispatch). So the largest span family in the trace
(previously 171,177 s of misclassified "rollout work") is confirmed to be almost entirely
**per-session admission backpressure** — `admission_wait` p50 0.000 s but p90 107.2 s, max
720.6 s, matching the earlier p90/max reported for the unsplit span.

### C2 result: queue/http sub-spans, landed and verified

`relax/agentic/rollout.py:1521` now emits `critical_path.reward.{component}.queue`
(`judge_{component}_queue_enter_at` → `judge_{component}_queue_acquired_at`, the client-side
`self._semaphores[spec.role]` wait in `relax/engine/rewards/dual_agentic_judge.py`) and
`critical_path.reward.{component}.http` (`judge_{component}_http_start_at` →
`judge_{component}_http_end_at`, last attempt only — these two keys are overwritten per
retry, so a multi-attempt sample's span covers only its final attempt; `attempt_count` is
carried as an attribute so this is identifiable), plus the full judge-trace sub-timings as
attributes on the parent `critical_path.reward.{component}` span (previously only
`critical_path.judge_request` in the direct-JSONL path carried these; the timeline path had
none). Mirrored in `analyze_latency.py`'s direct-JSONL ingestion as
`critical_path.judge_request.queue`/`.http`, which required widening
`_stage_for_event`'s `critical_path.judge_request` check from `==` to `startswith` — audited
against every other exact-name check in that file (`critical_path.reward`,
`critical_path.judge_request` in the reliability/cardinality validators) to confirm none of
them would double-count the new suffixed spans.

**Verification (both C1 and C2), stronger than a hand-replay:** imported
`_build_agentic_critical_path_timeline_events` from `relax/agentic/rollout.py` directly and
ran it against all 576 real samples from `3124850` (wrapping each row's `latency_trace` in a
`Sample(metadata={"agentic_trace": ...})`, then calling the function exactly as the rollout
process does). All 7 new span names are emitted, and every duration matches the
hand-computed numbers exactly: `admission_setup` sum 543.6 s, `admission_wait` sum
166,273.1 s, `dispatch` sum 1.2 s, `reward.multi_turn_reasoning.queue` p50 97.64 s,
`reward.multi_turn_reasoning.http` p50 47.74 s. This exercises the actual production code
path, not a re-implementation of its logic — the remaining gap to a live-cluster smoke test
is small (mainly: does the JSONL this was tested against still match what a *fresh* run
produces, given `analyze_latency.py`'s benchmark-invariant hash would catch a schema drift
anyway).

**Not doing:** `env_step` span. Measured env-gap union is only 7.3% of the window — not a
bottleneck; offline derivation from `chat_end_at` → next `chat_request_arrive_at` suffices.

---

## 7. Next measurement experiment

### E0 — offline judge microbenchmark (do first; no training GPUs)

Stand up `judge_multiturn_vlm` alone (4 GPUs) and replay real trajectories from
`rollout_result/train/*.jsonl`. Sweep, ≥50 requests per cell:

- `n_images ∈ {0, 1, 2, 4, 8}` — **`n_images=0` is the key control**
- `max_pixels_per_item ∈ {2764800, 1382400, 691200}`
- client concurrency ∈ {1, 2, 4, 8, 16, 32}
- also run **Qwen3-4B on the same text-only prompt** to separate model size from modality

Record per request: `queue` / `http` / `engine_http`, NVML util, `num_running_reqs`,
`gen_throughput`, real prompt tokens.

Decision rules:
- `n_images=0` returns to ~1 s ⇒ **modality is the cause**, model size excluded
- engine time grows linearly in `n_images` ⇒ **fixed per-image CPU cost**
- engine time grows with concurrency ⇒ **serialization inside the engine**

Output: a service-time curve `S(n_images, pixels, concurrency)` that feeds the queueing
model and sizes `max_concurrency` properly.

### E1 — instrumentation, landed before E2

C1, C2, M1, M4 from the TODO table. Small, all data already exists.

### E2 — main paired experiment

**Correction:** `G5_FULL24_ONLY=1` is **6 nodes / 24 GPUs**, not 4 nodes/16 GPUs as this
document said earlier — `submit_mobilegym_e2e.sh` asserts `NUM_NODES -eq 6` for this mode.
All reference runs analyzed in sections 1–4 (`3124850`/`3124159`/etc.) were 24-GPU runs;
every "16 GPU" reference earlier in this document should be read as 24 GPUs. This does not
change any measured latency number, only the resource-count framing.

Four arms, everything else fixed (model, data order, resources, round count). `SEED` is now
a script knob (`examples/mobilegym_agentic/run_mobilegym_e2e.sh`, previously hardcoded to
42) so replicates can vary it. `max_concurrency=24` is derived analytically, not from E0
(E2 is running before E0 per the execution decision below):

| arm | `reasoning_trigger` | `max_concurrency` | judge config | purpose |
|---|---|---|---|---|
| A1 | terminal_once | 8 | `judge_services_e2e_g5_terminal_once.json` (default) | baseline, reproduces current data |
| A2 | terminal_once | 24 | `judge_services_e2e_g5_terminal_once_mc24.json` | isolates the queueing amplifier |
| B1 | per_turn | 8 | `judge_services_e2e_g5_per_turn.json` (default) | reproduces the per_turn win |
| B2 | per_turn | 24 | `judge_services_e2e_g5_per_turn_mc24.json` | expected best |

`max_concurrency=24` derivation (no E0 needed): semaphore hold time (`http_elapsed_s` p50)
47.7 s; measured peak arrival rate 0.339 req/s (round 1 of `3124850`); break-even
`c = 0.339 × 47.7 = 16.2`; **24** gives 1.48× headroom. Risk: if the 38 s engine time is
single-threaded CPU work rather than genuine queueing, service time inflates and A1 vs A2
will be flat — that outcome is itself the trigger for E0.

Launch command per arm/replicate (from the repo root, on the login node running the nginx
gateway — `MOBILEGYM_ENV_URL` must be captured on that node):

```bash
sbatch --time=06:00:00 -p normal --nodes=6 \
  --export=ALL,REASONING_TRIGGER=<terminal_once|per_turn>,NUM_ROLLOUT=22,G5_FULL24_ONLY=1, \
SEED=<42|43|44>,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
[,JUDGE_SERVICES_CONFIG_OVERRIDE=examples/mobilegym_agentic/judge_services_e2e_g5_<terminal_once|per_turn>_mc24.json] \
  examples/mobilegym_agentic/submit_mobilegym_e2e.sh
```

12-run matrix (seed blocked across A/B and mc8/mc24 pairs so e.g. A1-rep1 and A2-rep1 share
a seed):

| replicate | seed | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| 1 | 42 | 4 attempts, 0 reached round 1 — see §8. `3138562` stuck (double-activation), `3138724` exit 2 (post-round-0 validation bug), `3139160` double-activation again, `3139260` exit 2 again. **E2 paused by user decision (2026-08-21) — not resuming without explicit direction.** | pending | pending | pending |
| 2 | 43 | pending | pending | pending | pending |
| 3 | 44 | pending | pending | pending | pending |

Protocol:
- **≥20 measured publication rounds** after ≥2 warmup rounds (`NUM_ROLLOUT=22`); the smoke
  test itself used `NUM_ROLLOUT=9` to match the reference run for a fast regression check,
  not for the final measurement
- **≥3 independent replicates** per arm, different seeds across replicates, same seed at a
  given replicate index across A1/A2/B1/B2; aggregate **one global delta per pair**, never K
  autocorrelated rounds as K samples (`examples/agentic_dual_judge/aggregate_latency_replicates.py`)
- fresh `--timeline-dump-dir`/`--rollout-result-dir`/`gpu_samples`/`latency_markers` per run
  (automatic — `EXP_DIR` defaults to `.../exp/${SLURM_JOB_ID}`)
- end with admission stop + drain + one final timeline flush, or in-flight judge work is
  right-censored
- randomise arm execution order to blunt cluster-temperature bias (12 runs is enough that
  submitting all remaining 11 together, letting SLURM interleave them, achieves this better
  than a fixed sequence would)
- keep `--use-metrics-service` on in every arm; run one trace-off pair via
  `analyze_ready_markers.py --require-trace-calibration` to bound tracing overhead

Primary metrics, in priority order:
1. **ready-to-ready makespan** per round (median + IQR) — the headline
2. **chain decomposition** B/C/D/E/F (`chain_decomposition.py`, §1 table) — where the tail went
3. `trainer_not_stalled` %, and with M4, measured actor GPU idle
4. judge GPU efficiency: NVML idle, `running_reqs`, GPU-seconds per training sample
5. within-round reward-completion spread — the straggler metric (§2's "round barrier", not
   a GRPO-group metric — see M6)

Success criteria:
- **A1** must reproduce `3124850` (median round ~587 s, tail ~556 s) — the gate before
  submitting the other 11 runs.
- **A1 vs A2** quantifies the semaphore amplifier. Prediction: reward branch 149 s → ~50 s.
  A flat result promotes E0 from contingency to required.
- **A1 vs B1** turns §3 into a quotable result with error bars.
- **B2** should be the fastest arm; if E is still >20% of the tail, that's expected (M6
  found E is gated by `num_iters_per_train_update`, which none of these 4 arms vary) and
  motivates a T1 follow-up rather than indicating a bug.
- If chain segments still sum to 100% with a small residual, the accounting holds; a large
  residual means instrumentation is missing an edge.

### Sequencing (as executed, differs from the original plan)

Decision: **E2 first**, E0 held as contingency (only if A1 vs A2 is flat), M6 resolved by
code reading (no cluster run needed). E1 (C1/C2/M4) landed and verified against historical
data before E2; the RAY_DEDUP_LOGS=0 fix and the SEED knob landed immediately before
submitting the first E2 job. `D1` is folded into E2 as arms A2/B2 rather than run
separately, so it costs no extra cluster time.

Preconditions verified before submitting: nginx gateway relaunched with the
`ModuleNotFoundError: No module named 'scripts'` fix from this document's own setup
instructions (the bundled `start_nginx_gateway.sh` needs the manual `PYTHONPATH` relaunch
every time, this is not a one-time fix); SLURM reservation `SD-69241-apertus-1-5-0` active
and large (600 nodes, account `infra01`, until 31 Aug); no conflicting jobs queued.

## 8. Bug found while running E2: a rollout dataflow-loop crash can silently stall the whole
job even with `--use-health-check` on

**Job `3138562` (A1, replicate 1, seed 42) hit this; killed after ~23 min stuck and
resubmitted as `3138724`.** Not a regression from any change in this session — the trigger
condition (an engine-init crash-and-replica-restart) is the same pre-existing,
environment-level `BlockingIOError` flakiness documented in section "Trainer GPU really is
idle" below (also seen 12× in the already-analyzed `3118665`), but this specific
*consequence* of it is new and worth a real writeup.

**Sequence of events**, all confirmed against the live cluster (`ray job list`/`py-spy dump`
via the job-submission HTTP API, `--address="http://<gcs-ip>:8265"` — direct `ray status`/
`ray list actors` against the raw GCS port from the login node hit a Ray client-version
mismatch and, separately, cross-node timeouts; the skill's documented workaround of routing
diagnostic commands through `ray job submit --working-dir <minimal-dir>` worked, once the
working-dir was trimmed below the API's 100 MB request-body cap):

1. `SGLangEngine.init()` hits the known `BlockingIOError` on one rollout engine rank during
   normal concurrent-engine-launch contention (12 rollout + 8 judge engines starting at
   once). Ray Serve auto-restarts the `Rollout` replica; a fresh `RolloutManager` comes up
   and all 12 rollout engines register healthy.
2. The fresh `RolloutManager`'s first batch admission for `rollout_id=0` hits
   `RuntimeError: RuntimeDomain activation did not take ownership of every leased request:
   leased_requests=64, activated_sessions=128, started_sessions=128` (raised at
   `relax/agentic/pipeline/runtime.py:2770`, `RuntimeDomain.start_batch`) — 128 is exactly
   double the expected 64 (`rollout_batch_size × n_samples_per_prompt`). Leading hypothesis:
   the crashed first `RolloutManager` had already leased/activated some sessions for
   `rollout_id=0` before dying in step 1, and the fresh replica re-leased/re-activated the
   same batch without full cleanup of state the dead one left on the (not-restarted) rollout
   SGLang engines or in the session forest. Not confirmed further — the practical fix was
   resubmitting, not chasing this specific double-activation to its root.
3. `relax/agentic/rollout.py:537` `_resident_dataflow_loop` catches the exception, records
   it (`self.resident_dataflow_error = exc`), logs `"Agentic resident pipeline dataflow loop
   failed"`, and **returns — permanently, with no internal retry**.
4. `relax/components/rollout.py:452` catches the propagated error, calls
   `self.healthy.report_error.remote("rollout", error_msg)` (implicitly `fatal=False`), and
   since `--use-health-check` is enabled for this launch script (confirmed:
   `run_mobilegym_e2e.sh:420` sets `--use-health-check`), takes the `break` branch — it does
   **not** re-raise, trusting the external `HealthChecker` to notice and restart.
5. **The `HealthChecker` never acts.** `relax/utils/health_system.py`'s `_check_loop` polls
   `get_unhealthy_services()` every `check_interval=1.0` s and should log
   `"Service {role} is unhealthy: ..., triggering restart"` the moment it sees the report.
   Over the ~9 minutes between the failure and the kill (≈540 poll cycles), that line never
   appeared anywhere in the log — nor did "restart", "_restarting", or "HealthChecker" in
   any form. The error report from step 4 never became visible to the central health-check
   loop. Not confirmed further whether `self.healthy` in the `Rollout` deployment and
   `HealthChecker.health_status` in the controller are the same actor handle (the most
   likely break point) — this needs someone with more context on the health-check wiring to
   pin down; flagging here rather than guessing further.
6. Meanwhile the top-level `training_loop` (`relax/core/controller.py:773`) sits blocked on
   `await [task_ref for task_ref in task_refs]` for **all** services, not just rollout —
   confirmed via `py-spy dump` on the driver process (`pid=226132`), MainThread idle at that
   exact line. Rollout's task exited early (step 3/4); the other services (actor, advantages,
   judges) were presumably idle waiting for data rollout would never produce again. All
   SGLang/GenRM engines stayed healthy throughout (confirmed via `/metrics` 200s and a
   6-node `ray job submit ... run_on_each_ray_node.py --list`) — this was a pure
   orchestration stall, not a resource or engine failure.

**Action taken:** `scancel 3138562`, resubmitted as `3138724` with identical arguments. No
code change made — the fix (if the wiring-gap hypothesis in step 5 holds) is outside the
scope of this latency measurement work and would need someone who owns the health-check
subsystem to confirm and fix the `self.healthy` vs. `HealthChecker.health_status` actor
identity.

**For E2 going forward:** if a run stalls with zero log growth beyond routine `/metrics`
polling for more than ~2 minutes after any `RuntimeError`/`resident pipeline dataflow loop
failed` line, don't wait longer — it will not self-heal. Kill and resubmit immediately
rather than burning the `--time` budget.

### Addendum: a second, different failure on the very next attempt

`3138724` (the resubmission) did **not** hit the double-activation bug — `rollout_id=0`
completed cleanly end to end: rollout, reward, transfer (`Total yielded: 8/8`), advantages
computed, `Rollout fully completed for rollout_id: 0`, 64 samples saved to
`rollout_result/train/0.jsonl`. The trainer's `StreamingTQIterator` polling log confirms
this was a normal wait, not a hang: `empty_streak` climbed steadily from 0 to 220 over
`elapsed=621.0s` while `rollout_id=0` was still generating, matching the `data_wait`
pattern already characterized in section 1 — nothing anomalous there.

**But the whole SLURM job then exited (`FAILED`, `head_exit_code=2`) within the same minute
round 0 finished**, after only 1 of 9 requested rounds, `--time=03:00:00` nowhere close to
expired. The driver's own stdout has **no Python traceback** — it just stops, which is the
signature of an uncaught `SystemExit` rather than an uncaught exception. `analyze_latency.py`
argparse-fails with exit code 2 by default, and `run_mobilegym_e2e.sh` invokes it
(`examples/mobilegym_agentic/run_mobilegym_e2e.sh:472`, gated on `G5_FULL24_ONLY=1`) checking
for `latency_markers/weight_serving_ready.jsonl` — a file that only gets written after a
round's weight sync completes, which round 0 never reached. That invocation is placed
**after** a plain `ray job submit --address="http://127.0.0.1:8265" ... | tee "$DRIVER_LOG"`
with no `--no-wait`, which should block until all 9 rounds finish — so this sequencing alone
doesn't explain why the analysis step would run after only 1 round. Checked and ruled out:
`RAY_NO_WAIT` was not set in the submitting shell (so `--export=ALL` didn't leak it in).

**Not resolved.** The two most likely explanations are (a) `ray job submit` returned early
for a reason not yet identified (the underlying Ray job itself may have been marked
STOPPED/FAILED by something else right at that moment, independent of the analysis script),
or (b) the analysis-script hypothesis is a red herring and something else entirely produced
exit code 2. Resubmitted a third time as `3139160`, this time with a live Monitor watching
for the exact transition (round-0-completion → exit) as it happens, rather than
reconstructing after the fact — forensic post-hoc log reading for this one hit diminishing
returns. If it recurs identically a third time, that's strong evidence of a systematic issue
in `run_mobilegym_e2e.sh`'s post-run validation sequencing for low `--num-rollout` counts,
worth a real fix rather than a workaround.

### Third attempt: the double-activation bug recurs, and resolves the open question from step 5

`3139160` (attempt 3) hit an `SGLangEngine.init()` engine-init-barrier crash again (same
`BlockingIOError`-class failure, different rank), Ray Serve redeployed a fresh `RolloutManager`
— and that fresh replica hit the **identical** double-activation error on its very first batch:
`leased_requests=64, activated_sessions=128, started_sessions=128`. This is now **2 of the 3
live attempts this session** where an engine-init crash-and-redeploy was directly observed, and
**both times** it was immediately followed by this exact error. Strong evidence this is a
deterministic (or near-deterministic) consequence of the crash-redeploy path, not a rare
coincidence — supports the hypothesis in step 2 above (stale leased/activated state surviving
across the `RolloutManager` replacement) over an unrelated independent cause.

This run resolved the step-5 open question, with the opposite answer from before: this time the
error correctly propagated all the way up — `relax.core.controller:200` logged `"Training loop
failed: Service task failed: ..."`, i.e. `training_loop`'s outer `except Exception as e:`
handler ran with `self._restarting == False`, took the final `else` branch
(`logger.exception(...); raise`), and the driver exited cleanly with **exit code 1** and a full
traceback — unlike `3138562`, which never unblocked at all. **So the HealthChecker's silence in
`3138562` was not this failure mode's normal behavior; it was a separate, still-unexplained
anomaly on top of it.** Two different outcomes (clean crash vs. permanent silent stall) from the
same triggering error is itself worth noting for whoever investigates this.

**Assessment:** this bug is now confirmed real and reproducible, not a one-off. Fixing it
properly means touching `relax/agentic/pipeline/runtime.py` (`RuntimeDomain`) and/or
`relax/distributed/ray/rollout.py` (`RolloutManager` replica lifecycle / session-forest cleanup
on replacement) — both are core distributed-orchestration code, squarely inside CLAUDE.md's Ask
First scope ("Controller / Service / Launcher logic"). Not attempted without explicit
confirmation. Continuing to retry in the meantime: the underlying trigger (engine-init
`BlockingIOError`) is itself the same pre-existing, already-documented flakiness that the
already-analyzed `3118665` hit 12 times and still completed successfully, so a clean run remains
plausible on a retry that either avoids the crash entirely or avoids hitting this specific
follow-on bug.

### Fourth attempt: a second bug, now also reproduced twice — stopping autonomous retries here

`3139260` (attempt 4) did **not** hit an engine-init crash and did **not** hit the
double-activation bug — `rollout_id=0` completed cleanly (`Total yielded: 8/8` at 14:35:02,
matching the earlier clean case in `3138724`). It then failed **identically to `3138724`**:
`analyze_latency.py --ready-markers ... weight_serving_ready.jsonl` argparse-errors (file
doesn't exist) → exit code 2 → `set -e` in `run_mobilegym_e2e.sh` propagates it → SPMD wrapper
force-stops the whole 6-node job. **This is now confirmed as a second, independent,
reproducible bug** (2/2 times a run completed round 0 cleanly, it hit this exact failure next),
distinct from the double-activation bug (which is tied to the engine-init-crash path and never
occurred in either of the two runs that reached round 0 cleanly).

Checked and ruled out this round: `python3 -m relax.entrypoints.train`'s own rollout loop never
logged `"All rollouts finished"` — it did not believe it was done, so the early exit is not a
round-counting bug in the training loop itself. No `MegatronTrainRayActor` training-step
activity (forward/backward/optimizer) appears anywhere in the log *after* round 0's rollout
completes — the trainer's `StreamingTQIterator` was still in its data-wait poll loop, per the
same pattern documented in section 1, when the job was cut off. No CUDA/NCCL/segfault-class
error strings appear in either `.out` or `.err`. So the failure is not (as far as this log can
show) a low-level training crash consuming the just-arrived data — it looks like `ray job
submit` (no `--no-wait`, which should block until the whole 9-round job finishes) returned
after only round 0, for a reason not identified. `RAY_NO_WAIT` is not set anywhere in the
submitting shell or found hardcoded for the G5 path in `run_mobilegym_e2e.sh`.

**Stopping the autonomous kill-and-resubmit loop here.** Four live attempts surfaced two
distinct, independently-reproducible bugs (both pre-existing — none of this session's C1/C2/M4
changes touch the code paths involved, and both bugs are exercised before any of those changes'
logic would even run):

| bug | occurrences | trigger | where |
|---|---|---|---|
| double-activation (`128` vs `64` sessions) | 2 of 4 (`3138562`, `3139160`) | follows an engine-init `BlockingIOError` crash-and-replica-restart | `relax/agentic/pipeline/runtime.py` / `relax/distributed/ray/rollout.py` |
| premature `analyze_latency.py` validation (exit 2) | 2 of 4 (`3138724`, `3139260`) | follows round 0 completing cleanly, before round 1 | `examples/mobilegym_agentic/run_mobilegym_e2e.sh` (example script, not core) |

Every one of the 4 attempts hit exactly one of these two; **zero reached round 1.** Continuing
to resubmit blind is unlikely to produce a different outcome without a code change. The second
bug lives in an example script (not Ask-First scope) and is more tractable — worth investigating
`ray job submit`'s exact return condition next — but doing that investigation justice needs
either significantly more diagnostic budget or someone who already knows this script's
history/intent. The first bug's fix is in Ask-First-scoped orchestration code regardless.

### Status: E2 paused (user decision, 2026-08-21)

Given the choice between investigating the script bug, investigating the `RuntimeDomain` bug,
blind retries, or pausing, the user chose to **pause E2 entirely** rather than spend more
cluster time or diagnostic budget right now. Nothing is running; all four SLURM jobs
(`3138562`/`3138724`/`3139160`/`3139260`) are confirmed terminal
(`CANCELLED+`/`FAILED`/`FAILED`/`FAILED`). Not resuming autonomously.

**To resume, in priority order:**
1. **Script bug first** — cheapest, no cluster time, fully in-scope for direct fixing. Read
   `run_mobilegym_e2e.sh`'s `ray job submit ... | tee "$DRIVER_LOG"` block and the
   `G5_FULL24_ONLY` validation block right after it; confirm whether `ray job submit` (no
   `--no-wait`) is truly blocking on this Ray/cluster version, or whether the job's Ray status
   reaches a terminal state after round 0 for a reason not yet found. `3138724` and `3139260`
   are complete log records of the failure to work from.
2. **`RuntimeDomain` double-activation bug** — needs sign-off (Ask First: Controller/Service
   logic). The hypothesis (stale leased/activated state surviving a `RolloutManager` replica
   restart) is written up in step 2 of the numbered list above but not confirmed by reading
   `RuntimeDomain.start_batch`'s actual session-tracking code.
3. Once either bug is fixed, rerun the A1 smoke test (`examples/mobilegym_agentic/LATENCY_FINDINGS.md`
   §7's launch command) before resuming the full E2 matrix — don't skip straight to the 4-arm ×
   3-replicate run.

**Independent of E2**, `M6` and the corrected `LATENCY_FINDINGS.md` §1–6 conclusions
(reward is the bottleneck, `max_concurrency` under-provisioned, `transfer` gating mechanism
understood) stand on their own and don't need E2 to be trusted — they're derived from
already-completed historical runs, not from the four failed smoke-test attempts.
