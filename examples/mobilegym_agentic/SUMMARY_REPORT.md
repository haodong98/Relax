# Agentic MLLM RL — Experiment Summary

A condensed, self-contained summary of the MobileGym agentic-multimodal RL latency
experiment on Relax. It consolidates the setup, goal, evaluation results, and the
post-hoc analysis (Gantt figure + CPU-codec/network latency) into one document.

Companion documents (full detail):

- `EVALUATION_REPORT.md` — organized by sub-question, the authoritative writeup.
- `LATENCY_FINDINGS.md` — chronological lab notebook with raw commands/logs/diffs.
- `README.md` — infrastructure & launch instructions.

______________________________________________________________________

## 1. Experiment setup

### 1.1 System under test

| axis          | value                                                                          |
| ------------- | ------------------------------------------------------------------------------ |
| framework     | Relax (Ray Serve + Megatron-LM + SGLang), fully-async                          |
| environment   | MobileGym (browser-hosted mobile-GUI simulator), `examples/mobilegym_agentic/` |
| training mode | fully-async (`--fully-async`), not colocate                                    |
| algorithm     | GRPO, `--kl-loss-coef 0.00`, `--true-on-policy-mode`                           |

### 1.2 Reward system (reproducing the Qwen3-VL agentic design)

| signal                   | Qwen3-VL design                         | this experiment                                   | code                                            |
| ------------------------ | --------------------------------------- | ------------------------------------------------- | ----------------------------------------------- |
| LLM outcome model (ORM)  | `answer_accuracy` (Qwen3-32B)           | `answer_accuracy`, Qwen3-8B, text-only            | `dual_agentic_judge.py`, `reward_projection.py` |
| VLM process reward (PRM) | `multi_turn_reasoning` (Qwen2.5-VL-72B) | `multi_turn_reasoning`, Qwen2.5-VL-7B, multimodal | `dual_agentic_judge.py`, `reward_projection.py` |
| Tool-calling reward      | offline expert target                   | **not implemented** (acknowledged gap)            | —                                               |

- Aggregate: `score = 0.8 * answer_accuracy + 0.2 * multi_turn_reasoning` (custom weights).
- Two PRM judgement modes (the central comparison axis):
  - **`terminal_once`** — the VLM reads the complete trajectory once after the episode.
  - **`per_turn`** — the VLM scores each `response -> observation` interaction as a
    **sidecar** while later turns continue; terminal materialization joins them via a barrier.

### 1.3 Model checkpoints (scaled-down workload generators, not authority)

| role                        | model                  | gpus           |
| --------------------------- | ---------------------- | -------------- |
| policy                      | Qwen3-VL-4B-Instruct   | actor 4 (TP=4) |
| rollout                     | policy engine (SGLang) | rollout 12 x 1 |
| ORM (`judge_accuracy`)      | Qwen3-8B               | 4 (TP=4)       |
| PRM (`judge_multiturn_vlm`) | Qwen2.5-VL-7B          | 4 (TP=4)       |

Resource layout (G5_FULL24_ONLY): **6 nodes x 4 GH200 = 24 GPUs**, fully-async.
Batch shape: `rollout_batch_size=8 x n_samples_per_prompt=8 = 64` samples/round = 8 GRPO groups.

### 1.4 Measurement instrumentation (built/verified during this work)

- `critical_path.*` spans for every stage (rollout/reward/transfer/training/weight-update)
  and per-judge `queue`/`http` sub-spans (**C1/C2**).
- `examples/agentic_dual_judge/analyze_latency.py` (3,211 lines) — fixed-K window selection,
  inclusive-occupancy, active-set, paired-workload validation.
- `examples/mobilegym_agentic/scripts/chain_decomposition.py` — per-sample V3 causal chain
  decomposition (residual = 0.000000 s).
- `relax/utils/metrics/judge_gpu_sampler.py` — NVML + SGLang-Prometheus sidecar for judge/rollout
  GPU occupancy; actor-side NVML added (**M4**).

### 1.5 Analyzed runs (all read-only, pre-existing)

| workload        | primary   | cross-checks         |
| --------------- | --------- | -------------------- |
| `terminal_once` | `3124850` | `3118393`, `3117772` |
| `per_turn`      | `3124159` | `3118665`, `3118224` |

`3124850` vs `3124159` are config-identical except `reasoning_trigger` (verified by diffing
the entrypoint argv) — a valid controlled pair. Clock skew audited: max pairwise offset 5.92 ms.

______________________________________________________________________

## 2. Goal

Reproduce the Qwen3-VL two-signal reward system (VLM PRM + LLM ORM) and measure, on a real
agentic MLLM RL pipeline:

1. **Execution time** of the stages: rollout generation, rewarding, training, weight update.
2. **Latency breakdown on the critical path** of those same stages.
3. **GPU utilization** of the corresponding reward GPUs.

Secondary goal: compare the **two VLM-PRM usage modes** (`terminal_once` vs `per_turn`), plus
an initial CPU-codec/network-transfer latency study.

______________________________________________________________________

## 3. Evaluation results

### 3.1 End-to-end time breakdown -> bottleneck is reward, not generation

Per-sample causal chain (means, `3124850` terminal_once / `3124159` per_turn):

| segment                                           | terminal_once | % post-gen tail |    per_turn | % post-gen tail |
| ------------------------------------------------- | ------------: | --------------: | ----------: | --------------: |
| A. trajectory generation (before tail)            |         363 s |               — |       429 s |               — |
| B. finalize (per_turn incl. `turn_judge_barrier`) |        14.0 s |            2.5% |     141.3 s |           40.6% |
| C. wait to enter reward pipeline                  |        73.6 s |           13.2% |      19.4 s |            5.6% |
| D. reward compute                                 |       168.1 s |           30.2% |       4.5 s |            1.3% |
| E. round barrier (straggler)                      |       195.5 s |           35.2% |      78.6 s |           22.6% |
| F. transfer release                               |       105.0 s |           18.9% |     104.2 s |           30.0% |
| **G. post-generation tail**                       |   **556.2 s** |            100% | **347.9 s** |            100% |

Tail attribution (terminal_once): **reward-related (C+D+E) = 78.6%**, transfer = 18.9%,
rollout generation = 0% (already finished in A). Corroborated by the V2 stall-coincidence
method (reward 59.18% / transfer 53.80% / generation 6.71%).

### 3.2 Trainer is idle, not generating

`trainer_not_stalled = 18.15%` (terminal_once) / `34.36%` (per_turn) — the trainer is idle
~82% / ~66%. `data_wait` never overlaps `optimizer_step` or `weight_update` (0.000%). 83% of
wait time sits in episodes >5 s (not CUDA drain). Fully-async removes lockstep barriers, not
the data dependency.

### 3.3 Reward GPUs 88% idle yet slow — self-inflicted queueing, not starvation

PRM (`multi_turn_reasoning`) branch decomposition (p50):

| layer                                      |     p50 | share |
| ------------------------------------------ | ------: | ----: |
| branch total                               | 149.4 s |  100% |
| client `max_concurrency=8` semaphore queue |  97.6 s | 65.1% |
| http RTT                                   |  47.7 s | 34.9% |
| -> engine time                             |  38.1 s | 24.5% |

NVML idle 88.41%, `num_running_reqs` mean 1.33/8. GPU-time/request ≈ 0.93 s vs 38.1 s engine
time => only 2.4% of engine time is GPU. Measured arrival rate 0.339 req/s exceeds capacity
0.168 req/s by 1.3–2.0x every round => the judge is overloaded against its own concurrency cap.

### 3.4 `per_turn` is faster end-to-end (direction solid, magnitude no error bars)

| run       | workload      | ready-to-ready (s)              | median  | throughput       |
| --------- | ------------- | ------------------------------- | ------- | ---------------- |
| `3124850` | terminal_once | 465,721,518,510,726,655,691,352 | **587** | 6.6 samples/min  |
| `3124159` | per_turn      | 321,435,1286,455,411,156        | **423** | 7.5 samples/min  |
| `3118665` | per_turn      | 375,352,345,412,274             | **352** | 10.9 samples/min |

Mechanism (corrected): **not** "judgement overlaps its own later generation" (only 22.1%).
It is: (i) work moves (D -163.6 s, B +127.3 s, net -27 s); (ii) **burst smoothing** — spreading
calls across the trajectory lifetime collapses queueing (C -54 s); (iii) **variance reduction** —
the round max over 64 samples gets cheaper, shrinking E by 117 s (largest single gain).
`per_turn` does **4.2x more total judge work (366,033 s)** yet halves its exposed cost
(stall:reward 59.18% -> 30.95%).

### 3.5 Data volume / network is not a bandwidth bottleneck

Screenshots p50 105.6 KB, ~8/trajectory (~990 KB), 20,400 vision tokens. Transport is base64
data-URIs over **two hops** (rollout -> GenRM Serve -> SGLang). Per round (64 samples): ORM
~0.6 MB, PRM-final ~173 MB, PRM-turn ~152 MB => ~0.3 MB/s — far below any bandwidth limit.

**Flagged:** `input_tokens` under-reports multimodal input ~7x (2,886 vs ~20,400).

### 3.6 Open / blocked

- PRM-vs-ORM **119x** engine-time gap (0.32 s vs 38.1 s) — **unattributed** (model size +
  modality + input size move together); needs the E0 offline microbenchmark.
- **E2** four-arm paired experiment (`mc 8/24` x `terminal/per_turn` x 3 replicates) —
  **blocked**: 4 attempts, 0 reached round 1. Two independent, pre-existing bugs:
  (a) `RuntimeDomain` double-activation (`leased=64, activated=128`) after an SGLang engine-init
  crash+replica-restart (2/4), and (b) premature `analyze_latency.py` post-run validation
  on a missing `weight_serving_ready.jsonl` (exit 2) after round 0 completed cleanly (2/4).
  Paused by user decision (2026-08-21).

______________________________________________________________________

## 4. Post-hoc analysis

### 4.1 Stage-activity Gantt (timeline.pdf style)

Recreated the timeline.pdf stage Gantt from the **real** Chrome-trace events of `3124850` and
`3124159` (rows = stages, x = time 0->T, blue bars = stage active; bar core = p10-90, light =
full min-max to expose stragglers; dashed lines = `weight_serving_ready` round boundaries):

- `latency_gantt_continuous_0_to_T.png` — continuous 0->T axis (closest to timeline.pdf).
- `latency_gantt_per_round.png` — per-publication-round normalized columns (more legible).

What it shows: **T = 4638 s (terminal_once) vs 3076 s (per_turn)**. Rollout and Training are
near-continuous (fully-async never stops), so they are **not** the bottleneck. Weight update is
a thin line (not a bottleneck). Reward blocks follow rollout and are heavy in terminal_once;
`per_turn` moves the VLM work to an orange sidecar row. **Data-wait (grey) blocks fill the
gaps = trainer idling.**

### 4.2 CPU-side base64 / network-transfer latency

Per-request decomposition of the PRM VLM judge (p50, 576 samples, `3124850`):

| phase                                   |      p50 | note                                        |
| --------------------------------------- | -------: | ------------------------------------------- |
| queue (client `max_concurrency=8`)      |   97.6 s | admission throttling, **not** codec/network |
| http (client RTT)                       |   47.6 s | —                                           |
| server_queue (GenRM internal semaphore) | ~0.000 s | **never binds**                             |
| media_restore (base64 decode + SHA-256) |  0.008 s | CPU decode                                  |
| tokenize (ChatML render + encode)       |  0.007 s | CPU tokenize                                |
| engine_http (GenRM -> SGLang)           |   38.1 s | **dominant; GPU only 0.93 s**               |
| transfer + framework (http - req_total) |   0.30 s | **base64 + network**                        |

Findings and critical-thinking caveats:

1. **CPU codec is negligible (~18 ms/request):** base64 encode 3 ms (client payload-prep),
   base64 decode 8 ms (server `media_restore`), tokenize 7 ms.
2. **base64 + per-request transfer ≈ 0.30 s**, and a text-only ORM control shows 0.023 s, so the
   multimodal payload adds ~0.28 s/request. Relative to the 38.1 s engine time, this is \<1%.
3. **"Bandwidth isn't a bottleneck" is a throughput statement, not a latency statement.** The
   0.30 s/request is real, scales with image count/size, and is inflated by base64 (4/3) **and
   double serialization over two hops** (GenRM decodes then forwards base64 to SGLang, which
   decodes again — a redundant decode).
4. **The dominant CPU cost is inside SGLang (~37 s/request)**, not the codec/transport — this is
   the un-attributed E0 question. Because GPU compute is only 2.4% of engine time, the ~37 s is
   CPU-side vision preprocessing (PIL decode / resize / patchify) inside the engine.
5. **The straggler tail is saturation, not transport:** `server_queue` is ~always 0, yet 34.9%
   of requests show a >5 s gap (up to 190 s) between client `http` and server `req_total`. This
   is **Ray Serve deployment-level ingress admission** on a replica saturated by 8 concurrent x
   38 s engine calls — not a per-request base64/network cost. It is a **throughput ceiling**.
   Consequence: raising the *client* `max_concurrency` (8 -> 24) may be flat if the replica/engine
   is the real ceiling — exactly the contingency trigger the report flagged for E0.

______________________________________________________________________

## 5. Key takeaways

1. **The bottleneck is reward, not generation** — but it is partly *self-inflicted*: an
   under-provisioned `max_concurrency=8` plus the transfer gate
   `num_iters_per_train_update=1` (which waits for the whole 64-sample round).
2. **`per_turn` wins on exposure (moves work off the critical path / smooths bursts / lowers the
   round max), at 4.2x the total judge compute.**
3. **The VLM-PRM engine is severely CPU-bound** (~37 s, only 0.93 s GPU) — the largest single
   cost and the least understood; E0 is the highest-value next step.
4. **Base64 + network are genuinely small** (~0.3 s/request) but are a latent, serialization-bound
   risk once the engine-side CPU work is reduced.
5. **The full four-arm E2 experiment remains blocked** on two pre-existing orchestration bugs; the
   conclusions above stand on the completed single-run historical data (directionally strong,
   not yet quotable with error bars — except section 3.1-3.3, which are accounting-identity-exact).
