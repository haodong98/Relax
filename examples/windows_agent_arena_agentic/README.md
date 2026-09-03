# WindowsAgentArena × Relax fully-async functional smoke

This example integrates the 154 public WindowsAgentArena (WAA) tasks with Relax managed-command rollouts without
changing Relax's Controller, Services, launchers, argument parser, or public APIs. The first hardware gate is deliberately
narrow: Qwen3-VL-2B-Instruct, one pinned Notepad task, three fully-async rounds, four nodes, and 16 GPUs for at most two
hours. It tests process functionality, not convergence, task quality, or throughput.

## Architecture and trust boundary

Each allocated host runs one authenticated, capacity-one broker and one ARM64/KVM Windows VM. Relax and SGLang stay in
the EDF container. Managed agent processes can call only the broker's high-level lease/action/evaluate/release endpoints;
the raw WAA `/execute` endpoint is bound to host loopback and is never advertised.

The model emits exactly one structured action:

```text
<action>{"type":"click","x":0.5,"y":0.5}</action>
```

The parser rejects prose, duplicate JSON keys, extra fields, NaN/Infinity, arbitrary code, and coordinates outside
`[0,1]`. The trusted broker renders validated actions to deterministic `pyautogui`; raw policy strings never reach WAA.
Only native terminal WAA evaluator scores become rewards. A valid `0.0` remains a training sample; VM, controller,
expected/gold, metric, or transport failures exit nonzero so Relax drops the whole GRPO group.

## Deterministic dataset

Generate the fixed 108/23/23 domain-stratified split:

```bash
/usr/bin/python3.11 examples/windows_agent_arena_agentic/scripts/build_dataset.py \
  --waa-repo "${WAA_REPO_DIR}" \
  --output-dir "${DATA_DIR}/waa"
```

The assignment digest for the pinned 154-task corpus is:

```text
5cf5a23d34c071720161b4181e735378a20de4b9e2cfacd4a8cce6ea155088b9
```

Policy JSONL rows contain only the instruction and safe task identity metadata. Full config/evaluator/expected data stays
in `trusted_registry.json`, which is read only by the host broker. `split_manifest.json` also records six upstream source
files whose internal `id` disagrees with the canonical `test_all.json` identity.

This v1 split is reproducible and domain-stratified, but it is not a semantic generalization split: three identical
instructions and several numbered task families cross split boundaries. Do not use its dev/test scores as evidence of
unseen-task generalization. A family-clustered split v2 is required before benchmark reporting; the pinned functional
smoke is unaffected because it does not estimate generalization.

Cache the pinned smoke task's gold file before launching:

```bash
/usr/bin/python3.11 examples/windows_agent_arena_agentic/scripts/cache_assets.py \
  --registry "${DATA_DIR}/waa/trusted_registry.json" \
  --output-dir "${DATA_DIR}/waa/assets" \
  --task-id 366de66e-cbae-4d72-b042-26390db2b145-WOS
```

## Four-node launch

The wrapper requests exactly `4 nodes × 4 GPUs/node × 2 hours = 32 GPU-hours`, below the 80 GPU-hour cap. It does not
contain a reservation and will not silently shrink the allocation. Required paths must be supplied explicitly:

```bash
sbatch --export=ALL,\
WAA_REPO_DIR="${WAA_REPO_DIR}",\
WAA_GOLDEN_STORAGE="${WAA_GOLDEN_STORAGE}",\
WAA_PYTHON="${WAA_PYTHON:-/usr/bin/python3.11}",\
RELAX_ENV_ROOT="${RELAX_ENV_ROOT}",\
EDF_TOML="${EDF_TOML}",\
MODEL_DIR="${MODEL_DIR}",\
DATA_DIR="${DATA_DIR}",\
SAVE_DIR="${SAVE_DIR}",\
EXP_DIR="${EXP_DIR}" \
examples/windows_agent_arena_agentic/submit_waa_4node.sh
```

The pinned Notepad smoke talks directly to the audited WAA screenshot, structured execution, setup, and file endpoints,
so its broker needs only `requests` (available in `/usr/bin/python3.11`). Full-corpus execution still uses the stock WAA
controllers and must set `WAA_PYTHON` to an interpreter containing all WindowsAgentArena optional dependencies.

The wrapper performs these hard gates before training:

1. exact four-node allocation and required path checks;
2. deterministic dataset generation and pinned gold caching;
3. Transformer Engine, TransferQueue, Bridge import and exact official 2B provider config on every EDF node;
4. one host broker per distinct node with atomic manifests;
5. all four broker endpoints reachable from every EDF container;
6. a Slurm-deadline-derived watchdog and exact-container/exact-overlay cleanup.

Each broker also appends fsync-backed, per-node `broker_events/*.jsonl` records for lease reservation, cold readiness,
native evaluation, release/cancellation, and failure. These records contain task/request/lease identity but no bearer
token, instruction, screenshot, or model output.
The EXIT trap removes the bearer token, requires the shared manifest directory to be empty, audits the exact Podman
container prefix and node-local overlay root on all four nodes, and writes four `cleanup_audit/*.json` records.

Before allocating GPUs, the same broker path can be validated on one CPU allocation with
`smoke_waa_broker_1node.sh`. It performs a live `lease -> wait -> done -> evaluate -> release` episode and requires the
clean golden image to return native reward `0.0`, then audits exact container and overlay cleanup.

The training topology is actor 4 GPUs (TP=1, DP=4), rollout 12 GPUs (12 TP=1 engines), and CPU advantages, with
`--fully-async --max-staleness 0 --num-iters-per-train-update 1`. The smoke uses rollout batch size 1, four GRPO samples,
global batch size 4, no reference role, no actor-forward role, KL coefficient zero, and eight GUI turns. The functional
contract fixes the run to exactly three rounds.

The launch uses the official 2B Instruct architecture contract (`rotary_base=5000000`, tied input/output embeddings).
The host broker changes its Linux process name to `waa-broker` before serving so the SPMD entrypoint's stale Python
cleanup cannot kill it. The Ray step receives 256 CPUs per node by default, rejects values below 64, forces synchronous
`ray job submit`, and writes to a fresh `EXP_DIR/job-${SLURM_JOB_ID}` so stale completion or broker manifests cannot pass
a retry gate. Brokers are probed again after the SPMD stale-process cleanup window. The training watchdog is derived from
the Slurm end time and reserves the final 15 minutes for Ray, VM, container, and overlay cleanup.
`RELAX_REQUIRE_WEIGHT_PUBLICATION=1` is enabled inside the EDF so a missing rollout weight consumer fails the smoke
instead of silently degrading to actor-only progress. The exact batch contract auto-enables true-on-policy mode, so the
absence of a separate actor-forward service is intentional and does not weaken this rollout publication gate.

## Validation and current boundary

Login-node tests:

```bash
/usr/bin/python3.11 -m pytest -q tests/agentic/test_windows_agent_arena_integration.py
bash -n examples/windows_agent_arena_agentic/*.sh \
  examples/windows_agent_arena_agentic/scripts/*.sh \
  scripts/training/multimodal/run-qwen3-vl-2b-waa-4node-async.sh
```

The 154-task ingestion, split, structured action protocol, append-only chat history, strict reward distinction, manifest,
dynamic port parser, and fake two-turn episode are CPU-tested. The GPU test is explicitly skipped outside the requested
allocation.

After Slurm reaches a terminal state, produce the strict functional report with:

```bash
/usr/bin/python3.11 examples/windows_agent_arena_agentic/scripts/verify_formal_run.py \
  --exp-dir "${EXP_DIR}/job-${SLURM_JOB_ID}" \
  --checkpoint-dir "${SAVE_DIR}/Qwen3-VL-2B-WAA-Checkpoint" \
  --job-id "${SLURM_JOB_ID}"
```

`formal_verification.json` passes only when Slurm accounting, all four EDF/provider gates, 12 fresh/evaluated/released
leases, 12 committed multimodal rows, three optimizer and rollout-weight publications, the final checkpoint, and four
zero-residual cleanup records all agree. Its claim scope explicitly excludes convergence and full-corpus readiness.

The pinned Notepad path is the only intended first hardware smoke. Some stock WAA getters directly reconnect to fixed
ports `5000/9222/8080`, and several non-Notepad setup methods swallow remote HTTP failures. Those domains are present in
the deterministic dataset but are not claimed production-ready until per-domain compute-node gates validate their guest
forwarding and strict failure classification. Likewise, static code and a queued Slurm job are not evidence of end-to-end
training success; success requires three completed rollout/update rounds, valid native rewards (including valid zero), no
group-wide infrastructure collapse, and cleanup artifacts from the actual allocation.
