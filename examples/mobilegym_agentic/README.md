# MobileGym x dual-judge agentic training (Phase 0: end-to-end on clariden)

Wires [MobileGym](https://github.com/Purewhiter/mobilegym) (a browser-hosted mobile GUI simulator
with a deterministic, code-level judge) into Relax's agentic rollout as the environment, alongside
the dual local-judge reward system (`examples/agentic_dual_judge/`). Goal of this phase: prove the
full pipeline -- MobileGym task -> rollout generation -> dual-judge reward computation -> training ->
weight update -- runs end to end on 4 GH200 nodes (16 GPUs). Measurement (latency exposure, judge GPU
utilization) is a separate, later phase; see `path/to/measurement-plan.md`.

## Architecture

Relax's agentic contract is minimal by design: launch a process, hand it an OpenAI-compatible
endpoint (`RELAX_BASE_URL`). MobileGym's own `bench_env.run` CLI already owns the *entire*
multi-turn agent\<->env loop (screenshot capture, action parsing, message history, the deterministic
state-diff judge) -- it just needs to be pointed at that endpoint. So `app/agent.py` does not drive
the loop itself; it is a thin subprocess wrapper: parse `RELAX_INPUT_JSON.metadata.task_id`, run
`bench_env.run --task-id ... --model-base-url "$OPENAI_BASE_URL"`, translate the single
`results.jsonl` row it produces into `RELAX_OUTPUT_JSON` (`reward.score` = MobileGym's own `progress`,
the fraction of `check_goals()` checks passed -- a dense signal from state diffing, not a VLM).

Every turn `bench_env.run` sends is transparently recorded by Relax's session-forest chat service on
the way through, exactly like any other agentic adapter (e.g. `examples/deepeyes_agentic/`) -- the
dual-judge system (terminal-once or per-turn `multi_turn_reasoning` VLM judge, terminal-only
`answer_accuracy` judge) reads that recorded trajectory the same way regardless of what produced it.

```
Relax rollout (SGLang, policy)
   |  RELAX_BASE_URL (per-session OpenAI-compatible endpoint)
   v
run_agent_app.sh -> app/agent.py
   |  subprocess: bench_env.run --model-base-url $OPENAI_BASE_URL --task-id <id> --agent generic_v2
   v
MobileGym bench_env (Playwright, dedicated venv) <---> MobileGym simulator (nginx gateway, :4180)
   |  results.jsonl: {progress, is_success, judge:{success,clean}, ...}
   v
RELAX_OUTPUT_JSON  ->  Relax session forest  ->  dual-judge reward (judge_accuracy + judge_multiturn_vlm)
```

## One-time environment setup (persists across job submissions)

Everything lives under a single scratch root so it's easy to `rm -rf` and redo:
`/iopsstor/scratch/cscs/$USER/mobilegym_e2e/`.

### 1. MobileGym itself (Node.js frontend + Python bench_env)

Login/interactive nodes have no `node`/`npm`/`nginx`/modern Python by default -- none of this needs
root:

```bash
# Node 22 (official arm64 tarball, no build needed)
curl -fsSL -o node.tar.xz https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-arm64.tar.xz
tar xf node.tar.xz -C mobilegym_e2e/node
export PATH="$PWD/mobilegym_e2e/node/node-v22.14.0-linux-arm64/bin:$PATH"

# nginx (conda-forge, dedicated prefix, not the base env)
/users/$USER/miniconda3/bin/conda create -y -p mobilegym_e2e/nginx_env -c conda-forge nginx

# MobileGym repo + companion dataset (~1.9GB, CC BY-NC 4.0)
git clone https://github.com/Purewhiter/mobilegym.git mobilegym_e2e/mobilegym
cd mobilegym_e2e/mobilegym && npm install && npm run build
curl -L -o mobilegym-data.tar.gz https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
tar -xzf mobilegym-data.tar.gz && rm mobilegym-data.tar.gz

# bench_env's own Python env (Playwright + Chromium, independent of Relax's Megatron/SGLang stack)
python3.11 -m venv mobilegym_e2e/venv
mobilegym_e2e/venv/bin/pip install -r bench_env/requirements.txt httpx starlette==0.37.2 uvicorn
mobilegym_e2e/venv/bin/python -m playwright install chromium
```

`starlette==0.37.2` is pinned deliberately: the repo's own `scripts/server/api_gateway.py` uses the
`Starlette(on_shutdown=[...])` constructor kwarg, which recent Starlette (>=0.40ish) removed in favor
of lifespan context managers. `httpx`/`starlette`/`uvicorn` are not in `bench_env/requirements.txt`
(that file covers `bench_env.run` itself, not the standalone gateway script) -- install them
separately as above. `bench_env/requirements.txt` is also missing `requests` (a transitive need of
some task files); install it too if you hit `ModuleNotFoundError: No module named 'requests'`.

**Start the simulator** (needed before any rollout, keep it running for the lifetime of all
experiments -- it is not part of any Slurm job's lifecycle):

```bash
cd mobilegym_e2e/mobilegym
export PATH="$PWD/../node/node-v22.14.0-linux-arm64/bin:$PATH"
NGINX_BIN=$PWD/../nginx_env/bin/nginx PYTHON_BIN=$PWD/../venv/bin/python \
  ./scripts/server/start_nginx_gateway.sh   # -> https://localhost:4180
```

The bundled `start_nginx_gateway.sh` launches `python scripts/server/api_gateway.py` directly rather
than via the `uvicorn module:app` CLI form its own docstring recommends for multi-worker mode; since
`uvicorn.run("scripts.server.api_gateway:app", workers=8, ...)` always re-imports the app by string
(regardless of worker count) for every worker subprocess, and plain `python some/path/script.py`
does **not** put the script's directory's *parent* (the repo root) on `sys.path` the way the `uvicorn`
CLI does, worker subprocesses fail with `ModuleNotFoundError: No module named 'scripts'` unless you
launch it directly with the repo root on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD   # mobilegym repo root
nohup mobilegym_e2e/venv/bin/python scripts/server/api_gateway.py --port 4181 --workers 8 \
  >> .nginx/logs/api_gateway.log 2>&1 &
```

(`start_nginx_gateway.sh` still starts nginx itself and generates the self-signed TLS cert; only the
API backend half needs the manual relaunch above. Verified end to end with a real Playwright episode
against `wechat.ReadMyWxid`, mock model endpoint: `curl -sk https://localhost:4180/` -> `200`.)

**`localhost` only works from the login node itself.** Compute nodes cannot reach the login node via
`localhost` (that resolves to the compute node's own loopback) -- confirmed empirically via a probe
`srun` job: the login node's plain hostname (e.g. `clariden-ln003`) and all of its `hsn0`-`hsn3` /
`nmn0` addresses are cluster-routable and respond from a compute node, `localhost` obviously does not.
Always pass `MOBILEGYM_ENV_URL=https://$(hostname):4180` (captured on the login node at `sbatch` submit
time) -- see "Launching" below. `submit_mobilegym_e2e.sh` has no `localhost` fallback for this reason;
it fails fast with a clear message if `MOBILEGYM_ENV_URL` is unset rather than silently using one.

### 2. Model checkpoints

```bash
pip install -U "huggingface_hub[cli]"
for repo in Qwen/Qwen3-VL-4B-Instruct Qwen/Qwen3-8B Qwen/Qwen2.5-VL-7B-Instruct; do
  hf download "$repo" --local-dir "mobilegym_e2e/models/$(basename "$repo")"
done
```

Qwen3-VL-4B-Instruct is the policy (MobileGym's own validated RL base model). Qwen3-8B /
Qwen2.5-VL-7B-Instruct are the dual-judge models -- deliberately smaller than the
`examples/agentic_dual_judge/` defaults (Qwen3-32B / Qwen2.5-VL-72B): for this experiment they are a
workload generator for the systems measurement, not the reward-correctness authority (MobileGym's own
deterministic `check_goals()` judge is -- see `judge_services_e2e_*.json`'s use of `dual_shadow`-style
training on the recorded env reward once the measurement phase starts).

### 3. Relax training environment (inside the container, one-time, idempotent)

`sglang_cuda13.sqsh` (the shared `infra01` container image used here) has Python 3.12 + torch +
sglang, but **no Megatron-LM, transformer_engine, or apex** -- building those from source is a
multi-hour compile this phase does not need, since `relax/backends/megatron/` does not hard-import
`transformer_engine` or `apex` directly. `run_mobilegym_e2e.sh` passes `--transformer-impl local` to
route around that dependency entirely (Megatron-Core's own well-supported non-TE code path).
`scripts/setup_relax_env.sh` (invoked automatically by `submit_mobilegym_e2e.sh`) creates a
`--system-site-packages` venv on top of the container's torch/sglang, installs Relax's
`requirements.txt` + `pip install -e .`, and clones Megatron-LM -- once, on `/iopsstor`, gated by a
marker file so later submissions skip it.

It also clones and `pip install -e`s `redai-infra/TransferQueue`: `relax/core/controller.py` imports
`transfer_queue` unconditionally, but it is not on PyPI, not vendored in this checkout (despite
`AGENTS.md` listing a `transfer_queue/` directory), and not in `requirements.txt` -- CI gets away
without it via a stub module (`.github/workflows/ci.yml`), which isn't an option for a real training
run. Use `redai-infra/TransferQueue` specifically, not the unrelated `Ascend/TransferQueue` also cited
in the top-level README's references section -- `redai-infra` is Relax's own GitHub org (`setup.py`),
and that fork's feature branches (`feat/rednote-ai/...`) track this codebase.

### 4. Container definition (EDF)

```toml
# mobilegym_e2e/edf/relax_sglang.toml
image = "/capstor/store/cscs/<org>/<team>/container-images/sglang_cuda13.sqsh"
mounts = ["/capstor", "/iopsstor", "/users"]
writable = true
workdir = "/workspace"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

### 5. Training data

```bash
mobilegym_e2e/venv/bin/python examples/mobilegym_agentic/scripts/build_tasks_jsonl.py \
  --mobilegym-repo mobilegym_e2e/mobilegym --split train --repeat 4 \
  --output mobilegym_e2e/data/mobilegym_train.jsonl
```

Reads MobileGym's own `bench_env/splits/train.txt` (160 task ids). Agentic mode needs no instruction
text or tool schema in the dataset -- each row is a placeholder message plus `metadata.task_id`
(mini_swe_agent's dummy-data pattern); MobileGym renders the actual instruction from the task's own
templates and owns tool/action parsing entirely.

## Launching

```bash
# Run from the login node that has the nginx gateway running (step 1 above) --
# MOBILEGYM_ENV_URL captures *this* node's hostname, cluster-routable from
# compute nodes (localhost is not -- see step 1's note).

# L1: single node, debug-rollout-only smoke (debug partition, 1:30 limit)
sbatch --time=00:30:00 -p debug --nodes=1 \
  --export=ALL,NUM_ROLLOUT=1,REASONING_TRIGGER=terminal_once,DEBUG_ROLLOUT_ONLY=1,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
  examples/mobilegym_agentic/submit_mobilegym_e2e.sh

# L2 / L3: full 16 GPUs (normal partition, 12h limit -- keep the ask well under that)
sbatch --time=02:00:00 -p normal \
  --export=ALL,REASONING_TRIGGER=terminal_once,NUM_ROLLOUT=3,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
  examples/mobilegym_agentic/submit_mobilegym_e2e.sh
sbatch --time=02:00:00 -p normal \
  --export=ALL,REASONING_TRIGGER=per_turn,NUM_ROLLOUT=3,MOBILEGYM_ENV_URL=https://$(hostname):4180 \
  examples/mobilegym_agentic/submit_mobilegym_e2e.sh
```

`REASONING_TRIGGER` selects `judge_services_e2e_terminal_once.json` vs `judge_services_e2e_per_turn.json`
(differ only in `reasoning_trigger`; same invariant hash -- verified). `answer_accuracy` is
terminal-only in both, per the dual-judge design: only `multi_turn_reasoning` has a per-turn mode.

GPU layout (16 GPUs): `actor` 4 (Qwen3-VL-4B, TP=4) / `rollout` 8 (8 SGLang engines x 1 GPU) /
`judge_accuracy` 2 (Qwen3-8B) / `judge_multiturn_vlm` 2 (Qwen2.5-VL-7B) / `advantages` 0 (CPU).
`rollout_batch_size(8) x n_samples_per_prompt(8) == global_batch_size(64)` auto-enables
`true_on_policy_mode` (skips `actor_fwd`); `--kl-loss-coef 0.00` means `reference` is not required
either. Breaking either equality without adding the corresponding role back to `--resource` makes
`advantages` poll TransferQueue for a field nobody produces -- a silent hang, not an error.

## Known gaps / next steps

- `submit_mobilegym_e2e.sh` bootstraps the Relax venv and resolves `MASTER_ADDR` correctly for the
  Slurm allocation model, but has not yet been exercised against a real allocation -- L1 is that
  first real test.
- Screenshot resolution sent to the model is whatever MobileGym's own `bench_env` uses internally;
  `judge_services_e2e_*.json` caps the VLM judge at `max_pixels_per_item: 700000` /
  `max_media_items: 12` on the judge side, but if raw episodes routinely exceed
  `--rollout-max-context-len 32768` at `--max-steps 8`, lower `MOBILEGYM_MAX_STEPS` (env var to
  `run_mobilegym_e2e.sh` via `--agent-env`) before scaling up rollout volume.
