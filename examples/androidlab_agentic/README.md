# AndroidLab × Relax fully-async adapter

This adapter treats AndroidLab as an online environment, not a static labelled
dataset. It uses Relax managed commands for model calls and broker-owned ADB
devices for state, actions, and evaluation. No Relax Controller, Service,
launcher, CLI argument, MobileGym, or managed-session public contract is
changed.

## Dataset boundary

Build deterministic task manifests with:

```bash
/usr/bin/python3.11 examples/androidlab_agentic/scripts/build_dataset.py \
  --androidlab-repo /iopsstor/scratch/cscs/${USER}/Multimodality-RL/AndroidLab/Android-Lab \
  --output-dir /path/to/data/androidlab
```

The builder verifies the pinned inventory: 138 tasks in nine apps, 93
operation-like tasks and 45 `query_detect` tasks. `map_14` is normalized from
the upstream spelling `operations`. Policy rows contain only the task text and
opaque identity; package names, evaluator modules, ADB queries and answers are
kept in `trusted_registry.json`.

## Trust boundary

The policy may emit exactly one `<action>{...}</action>` JSON action. The
broker accepts only bounded tap, swipe, text, navigation, wait, task-app launch
and terminal actions, then renders them to fixed ADB argv arrays. It never
passes policy text to Python `exec`, a shell, or arbitrary ADB/package commands.

For operation tasks, native AndroidLab `complete` maps to the scalar reward.
For query tasks, the broker retains the reference answer and calls a required
local Judge endpoint (`ANDROIDLAB_QUERY_JUDGE_URL`). A missing or failing Judge
is infrastructure failure, not reward zero. Valid task failure remains `0.0`.

## Runtime boundary

`ANDROIDLAB_BROKER_START_COMMAND_JSON` is a trusted argv JSON array. It is
called once per fresh lease with `ANDROIDLAB_LEASE_ID` and
`ANDROIDLAB_LEASE_ROOT` set, must launch one immutable Android runtime, wait for
ADB readiness, and print exactly `{"serial":"<adb serial>"}` to stdout. An
optional `ANDROIDLAB_BROKER_STOP_COMMAND_JSON` receives the lease root and must
stop that runtime. These commands are deliberately outside policy/Ray scope.

The current Alps ARM64 Cuttlefish image is suitable only for the existing
Settings action-compatibility smoke. It lacks benchmark applications and seed
state for the complete task suite; no task becomes trainable until its image,
package, seed, ADB and native evaluator capability gates pass.

## Four-node smoke

`submit_androidlab_4node.sh` requests exactly four nodes with four GPUs each
for at most two hours (32 GPU-hours). It starts one capacity-one broker per
node outside Ray and runs the Qwen3-VL-8B operation-only recipe in fully-async
mode for exactly three rounds. Query training is intentionally excluded from
this first GPU smoke unless a separately provisioned local Judge endpoint is
available. Do not submit while the shared Alps nodes are occupied.
