# OSWorld x Relax rollout-only adapter

This adapter connects the ARM64 OSWorld provider prepared under
`/iopsstor/scratch/cscs/${USER}/Multimodality-RL/osworldv2` to Relax's managed
agent command interface. It is intentionally limited to one broker lease and
one-node rollout generation; it does not claim multi-node capacity or RL
training readiness.

Build the leakage-resistant policy data with:

```bash
/usr/bin/python3.11 examples/osworld_agentic/scripts/build_dataset.py \
  --osworld-repo /iopsstor/scratch/cscs/${USER}/Multimodality-RL/osworldv2/external/OSWorld \
  --output-dir /path/to/data/osworld
```

The broker requires two trusted argv JSON commands:

- `OSWORLD_BROKER_START_COMMAND_JSON` starts one prepared VM and prints `{"server_url":"http://..."}`.
- `OSWORLD_EVALUATE_COMMAND_JSON` evaluates the trusted task config and prints `{"score":0..1}`.

Policy actions are validated before rendering to pyautogui. The policy cannot
provide either lifecycle or evaluator commands. Query/evaluator failures must
be treated as infrastructure failures, not reward zero, by the trusted
evaluator command.
