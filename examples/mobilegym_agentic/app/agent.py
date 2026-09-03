# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MobileGym agent mainline.

Unlike the deepeyes/mini_swe_agent adapters, this one does not drive the
model<->env loop itself. MobileGym's own ``bench_env.run`` CLI already owns
that entire loop (screenshot capture, action parsing, multi-turn message
history, and the deterministic state-diff judge) -- it just needs to be
pointed at an OpenAI-compatible endpoint. Relax's per-session endpoint
(``RELAX_BASE_URL``, mapped to ``OPENAI_BASE_URL`` by run_agent_app.sh) is
exactly that, so every turn ``bench_env.run`` sends is transparently recorded
by Relax's session-forest chat service on the way through -- this file is a
thin subprocess wrapper plus a translator from ``results.jsonl`` to
``RELAX_OUTPUT_JSON``.

``bench_env.run`` runs in a separate, dedicated Python environment
(MOBILEGYM_PYTHON) from the one this script itself runs in, because it needs
Playwright + a real Chromium build and its own pinned deps, independent of
Relax's Megatron/SGLang training stack.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_session_input(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_session_output(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set (see examples/mobilegym_agentic/README.md)")
    return value


def _find_results_row(runs_dir: Path, *, task_id: str) -> dict[str, Any] | None:
    """A dedicated ``--runs-dir`` per session contains exactly one timestamped
    run subdirectory with one ``results.jsonl`` row (one task, one trial)."""
    candidates = sorted(runs_dir.glob("*/results.jsonl"))
    if not candidates:
        return None
    # Only one run should exist under a session-scoped runs_dir; if bench_env.run
    # was invoked more than once here (e.g. a retried session), take the newest.
    results_path = candidates[-1]
    lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    row = json.loads(lines[-1])
    if row.get("id") != task_id:
        return None
    return row


def _mobilegym_outcome_evidence(row: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    """Extract only observable terminal evidence, never environment
    verdicts."""
    execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    checks = row.get("goal_checks", judge.get("goal_checks", judge.get("checks", judge.get("issues", []))))
    goal_checks: list[dict[str, Any]] = []
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            goal_checks.append({key: check.get(key) for key in ("field", "expected", "actual") if key in check})
    return {
        "schema_version": "mobilegym.outcome.v1",
        "task_id": row.get("id", task_id),
        "task_name": row.get("task_name"),
        "execution": {
            "stop_reason": execution.get("stop_reason"),
            "agent_message": execution.get("agent_message"),
            "agent_answer": execution.get("agent_answer"),
            "steps": execution.get("steps"),
        },
        "goal_checks": goal_checks,
    }


def run_mobilegym_episode(task_id: str, *, session_id: str, sample_seed: int) -> dict[str, Any]:
    mobilegym_python = _required_env("MOBILEGYM_PYTHON")
    mobilegym_repo = _required_env("MOBILEGYM_REPO_DIR")
    env_url = _required_env("MOBILEGYM_ENV_URL")
    runs_root = Path(_required_env("MOBILEGYM_RUNS_ROOT"))
    agent_name = os.environ.get("MOBILEGYM_AGENT", "generic_v2")
    max_steps = int(os.environ.get("MOBILEGYM_MAX_STEPS", "8"))
    timeout_s = float(os.environ.get("MOBILEGYM_TIMEOUT_S", "1200"))

    # Session-scoped runs_dir: avoids any cross-process collision under
    # concurrent rollout sessions and gives an unambiguous glob target below,
    # instead of racing on bench_env's default shared/timestamped ``runs/``.
    # A fresh subdirectory makes a result row attributable to this subprocess
    # invocation even after a session-level retry.
    runs_dir = runs_root / session_id / f"invoke-{time.time_ns()}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        mobilegym_python,
        "-m",
        "bench_env.run",
        "--task-id",
        task_id,
        "--env-url",
        env_url,
        "--model-base-url",
        os.environ["OPENAI_BASE_URL"],
        "--model-api-key",
        os.environ.get("OPENAI_API_KEY", ""),
        "--model-name",
        "relax-policy",
        "--agent",
        agent_name,
        "--headless",
        "--no-stream",
        "--max-steps",
        str(max_steps),
        "--sample-seed",
        str(sample_seed),
        "--runs-dir",
        str(runs_dir),
    ]

    log_path = runs_dir / "bench_env_run.log"
    started_at = time.time()
    error: str | None = None
    return_code: int | None = None
    # The EDF SGLang image omits desktop/browser system libraries (GLib, X11,
    # GBM, NSS) that Playwright's headless Chromium needs. Scope
    # LD_LIBRARY_PATH to just this subprocess -- see
    # examples/mobilegym_agentic/test_scripts/mobilegym_rollout_node.sh's
    # BROWSER_LD_LIBRARY_PATH for why it must not leak into Relax/SGLang.
    subprocess_env = dict(os.environ)
    browser_ld_library_path = os.environ.get("BROWSER_LD_LIBRARY_PATH")
    if browser_ld_library_path:
        subprocess_env["LD_LIBRARY_PATH"] = browser_ld_library_path
    # Opt-in browser-process diagnostics: makes Playwright log the browser's own
    # stderr and its final ``<process did exit: exitCode=..., signal=...>`` line
    # into bench_env_run.log (stderr is already redirected there). That exit
    # signal is the only reliable way to tell a renderer/GPU crash apart from an
    # external SIGKILL when a page dies mid-episode.
    if os.environ.get("MOBILEGYM_PW_DEBUG"):
        subprocess_env["DEBUG"] = "pw:browser*"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=mobilegym_repo,
                env=subprocess_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            return_code = completed.returncode
            if return_code != 0:
                error = f"bench_env.run exited with return code {return_code}"
    except subprocess.TimeoutExpired:
        error = f"bench_env.run exceeded MOBILEGYM_TIMEOUT_S={timeout_s}s"
    except Exception as exc:  # noqa: BLE001 -- surfaced via metadata, never raised to the caller
        error = f"{type(exc).__name__}: {exc}"

    row = _find_results_row(runs_dir, task_id=task_id)
    elapsed_s = time.time() - started_at

    if row is None:
        return {
            "metadata": {
                "task_id": task_id,
                "sample_seed": sample_seed,
                "elapsed_s": elapsed_s,
                "error": error or "bench_env.run produced no results.jsonl row",
                "subprocess_return_code": return_code,
                "log_path": str(log_path),
            },
            "reward": None,
        }

    # NOTE: results.jsonl's actual field names differ from docs/REFERENCE.md's
    # EpisodeResult table in a few places (verified against a real run, not
    # trusted from the docs alone) -- `id` not `task_id`, `is_success` not
    # `success`, `steps`/`error` nested under `execution`, and `goal_success`
    # (a documented *property*, not a stored field) must be read from
    # `judge.success` per REFERENCE.md's own definition of that property.
    judge = row.get("judge") or {}
    execution = row.get("execution") or {}
    return {
        "metadata": {
            "task_id": row.get("id", task_id),
            "sample_seed": sample_seed,
            "task_name": row.get("task_name"),
            "suite": row.get("suite"),
            "steps": execution.get("steps"),
            "max_steps": row.get("max_steps"),
            "is_success": row.get("is_success"),
            "goal_success": judge.get("success"),
            "clean": judge.get("clean"),
            "false_complete": row.get("false_complete"),
            "overdue_termination": row.get("overdue_termination"),
            "episode_error": execution.get("error"),
            "elapsed_s": elapsed_s,
            "subprocess_return_code": return_code,
            "mobilegym_outcome_evidence": _mobilegym_outcome_evidence(row, task_id=task_id),
            # `progress` (fraction of check_goals passed, 0.0-1.0) is a dense
            # signal from MobileGym's own deterministic state-diff judge. It is
            # reported as METADATA rather than as the sample reward on purpose:
            # relax/agentic/pipeline/reward.py dispatches to the judges only
            # when ``sample.reward is None`` (see _sample_needs_reward, and its
            # use in _start_group_sample_rewards), so returning a reward here
            # silently bypasses the dual judges entirely -- they would never
            # score a single trajectory. Leaving reward unset makes the judges
            # the reward source, while this field stays available to compare
            # their scores against the environment's ground truth.
            "env_progress": float(row.get("progress", 0.0)),
        },
        "reward": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MobileGym agent episode.")
    parser.add_argument("--input-json", required=True, help="Path to a JSON file containing RELAX_INPUT_JSON.")
    parser.add_argument("--output-json", required=True, help="Path to write the session output JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_input = read_session_input(args.input_json)
    metadata = session_input.get("metadata") or {}
    task_id = metadata.get("task_id")
    sample_seed = metadata.get("sample_seed")
    if not task_id:
        write_session_output(
            args.output_json,
            {"metadata": {"error": "RELAX_INPUT_JSON.metadata.task_id is required"}, "reward": None},
        )
        return
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int) or sample_seed < 0 or sample_seed > 2**32 - 1:
        write_session_output(
            args.output_json,
            {
                "metadata": {"error": "RELAX_INPUT_JSON.metadata.sample_seed must be a uint32 integer"},
                "reward": None,
            },
        )
        return
    session_id = os.environ.get("RELAX_SESSION_ID") or f"session-{os.getpid()}"
    output = run_mobilegym_episode(task_id, session_id=session_id, sample_seed=sample_seed)
    write_session_output(args.output_json, output)


if __name__ == "__main__":
    sys.exit(main())
