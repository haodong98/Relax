#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only MobileGym concurrency gate for the dual-judge experiment.

This deliberately uses a tiny local OpenAI-compatible policy which completes
after the first observation.  It measures browser/env concurrency and result
isolation without consuming rollout or judge GPUs.  It is not a quality or
reward experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MOCK_CONTENT = '<think>G1 CPU mock policy.</think><answer>{"action":"COMPLETE","return":"G1 mock terminal"}</answer>'


class _MockPolicyHandler(BaseHTTPRequestHandler):
    """Small enough OpenAI Chat Completions subset for GenericAgentV2."""

    request_count = 0
    request_lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return
        with self.request_lock:
            type(self).request_count += 1
        payload = {
            "id": "g1-cpu-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "g1-cpu-mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": MOCK_CONTENT},
                    "finish_reason": "stop",
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the benchmark log readable; requests are counted in summary."""


class MockPolicyServer:
    def __enter__(self) -> "MockPolicyServer":
        _MockPolicyHandler.request_count = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockPolicyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def request_count(self) -> int:
        return _MockPolicyHandler.request_count

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _children_by_parent() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            fields = (proc_path / "status").read_text(encoding="utf-8").splitlines()
            ppid = int(next(line.split()[1] for line in fields if line.startswith("PPid:")))
        except (FileNotFoundError, IndexError, StopIteration, ValueError):
            continue
        children.setdefault(ppid, []).append(int(proc_path.name))
    return children


def _process_descendants(root_pid: int) -> set[int]:
    children = _children_by_parent()
    descendants = {root_pid}
    pending = [root_pid]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def _proc_snapshot(pids: set[int]) -> tuple[int, int, int]:
    """Return aggregate RSS KiB, CPU jiffies, and Chromium-process count."""

    rss_kib = 0
    cpu_jiffies = 0
    chromium = 0
    for pid in pids:
        proc_path = Path("/proc") / str(pid)
        try:
            status = (proc_path / "status").read_text(encoding="utf-8")
            rss_line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
            rss_kib += int(rss_line.split()[1])
            stat = (proc_path / "stat").read_text(encoding="utf-8").split()
            cpu_jiffies += int(stat[13]) + int(stat[14])
            comm = (proc_path / "comm").read_text(encoding="utf-8").strip().lower()
            command = (proc_path / "cmdline").read_text(encoding="utf-8", errors="replace").lower()
            chromium += int("chrom" in comm or "chrome" in command or "chromium" in command)
        except (FileNotFoundError, IndexError, StopIteration, ValueError):
            continue
    return rss_kib, cpu_jiffies, chromium


class ProcessMonitor:
    def __init__(self, root_pid: int, interval_s: float = 0.5) -> None:
        self.root_pid = root_pid
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[dict[str, float | int]] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if len(self.samples) < 2:
            return {
                "sample_count": len(self.samples),
                "max_rss_mib": None,
                "max_cpu_percent": None,
                "max_pids": None,
                "max_chromium_pids": None,
            }
        max_cpu = max(sample["cpu_percent"] for sample in self.samples[1:])
        return {
            "sample_count": len(self.samples),
            "max_rss_mib": round(max(sample["rss_kib"] for sample in self.samples) / 1024, 2),
            "max_cpu_percent": round(float(max_cpu), 2),
            "max_pids": max(sample["pid_count"] for sample in self.samples),
            "max_chromium_pids": max(sample["chromium_pid_count"] for sample in self.samples),
        }

    def _run(self) -> None:
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        previous_time = time.monotonic()
        previous_cpu = 0
        while not self.stop_event.is_set():
            pids = _process_descendants(self.root_pid)
            rss_kib, cpu_jiffies, chromium = _proc_snapshot(pids)
            now = time.monotonic()
            elapsed = now - previous_time
            cpu_percent = (
                0.0
                if previous_cpu == 0 or elapsed <= 0
                else (cpu_jiffies - previous_cpu) / clock_ticks / elapsed * 100
            )
            self.samples.append(
                {
                    "at_s": round(now, 6),
                    "rss_kib": rss_kib,
                    "cpu_percent": cpu_percent,
                    "pid_count": len(pids),
                    "chromium_pid_count": chromium,
                }
            )
            previous_time = now
            previous_cpu = cpu_jiffies
            self.stop_event.wait(self.interval_s)


def _result_rows(case_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(case_dir.rglob("results.jsonl")):
        # MobileGym's MultiProcessRunner mirrors child rows into a top-level
        # file.  Count its top-level file only, while retaining the explicit
        # shard directories that this harness itself owns.
        if "shards" in path.relative_to(case_dir).parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _episode_signature(row: dict[str, Any]) -> dict[str, Any]:
    execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    return {
        "id": row.get("id"),
        "task_name": row.get("task_name"),
        "instruction": row.get("instruction"),
        "params": row.get("params"),
        "sample_seed": row.get("sample_seed"),
        "goal_checks": row.get("goal_checks", judge.get("goal_checks", judge.get("checks", judge.get("issues")))),
        "stop_reason": execution.get("stop_reason"),
        "agent_message": execution.get("agent_message"),
        "agent_answer": execution.get("agent_answer"),
        "steps": execution.get("steps"),
        "error": execution.get("error"),
    }


def _row_is_valid(row: dict[str, Any], task_id: str) -> bool:
    execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
    return (
        row.get("id") == task_id
        and execution.get("stop_reason") == "COMPLETE"
        and not execution.get("error")
        and isinstance(execution.get("agent_message"), str)
    )


def _browser_crash_markers(case_dir: Path) -> list[str]:
    """Return unambiguous browser-crash evidence, excluding normal teardown."""

    markers = ("PAGE CRASHED", "SIGSEGV", "SIGABRT", "SIGTRAP", "renderer process crashed")
    hits: list[str] = []
    for log_path in case_dir.rglob("*.log"):
        try:
            for line_number, line in enumerate(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if any(marker.lower() in line.lower() for marker in markers):
                    hits.append(f"{log_path.relative_to(case_dir)}:{line_number}:{line[:300]}")
        except OSError:
            continue
    return hits


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate only the process group created for this benchmark
    invocation."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def _run_case(args: argparse.Namespace, policy_url: str, concurrency: int, sequence: int) -> dict[str, Any]:
    case_name = f"{sequence:02d}-c{concurrency}"
    case_dir = args.output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    if args.topology == "per_episode":
        # Relax's agent wrapper invokes bench_env.run once per rollout session;
        # this is the production-equivalent topology, not MobileGym's pooled
        # --parallel convenience mode.
        shard_sizes = [1] * concurrency
    else:
        shard_sizes = [
            min(args.pages_per_browser, concurrency - offset)
            for offset in range(0, concurrency, args.pages_per_browser)
        ]
    shards = len(shard_sizes)
    environment = dict(os.environ)
    environment["MOBILEGYM_HISTORY_IMAGES"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    existing_cores = {path.resolve() for path in args.mobilegym_repo.glob("core*") if path.is_file()}
    started_at = time.time()
    started_mono = time.monotonic()
    commands: list[list[str]] = []
    log_files: list[Any] = []
    processes: list[subprocess.Popen[str]] = []
    return_codes: list[int] = []
    monitor = ProcessMonitor(os.getpid(), interval_s=args.monitor_interval_s)
    monitor.start()
    try:
        for shard_index, shard_size in enumerate(shard_sizes):
            shard_dir = case_dir / f"shard-{shard_index:02d}"
            shard_dir.mkdir()
            command = [
                str(args.mobilegym_python),
                "-m",
                "bench_env.run",
                "--task-id",
                args.task_id,
                "--repeat-n",
                str(shard_size),
                "--sample-seed",
                str(args.sample_seed),
                "--model-base-url",
                policy_url,
                "--model-api-key",
                "g1-cpu-mock",
                "--model-name",
                "g1-cpu-mock",
                "--agent",
                "generic_v2",
                "--env-url",
                args.env_url,
                "--headless",
                "--no-stream",
                "--max-steps",
                str(args.max_steps),
                "--parallel",
                str(shard_size),
                "--processes",
                "1",
                "--browsers",
                "1",
                "--isolation",
                "pages",
                "--runs-dir",
                str(shard_dir),
                "--quiet",
            ]
            log_file = (shard_dir / "bench_env.log").open("w", encoding="utf-8")
            log_files.append(log_file)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=args.mobilegym_repo,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
            commands.append(command)
        deadline = time.monotonic() + args.timeout_s
        timed_out = False
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                timed_out = True
                for process in processes:
                    if process.poll() is None:
                        _terminate_process_group(process)
                break
            time.sleep(0.2)
        return_codes = [process.wait() for process in processes]
    finally:
        for log_file in log_files:
            log_file.close()
        resources = monitor.stop()
    return_code = 0 if all(code == 0 for code in return_codes) else next(code for code in return_codes if code != 0)
    rows = _result_rows(case_dir)
    valid_rows = sum(_row_is_valid(row, args.task_id) for row in rows)
    error_rows = [
        (row.get("execution") or {}).get("error")
        for row in rows
        if isinstance(row.get("execution"), dict) and (row["execution"].get("error"))
    ]
    new_core_files = sorted(
        str(path)
        for path in {path.resolve() for path in args.mobilegym_repo.glob("core*") if path.is_file()} - existing_cores
    )
    browser_crash_markers = _browser_crash_markers(case_dir)
    result = {
        "case": case_name,
        "concurrency": concurrency,
        "topology": args.topology,
        "pages_per_browser": 1 if args.topology == "per_episode" else args.pages_per_browser,
        "processes": shards,
        "browsers": shards,
        "isolation": "pages",
        "sample_seed": args.sample_seed,
        "shard_commands": commands,
        "started_at": started_at,
        "elapsed_s": round(time.monotonic() - started_mono, 3),
        "return_code": return_code,
        "timed_out": timed_out,
        "result_rows": len(rows),
        "valid_rows": valid_rows,
        "error_rows": error_rows,
        "browser_crash_markers": browser_crash_markers,
        "new_core_files": new_core_files,
        "resources": resources,
        "signatures": [_episode_signature(row) for row in rows],
        "log_paths": [str(case_dir / f"shard-{index:02d}" / "bench_env.log") for index in range(shards)],
        "shard_return_codes": return_codes,
    }
    (case_dir / "g1_case_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _validate_args(args: argparse.Namespace) -> None:
    for path, description in (
        (args.mobilegym_repo, "--mobilegym-repo"),
        (args.mobilegym_python, "--mobilegym-python"),
    ):
        if not path.exists():
            raise SystemExit(f"{description} does not exist: {path}")
    if args.pages_per_browser < 1:
        raise SystemExit("--pages-per-browser must be at least one")
    if any(value < 1 for value in args.concurrency):
        raise SystemExit("--concurrency values must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobilegym-repo", type=Path, required=True)
    parser.add_argument("--mobilegym-python", type=Path, required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", default="alipay.CheckBalance")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, action="append", default=[])
    parser.add_argument(
        "--topology",
        choices=("per_episode", "pooled"),
        default="per_episode",
        help="per_episode matches Relax's one-bench_env-run-per-session wrapper; pooled is a MobileGym-only diagnostic.",
    )
    parser.add_argument("--pages-per-browser", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--monitor-interval-s", type=float, default=0.5)
    args = parser.parse_args()
    if not args.concurrency:
        args.concurrency = [1, 8, 32, 64]
    return args


def main() -> int:
    args = parse_args()
    _validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    with MockPolicyServer() as mock_policy:
        for sequence, concurrency in enumerate(args.concurrency, start=1):
            cases.append(_run_case(args, mock_policy.base_url, concurrency, sequence))
        mock_request_count = mock_policy.request_count
    reproducibility: dict[str, Any] = {"checked": False, "equal": None, "reason": "requires two c=1 cases"}
    c1_cases = [case for case in cases if case["concurrency"] == 1]
    if len(c1_cases) >= 2:
        reproducibility = {
            "checked": True,
            "equal": c1_cases[0]["signatures"] == c1_cases[1]["signatures"],
            "reason": "compares recorded task/evidence fields; MobileGym does not export a separate initial-state hash",
        }
    passed_cases = [
        case
        for case in cases
        if case["return_code"] == 0
        and not case["timed_out"]
        and case["result_rows"] == case["concurrency"]
        and case["valid_rows"] == case["concurrency"]
        and not case["browser_crash_markers"]
        and not case["new_core_files"]
    ]
    summary = {
        "schema_version": "relax.mobilegym.g1_cpu_sweep.v1",
        "hostname": socket.gethostname(),
        "task_id": args.task_id,
        "sample_seed": args.sample_seed,
        "mock_policy_requests": mock_request_count,
        "cases": cases,
        "reproducibility": reproducibility,
        "max_passing_parallelism": max((case["concurrency"] for case in passed_cases), default=0),
        "all_cases_passed": len(passed_cases) == len(cases),
        "limitations": [
            "This is a mock-policy browser/env gate, not a policy or judge throughput measurement.",
            f"Topology: {args.topology}.",
            "A task's deterministic environment outcome is recorded; MobileGym currently does not export a separate initial-state hash.",
            "The result parser treats terminal COMPLETE with a committed agent message and no episode error as valid, independent of task success.",
        ],
    }
    (args.output_root / "g1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if summary["all_cases_passed"] and reproducibility.get("equal") is not False else 1


if __name__ == "__main__":
    sys.exit(main())
