# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import socket
import threading
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from time import perf_counter_ns, time
from typing import Optional

from relax.utils.logging_utils import get_logger

from .misc import SingletonMeta


__all__ = ["Timer", "timer", "span_timer", "timeline_step", "TimelineEvent"]

logger = get_logger(__name__)
_timeline_step: ContextVar[Optional[int]] = ContextVar("relax_timeline_step", default=None)
_clock_host = socket.gethostname()


def _is_dist_rank0() -> bool:
    """Check if torch.distributed is initialized and current rank is 0.

    Returns False if torch is not available or distributed is not initialized.
    """
    try:
        import torch.distributed

        return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    except ImportError:
        return False


def get_default_pid() -> int:
    """Get default process ID for trace events."""
    return os.getpid()


def get_default_tid() -> int:
    """Get default thread ID for trace events."""
    return threading.get_ident()


def get_call_stack() -> str:
    """Get current call stack as a string."""
    return "".join(traceback.format_stack())


class TimelineEvent:
    """Represents a complete Timeline Trace event record.

    Corresponds to Chrome Trace Event format with ph='X' (complete event).
    """

    def __init__(
        self,
        name: str,
        start_ts: float,
        end_ts: float,
        pid: int = None,
        tid: int = None,
        step: int = None,
        call_stack: str = None,
        duration_s: float = None,
        attributes: dict | None = None,
    ):
        self.name = name
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.pid = pid if pid is not None else get_default_pid()
        self.tid = tid if tid is not None else get_default_tid()
        self.step = step
        self.call_stack = call_stack
        self.duration_s = duration_s
        self.attributes = deepcopy(attributes or {})

    def to_trace_event(self) -> dict:
        """Convert to Chrome Trace Event format (ph='X' complete event)."""
        event = {
            "name": self.name,
            "ph": "X",
            "ts": int(self.start_ts * 1e6),  # Convert to microseconds
            "dur": int(
                (self.duration_s if self.duration_s is not None else self.end_ts - self.start_ts) * 1e6
            ),  # Duration in microseconds
            "pid": self.pid,
            "tid": self.tid,
        }
        args = {**self.attributes, "clock_host": _clock_host}
        if self.step is not None:
            args["step"] = self.step
        if self.call_stack:
            args["call_stack"] = self.call_stack
        event["args"] = args
        return event

    def __repr__(self):
        return f"TimelineEvent(name={self.name}, duration={self.end_ts - self.start_ts:.3f}s, step={self.step})"


class Timer(metaclass=SingletonMeta):
    def __init__(self):
        self._lock = threading.RLock()
        self.timers = {}
        self.start_time = {}
        self.start_info = {}  # Store start info (timestamp, call_stack) for each timer
        # Store complete TimelineEvent records
        self.records: list[TimelineEvent] = []

    def start(self, name, **kwargs):
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()
        if _is_dist_rank0():
            logger.debug(f"Timer {name} start")

        # Store call stack for trace context
        self.start_info[name] = {
            "timestamp": self.start_time[name],
            "call_stack": get_call_stack(),
            "step": _timeline_step.get(),
        }

    def end(self, name, keep: bool = True, **kwargs):
        """
        - keep: wheather to keep this metric and report to WANDB. if False, only record event without reporting.
        """
        assert name in self.start_time, f"Timer {name} not started."
        elapsed_time = time() - self.start_time[name]
        self.add(name, elapsed_time)

        if _is_dist_rank0():
            logger.debug(f"Timer {name} end (elapsed: {elapsed_time:.1f}s)")

        # Create TimelineEvent record
        start_info = self.start_info.get(name, {})
        start_ts = start_info.get("timestamp", self.start_time[name])
        end_ts = time()
        call_stack = start_info.get("call_stack", "")

        record = TimelineEvent(
            name=name,
            start_ts=start_ts,
            end_ts=end_ts,
            pid=get_default_pid(),
            tid=get_default_tid(),
            step=start_info.get("step"),
            call_stack=call_stack,
        )
        with self._lock:
            self.records.append(record)

        del self.start_time[name]
        if name in self.start_info:
            del self.start_info[name]

        if not keep:
            del self.timers[name]

        return elapsed_time

    def reset(self, name=None):
        with self._lock:
            if name is None:
                self.timers = {}
            elif name in self.timers:
                del self.timers[name]

    def add(self, name, elapsed_time):
        with self._lock:
            self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def add_complete_span(
        self,
        name: str,
        *,
        start_wall_s: float,
        end_wall_s: float,
        duration_s: float,
        attributes: dict | None = None,
    ) -> None:
        """Record a monotonic duration with wall-clock placement for a
        timeline."""
        with self._lock:
            self.timers[name] = self.timers.get(name, 0) + duration_s
            self.records.append(
                TimelineEvent(
                    name=name,
                    start_ts=start_wall_s,
                    end_ts=end_wall_s,
                    duration_s=duration_s,
                    step=_timeline_step.get(),
                    attributes=attributes,
                )
            )

    def log_dict(self):
        """Return timer name -> elapsed time dict (original behavior)."""
        with self._lock:
            return self.timers.copy()

    def log_record_and_clear(
        self,
        step: int,
        *,
        include_critical_path: bool = True,
        critical_path_only: bool = False,
    ):
        """Return records and clear them atomically.

        This is the recommended method for retrieving timeline events to avoid
        potential duplication if log() is called multiple times.
        """
        with self._lock:
            if critical_path_only:
                records = [record for record in self.records if record.name.startswith("critical_path.")]
                self.records = [record for record in self.records if not record.name.startswith("critical_path.")]
            elif include_critical_path:
                records = self.records.copy()
                self.records.clear()
            else:
                records = [record for record in self.records if not record.name.startswith("critical_path.")]
                self.records = [record for record in self.records if record.name.startswith("critical_path.")]
            for r in records:
                if r.step is None:
                    r.step = step
        return records

    @contextmanager
    def context(self, name, **kwargs):
        self.start(name, **kwargs)
        try:
            yield
        finally:
            self.end(name, **kwargs)


def timer(name_or_func, **kwargs):
    """Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(name, **kwargs)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def span_timer(name: str, *, attributes: dict | None = None):
    """Concurrency-safe phase timer using monotonic duration and wall
    placement.

    Unlike :func:`timer`, this helper keeps its start state on the stack rather
    than in the process-global name map, so repeated/nested async-style phase
    names can coexist without colliding.
    """
    start_wall_s = time()
    started_ns = perf_counter_ns()
    try:
        yield
    finally:
        ended_ns = perf_counter_ns()
        Timer().add_complete_span(
            name,
            start_wall_s=start_wall_s,
            end_wall_s=time(),
            duration_s=(ended_ns - started_ns) / 1e9,
            attributes=attributes,
        )


@contextmanager
def timeline_step(step: int):
    """Bind an immutable step identifier to all spans created in this
    context."""
    token = _timeline_step.set(step)
    try:
        yield
    finally:
        _timeline_step.reset(token)


@contextmanager
def inverse_timer(name):
    Timer().end(name)
    try:
        yield
    finally:
        Timer().start(name)


def with_defer(deferred_func):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                deferred_func()

        return wrapper

    return decorator
