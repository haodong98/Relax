# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TimelineTrace Adapter for Chrome Trace Event format."""

import json
import os
from typing import Any, Dict, List, Optional

from relax.utils.timer import TimelineEvent


class TimelineTraceAdapter:
    """Adapter for generating Chrome Timeline Trace files.

    This adapter receives TimelineEvent records and generates Chrome Trace
    Event format JSON files that can be visualized in chrome://tracing or
    Perfetto.
    """

    def __init__(self, dump_dir: str, max_dump: Optional[int] = None):
        """Initialize the TimelineTraceAdapter.

        Args:
            dump_dir: Directory to dump timeline trace files. If None or empty,
                     the adapter is disabled.
        """
        self.dump_dir = dump_dir
        self.enabled = bool(dump_dir and dump_dir.strip())
        self._all_events: List[Dict[str, Any]] = []
        self._dump_cnt = 0
        self._max_dump = max_dump
        self._dumped_steps: set[int] = set()
        self._step_part_counts: Dict[int, int] = {}

        if self.enabled:
            # Ensure the directory exists
            os.makedirs(dump_dir, exist_ok=True)

    def is_enabled(self) -> bool:
        """Check if the adapter is enabled."""
        return self.enabled

    def add_events(self, events: List[TimelineEvent]):
        """Add TimelineEvent records.

        Args:
            events: List of TimelineEvent objects to add.
        """
        if not self.enabled:
            return

        for event in events:
            trace_event = event.to_trace_event()
            self._all_events.append(trace_event)

    def add_event_dicts(self, event_dicts: List[Dict[str, Any]]):
        """Add already-serialized event dictionaries.

        Args:
            event_dicts: List of event dictionaries in TimelineEvent format.
        """
        if not self.enabled:
            return

        self._all_events.extend(event_dicts)

    def dump(self, step: int):
        """Dump newly collected events to an incremental JSON shard.

        The first shard is ``timeline_step_{step}.json``. Later reports for
        the same step use ``timeline_step_{step}_part_N.json``.

        Args:
            step: The current step number.
        """
        if not self.enabled:
            return

        if not self._all_events:
            return

        if self._max_dump is not None and step not in self._dumped_steps and self._dump_cnt >= self._max_dump:
            # Honor the quota as a storage bound. Retaining rejected events
            # would leak memory and could mislabel them if an older step is
            # reported again later.
            self._all_events.clear()
            return

        # Sort only newly received events. Keeping every historical event and
        # rewriting an ever-growing file on each update makes trace overhead
        # quadratic in the run length and contaminates the latency benchmark.
        sorted_events = sorted(self._all_events, key=lambda e: e.get("ts", 0))

        part = self._step_part_counts.get(step, 0)
        filename = f"timeline_step_{step}.json" if part == 0 else f"timeline_step_{step}_part_{part}.json"
        filepath = os.path.join(self.dump_dir, filename)

        with open(filepath, "w") as f:
            json.dump(sorted_events, f)

        if step not in self._dumped_steps:
            self._dumped_steps.add(step)
            self._dump_cnt += 1
        self._step_part_counts[step] = part + 1
        self._all_events.clear()

    def dump_all(self, step: int):
        """Alias for dump() for backward compatibility."""
        self.dump(step)

    def clear(self):
        """Clear all stored events without dumping."""
        self._all_events.clear()

    def get_event_count(self) -> int:
        """Get the number of events currently stored."""
        return len(self._all_events)
