"""Timing instrumentation.

Each node records a TraceEvent with wall-clock start and end, which the UI
plots on a shared timeline. Overlapping bars are the evidence that the fan-out
actually runs in parallel.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from ..state import TraceEvent


@contextmanager
def timed(node: str, turn: int, label: str = "") -> Iterator[dict[str, Any]]:
    """Collect a trace event; the body may set ``event['detail']``."""
    event: dict[str, Any] = {
        "turn": turn,
        "node": node,
        "label": label or node,
        "started_at": time.time(),
        "detail": "",
    }
    try:
        yield event
    finally:
        event["finished_at"] = time.time()
        event["duration_ms"] = round((event["finished_at"] - event["started_at"]) * 1000, 1)


def make_event(
    node: str, turn: int, started_at: float, *, label: str = "", detail: str = ""
) -> TraceEvent:
    finished = time.time()
    return {
        "turn": turn,
        "node": node,
        "label": label or node,
        "started_at": started_at,
        "finished_at": finished,
        "duration_ms": round((finished - started_at) * 1000, 1),
        "detail": detail,
    }


def parallelism_stats(events: list[dict[str, Any]]) -> dict[str, float]:
    """Compare summed tool time against the wall clock the fan-out actually took.

    ``speedup`` above 1.0 means work genuinely overlapped.
    """
    if not events:
        return {"sequential_ms": 0.0, "wall_ms": 0.0, "saved_ms": 0.0, "speedup": 1.0}

    sequential = sum(float(e.get("duration_ms", 0.0)) for e in events)
    start = min(float(e.get("started_at", 0.0)) for e in events)
    end = max(float(e.get("finished_at", 0.0)) for e in events)
    wall = max((end - start) * 1000.0, 0.001)
    return {
        "sequential_ms": round(sequential, 1),
        "wall_ms": round(wall, 1),
        "saved_ms": round(max(sequential - wall, 0.0), 1),
        "speedup": round(sequential / wall, 2),
    }
