"""Tool execution.

One tool call per invocation: resolve the function in the registry, coerce the
model's arguments against the tool's JSON schema, run it, and write back a
ToolMessage carrying the originating tool_call_id.

Errors come back as error ToolMessages rather than being raised. A dead
upstream API should cost one section of the report, not the whole turn.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ...state import ToolTask
from ...tools import get_tool, validate_args
from ..tracing import make_event

# Marker used by the UI's chaos toggles to simulate a provider outage.
FAILURE_MESSAGE = "Injected failure for demonstration (chaos toggle enabled)"


async def execute_tool(task: ToolTask, config: RunnableConfig) -> dict[str, Any]:
    # config must be annotated exactly RunnableConfig, with no default.
    # LangGraph matches the parameter on its annotation text, and under
    # `from __future__ import annotations` the `RunnableConfig | None = None`
    # form silently fails to match: the node then runs with config=None and
    # ignores everything passed through configurable.
    started = time.time()
    turn = int(task.get("turn", 0))
    call = task.get("tool_call", {}) or {}
    name = call.get("name", "")
    call_id = call.get("id", "") or f"call_{name}"
    raw_args = call.get("args", {}) or {}

    configurable = (config or {}).get("configurable", {}) or {}
    failing: list[str] = list(configurable.get("fail_tools", []) or [])

    tool = get_tool(name)
    if tool is None:
        return _failure(
            turn, call_id, name, raw_args, started, f"unknown tool '{name}'"
        )

    args, notes = validate_args(tool, raw_args)
    if any(note.startswith("missing required") for note in notes):
        return _failure(
            turn, call_id, name, args, started, f"invalid arguments: {'; '.join(notes)}"
        )

    if name in failing:
        return _failure(turn, call_id, name, args, started, FAILURE_MESSAGE)

    try:
        result = await tool.func(**args)
    except Exception as exc:  # noqa: BLE001 - surface, don't propagate
        return _failure(turn, call_id, name, args, started, f"{type(exc).__name__}: {exc}")

    finished = time.time()
    payload = {
        "turn": turn,
        "tool_call_id": call_id,
        "name": name,
        "args": args,
        "ok": True,
        "result": result,
        "started_at": started,
        "finished_at": finished,
        "duration_ms": round((finished - started) * 1000, 1),
    }
    detail = f"ok in {payload['duration_ms']:.0f}ms"
    if notes:
        detail += f" ({'; '.join(notes)})"

    return {
        "tool_results": [payload],
        "messages": [
            ToolMessage(
                content=json.dumps(_compact(result), default=str)[:8000],
                tool_call_id=call_id,
                name=name,
            )
        ],
        "traces": [
            make_event("execute_tool", turn, started, label=name, detail=detail)
        ],
    }


def _failure(
    turn: int,
    call_id: str,
    name: str,
    args: dict[str, Any],
    started: float,
    error: str,
) -> dict[str, Any]:
    finished = time.time()
    return {
        "tool_results": [
            {
                "turn": turn,
                "tool_call_id": call_id,
                "name": name,
                "args": args,
                "ok": False,
                "error": error,
                "started_at": started,
                "finished_at": finished,
                "duration_ms": round((finished - started) * 1000, 1),
            }
        ],
        "messages": [
            ToolMessage(
                content=json.dumps({"error": error}),
                tool_call_id=call_id,
                name=name,
                status="error",
            )
        ],
        "warnings": [f"{name} failed: {error}"],
        "traces": [
            make_event("execute_tool", turn, started, label=f"{name} (failed)", detail=error)
        ],
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    """Trim tool output before it enters the message history.

    The full result stays in ``tool_results`` for the UI; the message copy only
    needs to be good enough for the model to reason over.
    """
    compact = dict(result)
    if isinstance(compact.get("days"), list):
        compact["days"] = compact["days"][:7]
    if isinstance(compact.get("images"), list):
        compact["images"] = [
            {"caption": i.get("caption", "")} for i in compact["images"][:6]
        ]
    return compact
