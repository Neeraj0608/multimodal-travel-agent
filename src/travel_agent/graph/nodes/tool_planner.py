"""Tool selection.

Schemas go to the provider as plain JSON and the tool_calls payload is kept
verbatim in state, so the UI can show what the model actually asked for rather
than a framework's rendering of it.

dispatch_tools turns that payload into one Send per call.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Send

from ...llm.factory import get_client
from ...state import AgentState
from ...tools import FETCH_TOOLS, tool_schemas
from ..prompts import TOOL_SYSTEM
from ..tracing import make_event


def _wanted_tools(plan: dict[str, Any]) -> list[str]:
    wanted: list[str] = []
    if plan.get("needs_weather", True):
        wanted.append("get_weather_forecast")
    if plan.get("needs_images", True):
        wanted.append("search_city_images")
    return [name for name in wanted if name in FETCH_TOOLS]


def _synthetic_calls(names: list[str], plan: dict[str, Any], city: str) -> list[dict[str, Any]]:
    """Deterministic fallback when the model declines to emit tool calls."""
    calls: list[dict[str, Any]] = []
    for name in names:
        if name == "get_weather_forecast":
            args: dict[str, Any] = {
                "city": city,
                "days": int(plan.get("forecast_days", 7) or 7),
                "start_offset_days": int(plan.get("start_offset_days", 0) or 0),
            }
        else:
            args = {"city": city, "count": 4}
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": name, "args": args})
    return calls


async def propose_tool_calls(state: AgentState) -> dict[str, Any]:
    started = time.time()
    turn = int(state.get("turn", 0))
    plan = state.get("plan", {})
    city = state.get("city", "")
    wanted = _wanted_tools(plan)

    if not wanted:
        return {
            "proposed_tool_calls": [],
            "traces": [
                make_event(
                    "tool_planner", turn, started, detail="nothing to fetch this turn"
                )
            ],
        }

    client = get_client()
    system = SystemMessage(
        content=TOOL_SYSTEM.format(
            city=city,
            offset=plan.get("start_offset_days", 0),
            days=plan.get("forecast_days", 7),
            needs=", ".join(wanted),
        )
    )
    history = [m for m in state.get("messages", []) if m.type in {"human", "system"}][-4:]

    raw_payload: dict[str, Any] = {}
    note = ""
    try:
        result = await client.achat(
            [system, *history], tools=tool_schemas(tuple(wanted)), tool_choice="required"
        )
        calls = [tc.to_langchain() for tc in result.tool_calls]
        raw_payload = _extract_raw_tool_calls(result.raw)
        note = f"model requested {len(calls)} tool call(s)"
    except Exception as exc:  # noqa: BLE001 - a provider outage must not end the turn
        calls = []
        note = f"tool planning failed ({type(exc).__name__}), using deterministic calls"

    calls = _repair(calls, wanted, plan, city)
    if not raw_payload:
        raw_payload = {"synthesised": True, "tool_calls": calls}

    ai_message = AIMessage(content="", tool_calls=calls)
    return {
        "proposed_tool_calls": calls,
        "raw_tool_payload": json.dumps(raw_payload, indent=2, default=str)[:6000],
        "messages": [ai_message],
        "traces": [
            make_event(
                "tool_planner",
                turn,
                started,
                label="propose tool calls",
                detail=f"{note}: {', '.join(c['name'] for c in calls) or 'none'}",
            )
        ],
    }


def _extract_raw_tool_calls(raw: dict[str, Any]) -> dict[str, Any]:
    """Pull just the tool_calls slice out of the provider payload for display."""
    try:
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if message.get("tool_calls"):
            return {"provider": "openai-compatible", "tool_calls": message["tool_calls"]}
        content = raw.get("content")
        if isinstance(content, list):
            blocks = [b for b in content if b.get("type") == "tool_use"]
            if blocks:
                return {"provider": "anthropic", "tool_use": blocks}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _repair(
    calls: list[dict[str, Any]], wanted: list[str], plan: dict[str, Any], city: str
) -> list[dict[str, Any]]:
    """Keep the model honest without silently discarding its intent.

    Models drop a tool, repeat one, or resolve the city differently from the
    planner. We fill gaps and align the city, but the call list still originates
    with the model.
    """
    seen: set[str] = set()
    repaired: list[dict[str, Any]] = []
    for call in calls:
        name = call.get("name", "")
        if name not in wanted or name in seen:
            continue
        seen.add(name)
        args = dict(call.get("args") or {})
        if city:
            args["city"] = city
        if name == "get_weather_forecast":
            args.setdefault("days", int(plan.get("forecast_days", 7) or 7))
            if not args.get("start_offset_days"):
                args["start_offset_days"] = int(plan.get("start_offset_days", 0) or 0)
        repaired.append({**call, "args": args, "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}"})

    missing = [name for name in wanted if name not in seen]
    repaired.extend(_synthetic_calls(missing, plan, city))
    return repaired


def dispatch_tools(state: AgentState) -> list[Send] | str:
    """One concurrent worker per tool call.

    A list of Send objects puts that many copies of execute_tool in the same
    superstep. The tool functions await real I/O, so they overlap instead of
    queueing behind each other.
    """
    calls = state.get("proposed_tool_calls", [])
    if not calls:
        return "composer"
    turn = int(state.get("turn", 0))
    return [
        Send("execute_tool", {"turn": turn, "tool_call": call, "city": state.get("city", "")})
        for call in calls
    ]
