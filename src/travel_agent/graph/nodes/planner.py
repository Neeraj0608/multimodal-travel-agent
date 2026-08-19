"""Works out what the current turn needs before anything gets fetched.

"Tell me about Tokyo" asks for everything. A follow-up like "what about next
week?" resolves the city from checkpointed state, clears needs_summary and
shifts the weather window, leaving only the weather tool to re-run.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import SystemMessage

from ...knowledge.seed_data import SEEDED_CITIES, normalise_city
from ...llm.factory import get_client
from ...schemas import TurnPlan
from ...state import AgentState
from ..prompts import PLANNER_SYSTEM
from ..tracing import make_event

PLAN_SCHEMA: dict[str, Any] = {
    "title": "TurnPlan",
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country_hint": {"type": "string"},
        "needs_summary": {"type": "boolean"},
        "needs_weather": {"type": "boolean"},
        "needs_images": {"type": "boolean"},
        "start_offset_days": {"type": "integer"},
        "forecast_days": {"type": "integer"},
        "is_followup": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["city", "needs_summary", "needs_weather", "needs_images"],
}


def _memory_block(state: AgentState) -> str:
    city = state.get("city") or "(none yet)"
    has_summary = bool(state.get("city_summary"))
    has_images = bool(state.get("image_urls"))
    return (
        f"- remembered city: {city}\n"
        f"- summary already produced: {has_summary}\n"
        f"- images already shown: {has_images}\n"
        f"- turns so far: {state.get('turn', 0)}"
    )


async def plan_turn(state: AgentState) -> dict[str, Any]:
    started = time.time()
    turn = int(state.get("turn", 0)) + 1
    client = get_client()

    messages = list(state.get("messages", []))
    system = SystemMessage(
        content=PLANNER_SYSTEM.format(
            memory=_memory_block(state), kb_cities=", ".join(SEEDED_CITIES)
        )
    )

    try:
        result = await client.achat(
            [system, *messages[-8:]], json_schema=PLAN_SCHEMA, json_object=True
        )
        plan = TurnPlan.model_validate(result.json_payload())
        detail = plan.rationale or "planned"
    except Exception as exc:  # noqa: BLE001 - never let planning kill the turn
        plan = _fallback_plan(state, messages)
        detail = f"planner fell back to heuristics: {type(exc).__name__}"

    # The model may forget the remembered city on a bare follow-up.
    remembered = state.get("city", "")
    if not plan.city and remembered:
        plan.city = remembered
        plan.is_followup = True
    if remembered and normalise_city(plan.city) == normalise_city(remembered):
        if state.get("city_summary") and plan.is_followup:
            plan.needs_summary = False
    else:
        # A different city always needs a fresh summary and gallery.
        plan.needs_summary = True
        plan.needs_images = True

    return {
        "turn": turn,
        "city": plan.city,
        "plan": plan.model_dump(),
        "traces": [
            make_event(
                "planner",
                turn,
                started,
                label="plan turn",
                detail=(
                    f"city={plan.city or 'unresolved'} summary={plan.needs_summary} "
                    f"weather={plan.needs_weather} images={plan.needs_images} "
                    f"offset={plan.start_offset_days}d :: {detail}"
                ),
            )
        ],
    }


def _fallback_plan(state: AgentState, messages: list[Any]) -> TurnPlan:
    """Heuristic plan used when the model is unavailable or returns garbage."""
    from ...llm.offline import OfflineClient

    result = OfflineClient().chat(messages, json_schema=PLAN_SCHEMA)
    try:
        return TurnPlan.model_validate(result.json_payload())
    except Exception:  # noqa: BLE001
        return TurnPlan(city=state.get("city", ""))


def route_after_plan(state: AgentState) -> str:
    """Conditional edge: clarify, reuse memory, or go to the knowledge base."""
    plan = state.get("plan", {})
    if not (plan.get("city") or "").strip():
        return "clarify"
    if not plan.get("needs_summary", True) and state.get("city_summary"):
        # Memory path: summary is already known, skip retrieval entirely.
        return "tool_planner"
    return "retrieve_knowledge"


async def clarify(state: AgentState) -> dict[str, Any]:
    """Terminal node for 'I could not work out which city you mean'."""
    started = time.time()
    turn = int(state.get("turn", 0))
    report = {
        "city": "",
        "city_summary": (
            "I could not work out which destination you meant. Tell me a city - "
            "for example \"Tell me about Kyoto\" - and I will pull together a "
            "summary, the forecast and a photo gallery."
        ),
        "highlights": [],
        "weather_forecast": [],
        "image_urls": [],
        "images": [],
        "knowledge_source": "memory",
        "sources": [],
        "warnings": ["No city could be resolved from this message."],
        "country": "",
        "best_time_to_visit": "",
    }
    return {
        "report": report,
        "traces": [make_event("clarify", turn, started, detail="no city resolved")],
    }
