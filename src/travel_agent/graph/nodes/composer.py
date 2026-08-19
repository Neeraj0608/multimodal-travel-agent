"""Builds the final TravelReport.

The split here matters: the model writes prose only. Every number and URL is
copied out of the tool results by this node, so a hallucinated temperature
cannot reach the chart and an invented URL cannot reach the gallery.

When the city has not changed, the prose comes from checkpointed state instead
of being regenerated.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import ValidationError

from ...llm.factory import get_client
from ...knowledge.seed_data import normalise_city
from ...schemas import TravelReport
from ...state import AgentState, results_for_turn
from ..prompts import COMPOSER_SYSTEM, COMPOSER_USER
from ..tracing import make_event

PROSE_SCHEMA: dict[str, Any] = {
    "title": "ReportProse",
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "city_summary": {"type": "string"},
        "highlights": {"type": "array", "items": {"type": "string"}},
        "best_time_to_visit": {"type": "string"},
    },
    "required": ["city", "city_summary"],
}


def _last_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _tool_payloads(state: AgentState, turn: int) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Split this turn's tool results into weather / images / failures."""
    weather: dict[str, Any] = {}
    images: dict[str, Any] = {}
    failures: list[str] = []

    for result in results_for_turn(state, turn):
        if not result.get("ok"):
            failures.append(f"{result.get('name')}: {result.get('error', 'failed')}")
            continue
        if result.get("name") == "get_weather_forecast":
            weather = result.get("result", {}) or {}
        elif result.get("name") == "search_city_images":
            images = result.get("result", {}) or {}
    return weather, images, failures


def _tool_summary(weather: dict[str, Any], images: dict[str, Any]) -> str:
    parts: list[str] = []
    days = weather.get("days") or []
    if days:
        highs = [d["temp_max_c"] for d in days]
        lows = [d["temp_min_c"] for d in days]
        parts.append(
            f"Forecast {days[0]['date']} to {days[-1]['date']}: highs "
            f"{min(highs):.0f}-{max(highs):.0f}C, lows {min(lows):.0f}-{max(lows):.0f}C, "
            f"conditions {', '.join(sorted({d['condition'] for d in days}))}."
        )
    if images.get("images"):
        parts.append(f"{len(images['images'])} photographs retrieved.")
    return "\n".join(parts) or "No tool data available this turn."


async def compose_report(state: AgentState) -> dict[str, Any]:
    started = time.time()
    turn = int(state.get("turn", 0))
    plan = state.get("plan", {})
    city = state.get("city", "")
    route = state.get("route", "vector_store")

    weather, images, failures = _tool_payloads(state, turn)
    reused: list[str] = []

    # Cached artefacts belong to whichever city produced them. Reusing them for
    # a different city would silently show Tokyo's forecast under "Rome" - so
    # every fallback below is gated on the cache still matching this city.
    cached_city = state.get("cached_for_city", "")
    same_city = bool(city) and normalise_city(cached_city) == normalise_city(city)

    # ---- weather: fresh if fetched, otherwise whatever we already had -------
    forecast = weather.get("days") or []
    if not forecast and same_city and state.get("weather_forecast"):
        forecast = state["weather_forecast"]
        reused.append("weather")

    # ---- images ------------------------------------------------------------
    image_items = images.get("images") or []
    if not image_items and same_city and state.get("image_urls"):
        image_items = [{"url": url, "caption": "", "credit": ""} for url in state["image_urls"]]
        reused.append("images")

    # ---- prose: generate, or reuse from memory -----------------------------
    needs_summary = (
        bool(plan.get("needs_summary", True)) or not state.get("city_summary") or not same_city
    )
    knowledge_source = route if route in {"vector_store", "web_search"} else "vector_store"
    warnings: list[str] = list(failures)
    detail_bits: list[str] = []

    if needs_summary:
        prose, prose_note = await _write_prose(state, city, route, weather, images)
        detail_bits.append(prose_note)
    else:
        prose = {
            "city": city,
            "country": state.get("country", ""),
            "city_summary": state.get("city_summary", ""),
            "highlights": state.get("highlights", []),
            "best_time_to_visit": state.get("best_time_to_visit", ""),
        }
        knowledge_source = "memory"
        reused.append("summary")
        detail_bits.append("summary reused from checkpointed state")

    if failures:
        warnings.append(
            "Some data could not be retrieved; the report below is partial."
        )

    report_data: dict[str, Any] = {
        "city": prose.get("city") or city,
        "country": prose.get("country") or state.get("country", "") or weather.get("country", ""),
        "city_summary": prose.get("city_summary", ""),
        "highlights": prose.get("highlights", []) or [],
        "best_time_to_visit": prose.get("best_time_to_visit", ""),
        "weather_forecast": forecast,
        "images": [
            {
                "url": item.get("url", ""),
                "caption": item.get("caption", ""),
                "credit": item.get("credit", ""),
            }
            for item in image_items
            if item.get("url")
        ],
        "image_urls": [item.get("url", "") for item in image_items if item.get("url")],
        "knowledge_source": knowledge_source,
        "sources": state.get("sources", []),
        "warnings": warnings,
    }

    try:
        report = TravelReport.model_validate(report_data)
    except ValidationError as exc:
        # Never hand the UI an object it cannot render: strip the offending
        # weather rows and try once more with a warning attached.
        report_data["weather_forecast"] = []
        report_data["warnings"] = warnings + [
            f"Forecast data failed validation and was dropped ({exc.error_count()} issue(s))."
        ]
        report = TravelReport.model_validate(report_data)

    if reused:
        detail_bits.append(f"reused from memory: {', '.join(sorted(set(reused)))}")

    return {
        "report": report.model_dump(),
        "cached_for_city": report.city,
        "city_summary": report.city_summary,
        "country": report.country,
        "highlights": report.highlights,
        "best_time_to_visit": report.best_time_to_visit,
        "weather_forecast": [point.model_dump() for point in report.weather_forecast],
        "image_urls": report.image_urls,
        "messages": [AIMessage(content=_chat_text(report))],
        "traces": [
            make_event(
                "composer",
                turn,
                started,
                label="compose report",
                detail="; ".join(b for b in detail_bits if b),
            )
        ],
    }


async def _write_prose(
    state: AgentState,
    city: str,
    route: str,
    weather: dict[str, Any],
    images: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    client = get_client()
    knowledge = state.get("knowledge", "") or "(no retrieved context available)"
    user = COMPOSER_USER.format(
        city=city,
        source=route,
        request=_last_user_text(state),
        knowledge=knowledge[:6000],
        tool_summary=_tool_summary(weather, images),
    )
    try:
        result = await client.achat(
            [SystemMessage(content=COMPOSER_SYSTEM), HumanMessage(content=user)],
            json_schema=PROSE_SCHEMA,
            json_object=True,
        )
        payload = result.json_payload()
        if not payload.get("city_summary"):
            raise ValueError("model returned no summary")
        payload.setdefault("city", city)
        return payload, "prose generated"
    except Exception as exc:  # noqa: BLE001 - fall back to the retrieved text
        # Strip "[overview]" style chunk headers and "[Things to do in Rome]"
        # style search-result titles before the text reaches the reader.
        fallback = re.sub(r"\[[^\]\n]{1,60}\]\s*", "", knowledge).strip()
        return (
            {
                "city": city,
                "country": state.get("country", ""),
                "city_summary": fallback[:1200]
                or f"No detailed information could be retrieved for {city}.",
                "highlights": [],
                "best_time_to_visit": "",
            },
            f"prose fell back to raw context ({type(exc).__name__})",
        )


def _chat_text(report: TravelReport) -> str:
    """Compact assistant message kept in history for conversational context."""
    lines = [f"**{report.city}**", report.city_summary]
    if report.weather_forecast:
        first, last = report.weather_forecast[0], report.weather_forecast[-1]
        lines.append(
            f"Forecast {first.date} to {last.date}; "
            f"{len(report.weather_forecast)} days, {len(report.image_urls)} images."
        )
    return "\n\n".join(lines)


def report_json(state: AgentState) -> str:
    return json.dumps(state.get("report", {}), indent=2, default=str)
