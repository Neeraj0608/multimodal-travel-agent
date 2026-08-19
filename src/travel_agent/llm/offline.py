"""Deterministic stand-in so the graph runs without an API key.

Keeps the app usable on a machine with no credentials, keeps the tests
hermetic, and gives the UI something sensible when a provider is down. It
speaks the same protocol as the real clients, tool_calls included.

The phase is inferred from the call shape: tools present means a tool-selection
turn, json_schema['title'] means structured output (plan or compose).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Sequence

from langchain_core.messages import AnyMessage, HumanMessage

from .base import BaseChatClient, ChatResult, ToolCall, _as_text

_FOLLOWUP_HINTS = (
    "next week",
    "next month",
    "what about",
    "how about",
    "and then",
    "later",
    "tomorrow",
    "weekend",
)

_STOPWORDS = {
    "Tell",
    "What",
    "How",
    "Show",
    "Give",
    "About",
    "Weather",
    "Please",
    "Trip",
    "Travel",
    "Plan",
    "Next",
    "Week",
    "Hi",
    "Hello",
    "I",
}


class OfflineClient(BaseChatClient):
    """Rule-based responder that mirrors the real wire protocol."""

    name = "offline"

    def __init__(self, model: str = "deterministic-stub", **_: Any) -> None:
        super().__init__(model)

    def chat(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        json_object: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        last_user = _last_human(messages)
        title = (json_schema or {}).get("title", "")

        if tools:
            return self._tool_turn(tools, last_user)
        if title == "TurnPlan":
            return self._plan_turn(messages, last_user)
        return self._compose_turn(messages, last_user)

    # ------------------------------------------------------------- phases
    def _tool_turn(self, tools: list[dict[str, Any]], last_user: str) -> ChatResult:
        """Emit one tool call per offered tool, filling arguments from context."""
        city = _guess_city(last_user) or "the requested city"
        offset = _guess_offset(last_user)
        calls: list[ToolCall] = []
        for tool in tools:
            name = tool["name"]
            if name == "get_weather_forecast":
                args: dict[str, Any] = {"city": city, "days": 7, "start_offset_days": offset}
            elif name == "search_city_images":
                args = {"city": city, "count": 4}
            else:
                args = {"city": city}
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, args=args))
        return ChatResult(
            text="",
            tool_calls=calls,
            model=self.model,
            finish_reason="tool_calls",
            raw={"offline": True, "reason": "deterministic tool plan"},
        )

    def _plan_turn(self, messages: Sequence[AnyMessage], last_user: str) -> ChatResult:
        city = _guess_city(last_user)
        prior_city = _prior_city(messages)
        followup = bool(prior_city) and (
            not city or any(h in last_user.lower() for h in _FOLLOWUP_HINTS)
        )
        resolved = city or prior_city or ""
        weather_only = followup and (
            "weather" in last_user.lower()
            or any(h in last_user.lower() for h in _FOLLOWUP_HINTS)
        )
        plan = {
            "city": resolved,
            "country_hint": "",
            "needs_summary": not (followup and resolved == prior_city),
            "needs_weather": True,
            "needs_images": not weather_only,
            "start_offset_days": _guess_offset(last_user),
            "forecast_days": 7,
            "is_followup": followup,
            "rationale": (
                "offline planner: reused remembered city, weather window shifted"
                if followup
                else "offline planner: new city request, full fetch"
            ),
        }
        return ChatResult(text=json.dumps(plan), model=self.model, finish_reason="stop")

    def _compose_turn(self, messages: Sequence[AnyMessage], last_user: str) -> ChatResult:
        """Write prose from whatever knowledge context the node supplied."""
        context = _as_text(messages[-1].content) if messages else ""
        knowledge = _section(context, "KNOWLEDGE")
        # Retrieved chunks carry "[section]" prefixes for the model's benefit;
        # they must not reach the reader.
        knowledge = re.sub(r"\[[^\]\n]{1,60}\]\s*", "", knowledge)
        city = _guess_city(context) or "This destination"
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", knowledge) if len(s.strip()) > 30]
        summary = " ".join(sentences[:6]) or (
            f"{city} is a destination worth exploring, though detailed notes were not "
            "available offline for this request."
        )
        highlights = [s.strip(" .")[:90] for s in sentences[6:10]] or [
            "Historic centre",
            "Local food markets",
            "Museums and galleries",
        ]
        payload = {
            "city": city,
            "country": "",
            "city_summary": summary,
            "highlights": highlights,
            "best_time_to_visit": "Shoulder seasons (spring and autumn) generally offer the best balance of weather and crowds.",
        }
        return ChatResult(text=json.dumps(payload), model=self.model, finish_reason="stop")


# ----------------------------------------------------------------- helpers
def _last_human(messages: Sequence[AnyMessage]) -> str:
    for msg in reversed(list(messages)):
        if isinstance(msg, HumanMessage):
            return _as_text(msg.content)
    return _as_text(messages[-1].content) if messages else ""


def _prior_city(messages: Sequence[AnyMessage]) -> str:
    """Scan backwards for a city the *user* mentioned earlier.

    Only human turns are considered. System prompts name the knowledge-base
    cities, so scanning them would invent a destination out of the prompt
    itself and skip the clarification path.
    """
    for msg in reversed(list(messages)[:-1]):
        if not isinstance(msg, HumanMessage):
            continue
        found = _guess_city(_as_text(msg.content))
        if found:
            return found
    return ""


def _guess_city(text: str) -> str:
    if not text:
        return ""
    from ..knowledge.seed_data import KNOWN_CITY_NAMES

    lowered = text.lower()
    for name in KNOWN_CITY_NAMES:
        if name.lower() in lowered:
            return name
    # Fall back to the first capitalised token that is not a sentence starter.
    for match in re.finditer(r"\b([A-Z][a-z]{2,})(?:\s+([A-Z][a-z]{2,}))?", text):
        candidate = match.group(0)
        if match.group(1) in _STOPWORDS:
            continue
        return candidate
    return ""


def _guess_offset(text: str) -> int:
    lowered = (text or "").lower()
    if "next week" in lowered:
        return 7
    if "next month" in lowered:
        return 30
    if "tomorrow" in lowered:
        return 1
    if "weekend" in lowered:
        return 5
    return 0


def _section(text: str, header: str) -> str:
    match = re.search(rf"{header}:\s*(.*?)(?:\n[A-Z_]+:|\Z)", text, re.S)
    return match.group(1).strip() if match else ""
