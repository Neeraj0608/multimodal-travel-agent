"""Knowledge retrieval and the routing decision that follows it.

retrieve_knowledge queries Chroma and records why the lookup hit or missed;
route_after_retrieval reads that record and either carries on or diverts to
web search. Keeping the reason in state means a surprising route can be
explained afterwards instead of guessed at.
"""

from __future__ import annotations

import time
from typing import Any

from ...knowledge.store import get_store
from ...state import AgentState
from ...tools.websearch import as_context, web_search
from ..tracing import make_event


def _last_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


async def retrieve_knowledge(state: AgentState) -> dict[str, Any]:
    """Query the curated vector store for the planned city."""
    started = time.time()
    turn = int(state.get("turn", 0))
    city = state.get("city", "")

    store = get_store()
    lookup = store.lookup(city, question=_last_user_text(state))

    update: dict[str, Any] = {
        "similarity": lookup.top_similarity,
        "knowledge_hits": [hit.as_dict() for hit in lookup.hits],
        "traces": [
            make_event(
                "retrieve_knowledge",
                turn,
                started,
                label="vector store lookup",
                detail=lookup.reason,
            )
        ],
    }
    if lookup.found:
        update["knowledge"] = lookup.text
        update["route"] = "vector_store"
        update["country"] = lookup.country or state.get("country", "")
    else:
        # Leave knowledge empty; the conditional edge will send us to the web.
        update["route"] = "web_search"
    return update


def route_after_retrieval(state: AgentState) -> str:
    """Conditional edge based on knowledge availability."""
    return "web_search" if state.get("route") == "web_search" else "tool_planner"


async def search_web(state: AgentState) -> dict[str, Any]:
    """Fallback path for cities outside the vector store."""
    started = time.time()
    turn = int(state.get("turn", 0))
    city = state.get("city", "")

    try:
        payload = await web_search(city, max_results=5)
        knowledge = as_context(payload)
        sources = [item.get("url", "") for item in payload.get("results", [])]
        detail = f"{payload.get('source')} returned {len(payload.get('results', []))} results"
        warnings: list[str] = []
        if payload.get("degraded_from_live"):
            warnings.append(f"Live search failed, used mock results: {payload['degraded_from_live']}")
    except Exception as exc:  # noqa: BLE001 - degrade to model prior knowledge
        knowledge = ""
        sources = []
        detail = f"web search failed: {type(exc).__name__}"
        warnings = [f"Web search unavailable ({type(exc).__name__}); answering from model priors."]

    return {
        "knowledge": knowledge,
        "route": "web_search",
        "sources": sources,
        "warnings": warnings,
        "traces": [
            make_event("web_search", turn, started, label="web search", detail=detail)
        ],
    }
