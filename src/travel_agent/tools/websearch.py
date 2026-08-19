"""Web search tool - the fallback path when the vector store has no coverage.

Mock mode synthesises plausible, clearly-labelled search snippets for any city
so the routing behaviour can be demonstrated without a search subscription.
Live mode uses Tavily when ``TAVILY_API_KEY`` is present.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
from typing import Any

from ..config import get_settings

_ANGLES = [
    (
        "{city} travel guide",
        "{city} is best approached on foot from its historic core, where the older street "
        "pattern survives. Visitors typically allow two to three days for the main sights "
        "and use regional transport for day trips.",
        "wikivoyage.org/wiki/{slug}",
    ),
    (
        "Things to do in {city}",
        "The most-recommended experiences in {city} combine one landmark site, one museum or "
        "gallery, and an evening in the district where residents actually eat. Guided walking "
        "tours run most mornings and are the fastest way to orient yourself.",
        "timeout.com/{slug}/things-to-do",
    ),
    (
        "Best time to visit {city}",
        "Shoulder seasons - roughly April to early June and September to October - offer the "
        "most comfortable conditions and the smallest crowds in {city}. Peak summer brings "
        "higher prices and longer queues at the headline attractions.",
        "lonelyplanet.com/{slug}/best-time-to-visit",
    ),
    (
        "{city} food scene",
        "Local specialities in {city} are cheapest and best at markets and neighbourhood "
        "counters rather than the central squares. Ask for the daily set menu at lunch, which "
        "is usually a third less than the same dishes at dinner.",
        "eater.com/{slug}",
    ),
    (
        "Getting around {city}",
        "Public transport in {city} is the practical choice; a stored-value or contactless "
        "card covers most operators. Taxis are plentiful but slower than rail at peak hours, "
        "and the compact centre makes walking viable for short hops.",
        "rome2rio.com/{slug}/transport",
    ),
]


def _slug(city: str) -> str:
    return (city or "unknown").strip().lower().replace(" ", "-")


def _rng(city: str) -> random.Random:
    digest = hashlib.blake2b(f"search:{city.lower()}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _mock_results(city: str, max_results: int) -> dict[str, Any]:
    slug = _slug(city)
    rng = _rng(city)
    angles = _ANGLES[:]
    rng.shuffle(angles)

    results = []
    for title, body, domain in angles[:max_results]:
        results.append(
            {
                "title": title.format(city=city),
                "url": f"https://www.{domain.format(slug=slug)}",
                "content": body.format(city=city),
                "score": round(rng.uniform(0.72, 0.97), 3),
            }
        )
    return {
        "query": f"{city} travel overview",
        "source": "mock-search",
        "results": results,
        "disclaimer": "Synthetic search results generated locally for demonstration.",
    }


async def _live_results(city: str, max_results: int) -> dict[str, Any]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    query = f"{city} travel guide overview best time to visit"
    raw = await asyncio.to_thread(
        client.search, query=query, max_results=max_results, search_depth="basic"
    )
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score"),
        }
        for item in raw.get("results", [])
    ]
    if not results:
        raise LookupError(f"Tavily returned nothing for '{city}'")
    return {"query": query, "source": "tavily", "results": results}


async def web_search(city: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for background on a city the knowledge base does not cover.

    Args:
        city: City name to research.
        max_results: How many results to return (1-8).
    """
    settings = get_settings()
    max_results = max(1, min(8, int(max_results or 5)))

    if settings.use_live_apis and os.getenv("TAVILY_API_KEY"):
        try:
            return await _live_results(city, max_results)
        except Exception as exc:  # noqa: BLE001
            result = _mock_results(city, max_results)
            result["degraded_from_live"] = str(exc)[:200]
            return result

    jitter = _rng(f"latency-{city}").uniform(0.8, 1.3)
    await asyncio.sleep(settings.mock_latency * jitter)
    return _mock_results(city, max_results)


def as_context(payload: dict[str, Any]) -> str:
    """Flatten search results into prompt context."""
    lines = []
    for item in payload.get("results", []):
        lines.append(f"[{item.get('title', '')}] {item.get('content', '')}")
    return "\n\n".join(lines)


SCHEMA: dict[str, Any] = {
    "name": "web_search",
    "description": "Search the web for background information about a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name to research"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["city"],
    },
}
