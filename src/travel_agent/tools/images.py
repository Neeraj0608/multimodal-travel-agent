"""Image search tool.

Mock mode returns curated Unsplash photo IDs for well-known cities and
deterministic seeded placeholders for everywhere else, so the gallery never
renders a broken tile.

Live mode asks Wikipedia's REST API for the article's lead image, which needs
no API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Any

from ..config import get_settings

_UNSPLASH = "https://images.unsplash.com/photo-{pid}?auto=format&fit=crop&w=1000&q=70"

# Verified photo IDs per city.
CURATED: dict[str, list[tuple[str, str]]] = {
    "paris": [
        ("1493976040374-85c8e12f0c0e", "Haussmann rooftops and the Eiffel Tower"),
        ("1502602898657-3e91760cbb34", "The Eiffel Tower from the Champ de Mars"),
        ("1431274172761-fca41d930114", "Evening light over the Seine"),
        ("1522093007474-d86e9bf7ba6f", "A classic Paris street corner"),
        ("1520939817895-060bdaf4fe1b", "Cafe terrace in the old quarter"),
    ],
    "tokyo": [
        ("1540959733332-eab4deabeeaf", "Neon-lit Tokyo street at night"),
        ("1503899036084-c55cdd92da26", "Tokyo skyline at dusk"),
        ("1480796927426-f609979314bd", "Crossing in the heart of the city"),
        ("1536098561742-ca998e48cbcc", "Temple gate framed by lanterns"),
        ("1513407030348-c983a97b98d8", "Rail lines threading the districts"),
    ],
    "new york": [
        ("1496442226666-8d4d0e62e6e9", "Grand Central Terminal concourse"),
        ("1518235506717-e1ed3306a89b", "Midtown towers from the street"),
        ("1534430480872-3498386e7856", "Manhattan skyline across the water"),
        ("1485871981521-5b1fd3805eee", "Yellow cabs on a Manhattan avenue"),
        ("1522083165195-3424ed129620", "Brooklyn Bridge cables at golden hour"),
    ],
    "kyoto": [
        ("1545569341-9eb8b30979d9", "Torii gates on the mountain path"),
        ("1524413840807-0c3cb6fa808d", "Pagoda above the old town roofs"),
        ("1493997181344-712f2f19d87a", "Temple garden in still water"),
        ("1478436127897-769e1b3f0f36", "Lantern-lit lane after rain"),
    ],
    "london": [
        ("1533929736458-ca588d08c8be", "The Thames and the city skyline"),
        ("1513635269975-59663e0ac1ad", "Westminster and the river at dusk"),
        ("1486299267070-83823f5448dd", "Red buses on a London street"),
    ],
    "rome": [
        ("1552832230-c0197dd311b5", "The Colosseum in morning light"),
        ("1515542622106-78bda8ba0e5b", "Roman rooftops and domes"),
        ("1531572753322-ad063cecc140", "A piazza fountain at midday"),
    ],
    "sydney": [
        ("1506973035872-a4ec16b8e8d9", "The harbour and the opera house"),
        ("1523482580672-f109ba8cb9be", "Harbour bridge from the quay"),
    ],
}

# Generic city photography for destinations without a curated set.
GENERIC: list[tuple[str, str]] = [
    ("1449824913935-59a10b8d2000", "City streets at street level"),
    ("1477959858617-67f85cf4f1df", "Downtown skyline"),
    ("1444723121867-7a241cacace9", "Skyline after sunset"),
    ("1470071459604-3b5ec3a7fe05", "Landscape on the city's edge"),
]


def _rng(city: str) -> random.Random:
    digest = hashlib.blake2b(city.lower().encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _mock_images(city: str, count: int) -> dict[str, Any]:
    key = (city or "").strip().lower()
    curated = CURATED.get(key)
    images: list[dict[str, str]] = []

    if curated:
        for pid, caption in curated[:count]:
            images.append(
                {
                    "url": _UNSPLASH.format(pid=pid),
                    "caption": caption,
                    "credit": "Unsplash",
                    "representative": True,
                }
            )
    else:
        rng = _rng(key or "unknown")
        pool = GENERIC[:]
        rng.shuffle(pool)
        for index in range(count):
            if index < len(pool):
                pid, caption = pool[index]
                url = _UNSPLASH.format(pid=pid)
                credit = "Unsplash"
            else:
                # Deterministic placeholder keyed to the city name.
                url = f"https://picsum.photos/seed/{key or 'city'}-{index}/1000/700"
                caption = f"Stock imagery for {city}"
                credit = "Lorem Picsum"
            images.append(
                {
                    "url": url,
                    "caption": f"{caption} (generic stock, not verified as {city})",
                    "credit": credit,
                    "representative": False,
                }
            )

    return {"city": city, "source": "mock", "count": len(images), "images": images}


async def _live_images(city: str, count: int) -> dict[str, Any]:
    """Wikipedia REST lead image + original, no API key required."""
    import httpx

    headers = {"User-Agent": "multimodal-travel-agent/0.1"}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        response = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{city.replace(' ', '_')}"
        )
        response.raise_for_status()
        data = response.json()

    images: list[dict[str, str]] = []
    for field, caption in (("originalimage", "Wikipedia lead image"), ("thumbnail", "Thumbnail")):
        source = (data.get(field) or {}).get("source")
        if source:
            images.append(
                {
                    "url": source,
                    "caption": f"{data.get('title', city)} - {caption}",
                    "credit": "Wikimedia Commons",
                    "representative": True,
                }
            )
    if not images:
        raise LookupError(f"No Wikipedia image for '{city}'")

    # Top up from the curated pool so the gallery still fills out.
    if len(images) < count:
        images.extend(_mock_images(city, count - len(images))["images"])
    return {"city": city, "source": "wikipedia", "count": len(images), "images": images[:count]}


async def search_city_images(city: str, count: int = 4) -> dict[str, Any]:
    """Find photographs of a city for the visual gallery.

    Args:
        city: City name, e.g. "Kyoto".
        count: How many images to return (1-6).
    """
    settings = get_settings()
    count = max(1, min(6, int(count or 4)))

    if settings.use_live_apis:
        try:
            return await _live_images(city, count)
        except Exception as exc:  # noqa: BLE001
            result = _mock_images(city, count)
            result["degraded_from_live"] = str(exc)[:200]
            return result

    jitter = _rng(f"latency-{city}").uniform(0.6, 1.15)
    await asyncio.sleep(settings.mock_latency * jitter)
    return _mock_images(city, count)


SCHEMA: dict[str, Any] = {
    "name": "search_city_images",
    "description": (
        "Find photographs of a city to display in the visual gallery. Call this "
        "when the user asks about a destination for the first time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Kyoto"},
            "count": {
                "type": "integer",
                "description": "How many images to return, 1-6. Default 4.",
                "minimum": 1,
                "maximum": 6,
            },
        },
        "required": ["city"],
    },
}
