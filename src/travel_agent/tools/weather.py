"""Weather forecast tool.

Mock mode is the default and is deterministic per (city, date): the same city
always yields the same forecast, which makes screenshots, tests and demos
reproducible while still producing diverse, realistic-looking data across
cities and seasons.

Live mode uses Open-Meteo, which needs no API key, and degrades back to the
mock if the network or the geocoder is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from datetime import date, timedelta
from typing import Any

from ..config import get_settings

CONDITIONS = [
    (0, "Clear"),
    (15, "Mostly sunny"),
    (30, "Partly cloudy"),
    (50, "Overcast"),
    (65, "Light rain"),
    (80, "Rain"),
    (92, "Thunderstorms"),
]

# Rough climate anchors so the mock is plausible rather than uniform noise.
# (baseline mean temp in C, seasonal amplitude, wetness 0-1, hemisphere)
CLIMATE: dict[str, tuple[float, float, float, int]] = {
    "paris": (11.5, 8.0, 0.45, 1),
    "tokyo": (16.0, 10.0, 0.50, 1),
    "new york": (13.0, 12.0, 0.40, 1),
    "kyoto": (15.5, 11.0, 0.55, 1),
    "london": (11.0, 7.0, 0.50, 1),
    "reykjavik": (5.0, 6.0, 0.60, 1),
    "cairo": (22.0, 9.0, 0.05, 1),
    "singapore": (27.5, 1.5, 0.65, 1),
    "sydney": (18.0, 6.0, 0.35, -1),
    "buenos aires": (17.0, 7.0, 0.40, -1),
    "cape town": (17.0, 6.0, 0.30, -1),
    "snohomish": (11.0, 8.0, 0.60, 1),
}


def _seed(*parts: Any) -> random.Random:
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return random.Random(int.from_bytes(digest.digest(), "big"))


def _climate_for(city: str) -> tuple[float, float, float, int]:
    key = (city or "").strip().lower()
    if key in CLIMATE:
        return CLIMATE[key]
    # Unknown city: derive a stable pseudo-climate from the name so results are
    # still deterministic and varied rather than identical everywhere.
    rng = _seed("climate", key)
    return (rng.uniform(6.0, 26.0), rng.uniform(5.0, 12.0), rng.uniform(0.1, 0.6), 1)


def _condition(precip: int) -> str:
    label = CONDITIONS[0][1]
    for threshold, name in CONDITIONS:
        if precip >= threshold:
            label = name
    return label


def _mock_forecast(city: str, days: int, start_offset_days: int) -> dict[str, Any]:
    mean, amplitude, wetness, hemisphere = _climate_for(city)
    start = date.today() + timedelta(days=start_offset_days)
    points: list[dict[str, Any]] = []

    for index in range(days):
        day = start + timedelta(days=index)
        rng = _seed(city.lower(), day.isoformat())
        # Seasonal sine wave peaking in July (northern) / January (southern).
        season = math.sin((day.timetuple().tm_yday - 100) / 365.0 * 2 * math.pi) * hemisphere
        base = mean + amplitude * season
        swing = rng.uniform(-3.2, 3.2)
        high = base + 4.0 + swing
        low = base - 4.0 + swing * 0.6
        precip = int(max(0, min(95, rng.gauss(wetness * 70, 18))))
        points.append(
            {
                "date": day.isoformat(),
                "weekday": day.strftime("%a"),
                "temp_max_c": round(high, 1),
                "temp_min_c": round(min(low, high - 1.5), 1),
                "precipitation_chance": precip,
                "condition": _condition(precip),
            }
        )

    return {
        "city": city,
        "unit": "celsius",
        "source": "mock",
        "start_offset_days": start_offset_days,
        "days": points,
    }


async def _live_forecast(city: str, days: int, start_offset_days: int) -> dict[str, Any]:
    """Open-Meteo: keyless geocoding + daily forecast."""
    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        results = (geo.json() or {}).get("results") or []
        if not results:
            raise LookupError(f"Open-Meteo could not geocode '{city}'")
        place = results[0]

        forecast = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": min(16, days + start_offset_days),
                "timezone": "auto",
            },
        )
        daily = (forecast.json() or {}).get("daily") or {}

    dates = daily.get("time", [])[start_offset_days : start_offset_days + days]
    highs = daily.get("temperature_2m_max", [])[start_offset_days : start_offset_days + days]
    lows = daily.get("temperature_2m_min", [])[start_offset_days : start_offset_days + days]
    precs = daily.get("precipitation_probability_max", [])[
        start_offset_days : start_offset_days + days
    ]

    points = []
    for iso, high, low, precip in zip(dates, highs, lows, precs):
        chance = int(precip or 0)
        points.append(
            {
                "date": iso,
                "weekday": date.fromisoformat(iso).strftime("%a"),
                "temp_max_c": float(high),
                "temp_min_c": float(low),
                "precipitation_chance": chance,
                "condition": _condition(chance),
            }
        )
    if not points:
        raise LookupError("Open-Meteo returned an empty forecast window")

    return {
        "city": place.get("name", city),
        "country": place.get("country", ""),
        "unit": "celsius",
        "source": "open-meteo",
        "start_offset_days": start_offset_days,
        "days": points,
    }


async def get_weather_forecast(
    city: str, days: int = 7, start_offset_days: int = 0
) -> dict[str, Any]:
    """Daily forecast for a city.

    Args:
        city: City name, e.g. "Kyoto".
        days: How many days to return (1-14).
        start_offset_days: 0 for today, 7 to start a week from now.
    """
    settings = get_settings()
    days = max(1, min(14, int(days or 7)))
    start_offset_days = max(0, min(60, int(start_offset_days or 0)))

    if settings.use_live_apis:
        try:
            return await _live_forecast(city, days, start_offset_days)
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the turn
            result = _mock_forecast(city, days, start_offset_days)
            result["degraded_from_live"] = str(exc)[:200]
            return result

    # Simulated network latency, so the fan-out has real work to overlap.
    jitter = _seed("latency", city, start_offset_days).uniform(0.75, 1.35)
    await asyncio.sleep(settings.mock_latency * jitter)
    return _mock_forecast(city, days, start_offset_days)


SCHEMA: dict[str, Any] = {
    "name": "get_weather_forecast",
    "description": (
        "Get the daily weather forecast for a city. Use this whenever the user "
        "asks about a destination, its weather, or a future travel window."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Kyoto"},
            "days": {
                "type": "integer",
                "description": "Number of days to forecast, 1-14. Default 7.",
                "minimum": 1,
                "maximum": 14,
            },
            "start_offset_days": {
                "type": "integer",
                "description": (
                    "Days from today the window starts. 0 = today, 7 = next week. "
                    "Use this for follow-ups like 'what about next week?'."
                ),
                "minimum": 0,
                "maximum": 60,
            },
        },
        "required": ["city"],
    },
}
