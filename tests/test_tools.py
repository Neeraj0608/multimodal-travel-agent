"""Tools and the manual argument-validation layer."""

from __future__ import annotations

import asyncio

import pytest

from travel_agent.schemas import WeatherPoint
from travel_agent.tools import FETCH_TOOLS, TOOL_REGISTRY, get_tool, tool_schemas, validate_args
from travel_agent.tools.images import search_city_images
from travel_agent.tools.weather import get_weather_forecast


def test_registry_exposes_valid_json_schemas():
    for schema in tool_schemas(tuple(TOOL_REGISTRY)):
        assert schema["name"] and schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert "city" in schema["parameters"]["properties"]


def test_fetch_tools_exclude_web_search():
    # Whether to search the web is a routing decision, not a model choice.
    assert "web_search" not in FETCH_TOOLS
    assert "web_search" in TOOL_REGISTRY


def test_weather_is_deterministic_and_valid():
    first = asyncio.run(get_weather_forecast("Kyoto", days=7))
    second = asyncio.run(get_weather_forecast("Kyoto", days=7))
    assert first == second

    assert len(first["days"]) == 7
    for day in first["days"]:
        point = WeatherPoint.model_validate(day)  # must satisfy the UI contract
        assert point.temp_max_c > point.temp_min_c
        assert 0 <= point.precipitation_chance <= 100


def test_weather_differs_between_cities():
    cairo = asyncio.run(get_weather_forecast("Cairo", days=5))
    reykjavik = asyncio.run(get_weather_forecast("Reykjavik", days=5))
    cairo_mean = sum(d["temp_max_c"] for d in cairo["days"]) / 5
    reykjavik_mean = sum(d["temp_max_c"] for d in reykjavik["days"]) / 5
    assert cairo_mean > reykjavik_mean


def test_weather_offset_shifts_the_window():
    today = asyncio.run(get_weather_forecast("Tokyo", days=3, start_offset_days=0))
    next_week = asyncio.run(get_weather_forecast("Tokyo", days=3, start_offset_days=7))
    assert today["days"][0]["date"] != next_week["days"][0]["date"]


def test_weather_clamps_out_of_range_arguments():
    result = asyncio.run(get_weather_forecast("Paris", days=99))
    assert len(result["days"]) == 14


def test_images_return_usable_urls():
    result = asyncio.run(search_city_images("Paris", count=4))
    assert len(result["images"]) == 4
    for image in result["images"]:
        assert image["url"].startswith("https://")


def test_images_fall_back_for_unknown_cities():
    result = asyncio.run(search_city_images("Ouagadougou", count=4))
    assert result["images"]
    # Generic stock must be labelled as such rather than passed off as the city.
    assert all(not i["representative"] for i in result["images"])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"city": "Kyoto", "days": "5"}, {"city": "Kyoto", "days": 5}),
        ({"city": "Kyoto", "bogus": 1}, {"city": "Kyoto"}),
        ({"city": 42}, {"city": "42"}),
    ],
)
def test_validate_args_coerces_model_output(raw, expected):
    tool = get_tool("get_weather_forecast")
    cleaned, _ = validate_args(tool, raw)
    assert cleaned == expected


def test_validate_args_flags_missing_required():
    tool = get_tool("get_weather_forecast")
    _, notes = validate_args(tool, {"days": 3})
    assert any("missing required" in note for note in notes)
