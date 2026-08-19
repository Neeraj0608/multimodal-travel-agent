"""Tool registry.

One place that maps a tool name to its callable and to the JSON schema shown to
the model. The executor resolves tools through here at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from . import images, websearch, weather


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Awaitable[dict[str, Any]]]

    def schema(self) -> dict[str, Any]:
        """OpenAI/Groq function-schema shape; adapted per provider downstream."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _tool_from(module: Any, func: Callable[..., Awaitable[dict[str, Any]]]) -> Tool:
    schema = module.SCHEMA
    return Tool(
        name=schema["name"],
        description=schema["description"],
        parameters=schema["parameters"],
        func=func,
    )


TOOL_REGISTRY: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        _tool_from(weather, weather.get_weather_forecast),
        _tool_from(images, images.search_city_images),
        _tool_from(websearch, websearch.web_search),
    )
}

# Tools offered to the model during the fetch step. ``web_search`` is not in
# this set: whether to search the web is a *routing* decision made by a
# conditional edge, not something the model should pick opportunistically.
FETCH_TOOLS: tuple[str, ...] = ("get_weather_forecast", "search_city_images")


def get_tool(name: str) -> Tool | None:
    return TOOL_REGISTRY.get(name)


def tool_schemas(names: tuple[str, ...] | list[str] = FETCH_TOOLS) -> list[dict[str, Any]]:
    return [TOOL_REGISTRY[n].schema() for n in names if n in TOOL_REGISTRY]


def validate_args(tool: Tool, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce and filter model-supplied arguments against the tool schema.

    Models send numbers as strings, invent extra keys and drop optional ones.
    Handling that here keeps the tool functions plain Python with ordinary
    type hints.
    """
    schema = tool.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    cleaned: dict[str, Any] = {}
    notes: list[str] = []

    for key, value in (args or {}).items():
        if key not in properties:
            notes.append(f"dropped unknown argument '{key}'")
            continue
        expected = properties[key].get("type")
        try:
            if expected == "integer" and not isinstance(value, int):
                cleaned[key] = int(float(value))
            elif expected == "number" and not isinstance(value, (int, float)):
                cleaned[key] = float(value)
            elif expected == "string" and not isinstance(value, str):
                cleaned[key] = str(value)
            else:
                cleaned[key] = value
        except (TypeError, ValueError):
            notes.append(f"could not coerce '{key}'={value!r} to {expected}")

    for key in required:
        if not cleaned.get(key):
            notes.append(f"missing required argument '{key}'")

    return cleaned, notes


__all__ = [
    "Tool",
    "TOOL_REGISTRY",
    "FETCH_TOOLS",
    "get_tool",
    "tool_schemas",
    "validate_args",
]
