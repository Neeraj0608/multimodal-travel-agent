"""Pydantic contract between the graph and the UI.

Streamlit renders fields, never free-form markdown, so a structural mistake by
the model surfaces here as a ValidationError instead of as a broken layout
three layers away.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

KnowledgeSource = Literal["vector_store", "web_search", "memory", "model_prior"]


class WeatherPoint(BaseModel):
    """One day of forecast. Units are metric and explicit in the field names."""

    date: str = Field(description="ISO-8601 date, e.g. 2026-08-19")
    weekday: str = Field(default="", description="Short weekday label, e.g. Tue")
    temp_max_c: float
    temp_min_c: float
    precipitation_chance: int = Field(default=0, ge=0, le=100)
    condition: str = Field(default="", description="Short human label, e.g. Light rain")

    @field_validator("date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        _date.fromisoformat(v)  # raises for malformed input
        return v

    @field_validator("temp_max_c", "temp_min_c", mode="before")
    @classmethod
    def _coerce_number(cls, v: object) -> object:
        if isinstance(v, str):
            return float(v.replace("°C", "").replace("C", "").strip())
        return v


class ImageItem(BaseModel):
    url: str
    caption: str = ""
    credit: str = ""


class TravelReport(BaseModel):
    """The object the UI renders. Nothing reaches the view layer unvalidated."""

    city: str
    country: str = ""
    city_summary: str = Field(description="2-4 paragraph prose summary for the traveller")
    highlights: list[str] = Field(default_factory=list, description="Short bullet attractions")
    best_time_to_visit: str = ""
    weather_forecast: list[WeatherPoint] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    images: list[ImageItem] = Field(default_factory=list)
    knowledge_source: KnowledgeSource = "vector_store"
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Degraded-mode notices surfaced to the user, e.g. a failed tool",
    )

    def model_post_init(self, _ctx: object) -> None:
        # Keep the two image representations consistent regardless of which one
        # the model chose to populate.
        if self.images and not self.image_urls:
            self.image_urls = [i.url for i in self.images]
        elif self.image_urls and not self.images:
            self.images = [ImageItem(url=u) for u in self.image_urls]


class TurnPlan(BaseModel):
    """Planner output: what this turn is actually asking for.

    ``needs_summary`` is the memory switch that powers the follow-up case
    ("Tokyo" -> "what about next week?"): the city is unchanged, so the summary
    and gallery are reused from the checkpointed state and only the weather tool
    re-runs.
    """

    city: str = Field(default="", description="Resolved city name, '' if none could be found")
    country_hint: str = ""
    needs_summary: bool = True
    needs_weather: bool = True
    needs_images: bool = True
    start_offset_days: int = Field(
        default=0, description="0 = today, 7 = 'next week', used for the weather window"
    )
    forecast_days: int = Field(default=7, ge=1, le=14)
    rationale: str = ""
    is_followup: bool = False
