"""The switch: vector store vs web search."""

from __future__ import annotations

import pytest

from travel_agent.knowledge.seed_data import SEEDED_CITIES, normalise_city


@pytest.mark.parametrize("city", SEEDED_CITIES)
def test_seeded_cities_are_found(store, city):
    lookup = store.lookup(city, "what should I know before visiting?")
    assert lookup.found
    assert normalise_city(lookup.city) == normalise_city(city)
    assert len(lookup.text) > 200
    assert lookup.hits


@pytest.mark.parametrize("city", ["Snohomish", "Kyoto", "Ouagadougou"])
def test_unseeded_cities_miss(store, city):
    lookup = store.lookup(city)
    assert not lookup.found
    assert "floor" in lookup.reason


def test_alias_resolves_to_seeded_city(store):
    assert store.lookup("New York City").found


def test_lookup_reports_evidence_for_the_router(store):
    lookup = store.lookup("Paris")
    assert lookup.top_similarity > 0
    assert all(0.0 <= hit.similarity <= 1.0 for hit in lookup.hits)


def test_route_function_reads_state():
    from travel_agent.graph.nodes.knowledge import route_after_retrieval

    assert route_after_retrieval({"route": "web_search"}) == "web_search"
    assert route_after_retrieval({"route": "vector_store"}) == "tool_planner"


def test_plan_routing_branches():
    from travel_agent.graph.nodes.planner import route_after_plan

    assert route_after_plan({"plan": {"city": ""}}) == "clarify"
    assert (
        route_after_plan({"plan": {"city": "Tokyo", "needs_summary": True}})
        == "retrieve_knowledge"
    )
    assert (
        route_after_plan(
            {"plan": {"city": "Tokyo", "needs_summary": False}, "city_summary": "cached"}
        )
        == "tool_planner"
    )
