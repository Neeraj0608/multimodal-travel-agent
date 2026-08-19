"""End-to-end graph behaviour: routing, fan-out, memory, degradation."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from travel_agent.graph.tracing import parallelism_stats
from travel_agent.schemas import TravelReport
from travel_agent.state import results_for_turn, traces_for_turn


def run_turn(graph, text, thread="t1", fail_tools=None):
    config = {"configurable": {"thread_id": thread, "fail_tools": fail_tools or []}}
    return asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=text)]}, config))


def test_vector_store_turn_produces_a_valid_report(graph):
    state = run_turn(graph, "Tell me about Tokyo")

    assert state["route"] == "vector_store"
    report = TravelReport.model_validate(state["report"])
    assert report.city == "Tokyo"
    assert report.knowledge_source == "vector_store"
    assert len(report.weather_forecast) == 7
    assert len(report.image_urls) >= 1
    assert not report.warnings


def test_web_search_fallback(graph):
    state = run_turn(graph, "Tell me about Snohomish", thread="t-web")
    assert state["route"] == "web_search"
    assert TravelReport.model_validate(state["report"]).knowledge_source == "web_search"


def test_manual_tool_protocol_round_trip(graph):
    """AIMessage(tool_calls) -> our executor -> ToolMessage with matching id."""
    state = run_turn(graph, "Tell me about Paris", thread="t-proto")

    ai_messages = [
        m for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls
    ]
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert ai_messages, "the model must emit tool_calls"

    requested = {c["id"] for m in ai_messages for c in m.tool_calls}
    answered = {m.tool_call_id for m in tool_messages}
    assert requested == answered, "every tool_call must get a matching ToolMessage"


def test_tools_run_in_parallel(graph):
    state = run_turn(graph, "Tell me about Tokyo", thread="t-par")
    turn = state["turn"]
    tool_traces = [t for t in traces_for_turn(state, turn) if t["node"] == "execute_tool"]
    assert len(tool_traces) == 2

    stats = parallelism_stats(tool_traces)
    # Two ~1s tools that overlap finish in materially less than their sum.
    assert stats["speedup"] > 1.4, stats
    first, second = sorted(tool_traces, key=lambda t: t["started_at"])
    assert second["started_at"] < first["finished_at"], "executions did not overlap"


def test_followup_reuses_memory_and_refetches_only_weather(graph):
    first = run_turn(graph, "Tell me about Tokyo", thread="t-mem")
    second = run_turn(graph, "What about next week?", thread="t-mem")

    assert second["city"] == "Tokyo", "city must survive from the checkpoint"
    called = [c["name"] for c in second["proposed_tool_calls"]]
    assert called == ["get_weather_forecast"], f"expected weather only, got {called}"

    # The summary was not regenerated, and no retrieval node ran again.
    assert second["report"]["city_summary"] == first["report"]["city_summary"]
    assert second["report"]["knowledge_source"] == "memory"
    nodes = {t["node"] for t in traces_for_turn(second, second["turn"])}
    assert "retrieve_knowledge" not in nodes

    # ...but the forecast window did move.
    assert second["report"]["weather_forecast"][0]["date"] != first["report"][
        "weather_forecast"
    ][0]["date"]


def test_failed_tool_degrades_instead_of_crashing(graph):
    state = run_turn(
        graph, "Tell me about Paris", thread="t-fail", fail_tools=["get_weather_forecast"]
    )

    report = TravelReport.model_validate(state["report"])
    assert report.warnings, "the failure must be surfaced to the user"
    assert report.weather_forecast == []
    assert report.image_urls, "the independent tool must still have succeeded"
    assert report.city_summary, "the text answer must survive a tool failure"

    failed = [r for r in results_for_turn(state, state["turn"]) if not r["ok"]]
    assert len(failed) == 1


def test_unresolvable_city_asks_for_clarification(graph):
    state = run_turn(graph, "hello there", thread="t-clarify")
    assert state["report"]["warnings"]
    assert not state["report"]["weather_forecast"]


def test_checkpoint_history_supports_time_travel(graph):
    run_turn(graph, "Tell me about Paris", thread="t-history")
    history = list(graph.get_state_history({"configurable": {"thread_id": "t-history"}}))

    assert len(history) > 3
    assert any(snap.next for snap in history), "intermediate checkpoints must exist"
    assert all(
        snap.config["configurable"].get("checkpoint_id") for snap in history
    ), "each checkpoint must be addressable for replay"


def test_cached_data_is_not_reused_for_a_different_city(graph):
    """A second city must not inherit the first city's data.

    If nothing fresh arrives, the composer falls back to cached artefacts; that
    fallback has to be gated on the city still matching.
    """
    first = run_turn(graph, "Tell me about Tokyo", thread="t-stale")
    assert first["report"]["weather_forecast"]

    # Same thread, new city, and both tools broken so nothing fresh arrives.
    second = run_turn(
        graph,
        "Tell me about Paris",
        thread="t-stale",
        fail_tools=["get_weather_forecast", "search_city_images"],
    )

    assert second["report"]["city"] == "Paris"
    assert second["report"]["weather_forecast"] == [], "must not show Tokyo's forecast"
    assert second["report"]["image_urls"] == [], "must not show Tokyo's photos"
    assert second["report"]["warnings"]


def test_summary_is_not_reused_across_cities(graph):
    run_turn(graph, "Tell me about Tokyo", thread="t-prose")
    second = run_turn(graph, "Tell me about Paris", thread="t-prose")
    assert "Tokyo" not in second["report"]["city_summary"]
    assert second["report"]["knowledge_source"] == "vector_store"
