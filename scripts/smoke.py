"""Runs the graph end to end without Streamlit.

Four scenarios, each with assertions:

1. vector-store hit    - "Tell me about Tokyo"
2. follow-up / memory  - "What about next week?"  (weather only, summary reused)
3. web-search fallback - "Tell me about Snohomish"
4. degraded mode       - weather tool forced to fail

Usage:  python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Real models emit typographic characters (en dashes, non-breaking hyphens) that
# the default Windows console codec cannot encode. Never let printing crash a run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

from travel_agent import bootstrap  # noqa: E402

bootstrap()

from langchain_core.messages import HumanMessage  # noqa: E402

from travel_agent.config import get_settings  # noqa: E402
from travel_agent.graph.builder import build_graph  # noqa: E402
from travel_agent.graph.tracing import parallelism_stats  # noqa: E402
from travel_agent.state import results_for_turn, traces_for_turn  # noqa: E402


def show(state: dict, label: str) -> None:
    turn = state.get("turn", 0)
    report = state.get("report", {}) or {}
    traces = traces_for_turn(state, turn)
    tool_traces = [t for t in traces if t["node"] == "execute_tool"]
    stats = parallelism_stats(tool_traces)

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"route            : {state.get('route')}")
    print(f"knowledge_source : {report.get('knowledge_source')}")
    print(f"city             : {report.get('city')!r}  country={report.get('country')!r}")
    print(f"summary          : {(report.get('city_summary') or '')[:140]}...")
    print(f"forecast days    : {len(report.get('weather_forecast', []))}")
    print(f"images           : {len(report.get('image_urls', []))}")
    print(f"warnings         : {report.get('warnings')}")
    print(f"tool calls       : {[c['name'] for c in state.get('proposed_tool_calls', [])]}")
    for result in results_for_turn(state, turn):
        flag = "ok " if result.get("ok") else "ERR"
        print(f"  - {flag} {result['name']:<24} {result['duration_ms']:>8.0f} ms")
    print(
        f"fan-out          : sequential {stats['sequential_ms']:.0f} ms vs wall "
        f"{stats['wall_ms']:.0f} ms  ->  {stats['speedup']}x"
    )
    for trace in traces:
        print(f"  [{trace['node']:<18}] {trace['duration_ms']:>7.0f} ms  {trace['detail'][:90]}")


async def main() -> int:
    settings = get_settings()
    print(f"provider={settings.describe()}  live_apis={settings.use_live_apis}  "
          f"embeddings={settings.embedding_backend}")

    graph = build_graph()
    config = {"configurable": {"thread_id": "smoke-1"}}

    state = await graph.ainvoke({"messages": [HumanMessage(content="Tell me about Tokyo")]}, config)
    show(state, "1. Vector-store path: 'Tell me about Tokyo'")
    assert state.get("route") == "vector_store", "expected the vector store to answer Tokyo"

    state = await graph.ainvoke({"messages": [HumanMessage(content="What about next week?")]}, config)
    show(state, "2. Memory path: 'What about next week?'")
    called = [c["name"] for c in state.get("proposed_tool_calls", [])]
    assert "get_weather_forecast" in called, "follow-up must refresh the weather"

    config2 = {"configurable": {"thread_id": "smoke-2"}}
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Tell me about Snohomish")]}, config2
    )
    show(state, "3. Web-search fallback: 'Tell me about Snohomish'")
    assert state.get("route") == "web_search", "Snohomish is not in the store; expected web search"

    config3 = {"configurable": {"thread_id": "smoke-3", "fail_tools": ["get_weather_forecast"]}}
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Tell me about Paris")]}, config3
    )
    show(state, "4. Degraded mode: weather tool forced to fail")
    assert state["report"]["warnings"], "a failed tool must surface a warning"
    assert state["report"]["image_urls"], "images must still render when weather fails"

    print("\nAll smoke assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
