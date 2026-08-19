"""Graph state and its reducers.

Keys written by more than one concurrent node need a reducer or LangGraph
raises InvalidUpdateError. The fan-out writes messages, tool_results, traces
and warnings; everything else has a single writer per superstep.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["vector_store", "web_search", "memory"]


class ToolResult(TypedDict, total=False):
    """One manual tool execution, tagged with the turn that requested it."""

    turn: int
    tool_call_id: str
    name: str
    args: dict[str, Any]
    ok: bool
    result: Any
    error: str
    started_at: float
    finished_at: float
    duration_ms: float


class TraceEvent(TypedDict, total=False):
    """Timeline entry used by the UI to prove parallelism and explain routing."""

    turn: int
    node: str
    label: str
    started_at: float
    finished_at: float
    duration_ms: float
    detail: str


class ToolTask(TypedDict):
    """Payload handed to a fan-out worker via ``Send``."""

    turn: int
    tool_call: dict[str, Any]
    city: str


class AgentState(TypedDict, total=False):
    # conversation
    messages: Annotated[list[AnyMessage], add_messages]
    turn: int

    # planner output (persisted across turns by the checkpointer)
    city: str
    plan: dict[str, Any]
    route: Route

    # knowledge
    knowledge: str
    knowledge_hits: list[dict[str, Any]]
    similarity: float
    sources: list[str]

    # cached artefacts reused on follow-up turns, valid only for cached_for_city
    cached_for_city: str
    city_summary: str
    country: str
    highlights: list[str]
    best_time_to_visit: str
    image_urls: list[str]
    weather_forecast: list[dict[str, Any]]

    # manual tool protocol
    proposed_tool_calls: list[dict[str, Any]]
    raw_tool_payload: str
    tool_results: Annotated[list[ToolResult], operator.add]

    # observability / UX
    traces: Annotated[list[TraceEvent], operator.add]
    warnings: Annotated[list[str], operator.add]

    # final structured object handed to Streamlit
    report: dict[str, Any]


def current_turn(state: AgentState) -> int:
    return int(state.get("turn", 0))


def results_for_turn(state: AgentState, turn: int) -> list[ToolResult]:
    return [r for r in state.get("tool_results", []) if r.get("turn") == turn]


def traces_for_turn(state: AgentState, turn: int) -> list[TraceEvent]:
    return [t for t in state.get("traces", []) if t.get("turn") == turn]
