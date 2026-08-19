"""Graph topology.

    START
      |
    planner ---------------------------> clarify --> END      (no city resolved)
      |  \\
      |   \\--(city known, summary cached)--\\
      v                                      v
    retrieve_knowledge --(miss)--> web_search --> tool_planner
      |  (hit)                                       |
      \\----------------------------------------------/
                                                     |
                              Send x N (parallel) --> execute_tool
                                                     |
                                                  composer --> END

Three conditional edges:

* route_after_plan      - clarify / reuse memory / consult the knowledge base
* route_after_retrieval - vector store hit vs web-search fallback
* dispatch_tools        - fan out one worker per tool call, or skip to composer
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..state import AgentState
from .nodes import (
    clarify,
    compose_report,
    dispatch_tools,
    execute_tool,
    plan_turn,
    propose_tool_calls,
    retrieve_knowledge,
    route_after_plan,
    route_after_retrieval,
    search_web,
)


@lru_cache(maxsize=1)
def get_checkpointer() -> MemorySaver:
    """One checkpointer per process.

    Streamlit re-runs the script on every interaction, so this must be cached or
    conversation memory (and time travel) would reset on each keystroke. Swap
    for ``SqliteSaver`` to persist across restarts - the graph code is unchanged.
    """
    return MemorySaver()


def build_graph(*, interrupt_before_tools: bool = False, checkpointer=None):
    """Compile the travel-assistant graph.

    Args:
        interrupt_before_tools: pause before the fan-out so a human can inspect,
            approve or reject the proposed tool calls (human-in-the-loop).
        checkpointer: override for tests; defaults to the process checkpointer.
    """
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_turn)
    builder.add_node("clarify", clarify)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("web_search", search_web)
    builder.add_node("tool_planner", propose_tool_calls)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("composer", compose_report)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_plan,
        {
            "clarify": "clarify",
            "retrieve_knowledge": "retrieve_knowledge",
            "tool_planner": "tool_planner",
        },
    )
    builder.add_conditional_edges(
        "retrieve_knowledge",
        route_after_retrieval,
        {"web_search": "web_search", "tool_planner": "tool_planner"},
    )
    builder.add_edge("web_search", "tool_planner")
    builder.add_conditional_edges(
        "tool_planner", dispatch_tools, ["execute_tool", "composer"]
    )
    builder.add_edge("execute_tool", "composer")
    builder.add_edge("composer", END)
    builder.add_edge("clarify", END)

    return builder.compile(
        checkpointer=checkpointer or get_checkpointer(),
        interrupt_before=["execute_tool"] if interrupt_before_tools else None,
    )


@lru_cache(maxsize=2)
def get_graph(interrupt_before_tools: bool = False):
    """Cached compiled graph, keyed by the human-in-the-loop setting."""
    return build_graph(interrupt_before_tools=interrupt_before_tools)
