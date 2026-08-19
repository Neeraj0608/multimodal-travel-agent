"""Multi-modal travel assistant - Streamlit front end.

Run with:  streamlit run app.py

The app owns no agent logic. It sends a message into the LangGraph graph, reads
the structured ``TravelReport`` back out of state, and renders it. Everything
interesting (routing, tool calling, parallelism, memory) lives in
``src/travel_agent/graph``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from travel_agent import bootstrap  # noqa: E402

bootstrap()

import streamlit as st  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402


def _adopt_streamlit_secrets() -> None:
    """Copy st.secrets into the environment before settings are resolved.

    Locally the configuration comes from .env. When deployed there is no .env,
    and the keys arrive through Streamlit's secrets manager instead. Reading
    them into os.environ keeps config.py as the single place that resolves
    settings, rather than teaching it about two different sources.
    """
    keys = (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "USE_LIVE_APIS",
        "MOCK_LATENCY",
        "EMBEDDING_BACKEND",
        "SIMILARITY_FLOOR",
    )
    # Touching st.secrets with no secrets.toml on disk makes Streamlit render an
    # error box, so check for the file rather than catching the failure.
    locations = (
        Path.home() / ".streamlit" / "secrets.toml",
        ROOT / ".streamlit" / "secrets.toml",
    )
    if not any(path.exists() for path in locations):
        return

    try:
        secrets = st.secrets
    except Exception:
        return
    for key in keys:
        try:
            value = secrets[key]
        except Exception:
            continue
        if value not in (None, ""):
            os.environ.setdefault(key, str(value))


_adopt_streamlit_secrets()

from travel_agent.config import API_KEY_VARS, get_settings  # noqa: E402
from travel_agent.graph.builder import get_graph  # noqa: E402
from travel_agent.graph.tracing import parallelism_stats  # noqa: E402
from travel_agent.knowledge.store import get_store  # noqa: E402
from travel_agent.state import results_for_turn, traces_for_turn  # noqa: E402
from travel_agent.ui.render import render_report, render_trace  # noqa: E402

st.set_page_config(
    page_title="Multi-Modal Travel Assistant",
    page_icon=":material/travel_explore:",
    layout="wide",
)

EXAMPLES = [
    ("Tell me about Tokyo", "vector store"),
    ("What about next week?", "memory + weather only"),
    ("Tell me about Snohomish", "web search"),
]


# --------------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Preparing the city knowledge base...")
def warm_store() -> dict[str, Any]:
    store = get_store()
    return {"cities": store.cities(), "documents": store.count(), "backend": store.embedder.name}


def init_session() -> None:
    st.session_state.setdefault("thread_id", f"thread-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("pending", None)


def reset_conversation() -> None:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    st.session_state.turns = []
    st.session_state.pending = None


def run(coro: Any) -> Any:
    """Streamlit scripts are synchronous; the graph is async."""
    return asyncio.run(coro)


def collect_turn(state: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Flatten the graph state into what a single UI card needs."""
    turn = int(state.get("turn", 0))
    traces = traces_for_turn(state, turn)
    return {
        "user_text": user_text,
        "turn": turn,
        "report": state.get("report", {}) or {},
        "route": state.get("route", "-"),
        "plan": state.get("plan", {}),
        "tool_calls": state.get("proposed_tool_calls", []),
        "raw_tool_payload": state.get("raw_tool_payload", ""),
        "tool_results": results_for_turn(state, turn),
        "traces": traces,
        "stats": parallelism_stats([t for t in traces if t["node"] == "execute_tool"]),
    }


# ------------------------------------------------------------------- sidebar
def sidebar(settings: Any, store_info: dict[str, Any]) -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Runtime")
        if settings.is_offline:
            expected = ", ".join(f"`{v}`" for v in API_KEY_VARS.values())
            st.warning(
                "No API key found, running the **deterministic offline model**. "
                "The graph, routing, tools and memory all work; only the prose is "
                f"canned. Set one of {expected} in `.env` locally, or in the app "
                "secrets when deployed.",
                icon=":material/key_off:",
            )
        else:
            st.success(f"{settings.provider} · `{settings.model}`", icon=":material/bolt:")

        st.caption(
            f"Knowledge base: {store_info['documents']} documents · "
            f"{', '.join(store_info['cities'])} · {store_info['backend']} embeddings"
        )
        st.caption(f"Thread `{st.session_state.thread_id}`")

        st.divider()
        st.markdown("### Behaviour")
        hitl = st.toggle(
            "Approve tool calls before running",
            value=False,
            help="Human-in-the-loop: the graph pauses after the model proposes tool "
            "calls so you can inspect or reject them.",
        )
        fail_weather = st.toggle("Break the weather API", value=False)
        fail_images = st.toggle("Break the image API", value=False)
        if fail_weather or fail_images:
            st.caption("Chaos mode on - the report should degrade, not crash.")

        st.divider()
        st.markdown("### Try")
        for prompt, why in EXAMPLES:
            if st.button(prompt, use_container_width=True, help=f"{why} path"):
                st.session_state.pending = prompt
                st.rerun()

        st.divider()
        if st.button("New conversation", type="primary", use_container_width=True):
            reset_conversation()
            st.rerun()

        with st.expander("Time travel"):
            st.caption(
                "Every superstep is checkpointed. Pick an earlier checkpoint to "
                "inspect the exact state the graph held at that moment."
            )
            history = list(
                get_graph(False).get_state_history(
                    {"configurable": {"thread_id": st.session_state.thread_id}}
                )
            )
            if not history:
                st.caption("No checkpoints yet - ask something first.")
            else:
                options = {
                    f"{i}: next={snap.next or ('END',)} · turn {snap.values.get('turn', 0)}": snap
                    for i, snap in enumerate(history[:25])
                }
                chosen = st.selectbox("Checkpoint", list(options), index=0)
                snapshot = options[chosen]
                st.json(
                    {
                        "next": list(snapshot.next),
                        "city": snapshot.values.get("city"),
                        "route": snapshot.values.get("route"),
                        "turn": snapshot.values.get("turn"),
                        "tool_calls": [
                            c.get("name") for c in snapshot.values.get("proposed_tool_calls", [])
                        ],
                        "checkpoint_id": snapshot.config.get("configurable", {}).get(
                            "checkpoint_id"
                        ),
                    },
                    expanded=True,
                )

    fail_tools = [
        name
        for name, broken in (
            ("get_weather_forecast", fail_weather),
            ("search_city_images", fail_images),
        )
        if broken
    ]
    return {"hitl": hitl, "fail_tools": fail_tools}


# ---------------------------------------------------------------- graph turn
def execute_turn(prompt: str, options: dict[str, Any]) -> None:
    graph = get_graph(options["hitl"])
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "fail_tools": options["fail_tools"],
        }
    }

    with st.spinner("Routing, fetching and composing..."):
        state = run(graph.ainvoke({"messages": [HumanMessage(content=prompt)]}, config))

    snapshot = graph.get_state(config)
    if snapshot.next and "execute_tool" in snapshot.next:
        # Human-in-the-loop pause: park the turn until the user decides.
        st.session_state.pending_approval = {
            "prompt": prompt,
            "tool_calls": state.get("proposed_tool_calls", []),
            "raw": state.get("raw_tool_payload", ""),
            "hitl": options["hitl"],
            "fail_tools": options["fail_tools"],
        }
        return

    st.session_state.turns.append(collect_turn(state, prompt))


def resume_turn(approved: bool, options: dict[str, Any]) -> None:
    parked = st.session_state.pop("pending_approval")
    graph = get_graph(parked["hitl"])
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "fail_tools": parked["fail_tools"],
        }
    }

    if not approved:
        # Clear the proposed calls and re-enter the dispatch edge, which will
        # then route straight to the composer.
        graph.update_state(config, {"proposed_tool_calls": []}, as_node="tool_planner")

    with st.spinner("Resuming graph..."):
        state = run(graph.ainvoke(None, config))
    st.session_state.turns.append(collect_turn(state, parked["prompt"]))


# -------------------------------------------------------------------- layout
def main() -> None:
    init_session()
    settings = get_settings()
    store_info = warm_store()
    options = sidebar(settings, store_info)

    st.title("Multi-Modal Travel Assistant")
    st.caption(
        "LangGraph decides where knowledge comes from, calls the weather and image "
        "tools in parallel, and returns one validated structured object."
    )

    for index, turn in enumerate(st.session_state.turns):
        with st.chat_message("user"):
            st.markdown(turn["user_text"])
        with st.chat_message("assistant"):
            render_report(turn["report"], key=f"turn-{index}")
            render_trace(turn, key=f"turn-{index}")

    parked = st.session_state.get("pending_approval")
    if parked:
        with st.chat_message("assistant"):
            st.info(
                "The model proposed these tool calls. Nothing has run yet.",
                icon=":material/pan_tool:",
            )
            for call in parked["tool_calls"]:
                st.markdown(f"**{call['name']}**")
                st.json(call.get("args", {}))
            approve, reject = st.columns(2)
            if approve.button("Approve and run", type="primary", use_container_width=True):
                resume_turn(True, options)
                st.rerun()
            if reject.button("Reject", use_container_width=True):
                resume_turn(False, options)
                st.rerun()

    prompt = st.chat_input("Ask about a city, e.g. 'Tell me about Kyoto'")
    if not prompt and st.session_state.get("pending"):
        prompt = st.session_state.pop("pending")

    if prompt and not parked:
        try:
            execute_turn(prompt, options)
        except Exception as exc:  # noqa: BLE001 - last-resort UI guard
            st.error(f"The turn failed: {type(exc).__name__}: {exc}")
        st.rerun()


main()
