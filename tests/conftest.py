from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Tests must never touch a provider or the network: force the deterministic
# client and a fast, isolated vector store before anything imports config.
os.environ["LLM_PROVIDER"] = "offline"
os.environ["USE_LIVE_APIS"] = "false"
os.environ["MOCK_LATENCY"] = "0.05"
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ.setdefault("CHROMA_PATH", str(ROOT / "data" / "chroma-test"))

from travel_agent import bootstrap  # noqa: E402

bootstrap()


@pytest.fixture(scope="session")
def store():
    from travel_agent.knowledge.store import get_store

    return get_store()


@pytest.fixture
def graph():
    from langgraph.checkpoint.memory import MemorySaver

    from travel_agent.graph.builder import build_graph

    # A fresh checkpointer per test keeps threads from leaking between cases.
    return build_graph(checkpointer=MemorySaver())
