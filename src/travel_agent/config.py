"""Typed, single-source-of-truth configuration read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

Provider = Literal["groq", "openai", "anthropic", "offline"]

ROOT = Path(__file__).resolve().parents[2]

# Default model per provider. LLM_PROVIDER switches between them with no code
# change; gpt-oss-120b is OpenAI's open-weight model served on Groq.
DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "offline": "deterministic-stub",
}

API_KEY_VARS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: Provider
    model: str
    api_key: str | None
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout: float = 60.0

    # knowledge base
    chroma_path: Path = field(default_factory=lambda: ROOT / "data" / "chroma")
    collection: str = "city-knowledge"
    embedding_backend: Literal["hashing", "minilm"] = "hashing"
    #: cosine similarity below which the vector store is considered a miss and
    #: the graph routes to web search instead.
    similarity_floor: float = 0.35

    # tool behaviour
    use_live_apis: bool = False
    mock_latency: float = 1.0
    forecast_days: int = 7

    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve settings once per process.

    Provider resolution order: explicit ``LLM_PROVIDER`` wins; otherwise the
    first provider with a key present; otherwise the deterministic offline
    client so the graph is always runnable (demo, CI, no-key laptop).
    """
    requested = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    provider: Provider | None = requested if requested in DEFAULT_MODELS else None

    if provider is None or provider == "offline":
        if provider is None:
            for candidate, var in API_KEY_VARS.items():
                if os.getenv(var):
                    provider = candidate  # type: ignore[assignment]
                    break
    if provider is None:
        provider = "offline"

    key = os.getenv(API_KEY_VARS.get(provider, ""), None)
    if provider != "offline" and not key:
        provider = "offline"
        key = None

    backend = (os.getenv("EMBEDDING_BACKEND") or "hashing").strip().lower()
    if backend not in {"hashing", "minilm"}:
        backend = "hashing"

    return Settings(
        provider=provider,  # type: ignore[arg-type]
        model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
        api_key=key,
        temperature=_num("LLM_TEMPERATURE", 0.2),
        max_tokens=int(_num("LLM_MAX_TOKENS", 2048)),
        request_timeout=_num("LLM_TIMEOUT", 60.0),
        chroma_path=Path(os.getenv("CHROMA_PATH") or (ROOT / "data" / "chroma")),
        collection=os.getenv("CHROMA_COLLECTION") or "city-knowledge",
        embedding_backend=backend,  # type: ignore[arg-type]
        similarity_floor=_num("SIMILARITY_FLOOR", 0.35),
        use_live_apis=_flag("USE_LIVE_APIS", False),
        mock_latency=_num("MOCK_LATENCY", 1.0),
        forecast_days=int(_num("FORECAST_DAYS", 7)),
    )
