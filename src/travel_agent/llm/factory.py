"""Build the chat client for the configured provider.

Swapping providers is an env-var change, never a code change:
``LLM_PROVIDER=groq|openai|anthropic|offline``.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import Settings, get_settings
from .base import BaseChatClient
from .offline import OfflineClient


def build_client(settings: Settings | None = None) -> BaseChatClient:
    settings = settings or get_settings()

    if settings.provider == "offline" or not settings.api_key:
        return OfflineClient()

    try:
        if settings.provider in {"groq", "openai"}:
            from .openai_compat import OpenAICompatClient

            return OpenAICompatClient(
                settings.model,
                settings.api_key,
                flavour=settings.provider,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=settings.request_timeout,
            )
        if settings.provider == "anthropic":
            from .anthropic_client import AnthropicClient

            return AnthropicClient(
                settings.model,
                settings.api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=settings.request_timeout,
            )
    except Exception:  # noqa: BLE001 - a missing SDK must not break the app
        return OfflineClient()

    return OfflineClient()


@lru_cache(maxsize=4)
def get_client() -> BaseChatClient:
    """Process-wide singleton; SDK clients hold connection pools worth reusing."""
    return build_client()
