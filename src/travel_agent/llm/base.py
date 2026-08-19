"""Provider-agnostic chat contract.

Talks to each vendor's wire format directly rather than going through
langchain.chat_models. Provider differences stay explicit instead of hidden
behind a wrapper, and the tool-calling payloads remain inspectable.

LangChain message objects are still the transport type, because LangGraph's
add_messages reducer knows how to merge them.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation request as emitted by the model."""

    id: str
    name: str
    args: dict[str, Any]

    def to_langchain(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": self.args, "type": "tool_call"}


@dataclass(slots=True)
class ChatResult:
    """Normalised model response plus the untouched provider payload.

    ``raw`` is surfaced in the UI so a reviewer can see the actual
    ``tool_calls`` JSON the model produced, not a framework's rendering of it.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def to_ai_message(self) -> AIMessage:
        return AIMessage(
            content=self.text,
            tool_calls=[tc.to_langchain() for tc in self.tool_calls],
            response_metadata={"model": self.model, "finish_reason": self.finish_reason},
        )

    def json_payload(self) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from the response text."""
        return extract_json(self.text)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model text, tolerating fences and prose."""
    if not text:
        return {}
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost balanced {...} span.
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def to_openai_messages(messages: Sequence[AnyMessage]) -> list[dict[str, Any]]:
    """LangChain messages -> OpenAI/Groq chat-completions wire format."""
    wire: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            wire.append({"role": "system", "content": _as_text(msg.content)})
        elif isinstance(msg, HumanMessage):
            wire.append({"role": "user", "content": _as_text(msg.content)})
        elif isinstance(msg, ToolMessage):
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": _as_text(msg.content),
                }
            )
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": _as_text(msg.content) or None}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            wire.append(entry)
    return wire


def to_anthropic_messages(
    messages: Sequence[AnyMessage],
) -> tuple[str, list[dict[str, Any]]]:
    """LangChain messages -> (system prompt, Anthropic messages)."""
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(_as_text(msg.content))
        elif isinstance(msg, HumanMessage):
            wire.append({"role": "user", "content": _as_text(msg.content)})
        elif isinstance(msg, ToolMessage):
            wire.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": _as_text(msg.content),
                        }
                    ],
                }
            )
        elif isinstance(msg, AIMessage):
            blocks: list[dict[str, Any]] = []
            text = _as_text(msg.content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("args", {}),
                    }
                )
            if blocks:
                wire.append({"role": "assistant", "content": blocks})
    return "\n\n".join(p for p in system_parts if p), wire


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


class BaseChatClient(ABC):
    """Minimal surface the graph needs from any provider."""

    name: str = "base"

    def __init__(self, model: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        json_object: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Single completion. Implementations must never raise for tool parsing."""

    async def achat(self, messages: Sequence[AnyMessage], **kwargs: Any) -> ChatResult:
        """Default async: offload the blocking SDK call to a worker thread."""
        return await asyncio.to_thread(lambda: self.chat(messages, **kwargs))
