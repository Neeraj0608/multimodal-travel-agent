"""Anthropic Messages API client behind the same normalised contract."""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import AnyMessage

from .base import BaseChatClient, ChatResult, ToolCall, to_anthropic_messages


class AnthropicClient(BaseChatClient):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        json_object: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        system, wire = to_anthropic_messages(messages)
        if json_schema or json_object:
            # Anthropic has no response_format parameter; steer via the system
            # prompt and let ChatResult.json_payload() handle extraction.
            system = (system + "\n\nRespond with a single JSON object and nothing else.").strip()

        payload: dict[str, Any] = {
            "model": self.model,
            "system": system or "You are a helpful assistant.",
            "messages": wire,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
            if tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice and tool_choice not in {"auto", "none"}:
                payload["tool_choice"] = {"type": "tool", "name": tool_choice}

        response = self._client.messages.create(**payload)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = (
                    block.input
                    if isinstance(block.input, dict)
                    else json.loads(block.input or "{}")
                )
                calls.append(ToolCall(id=block.id, name=block.name, args=args))

        try:
            raw_dump = response.model_dump()
        except Exception:  # pragma: no cover
            raw_dump = {}

        return ChatResult(
            text="".join(text_parts),
            tool_calls=calls,
            raw=raw_dump,
            model=response.model,
            finish_reason=response.stop_reason or "",
            usage={
                "prompt_tokens": getattr(response.usage, "input_tokens", None),
                "completion_tokens": getattr(response.usage, "output_tokens", None),
            },
        )
