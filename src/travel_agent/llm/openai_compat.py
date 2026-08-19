"""Groq and OpenAI share the chat-completions wire format, so they share code.

Groq's free tier is served through an OpenAI-compatible endpoint; the only
differences are the SDK class, the default model, and which structured-output
mode the model supports. The JSON ladder below degrades gracefully:
``json_schema`` -> ``json_object`` -> prompt-only, so a model that lacks strict
structured output still returns parseable JSON.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import AnyMessage

from .base import BaseChatClient, ChatResult, ToolCall, to_openai_messages


class OpenAICompatClient(BaseChatClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        flavour: str = "groq",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self.name = flavour
        self._supports_json_schema = True

        if flavour == "groq":
            from groq import Groq

            self._client: Any = Groq(api_key=api_key, timeout=timeout)
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, timeout=timeout)

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: Sequence[AnyMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        json_object: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            if tool_choice:
                payload["tool_choice"] = (
                    tool_choice
                    if tool_choice in {"auto", "none", "required"}
                    else {"type": "function", "function": {"name": tool_choice}}
                )

        response = self._request(payload, json_object=json_object, json_schema=json_schema)
        return self._parse(response)

    def _request(
        self,
        payload: dict[str, Any],
        *,
        json_object: bool,
        json_schema: dict[str, Any] | None,
    ) -> Any:
        """Try the strictest structured-output mode the endpoint accepts."""
        attempts: list[dict[str, Any] | None] = []
        if json_schema and self._supports_json_schema:
            attempts.append(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": json_schema, "strict": False},
                }
            )
        if json_schema or json_object:
            attempts.append({"type": "json_object"})
        attempts.append(None)

        last_error: Exception | None = None
        for response_format in attempts:
            body = dict(payload)
            if response_format is not None:
                body["response_format"] = response_format
            try:
                return self._client.chat.completions.create(**body)
            except Exception as exc:  # noqa: BLE001 - degrade instead of killing the graph
                last_error = exc
                if response_format and response_format.get("type") == "json_schema":
                    self._supports_json_schema = False
                continue
        raise RuntimeError(f"{self.name} request failed: {last_error}") from last_error

    @staticmethod
    def _parse(response: Any) -> ChatResult:
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            # Arguments arrive as a JSON string on the wire, not as an object.
            raw_args = raw_call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"__unparsed__": raw_args}
            calls.append(ToolCall(id=raw_call.id, name=raw_call.function.name, args=args))

        try:
            raw_dump = response.model_dump()
        except Exception:  # pragma: no cover
            raw_dump = {}

        usage = raw_dump.get("usage") or {}
        return ChatResult(
            text=message.content or "",
            tool_calls=calls,
            raw=raw_dump,
            model=getattr(response, "model", ""),
            finish_reason=getattr(choice, "finish_reason", "") or "",
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        )
