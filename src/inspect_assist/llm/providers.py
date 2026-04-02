"""OpenAI and Azure OpenAI LLM provider implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from inspect_assist.config import LLMProvider, Settings
from inspect_assist.llm import (
    ImageContent,
    LLMResponse,
    Message,
    Role,
    ToolCallRequest,
)


class OpenAIProvider:
    """Unified provider for OpenAI, Azure OpenAI, and Ollama APIs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.llm_provider == LLMProvider.AZURE_OPENAI:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            self._model = settings.azure_openai_deployment
        elif settings.llm_provider == LLMProvider.OLLAMA:
            self._client = AsyncOpenAI(
                base_url=settings.ollama_base_url,
                api_key="ollama",  # Ollama doesn't need a real key
            )
            self._model = settings.ollama_model
        else:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_model

    @property
    def provider_name(self) -> str:
        return self._settings.llm_provider.value

    @property
    def data_locality(self) -> str:
        return "local" if self._settings.llm_provider == LLMProvider.OLLAMA else "cloud"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        api_messages = [self._convert_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCallRequest] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        function_name=tc.function.name,
                        arguments_json=tc.function.arguments,
                    )
                )

        usage_dict: dict[str, int] = {}
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage=usage_dict,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream tokens as strings, then yield a final LLMResponse with tool calls (if any)."""
        api_messages = [self._convert_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)

        content_buffer = ""
        tool_calls_buffer: dict[int, dict[str, str]] = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # Text content
            if delta.content:
                content_buffer += delta.content
                yield delta.content

            # Accumulate tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": "",
                            "function_name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_buffer[idx]["function_name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc.function.arguments

        # Build final tool call list
        tool_calls: list[ToolCallRequest] = []
        for idx in sorted(tool_calls_buffer.keys()):
            buf = tool_calls_buffer[idx]
            tool_calls.append(
                ToolCallRequest(
                    id=buf["id"],
                    function_name=buf["function_name"],
                    arguments_json=buf["arguments"],
                )
            )

        # Final yield: complete response summary
        yield LLMResponse(
            content=content_buffer,
            tool_calls=tool_calls,
            usage={},  # streaming doesn't return usage in chunks
        )

    @staticmethod
    def _convert_message(m: Message) -> dict[str, Any]:
        if m.role == Role.TOOL:
            return {
                "role": "tool",
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            }

        if m.role == Role.ASSISTANT and m.tool_calls:
            return {
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function_name,
                            "arguments": tc.arguments_json,
                        },
                    }
                    for tc in m.tool_calls
                ],
            }

        # Handle images for vision
        if m.images:
            content_parts: list[dict[str, Any]] = []
            if m.content:
                content_parts.append({"type": "text", "text": m.content})
            for img in m.images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.media_type};base64,{img.base64_data}",
                            "detail": "low",
                        },
                    }
                )
            return {"role": m.role.value, "content": content_parts}

        return {"role": m.role.value, "content": m.content}


def create_llm_provider(settings: Settings) -> OpenAIProvider:
    return OpenAIProvider(settings)
