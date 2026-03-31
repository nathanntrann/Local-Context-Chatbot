"""OpenAI and Azure OpenAI LLM provider implementations."""

from __future__ import annotations

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
    """Unified provider for both OpenAI and Azure OpenAI APIs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.llm_provider == LLMProvider.AZURE_OPENAI:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
            self._model = settings.azure_openai_deployment
        else:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_model

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
