"""Abstract LLM provider protocol and message types."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ImageContent:
    """An image to send to a vision-capable model."""

    base64_data: str
    media_type: str = "image/png"

    @classmethod
    def from_path(cls, path: Path, *, max_size_px: int = 1024) -> ImageContent:
        from PIL import Image
        import io

        img = Image.open(path)
        # Resize if needed, preserving aspect ratio
        if max(img.size) > max_size_px:
            img.thumbnail((max_size_px, max_size_px), Image.LANCZOS)

        buf = io.BytesIO()
        fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        media = "image/png" if fmt == "PNG" else "image/jpeg"
        img.save(buf, format=fmt)
        return cls(base64_data=base64.b64encode(buf.getvalue()).decode(), media_type=media)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCallRequest] | None = None
    images: list[ImageContent] | None = None
    name: str | None = None


@dataclass
class ToolCallRequest:
    id: str
    function_name: str
    arguments_json: str


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@runtime_checkable
class LLMProviderProtocol(Protocol):
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse: ...
