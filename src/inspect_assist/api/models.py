"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class StatsResponse(BaseModel):
    active_conversations: int
    total_conversations: int


class ToolInfo(BaseModel):
    name: str
    description: str
