"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None


class Attachment(BaseModel):
    type: str = "image"
    data: str  # base64-encoded
    media_type: str = "image/png"
    label: str = ""


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    provider: str = ""
    data_locality: str = ""  # "local" or "cloud"
    attachments: list[Attachment] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    model_tier: str = ""  # "fast" or "strong" when routing is enabled


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class StatsResponse(BaseModel):
    active_conversations: int
    total_conversations: int
    persisted_conversations: int | None = None


class ToolInfo(BaseModel):
    name: str
    description: str


class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    active: bool = False


class ModelSwitchRequest(BaseModel):
    provider: str = Field(..., pattern="^(ollama|openai|azure_openai|anthropic)$")
    model: str = Field(..., min_length=1, max_length=200)
    api_key: str | None = None
    endpoint: str | None = None


class ModelSwitchResponse(BaseModel):
    provider: str
    model: str
    status: str = "ok"


class ConversationSummary(BaseModel):
    id: str
    title: str
    model: str = ""
    created_at: str
    updated_at: str


class ConversationDetail(BaseModel):
    id: str
    title: str
    model: str = ""
    created_at: str
    updated_at: str
    messages: list[dict] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: bool


class FeedbackRequest(BaseModel):
    message_index: int = Field(..., ge=0)
    rating: int = Field(..., ge=-1, le=1)  # 1 = helpful, -1 = not helpful


class FeedbackResponse(BaseModel):
    id: int
    status: str = "ok"


class FeedbackSummary(BaseModel):
    total_feedback: int = 0
    positive: int = 0
    negative: int = 0
    satisfaction_rate: float = 0.0
    recent_negative_queries: list[dict] = Field(default_factory=list)
