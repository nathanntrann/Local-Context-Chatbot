"""API routes for InspectAssist."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from inspect_assist.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StatsResponse,
    ToolInfo,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse()


@router.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, body: ChatRequest):
    orchestrator = request.app.state.orchestrator
    response_text, conversation_id = await orchestrator.chat(
        user_message=body.message,
        conversation_id=body.conversation_id,
    )
    return ChatResponse(response=response_text, conversation_id=conversation_id)


@router.get("/api/v1/tools", response_model=list[ToolInfo], tags=["system"])
async def list_tools(request: Request):
    registry = request.app.state.tool_registry
    return [
        ToolInfo(name=t.name, description=t.description)
        for t in registry.all_tools
    ]


@router.get("/api/v1/stats", response_model=StatsResponse, tags=["system"])
async def stats(request: Request):
    orchestrator = request.app.state.orchestrator
    data = orchestrator.get_stats()
    return StatsResponse(**data)
