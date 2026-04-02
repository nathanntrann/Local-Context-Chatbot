"""API routes for InspectAssist."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inspect_assist.api.models import (
    Attachment,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    DeleteResponse,
    HealthResponse,
    ModelInfo,
    ModelSwitchRequest,
    ModelSwitchResponse,
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
    result = await orchestrator.chat(
        user_message=body.message,
        conversation_id=body.conversation_id,
    )
    return ChatResponse(
        response=result.response,
        conversation_id=result.conversation_id,
        provider=result.provider,
        data_locality=result.data_locality,
        attachments=[Attachment(**a) for a in result.attachments],
        suggestions=result.suggestions,
    )


@router.post("/api/v1/chat/stream", tags=["chat"])
async def chat_stream(request: Request, body: ChatRequest):
    """Stream chat response tokens via Server-Sent Events."""
    orchestrator = request.app.state.orchestrator
    return StreamingResponse(
        orchestrator.chat_stream(
            user_message=body.message,
            conversation_id=body.conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
    data = await orchestrator.get_stats()
    return StatsResponse(**data)


@router.get("/api/v1/models", response_model=list[ModelInfo], tags=["models"])
async def list_models(request: Request):
    """List available models — local Ollama models + cloud provider options."""
    settings = request.app.state.settings
    current_provider = settings.llm_provider.value
    current_model = _get_current_model(settings)

    models: list[ModelInfo] = []

    # Fetch local Ollama models
    ollama_url = settings.ollama_base_url.replace("/v1", "")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    model_name = m["name"]
                    models.append(ModelInfo(
                        id=f"ollama/{model_name}",
                        name=model_name,
                        provider="ollama",
                        active=(current_provider == "ollama" and current_model == model_name),
                    ))
    except httpx.ConnectError:
        pass  # Ollama not running — skip local models

    # Cloud provider options (always shown, user provides key in UI)
    cloud_models = [
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("azure_openai", "gpt-4o"),
    ]
    for provider, model_name in cloud_models:
        models.append(ModelInfo(
            id=f"{provider}/{model_name}",
            name=model_name,
            provider=provider,
            active=(current_provider == provider and current_model == model_name),
        ))

    return models


@router.post("/api/v1/models/switch", response_model=ModelSwitchResponse, tags=["models"])
async def switch_model(request: Request, body: ModelSwitchRequest):
    """Switch the active LLM model at runtime."""
    from inspect_assist.config import LLMProvider, Settings
    from inspect_assist.llm.providers import create_llm_provider

    settings: Settings = request.app.state.settings

    # Cloud providers require an API key
    if body.provider == "openai" and not body.api_key:
        raise HTTPException(status_code=400, detail="API key required for OpenAI")
    if body.provider == "azure_openai" and (not body.api_key or not body.endpoint):
        raise HTTPException(status_code=400, detail="API key and endpoint required for Azure OpenAI")

    # Update settings for the new provider
    settings.llm_provider = LLMProvider(body.provider)

    if body.provider == "ollama":
        settings.ollama_model = body.model
    elif body.provider == "openai":
        settings.openai_model = body.model
        settings.openai_api_key = body.api_key or ""
    elif body.provider == "azure_openai":
        settings.azure_openai_deployment = body.model
        settings.azure_openai_api_key = body.api_key or ""
        settings.azure_openai_endpoint = body.endpoint or ""

    # Rebuild the LLM provider and rewire
    new_llm = create_llm_provider(settings)
    orchestrator = request.app.state.orchestrator
    orchestrator._llm = new_llm

    # Update vision tools with new LLM
    from inspect_assist.tools import vision_tools
    vision_tools._llm_provider = new_llm

    return ModelSwitchResponse(provider=body.provider, model=body.model)


@router.get("/api/v1/conversations", response_model=list[ConversationSummary], tags=["conversations"])
async def list_conversations(request: Request):
    """List all persisted conversations."""
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        return []
    rows = await store.list_conversations()
    return [ConversationSummary(**r) for r in rows]


@router.get("/api/v1/conversations/search", response_model=list[ConversationSummary], tags=["conversations"])
async def search_conversations(request: Request, q: str = ""):
    """Full-text search across persisted conversations."""
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        return []
    if not q.strip():
        return []
    rows = await store.search(q.strip())
    return [ConversationSummary(**r) for r in rows]


@router.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetail, tags=["conversations"])
async def get_conversation(request: Request, conversation_id: str):
    """Load a persisted conversation with its messages."""
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(status_code=404, detail="Persistence not enabled")
    conv = await store.load_detail(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=conv["id"],
        title=conv["title"],
        model=conv["model"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=[{"role": m["role"], "content": m["content"]} for m in conv["messages"]],
    )


@router.delete("/api/v1/conversations/{conversation_id}", response_model=DeleteResponse, tags=["conversations"])
async def delete_conversation(request: Request, conversation_id: str):
    """Delete a persisted conversation."""
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(status_code=404, detail="Persistence not enabled")
    deleted = await store.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return DeleteResponse(deleted=True)


@router.get("/api/v1/conversations/{conversation_id}/export", tags=["conversations"])
async def export_conversation(request: Request, conversation_id: str):
    """Export a conversation as a downloadable JSON report."""
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(status_code=404, detail="Persistence not enabled")
    conv = await store.load_detail(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    export = {
        "export_type": "conversation_report",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "application": "InspectAssist",
        "conversation": {
            "id": conv["id"],
            "title": conv["title"],
            "model": conv["model"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "message_count": len(conv["messages"]),
            "messages": conv["messages"],
        },
    }

    filename = f"inspectassist_export_{conversation_id[:8]}.json"
    return JSONResponse(
        content=export,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def _get_current_model(settings) -> str:
    from inspect_assist.config import LLMProvider
    if settings.llm_provider == LLMProvider.OLLAMA:
        return settings.ollama_model
    elif settings.llm_provider == LLMProvider.OPENAI:
        return settings.openai_model
    else:
        return settings.azure_openai_deployment
